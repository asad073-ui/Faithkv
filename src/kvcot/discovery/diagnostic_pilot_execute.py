"""Coordinator, dry-run preflight, execution, and reconstruction."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid
from typing import Any

from kvcot.discovery.attempt_artifacts import atomic_write_json, atomic_write_text, sha256_file
from kvcot.discovery.diagnostic_pilot_authorization import (
    claim_authorization_once,
    parse_authorization_document,
)
from kvcot.discovery.diagnostic_pilot_contract import (
    BRANCH,
    CANDIDATE_MANIFEST_CANONICAL_SHA256,
    CONFIG_PATH,
    EXACT_EXECUTION_COMMAND,
    MAXIMUM_QUALIFICATION_CANDIDATES,
    MAXIMUM_SELECTED_EXAMPLES,
    MODEL_REVISION,
    PROTOCOL_DOCUMENT_PATH,
    RKV_REVISION,
    RUNTIME_LIMIT_SECONDS,
    SWAP_GAIN_THRESHOLD_NATS,
    TOKENIZER_REVISION,
    VRAM_LIMIT_BYTES,
    CLASSIFICATION_TEXT,
    attach_canonical_hash,
    classify_pilot,
    material_margin_change,
)
from kvcot.discovery.diagnostic_pilot_manifest import (
    bounded_candidate_upper_bound,
    mechanically_qualifies,
    select_first_three_qualified,
)
from kvcot.discovery.diagnostic_pilot_prepare import verify_runtime_inputs


class DiagnosticExecutionRefused(RuntimeError):
    pass


def _git(repository_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repository_root, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise DiagnosticExecutionRefused(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _preflight(repository_root: Path, authorization: Any) -> dict[str, Any]:
    payload = authorization.payload
    runtime = verify_runtime_inputs(payload["runtime_config_path"])
    head = _git(repository_root, "rev-parse", "HEAD")
    remote_head = _git(repository_root, "rev-parse", f"origin/{BRANCH}")
    branch = _git(repository_root, "branch", "--show-current")
    status = _git(repository_root, "status", "--porcelain=v1", "--untracked-files=all")
    parent = _git(repository_root, "rev-parse", "HEAD^")
    changed = _git(repository_root, "diff", "--name-only", "HEAD^", "HEAD").splitlines()
    rkv = _git(repository_root / "third_party/R-KV", "rev-parse", "HEAD")
    if branch != BRANCH:
        raise DiagnosticExecutionRefused(f"branch {branch!r} does not match {BRANCH!r}")
    if remote_head != head:
        raise DiagnosticExecutionRefused("local HEAD does not match the fetched remote branch")
    if status:
        raise DiagnosticExecutionRefused("worktree is not clean")
    if parent != payload["authorized_implementation_sha"]:
        raise DiagnosticExecutionRefused("authorization commit parent is not the authorized implementation SHA")
    expected_document = Path(authorization.document_path).resolve().relative_to(repository_root).as_posix()
    if changed != [expected_document]:
        raise DiagnosticExecutionRefused("authorization commit must change only the execution authorization document")
    if rkv != RKV_REVISION:
        raise DiagnosticExecutionRefused("R-KV submodule revision mismatch")
    if runtime["canonical_sha256"] != payload["runtime_config_canonical_sha256"]:
        raise DiagnosticExecutionRefused("runtime-config canonical hash mismatch")
    if runtime["protocol_document_sha256"] != payload["protocol_document_sha256"]:
        raise DiagnosticExecutionRefused("protocol document hash mismatch")
    if payload["candidate_manifest_canonical_sha256"] != CANDIDATE_MANIFEST_CANONICAL_SHA256:
        raise DiagnosticExecutionRefused("candidate-manifest canonical hash mismatch")
    if payload["model_revision"] != MODEL_REVISION or payload["tokenizer_revision"] != TOKENIZER_REVISION:
        raise DiagnosticExecutionRefused("model/tokenizer revision mismatch")
    if Path(payload["claim_path"]).exists():
        raise DiagnosticExecutionRefused("one-use authorization claim already exists")
    output_root = Path(payload["output_root"])
    attempt_prefix = f"diagnostic-pilot-attempt-{payload['authorization_id']}-"
    existing_attempts = sorted(path for path in output_root.glob(f"{attempt_prefix}*") if path.is_dir()) if output_root.exists() else []
    if existing_attempts:
        raise DiagnosticExecutionRefused("authorized output attempt already exists")
    return {
        "passed": True,
        "repository_head": head,
        "remote_branch_head": remote_head,
        "authorization_commit_parent": parent,
        "branch": branch,
        "worktree_clean": True,
        "authorization_only_changed_path": expected_document,
        "rkv_revision": rkv,
        "runtime_config_canonical_sha256": runtime["canonical_sha256"],
        "model_snapshot_path": runtime["model_snapshot_path"],
        "tokenizer_snapshot_path": runtime["tokenizer_snapshot_path"],
        "claim_absent": True,
        "attempt_absent": True,
        "would_initialize_cuda": False,
        "would_load_model_weights": False,
        "would_consume_claim": False,
        "would_create_attempt": False,
    }


def dry_run_diagnostic_pilot(
    *, repository_root: str | Path, authorization_document: str | Path
) -> dict[str, Any]:
    repository_root = Path(repository_root).resolve()
    authorization = parse_authorization_document(authorization_document)
    return _preflight(repository_root, authorization)


def _worker_env(attempt_id: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": "13",
            "CUDA_VISIBLE_DEVICES": "0",
            "KVCOT_DIAGNOSTIC_ATTEMPT_ID": attempt_id,
        }
    )
    return env


def _launch_worker(
    *, role: str, ordinal: int, output: Path, config_path: Path, prompts_path: Path,
    attempt_id: str, timeout: float, fullkv_result: Path | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "kvcot.discovery.diagnostic_pilot_worker_entry",
        "--role",
        role,
        "--config",
        str(config_path),
        "--prompts",
        str(prompts_path),
        "--candidate-ordinal",
        str(ordinal),
        "--output",
        str(output),
    ]
    if fullkv_result is not None:
        command.extend(["--fullkv-result", str(fullkv_result)])
    completed = subprocess.run(
        command,
        env=_worker_env(attempt_id),
        text=True,
        capture_output=True,
        timeout=max(1.0, timeout),
        check=False,
    )
    atomic_write_text(output.with_suffix(".stdout.log"), completed.stdout)
    atomic_write_text(output.with_suffix(".stderr.log"), completed.stderr)
    if completed.returncode != 0 or not output.is_file():
        raise DiagnosticExecutionRefused(
            f"{role} worker failed for candidate {ordinal}; retry is permanently prohibited"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _attempt_references(attempt: Path) -> dict[str, Any]:
    references = []
    for path in sorted(attempt.rglob("*")):
        if path.is_file() and path.name != "final.json":
            references.append(
                {
                    "relative_path": path.relative_to(attempt).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {"files": references}


def _pair_population_valid(result: dict[str, Any]) -> bool:
    selected_event = result.get("selected_event") or {}
    pool = selected_event.get("candidate_pool") or []
    pool_positions = [row.get("absolute_token_position") for row in pool]
    pairs = result.get("pair_records") or []
    arm_a = [pair for pair in pairs if pair.get("arm") == "candidate_upper_bound"]
    arm_b = [pair for pair in pairs if pair.get("arm") == "restore_width"]
    noop = [pair for pair in pairs if pair.get("arm") == "no_op"]
    if [pair.get("candidate_absolute_position") for pair in arm_a] != pool_positions:
        return False
    if len(arm_b) != 1 or len(noop) != 1 or not pool_positions:
        return False
    if arm_b[0].get("candidate_absolute_position") != pool_positions[0]:
        return False
    if arm_b[0].get("restore_width") != 2 or arm_b[0].get("kv_head_indices") != [0, 1]:
        return False
    if any(
        pair.get("restore_width") != 1
        or pair.get("diagnostic_only") is not True
        or pair.get("deployable_performance") is not False
        for pair in arm_a
    ):
        return False
    donor = selected_event.get("donor_absolute_position")
    if noop[0].get("candidate_absolute_position") != donor or noop[0].get("donor_absolute_position") != donor:
        return False
    for pair in pairs:
        baseline_ids = pair.get("baseline_scored_token_ids")
        intervention_ids = pair.get("intervention_scored_token_ids")
        baseline_nll = pair.get("baseline_per_token_nll")
        intervention_nll = pair.get("intervention_per_token_nll")
        if (
            not isinstance(baseline_ids, list)
            or baseline_ids != intervention_ids
            or len(baseline_ids) != 48
            or not isinstance(baseline_nll, list)
            or not isinstance(intervention_nll, list)
            or len(baseline_nll) != 48
            or len(intervention_nll) != 48
        ):
            return False
        baseline_mean = sum(float(value) for value in baseline_nll) / len(baseline_nll)
        intervention_mean = sum(float(value) for value in intervention_nll) / len(intervention_nll)
        if pair.get("baseline_mean_nll") != baseline_mean:
            return False
        if pair.get("intervention_mean_nll") != intervention_mean:
            return False
        if pair.get("swap_gain") != baseline_mean - intervention_mean:
            return False
        before = {
            row["absolute_position"]: row["margin"]
            for row in pair.get("baseline_answer_token_margins", [])
        }
        after = {
            row["absolute_position"]: row["margin"]
            for row in pair.get("intervention_answer_token_margins", [])
        }
        reconstructed_margin_change = any(
            material_margin_change(before.get(position), after.get(position))
            for position in set(before) | set(after)
        )
        if pair.get("answer_margin_sign_change") is not reconstructed_margin_change:
            return False
        extracted_changed = (
            pair.get("baseline_fixed_trace_extracted_answer")
            != pair.get("intervention_fixed_trace_extracted_answer")
        )
        correctness_changed = (
            pair.get("baseline_fixed_trace_correctness_status")
            != pair.get("intervention_fixed_trace_correctness_status")
        )
        if pair.get("fixed_trace_extracted_answer_change") is not extracted_changed:
            return False
        if pair.get("fixed_trace_correctness_change") is not correctness_changed:
            return False
    no_op = noop[0]
    return (
        no_op.get("baseline_per_token_nll") == no_op.get("intervention_per_token_nll")
        and no_op.get("swap_gain") == 0.0
        and no_op.get("baseline_final_state_sha256") == no_op.get("intervention_final_state_sha256")
        and (no_op.get("mutation") or {}).get("is_noop") is True
    )


def _summarize(qualification_rows: list[dict[str, Any]], rkv_results: list[dict[str, Any]]) -> dict[str, Any]:
    selected = select_first_three_qualified(qualification_rows)
    selected_ordinals = {row["candidate_ordinal"] for row in selected}
    selected_results = [row for row in rkv_results if row["candidate_ordinal"] in selected_ordinals]
    all_pairs = [pair for result in selected_results for pair in result["pair_records"]]
    arm_a_records = [pair for pair in all_pairs if pair["arm"] == "candidate_upper_bound"]
    arm_b_records = [pair for pair in all_pairs if pair["arm"] == "restore_width"]
    noop_records = [pair for pair in all_pairs if pair["arm"] == "no_op"]
    arm_a_by_example: list[float] = []
    for result in selected_results:
        candidates = [pair for pair in result["pair_records"] if pair["arm"] == "candidate_upper_bound"]
        arm_a_by_example.append(bounded_candidate_upper_bound(candidates))
    arm_b_gains = [float(pair["swap_gain"]) for pair in arm_b_records]
    behavioural_change = any(
        pair["answer_margin_sign_change"]
        or pair["fixed_trace_extracted_answer_change"]
        or pair["fixed_trace_correctness_change"]
        for pair in arm_a_records + arm_b_records
    )
    primitive_populations_valid = all(_pair_population_valid(result) for result in selected_results)
    noop_exact = (
        len(noop_records) == len(selected_results)
        and all(result["noop_exact"] is True for result in selected_results)
        and primitive_populations_valid
    )
    expected_pairs = sum(len(result["selected_event"]["candidate_pool"]) + 2 for result in selected_results)
    complete = (
        len(selected_results) == len(selected)
        and len(all_pairs) == expected_pairs
        and primitive_populations_valid
    )
    classification = classify_pilot(
        qualified_examples=len(selected),
        arm_a_gains=arm_a_by_example,
        arm_b_gains=arm_b_gains,
        behavioural_change=behavioural_change,
        noop_exact=noop_exact,
        complete=complete,
    )
    return attach_canonical_hash(
        {
            "artifact_schema_version": "faithkv-post-stage-c-diagnostic-pilot-summary-v1",
            "qualified_example_count": len(selected),
            "selected_candidate_ordinals": sorted(selected_ordinals),
            "selected_unique_ids": [row["unique_id"] for row in selected],
            "completed_arm_a_pairs": len(arm_a_records),
            "completed_arm_b_pairs": len(arm_b_records),
            "completed_noop_pairs": len(noop_records),
            "expected_total_pairs": expected_pairs,
            "completed_total_pairs": len(all_pairs),
            "noop_exact": noop_exact,
            "primitive_populations_reconstructed": primitive_populations_valid,
            "arm_a_bounded_upper_bounds": arm_a_by_example,
            "arm_b_gains": arm_b_gains,
            "largest_arm_a_gain": max(arm_a_by_example) if arm_a_by_example else None,
            "largest_arm_b_gain": max(arm_b_gains) if arm_b_gains else None,
            "arm_a_gains_above_0_01": sum(value > SWAP_GAIN_THRESHOLD_NATS for value in arm_a_by_example),
            "arm_b_gains_above_0_01": sum(value > SWAP_GAIN_THRESHOLD_NATS for value in arm_b_gains),
            "behavioural_change": behavioural_change,
            "answer_margin_sign_changes": sum(pair["answer_margin_sign_change"] for pair in arm_a_records + arm_b_records),
            "fixed_trace_answer_flips": 0,
            "correctness_flips": 0,
            "classification": classification.value,
            "classification_text": CLASSIFICATION_TEXT[classification],
            "scale_limitation": "The result does not confirm or refute the immutable 8B R2 null.",
            "b2b_status": "blocked",
        }
    )


def run_diagnostic_pilot(
    *, repository_root: str | Path, authorization_document: str | Path
) -> dict[str, Any]:
    repository_root = Path(repository_root).resolve()
    authorization = parse_authorization_document(authorization_document)
    preflight = _preflight(repository_root, authorization)
    payload = authorization.payload
    attempt_id = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt = Path(payload["output_root"]) / (
        f"diagnostic-pilot-attempt-{payload['authorization_id']}-{timestamp}-{attempt_id}"
    )
    claim_path, claim = claim_authorization_once(
        authorization, attempt_id=attempt_id, attempt_directory=str(attempt)
    )
    started = time.perf_counter()
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "fullkv").mkdir()
    (attempt / "rkv").mkdir()
    atomic_write_json(
        attempt / "invocation.json",
        {
            "command": EXACT_EXECUTION_COMMAND,
            "authorization_document": str(authorization.document_path),
            "execute": True,
            "invocation_ordinal": 1,
            "automatic_retries": 0,
            "attempt_id": attempt_id,
        },
    )
    atomic_write_json(attempt / "authorization_claim.json", claim)
    atomic_write_json(attempt / "preflight.json", preflight)
    runtime = verify_runtime_inputs(payload["runtime_config_path"])
    atomic_write_json(
        attempt / "protocol_binding.json",
        {
            "protocol_document_path": PROTOCOL_DOCUMENT_PATH,
            "protocol_document_sha256": runtime["protocol_document_sha256"],
            "runtime_config_path": payload["runtime_config_path"],
            "runtime_config_canonical_sha256": runtime["canonical_sha256"],
            "candidate_manifest_canonical_sha256": CANDIDATE_MANIFEST_CANONICAL_SHA256,
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": TOKENIZER_REVISION,
            "rkv_revision": RKV_REVISION,
        },
    )
    atomic_write_json(
        attempt / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
    )
    qualification_rows: list[dict[str, Any]] = []
    fullkv_results: list[dict[str, Any]] = []
    rkv_results: list[dict[str, Any]] = []
    try:
        config_path = repository_root / CONFIG_PATH
        prompts_path = Path(runtime["candidate_prompts_path"])
        qualified_count = 0
        for ordinal in range(MAXIMUM_QUALIFICATION_CANDIDATES):
            if qualified_count >= MAXIMUM_SELECTED_EXAMPLES:
                break
            remaining = RUNTIME_LIMIT_SECONDS - (time.perf_counter() - started)
            if remaining <= 0:
                raise DiagnosticExecutionRefused("pilot wall-time limit exhausted")
            fullkv_path = attempt / "fullkv" / f"candidate-{ordinal}.json"
            fullkv = _launch_worker(
                role="fullkv",
                ordinal=ordinal,
                output=fullkv_path,
                config_path=config_path,
                prompts_path=prompts_path,
                attempt_id=attempt_id,
                timeout=remaining,
            )
            fullkv_peak = max(
                int(fullkv.get("peak_cuda_allocated_bytes", VRAM_LIMIT_BYTES + 1)),
                int(fullkv.get("peak_cuda_reserved_bytes", VRAM_LIMIT_BYTES + 1)),
            )
            if fullkv_peak > VRAM_LIMIT_BYTES:
                raise DiagnosticExecutionRefused("FullKV peak allocated/reserved VRAM exceeded 22 GiB")
            fullkv_results.append(fullkv)
            remaining = RUNTIME_LIMIT_SECONDS - (time.perf_counter() - started)
            rkv_path = attempt / "rkv" / f"candidate-{ordinal}.json"
            rkv = _launch_worker(
                role="diagnostic-rkv",
                ordinal=ordinal,
                output=rkv_path,
                config_path=config_path,
                prompts_path=prompts_path,
                attempt_id=attempt_id,
                timeout=remaining,
                fullkv_result=fullkv_path,
            )
            qualification = dict(rkv["qualification"])
            qualification["candidate_ordinal"] = ordinal
            qualification_rows.append(qualification)
            rkv_results.append(rkv)
            if mechanically_qualifies(qualification):
                qualified_count += 1

        summary = _summarize(qualification_rows, rkv_results)
        selected_ordinals = set(summary["selected_candidate_ordinals"])
        selected_results = [row for row in rkv_results if row["candidate_ordinal"] in selected_ordinals]
        all_pairs = [pair for row in selected_results for pair in row["pair_records"]]
        atomic_write_json(attempt / "qualification.json", qualification_rows)
        atomic_write_json(
            attempt / "selected_examples.json",
            [
                {
                    "candidate_ordinal": row["candidate_ordinal"],
                    "unique_id": row["unique_id"],
                    "selected_event": row["selected_event"],
                }
                for row in selected_results
            ],
        )
        atomic_write_json(
            attempt / "candidate_pools.json",
            [row["selected_event"] for row in selected_results],
        )
        atomic_write_json(attempt / "pair_records.json", all_pairs)
        atomic_write_json(
            attempt / "behavioural_readouts.json",
            [
                {
                    "candidate_ordinal": row["candidate_ordinal"],
                    "unique_id": row["unique_id"],
                    "fixed_trace_extracted_answer": row["fixed_trace_extracted_answer"],
                    "fixed_trace_correctness_status": row["fixed_trace_correctness_status"],
                    "free_running_answer_flip_available": False,
                    "pairs": [
                        {
                            "arm": pair["arm"],
                            "candidate_absolute_position": pair["candidate_absolute_position"],
                            "baseline_answer_token_margins": pair["baseline_answer_token_margins"],
                            "intervention_answer_token_margins": pair["intervention_answer_token_margins"],
                            "answer_margin_sign_change": pair["answer_margin_sign_change"],
                            "baseline_fixed_trace_extracted_answer": pair[
                                "baseline_fixed_trace_extracted_answer"
                            ],
                            "intervention_fixed_trace_extracted_answer": pair[
                                "intervention_fixed_trace_extracted_answer"
                            ],
                            "baseline_fixed_trace_correctness_status": pair[
                                "baseline_fixed_trace_correctness_status"
                            ],
                            "intervention_fixed_trace_correctness_status": pair[
                                "intervention_fixed_trace_correctness_status"
                            ],
                        }
                        for pair in row["pair_records"]
                    ],
                }
                for row in selected_results
            ],
        )
        atomic_write_json(attempt / "scientific_summary.json", summary)
        runtime_seconds = time.perf_counter() - started
        peak_vram = max(
            [max(int(row["peak_cuda_allocated_bytes"]), int(row["peak_cuda_reserved_bytes"])) for row in rkv_results]
            + [max(int(row["peak_cuda_allocated_bytes"]), int(row["peak_cuda_reserved_bytes"])) for row in fullkv_results]
            + [0]
        )
        completion = {
            "completed": True,
            "runtime_seconds": runtime_seconds,
            "peak_cuda_allocated_bytes": peak_vram,
            "peak_tracked_cuda_bytes": peak_vram,
            "claim_path": str(claim_path),
            "retry_allowed": False,
        }
        atomic_write_json(attempt / "completion.json", completion)
        final = {
            "attempt_id": attempt_id,
            "claim_path": str(claim_path),
            "claim_canonical_sha256": claim["canonical_sha256"],
            "scientific_summary": summary,
            "completion": completion,
            "references": _attempt_references(attempt),
        }
        atomic_write_json(attempt / "final.json", final)
        return {
            "attempt_id": attempt_id,
            "attempt_directory": str(attempt),
            "claim_path": str(claim_path),
            "claim_canonical_sha256": claim["canonical_sha256"],
            "classification": summary["classification"],
            "retry_allowed": False,
        }
    except BaseException as exc:
        failure = {
            "completed": False,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "claim_path": str(claim_path),
            "authorization_consumed": True,
            "retry_allowed": False,
            "runtime_seconds": time.perf_counter() - started,
        }
        try:
            atomic_write_json(attempt / "completion.json", failure)
            atomic_write_json(
                attempt / "final.json",
                {
                    "attempt_id": attempt_id,
                    "claim_path": str(claim_path),
                    "classification": "void",
                    "failure": failure,
                    "references": _attempt_references(attempt),
                },
            )
        finally:
            raise


def verify_attempt(attempt_directory: str | Path) -> dict[str, Any]:
    from kvcot.discovery.diagnostic_pilot_authorization import verify_claim

    attempt = Path(attempt_directory).resolve()
    final = json.loads((attempt / "final.json").read_text(encoding="utf-8"))
    if _attempt_references(attempt) != final["references"]:
        raise DiagnosticExecutionRefused("final reference manifest is incomplete or has unexpected files")
    for reference in final["references"]["files"]:
        path = attempt / reference["relative_path"]
        if sha256_file(path) != reference["sha256"] or path.stat().st_size != reference["size_bytes"]:
            raise DiagnosticExecutionRefused(f"attempt reference mismatch: {reference['relative_path']}")
    claim = verify_claim(attempt / "authorization_claim.json")
    if claim.get("attempt_id") != final.get("attempt_id"):
        raise DiagnosticExecutionRefused("claim attempt ID does not match final")
    if Path(claim.get("attempt_directory", "")).resolve() != attempt:
        raise DiagnosticExecutionRefused("claim attempt-directory binding mismatch")
    external_claim = Path(final["claim_path"])
    if not external_claim.is_file() or sha256_file(external_claim) != sha256_file(
        attempt / "authorization_claim.json"
    ):
        raise DiagnosticExecutionRefused("external claim and attempt claim copy differ")
    if (attempt / "scientific_summary.json").exists():
        qualification = json.loads((attempt / "qualification.json").read_text(encoding="utf-8"))
        rkv_results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((attempt / "rkv").glob("candidate-*.json"))]
        reconstructed = _summarize(qualification, rkv_results)
        stored = json.loads((attempt / "scientific_summary.json").read_text(encoding="utf-8"))
        if reconstructed != stored:
            raise DiagnosticExecutionRefused("primitive records do not reconstruct the stored scientific summary")
    return {"verified": True, "attempt_directory": str(attempt), "final_sha256": sha256_file(attempt / "final.json")}
