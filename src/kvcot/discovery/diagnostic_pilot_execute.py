"""Coordinator, dry-run preflight, execution, and reconstruction."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
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
    DiagnosticAuthorizationConsumed,
    claim_authorization_once,
    parse_authorization_document,
)
from kvcot.discovery.diagnostic_pilot_contract import (
    BRANCH,
    CANDIDATE_MANIFEST_BYTE_SHA256,
    CANDIDATE_MANIFEST_CANONICAL_SHA256,
    CANDIDATE_MANIFEST_PATH,
    CONFIG_PATH,
    EXACT_EXECUTION_COMMAND,
    MAXIMUM_CANDIDATE_POOL_SIZE,
    MAXIMUM_QUALIFICATION_CANDIDATES,
    MAXIMUM_SELECTED_EXAMPLES,
    MODEL_REVISION,
    PAIR_SCHEMA_VERSION,
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
    runtime = verify_runtime_inputs(
        payload["runtime_config_path"], repository_root=repository_root
    )
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
    if runtime["output_root"] != payload["output_root"]:
        raise DiagnosticExecutionRefused("authorization output root differs from runtime binding")
    if runtime["protocol_document_sha256"] != payload["protocol_document_sha256"]:
        raise DiagnosticExecutionRefused("protocol document hash mismatch")
    if sha256_file(repository_root / PROTOCOL_DOCUMENT_PATH) != runtime[
        "protocol_document_sha256"
    ]:
        raise DiagnosticExecutionRefused("checked-out protocol document differs from runtime binding")
    if sha256_file(repository_root / CONFIG_PATH) != runtime["config_byte_sha256"]:
        raise DiagnosticExecutionRefused("checked-out diagnostic config differs from runtime binding")
    manifest_path = repository_root / CANDIDATE_MANIFEST_PATH
    if sha256_file(manifest_path) != CANDIDATE_MANIFEST_BYTE_SHA256:
        raise DiagnosticExecutionRefused("checked-out candidate manifest byte hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("canonical_sha256") != CANDIDATE_MANIFEST_CANONICAL_SHA256:
        raise DiagnosticExecutionRefused("checked-out candidate manifest canonical hash mismatch")
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
    if timeout <= 0:
        raise DiagnosticExecutionRefused(
            f"{role} worker was not launched because the pilot wall-time limit is exhausted"
        )
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
        timeout=timeout,
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
    architecture = result.get("architecture") or {}
    if (
        result.get("role") != "diagnostic_rkv"
        or result.get("model_revision") != MODEL_REVISION
        or result.get("tokenizer_revision") != TOKENIZER_REVISION
        or result.get("rkv_revision") != RKV_REVISION
        or architecture.get("num_attention_heads") != 12
        or architecture.get("num_key_value_heads") != 2
        or result.get("free_running_answer_flip_available") is not False
    ):
        return False
    selected_event = result.get("selected_event") or {}
    pool = selected_event.get("candidate_pool") or []
    if not 2 <= len(pool) <= MAXIMUM_CANDIDATE_POOL_SIZE:
        return False
    if any(
        type(row.get("absolute_token_position")) is not int
        or type(row.get("deployable_score")) not in (float, int)
        or not math.isfinite(float(row["deployable_score"]))
        for row in pool
    ):
        return False
    if len({row["absolute_token_position"] for row in pool}) != len(pool):
        return False
    if pool != sorted(
        pool,
        key=lambda row: (-float(row["deployable_score"]), row["absolute_token_position"]),
    ):
        return False
    evidence = result.get("eligible_event_score_evidence") or []
    if not evidence:
        return False
    for event in evidence:
        event_pool = event.get("candidate_pool") or []
        if not 2 <= len(event_pool) <= MAXIMUM_CANDIDATE_POOL_SIZE:
            return False
        if any(
            type(row.get("absolute_token_position")) is not int
            or type(row.get("deployable_score")) not in (float, int)
            or not math.isfinite(float(row["deployable_score"]))
            for row in event_pool
        ):
            return False
        if len({row["absolute_token_position"] for row in event_pool}) != len(event_pool):
            return False
        if event_pool != sorted(
            event_pool,
            key=lambda row: (-float(row["deployable_score"]), row["absolute_token_position"]),
        ):
            return False
        if event.get("deployable_event_score") != event_pool[0].get("deployable_score"):
            return False
        if event.get("captured_before_intervention") is not True:
            return False
        if event.get("rank_zero_candidate_available_in_all_kv_heads") is not True:
            return False
    expected_event = min(
        evidence,
        key=lambda row: (
            -float(row["deployable_event_score"]),
            row["event_index"],
            row["layer_index"],
            row["selected_kv_head"],
        ),
    )
    for key in (
        "event_index",
        "absolute_event_position",
        "layer_index",
        "selected_kv_head",
        "deployable_event_score",
        "candidate_pool",
    ):
        if selected_event.get(key) != expected_event.get(key):
            return False
    if selected_event.get("candidate_pool_frozen_before_intervention") is not True:
        return False
    if selected_event.get("candidate_pool_diagnostic_only") is not True:
        return False
    if result.get("score_replay_retained_full_snapshots") != 0:
        return False
    if result.get("selected_snapshot_count") != 1:
        return False
    pool_positions = [row.get("absolute_token_position") for row in pool]
    pairs = result.get("pair_records") or []
    if any(pair.get("arm") not in {"candidate_upper_bound", "restore_width", "no_op"} for pair in pairs):
        return False
    arm_a = [pair for pair in pairs if pair.get("arm") == "candidate_upper_bound"]
    arm_b = [pair for pair in pairs if pair.get("arm") == "restore_width"]
    noop = [pair for pair in pairs if pair.get("arm") == "no_op"]
    if [pair.get("candidate_absolute_position") for pair in arm_a] != pool_positions:
        return False
    if [pair.get("candidate_pool_rank") for pair in arm_a] != list(range(len(pool_positions))):
        return False
    if [pair.get("is_control_candidate") for pair in arm_a] != [
        index == 0 for index in range(len(pool_positions))
    ]:
        return False
    if len(arm_b) != 1 or len(noop) != 1 or not pool_positions:
        return False
    if arm_b[0].get("candidate_absolute_position") != pool_positions[0]:
        return False
    if arm_b[0].get("restore_width") != 2 or arm_b[0].get("kv_head_indices") != [0, 1]:
        return False
    if any(
        pair.get("restore_width") != 1
        or pair.get("kv_head_indices") != [selected_event.get("selected_kv_head")]
        or pair.get("diagnostic_only") is not True
        or pair.get("deployable_performance") is not False
        for pair in arm_a
    ):
        return False
    if arm_b[0].get("diagnostic_only") is not False or arm_b[0].get(
        "deployable_performance"
    ) is not False:
        return False
    donor = selected_event.get("donor_absolute_position")
    if noop[0].get("candidate_absolute_position") != donor or noop[0].get("donor_absolute_position") != donor:
        return False
    if (
        noop[0].get("restore_width") != 1
        or noop[0].get("kv_head_indices") != [selected_event.get("selected_kv_head")]
        or noop[0].get("diagnostic_only") is not False
        or noop[0].get("deployable_performance") is not False
    ):
        return False
    for pair in pairs:
        if pair.get("artifact_schema_version") != PAIR_SCHEMA_VERSION:
            return False
        if pair.get("compaction_event_id") != selected_event.get("event_index"):
            return False
        if pair.get("layer_index") != selected_event.get("layer_index"):
            return False
        if pair.get("selected_kv_head") != selected_event.get("selected_kv_head"):
            return False
        if pair.get("donor_absolute_position") != donor:
            return False
        mutation = pair.get("mutation") or {}
        if mutation.get("layer_index") != pair.get("layer_index"):
            return False
        if mutation.get("kv_head_indices") != pair.get("kv_head_indices"):
            return False
        if mutation.get("cache_shape_unchanged") is not True:
            return False
        if mutation.get("absolute_position_unchanged") is not True:
            return False
        if mutation.get("provenance_valid") is not True:
            return False
        donor_slots = pair.get("donor_post_storage_positions_by_head") or {}
        expected_slots = [
            [head, donor_slots.get(str(head), donor_slots.get(head))]
            for head in pair.get("kv_head_indices", [])
        ]
        if any(type(slot) is not int or slot < 0 for _head, slot in expected_slots):
            return False
        if mutation.get("token_positions_by_head") != expected_slots:
            return False
        expected_changes = 0 if pair.get("arm") == "no_op" else pair.get("restore_width")
        if mutation.get("key_slots_changed") != expected_changes:
            return False
        if mutation.get("value_slots_changed") != expected_changes:
            return False
        if mutation.get("is_noop") is not (pair.get("arm") == "no_op"):
            return False
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
        baseline_margin_rows = pair.get("baseline_answer_token_margins", [])
        intervention_margin_rows = pair.get("intervention_answer_token_margins", [])
        if not _answer_margin_evidence_valid(baseline_margin_rows):
            return False
        if not _answer_margin_evidence_valid(intervention_margin_rows):
            return False
        baseline_correct = {
            row["absolute_position"]: row
            for row in baseline_margin_rows
            if row["is_correct_answer_token"] is True
        }
        intervention_correct = {
            row["absolute_position"]: row
            for row in intervention_margin_rows
            if row["is_correct_answer_token"] is True
        }
        reconstructed_margin_change = any(
            material_margin_change(
                baseline_correct.get(position, {}).get("margin"),
                intervention_correct.get(position, {}).get("margin"),
            )
            for position in set(baseline_correct) | set(intervention_correct)
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


def _answer_margin_evidence_valid(rows: Any) -> bool:
    if not isinstance(rows, list):
        return False
    positions: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            return False
        position = row.get("absolute_position")
        reference_id = row.get("reference_token_id")
        alternative_id = row.get("strongest_alternative_token_id")
        reference_logp = row.get("reference_log_probability")
        alternative_logp = row.get("strongest_alternative_log_probability")
        margin = row.get("margin")
        if (
            type(position) is not int
            or position in positions
            or type(reference_id) is not int
            or type(alternative_id) is not int
            or reference_id == alternative_id
            or type(reference_logp) not in (float, int)
            or type(alternative_logp) not in (float, int)
            or type(margin) not in (float, int)
            or not all(
                math.isfinite(float(value))
                for value in (reference_logp, alternative_logp, margin)
            )
            or float(margin) != float(reference_logp) - float(alternative_logp)
            or type(row.get("is_correct_answer_token")) is not bool
        ):
            return False
        correct_logp = row.get("correct_answer_log_probability")
        if row["is_correct_answer_token"] is True:
            if correct_logp != reference_logp:
                return False
        elif correct_logp is not None:
            return False
        positions.add(position)
    return True


def _raw_worker_binding_valid(fullkv: dict[str, Any], rkv: dict[str, Any]) -> bool:
    from kvcot.discovery.strict_device import (
        _single_worker_placement_ok,
        verify_device_gate_from_raw_evidence,
    )

    ordinal = rkv.get("candidate_ordinal")
    fullkv_identity = fullkv.get("dataset_row_identity") or {}
    qualification = dict(rkv.get("qualification") or {})
    qualification["candidate_ordinal"] = ordinal
    try:
        qualified = mechanically_qualifies(qualification)
    except (TypeError, ValueError):
        return False
    cache_lengths = rkv.get("final_cache_length_per_layer") or {}
    full_token_count = rkv.get("natural_full_token_count")
    meaningful_compression = (
        type(full_token_count) is int
        and full_token_count > 0
        and type(qualification.get("observed_compaction_event_count")) is int
        and qualification["observed_compaction_event_count"] > 0
        and isinstance(cache_lengths, dict)
        and bool(cache_lengths)
        and all(type(length) is int and length >= 0 for length in cache_lengths.values())
        and all(length <= full_token_count for length in cache_lengths.values())
        and any(length < full_token_count for length in cache_lengths.values())
    )
    selected_event = rkv.get("selected_event")
    derived = {
        "rkv_replay_mechanically_valid": rkv.get("selected_event_replay_valid") is True
        and rkv.get("rkv_natural_cap_hit") is False,
        "correctness_status_matched": fullkv.get("natural_answer_status")
        == rkv.get("fixed_trace_correctness_status"),
        "meaningful_compression": meaningful_compression,
        "eligible_event_exists": bool(rkv.get("eligible_event_score_evidence")),
        "eligible_scored_event_count": len(rkv.get("eligible_event_score_evidence") or []),
        "selected_event_has_two_candidates": bool(
            selected_event and len(selected_event.get("candidate_pool") or []) >= 2
        ),
        "selected_event_count": 1 if selected_event is not None else 0,
    }
    if any(qualification.get(key) != value for key, value in derived.items()):
        return False
    fullkv_valid = (
        fullkv.get("cap_hit") is False
        and fullkv.get("actual_batch_size_verified") is True
        and _single_worker_placement_ok(fullkv.get("parameter_placement"))
        and verify_device_gate_from_raw_evidence(
            fullkv.get("device_evidence", {}), rkv.get("device_evidence", {})
        )
    )
    return (
        type(ordinal) is int
        and fullkv.get("candidate_ordinal") == ordinal
        and fullkv.get("role") == "fullkv"
        and fullkv.get("model_revision") == MODEL_REVISION
        and fullkv.get("tokenizer_revision") == TOKENIZER_REVISION
        and rkv.get("role") == "diagnostic_rkv"
        and rkv.get("model_revision") == MODEL_REVISION
        and rkv.get("tokenizer_revision") == TOKENIZER_REVISION
        and rkv.get("rkv_revision") == RKV_REVISION
        and (rkv.get("architecture") or {}).get("num_attention_heads") == 12
        and (rkv.get("architecture") or {}).get("num_key_value_heads") == 2
        and fullkv_identity.get("unique_id") == rkv.get("unique_id")
        and fullkv.get("dataset_repo") == rkv.get("dataset_repo")
        and fullkv.get("dataset_revision") == rkv.get("dataset_revision")
        and fullkv.get("manifest_hash") == rkv.get("manifest_hash")
        and fullkv.get("prompt_token_ids_sha256") == rkv.get("prompt_token_ids_sha256")
        and fullkv.get("prompt_token_count") == rkv.get("prompt_token_count")
        and qualification.get("unique_id") == rkv.get("unique_id")
        and qualification.get("fullkv_correctness_status")
        == fullkv.get("natural_answer_status")
        and qualification.get("rkv_correctness_status")
        == rkv.get("fixed_trace_correctness_status")
        and qualification.get("rkv_natural_cap_hit") == rkv.get("rkv_natural_cap_hit")
        and qualification.get("fullkv_execution_valid") is fullkv_valid
        and rkv.get("qualified") is qualified
        and qualification.get("intervention_not_evaluated") is True
        and _single_worker_placement_ok(rkv.get("parameter_placement"))
        and (qualified or (rkv.get("pair_records") == [] and rkv.get("noop_exact") is None))
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
            "arm_a_gains_above_0_01": sum(
                float(pair["swap_gain"]) > SWAP_GAIN_THRESHOLD_NATS
                for pair in arm_a_records
            ),
            "arm_a_bounded_maxima_above_0_01": sum(
                value > SWAP_GAIN_THRESHOLD_NATS for value in arm_a_by_example
            ),
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


def _preserve_consumed_failure(
    *, attempt: Path, attempt_id: str, claim_path: Path, started: float, exc: BaseException
) -> None:
    """Best-effort immutable failure evidence after the claim exists.

    The attempt directory is preferred.  If setup failed before it could be
    created or made writable, the claim directory is the last known writable
    location because the atomic claim was just created there successfully.
    """
    try:
        claim_canonical_sha256 = json.loads(claim_path.read_text(encoding="utf-8")).get(
            "canonical_sha256"
        )
    except BaseException:
        claim_canonical_sha256 = None
    failure = {
        "completed": False,
        "failure_type": type(exc).__name__,
        "failure_message": str(exc),
        "claim_path": str(claim_path),
        "attempt_id": attempt_id,
        "intended_attempt_directory": str(attempt),
        "authorization_consumed": True,
        "retry_allowed": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    preservation_errors: list[str] = []
    if attempt.is_dir():
        failure_path = attempt / "failure.json"
        completion_path = attempt / "completion.json"
        final_path = attempt / "final.json"
        for label, path, artifact in (
            ("failure", failure_path, failure),
            ("completion", completion_path, failure),
        ):
            try:
                if not path.exists():
                    atomic_write_json(path, artifact)
            except BaseException as preserve_exc:
                preservation_errors.append(
                    f"attempt-{label}:{type(preserve_exc).__name__}:{preserve_exc}"
                )
        try:
            if not final_path.exists():
                atomic_write_json(
                    final_path,
                    {
                        "attempt_id": attempt_id,
                        "claim_path": str(claim_path),
                        "claim_canonical_sha256": claim_canonical_sha256,
                        "classification": "void",
                        "failure": failure,
                        "references": _attempt_references(attempt),
                    },
                )
        except BaseException as preserve_exc:
            preservation_errors.append(
                f"attempt-final:{type(preserve_exc).__name__}:{preserve_exc}"
            )
        if not preservation_errors:
            return
    fallback = claim_path.with_name(f"{claim_path.name}.failure.json")
    fallback_payload = dict(failure, preservation_errors=preservation_errors)
    try:
        atomic_write_json(fallback, fallback_payload)
    except BaseException:
        # The immutable claim still proves consumption. There is no further
        # known-safe writable boundary; never mask the original exception.
        pass


def _derived_projection_payloads(selected_results: list[dict[str, Any]]) -> dict[str, Any]:
    selected_examples = [
        {
            "candidate_ordinal": row["candidate_ordinal"],
            "unique_id": row["unique_id"],
            "selected_event": row["selected_event"],
        }
        for row in selected_results
    ]
    candidate_pools = [row["selected_event"] for row in selected_results]
    pair_records = [pair for row in selected_results for pair in row["pair_records"]]
    behavioural_readouts = [
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
                    "intervention_answer_token_margins": pair[
                        "intervention_answer_token_margins"
                    ],
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
    ]
    return {
        "selected_examples.json": selected_examples,
        "candidate_pools.json": candidate_pools,
        "pair_records.json": pair_records,
        "behavioural_readouts.json": behavioural_readouts,
    }


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
    started = time.perf_counter()
    claim_path = Path(payload["claim_path"]).resolve()
    try:
        claim_path, claim = claim_authorization_once(
            authorization, attempt_id=attempt_id, attempt_directory=str(attempt)
        )
    except DiagnosticAuthorizationConsumed:
        raise
    except BaseException as exc:
        # If atomic claim creation succeeded but its subsequent write/fsync
        # failed, the authorization is nevertheless permanently consumed.
        # Preserve that fact at the only safe boundary available here.
        if claim_path.exists():
            _preserve_consumed_failure(
                attempt=attempt,
                attempt_id=attempt_id,
                claim_path=claim_path,
                started=started,
                exc=exc,
            )
        raise
    try:
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
        runtime = verify_runtime_inputs(
            payload["runtime_config_path"], repository_root=repository_root
        )
        atomic_write_json(
            attempt / "protocol_binding.json",
            {
                "protocol_document_path": PROTOCOL_DOCUMENT_PATH,
                "protocol_document_sha256": runtime["protocol_document_sha256"],
                "authorization_id": payload["authorization_id"],
                "authorization_document_sha256": authorization.document_sha256,
                "authorized_implementation_sha": payload["authorized_implementation_sha"],
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
            if remaining <= 0:
                raise DiagnosticExecutionRefused(
                    "pilot wall-time limit exhausted before diagnostic R-KV launch"
                )
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

        if time.perf_counter() - started > RUNTIME_LIMIT_SECONDS:
            raise DiagnosticExecutionRefused("pilot wall-time limit exhausted after worker completion")
        summary = _summarize(qualification_rows, rkv_results)
        selected_ordinals = set(summary["selected_candidate_ordinals"])
        selected_results = [row for row in rkv_results if row["candidate_ordinal"] in selected_ordinals]
        projections = _derived_projection_payloads(selected_results)
        atomic_write_json(attempt / "qualification.json", qualification_rows)
        for name, projection in projections.items():
            atomic_write_json(attempt / name, projection)
        atomic_write_json(attempt / "scientific_summary.json", summary)
        runtime_seconds = time.perf_counter() - started
        if runtime_seconds > RUNTIME_LIMIT_SECONDS:
            raise DiagnosticExecutionRefused("pilot wall-time limit exceeded before completion")
        all_worker_results = rkv_results + fullkv_results
        peak_allocated = max(
            [int(row["peak_cuda_allocated_bytes"]) for row in all_worker_results] + [0]
        )
        peak_reserved = max(
            [int(row["peak_cuda_reserved_bytes"]) for row in all_worker_results] + [0]
        )
        peak_vram = max(peak_allocated, peak_reserved)
        completion = {
            "completed": True,
            "runtime_seconds": runtime_seconds,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "peak_tracked_cuda_bytes": peak_vram,
            "claim_path": str(claim_path),
            "claim_canonical_sha256": claim["canonical_sha256"],
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
        _preserve_consumed_failure(
            attempt=attempt,
            attempt_id=attempt_id,
            claim_path=claim_path,
            started=started,
            exc=exc,
        )
        raise


def verify_attempt(attempt_directory: str | Path) -> dict[str, Any]:
    from kvcot.discovery.diagnostic_pilot_authorization import verify_claim

    attempt = Path(attempt_directory).resolve()
    final = json.loads((attempt / "final.json").read_text(encoding="utf-8"))
    completion = json.loads((attempt / "completion.json").read_text(encoding="utf-8"))
    if _attempt_references(attempt) != final["references"]:
        raise DiagnosticExecutionRefused("final reference manifest is incomplete or has unexpected files")
    for reference in final["references"]["files"]:
        path = attempt / reference["relative_path"]
        if sha256_file(path) != reference["sha256"] or path.stat().st_size != reference["size_bytes"]:
            raise DiagnosticExecutionRefused(f"attempt reference mismatch: {reference['relative_path']}")
    claim = verify_claim(attempt / "authorization_claim.json")
    if final.get("claim_canonical_sha256") != claim.get("canonical_sha256"):
        raise DiagnosticExecutionRefused("final stored claim canonical hash mismatch")
    if completion.get("retry_allowed") is not False:
        raise DiagnosticExecutionRefused("completion must permanently prohibit retry")
    if claim.get("attempt_id") != final.get("attempt_id"):
        raise DiagnosticExecutionRefused("claim attempt ID does not match final")
    if Path(claim.get("attempt_directory", "")).resolve() != attempt:
        raise DiagnosticExecutionRefused("claim attempt-directory binding mismatch")
    external_claim = Path(final["claim_path"])
    if not external_claim.is_file() or sha256_file(external_claim) != sha256_file(
        attempt / "authorization_claim.json"
    ):
        raise DiagnosticExecutionRefused("external claim and attempt claim copy differ")
    authorization = parse_authorization_document(claim["authorization_document_path"])
    if authorization.document_sha256 != claim.get("authorization_document_sha256"):
        raise DiagnosticExecutionRefused("claim authorization-document hash mismatch")
    authorization_payload = authorization.payload
    if (
        authorization_payload.get("authorization_id") != claim.get("authorization_id")
        or authorization_payload.get("authorized_implementation_sha")
        != claim.get("authorized_implementation_sha")
        or Path(authorization_payload.get("claim_path", "")).resolve() != external_claim.resolve()
        or attempt.parent != Path(authorization_payload.get("output_root", "")).resolve()
    ):
        raise DiagnosticExecutionRefused("claim differs from the verified execution authorization")
    binding_path = attempt / "protocol_binding.json"
    if binding_path.exists():
        binding = json.loads((attempt / "protocol_binding.json").read_text(encoding="utf-8"))
        if binding.get("authorization_id") != claim.get("authorization_id"):
            raise DiagnosticExecutionRefused("protocol binding authorization ID mismatch")
        if binding.get("authorization_document_sha256") != claim.get(
            "authorization_document_sha256"
        ):
            raise DiagnosticExecutionRefused("protocol binding authorization document mismatch")
        if binding.get("authorized_implementation_sha") != claim.get(
            "authorized_implementation_sha"
        ):
            raise DiagnosticExecutionRefused("protocol binding implementation SHA mismatch")
    if (attempt / "scientific_summary.json").exists():
        if not binding_path.is_file():
            raise DiagnosticExecutionRefused("successful attempt lacks protocol binding")
        expected_binding = {
            "protocol_document_sha256": authorization_payload["protocol_document_sha256"],
            "runtime_config_canonical_sha256": authorization_payload[
                "runtime_config_canonical_sha256"
            ],
            "candidate_manifest_canonical_sha256": authorization_payload[
                "candidate_manifest_canonical_sha256"
            ],
            "model_revision": authorization_payload["model_revision"],
            "tokenizer_revision": authorization_payload["tokenizer_revision"],
            "rkv_revision": authorization_payload["rkv_revision"],
        }
        if any(binding.get(key) != value for key, value in expected_binding.items()):
            raise DiagnosticExecutionRefused("protocol binding differs from execution authorization")
        qualification = json.loads((attempt / "qualification.json").read_text(encoding="utf-8"))
        rkv_results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((attempt / "rkv").glob("candidate-*.json"))]
        fullkv_results = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((attempt / "fullkv").glob("candidate-*.json"))
        ]
        if len(fullkv_results) != len(rkv_results) or not all(
            _raw_worker_binding_valid(fullkv, rkv)
            for fullkv, rkv in zip(fullkv_results, rkv_results)
        ):
            raise DiagnosticExecutionRefused("FullKV/R-KV raw worker binding mismatch")
        reconstructed_qualification = []
        for row in rkv_results:
            qualification_row = dict(row["qualification"])
            qualification_row["candidate_ordinal"] = row["candidate_ordinal"]
            reconstructed_qualification.append(qualification_row)
        if qualification != reconstructed_qualification:
            raise DiagnosticExecutionRefused("qualification projection differs from worker primitives")
        reconstructed = _summarize(qualification, rkv_results)
        stored = json.loads((attempt / "scientific_summary.json").read_text(encoding="utf-8"))
        if reconstructed != stored:
            raise DiagnosticExecutionRefused("primitive records do not reconstruct the stored scientific summary")
        if final.get("scientific_summary") != stored:
            raise DiagnosticExecutionRefused("final embedded scientific summary differs from referenced file")
        if final.get("completion") != completion:
            raise DiagnosticExecutionRefused("final embedded completion differs from referenced file")
        if (
            completion.get("completed") is not True
            or completion.get("claim_canonical_sha256") != claim.get("canonical_sha256")
            or type(completion.get("runtime_seconds")) not in (float, int)
            or completion["runtime_seconds"] > RUNTIME_LIMIT_SECONDS
            or type(completion.get("peak_tracked_cuda_bytes")) is not int
            or completion["peak_tracked_cuda_bytes"] > VRAM_LIMIT_BYTES
        ):
            raise DiagnosticExecutionRefused("successful completion violates frozen runtime evidence")
        selected_ordinals = set(reconstructed["selected_candidate_ordinals"])
        selected_results = [
            row for row in rkv_results if row["candidate_ordinal"] in selected_ordinals
        ]
        for name, expected in _derived_projection_payloads(selected_results).items():
            observed = json.loads((attempt / name).read_text(encoding="utf-8"))
            if observed != expected:
                raise DiagnosticExecutionRefused(f"{name} differs from worker primitives")
    elif final.get("failure") != completion:
        raise DiagnosticExecutionRefused("final embedded failure differs from referenced completion")
    return {"verified": True, "attempt_directory": str(attempt), "final_sha256": sha256_file(attempt / "final.json")}
