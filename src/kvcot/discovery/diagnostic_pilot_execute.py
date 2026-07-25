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
    EXECUTING_GENERATION,
    MAXIMUM_CANDIDATE_POOL_SIZE,
    MAXIMUM_PAIRS_PER_SELECTED_EXAMPLE,
    MAXIMUM_QUALIFICATION_CANDIDATES,
    MAXIMUM_SELECTED_EXAMPLES,
    MAXIMUM_TOTAL_PAIRS,
    MODEL_REVISION,
    NOOP_ARM,
    PAIR_SCHEMA_VERSION,
    PILOT_ARMS,
    RESTORE_ARM,
    RESTORE_WIDTHS,
    RKV_REVISION,
    RUNTIME_LIMIT_SECONDS,
    SUMMARY_SCHEMA_VERSION,
    SWAP_GAIN_THRESHOLD_NATS,
    TOKENIZER_REVISION,
    VRAM_LIMIT_BYTES,
    CLASSIFICATION_LETTER,
    CLASSIFICATION_TEXT,
    attach_canonical_hash,
    classify_pilot,
    generation_for_authorization_document,
    generation_for_protocol_document_path,
    material_margin_change,
)
from kvcot.discovery.diagnostic_pilot_manifest import (
    bounded_local_candidate_maximum,
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
    generation = generation_for_authorization_document(authorization.document_path)
    if generation is not EXECUTING_GENERATION:
        # The consumed R1 authorization stays permanently parseable and
        # verifiable, but this implementation executes only the R2 grid and
        # must never re-enter a superseded generation's execution path.
        raise DiagnosticExecutionRefused(
            f"authorization generation {generation.label!r} is not executable by this "
            f"implementation; only {EXECUTING_GENERATION.label!r} may execute"
        )
    runtime = verify_runtime_inputs(
        payload["runtime_config_path"], repository_root=repository_root
    )
    runtime_generation = generation_for_protocol_document_path(runtime["protocol_document_path"])
    if runtime_generation is not generation:
        raise DiagnosticExecutionRefused(
            "runtime protocol document belongs to a different diagnostic-pilot generation"
        )
    if runtime.get("implementation_sha") != payload["authorized_implementation_sha"]:
        raise DiagnosticExecutionRefused(
            "runtime binding was prepared against a different implementation SHA"
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
    if sha256_file(repository_root / generation.protocol_document_path) != runtime[
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
        "generation": generation.label,
        "protocol_document_path": generation.protocol_document_path,
        "output_root": runtime["output_root"],
        "maximum_pairs_per_selected_example": generation.maximum_pairs_per_selected_example,
        "maximum_total_pairs": generation.maximum_total_pairs,
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
    if any(pair.get("arm") not in set(PILOT_ARMS) for pair in pairs):
        return False
    if not pool_positions:
        return False
    selected_head = selected_event.get("selected_kv_head")
    # The complete bounded factorial grid, rank-major, then exactly one
    # no-op.  Both the population and its order are reconstructed here from
    # the frozen pool rather than trusted from the worker.
    expected_grid = [
        (rank, position, width)
        for rank, position in enumerate(pool_positions)
        for width in RESTORE_WIDTHS
    ]
    if len(pairs) != len(expected_grid) + 1:
        return False
    if len(pairs) > MAXIMUM_PAIRS_PER_SELECTED_EXAMPLE:
        return False
    restore_pairs = pairs[: len(expected_grid)]
    noop_pairs = pairs[len(expected_grid) :]
    if len(noop_pairs) != 1:
        return False
    if any(pair.get("arm") != RESTORE_ARM for pair in restore_pairs):
        return False
    if noop_pairs[0].get("arm") != NOOP_ARM:
        return False
    for pair, (rank, position, width) in zip(restore_pairs, expected_grid):
        expected_heads = [selected_head] if width == 1 else [0, 1]
        if (
            pair.get("candidate_pool_rank") != rank
            or pair.get("is_rank_zero_candidate") is not (rank == 0)
            or pair.get("candidate_absolute_position") != position
            or pair.get("restore_width") != width
            or pair.get("kv_head_indices") != expected_heads
            or pair.get("diagnostic_only") is not True
            or pair.get("deployable_performance") is not False
        ):
            return False
    noop = noop_pairs[0]
    donor = selected_event.get("donor_absolute_position")
    if noop.get("candidate_absolute_position") != donor or noop.get("donor_absolute_position") != donor:
        return False
    if (
        noop.get("restore_width") != 1
        or noop.get("kv_head_indices") != [selected_head]
        or noop.get("candidate_pool_rank") is not None
        or noop.get("is_rank_zero_candidate") is not False
        or noop.get("diagnostic_only") is not False
        or noop.get("deployable_performance") is not False
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
    return (
        noop.get("baseline_per_token_nll") == noop.get("intervention_per_token_nll")
        and noop.get("swap_gain") == 0.0
        and noop.get("baseline_final_state_sha256") == noop.get("intervention_final_state_sha256")
        and (noop.get("mutation") or {}).get("is_noop") is True
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


def _example_gain_matrix(result: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct one example's candidate-rank by restore-width matrix.

    Every cell is read back from the primitive pair records; no cell is
    ever taken from a worker-side summary.
    """
    pool = result["selected_event"]["candidate_pool"]
    restore_pairs = [pair for pair in result["pair_records"] if pair["arm"] == RESTORE_ARM]
    by_cell = {
        (pair["candidate_pool_rank"], pair["restore_width"]): pair for pair in restore_pairs
    }
    rows = []
    for rank, candidate in enumerate(pool):
        cells = {width: by_cell.get((rank, width)) for width in RESTORE_WIDTHS}
        gains = {
            width: (None if pair is None else float(pair["swap_gain"]))
            for width, pair in cells.items()
        }
        behaviour = {
            width: (
                None
                if pair is None
                else bool(
                    pair["answer_margin_sign_change"]
                    or pair["fixed_trace_extracted_answer_change"]
                    or pair["fixed_trace_correctness_change"]
                )
            )
            for width, pair in cells.items()
        }
        increment = (
            None
            if gains[1] is None or gains[2] is None
            else gains[2] - gains[1]
        )
        rows.append(
            {
                "candidate_pool_rank": rank,
                "is_rank_zero_candidate": rank == 0,
                "candidate_absolute_position": candidate["absolute_token_position"],
                "deployable_score": candidate["deployable_score"],
                "width_1_gain": gains[1],
                "width_2_gain": gains[2],
                "width_increment": increment,
                "width_1_behavioural_change": behaviour[1],
                "width_2_behavioural_change": behaviour[2],
            }
        )
    return {
        "candidate_ordinal": result["candidate_ordinal"],
        "unique_id": result["unique_id"],
        "event_index": result["selected_event"]["event_index"],
        "layer_index": result["selected_event"]["layer_index"],
        "selected_kv_head": result["selected_event"]["selected_kv_head"],
        "donor_absolute_position": result["selected_event"]["donor_absolute_position"],
        "candidate_pool_size": len(pool),
        "rows": rows,
    }


def _cell_gains(matrices: list[dict[str, Any]], *, rank_zero: bool, width: int) -> list[float]:
    key = "width_1_gain" if width == 1 else "width_2_gain"
    return [
        row[key]
        for matrix in matrices
        for row in matrix["rows"]
        if row["is_rank_zero_candidate"] is rank_zero and row[key] is not None
    ]


def _summarize(qualification_rows: list[dict[str, Any]], rkv_results: list[dict[str, Any]]) -> dict[str, Any]:
    selected = select_first_three_qualified(qualification_rows)
    selected_ordinals = {row["candidate_ordinal"] for row in selected}
    selected_results = [row for row in rkv_results if row["candidate_ordinal"] in selected_ordinals]
    all_pairs = [pair for result in selected_results for pair in result["pair_records"]]
    restore_records = [pair for pair in all_pairs if pair["arm"] == RESTORE_ARM]
    width_one_records = [pair for pair in restore_records if pair["restore_width"] == 1]
    width_two_records = [pair for pair in restore_records if pair["restore_width"] == 2]
    noop_records = [pair for pair in all_pairs if pair["arm"] == NOOP_ARM]
    primitive_populations_valid = all(_pair_population_valid(result) for result in selected_results)

    matrices = [_example_gain_matrix(result) for result in selected_results]
    bounded_maxima_by_example = [
        bounded_local_candidate_maximum(
            [pair for pair in result["pair_records"] if pair["arm"] == RESTORE_ARM]
        )
        for result in selected_results
    ]
    rank_zero_width_one = _cell_gains(matrices, rank_zero=True, width=1)
    rank_zero_width_two = _cell_gains(matrices, rank_zero=True, width=2)
    other_width_one = _cell_gains(matrices, rank_zero=False, width=1)
    other_width_two = _cell_gains(matrices, rank_zero=False, width=2)
    all_gains = rank_zero_width_one + rank_zero_width_two + other_width_one + other_width_two
    increments = [
        row["width_increment"]
        for matrix in matrices
        for row in matrix["rows"]
        if row["width_increment"] is not None
    ]

    behavioural_change = any(
        pair["answer_margin_sign_change"]
        or pair["fixed_trace_extracted_answer_change"]
        or pair["fixed_trace_correctness_change"]
        for pair in restore_records
    )
    noop_maximum_absolute_difference = max(
        (
            max(
                (
                    abs(float(baseline) - float(intervention))
                    for baseline, intervention in zip(
                        pair["baseline_per_token_nll"], pair["intervention_per_token_nll"]
                    )
                ),
                default=0.0,
            )
            for pair in noop_records
        ),
        default=None,
    )
    noop_exact = (
        len(noop_records) == len(selected_results)
        and all(result["noop_exact"] is True for result in selected_results)
        and primitive_populations_valid
    )
    expected_pairs = sum(
        len(result["selected_event"]["candidate_pool"]) * len(RESTORE_WIDTHS) + 1
        for result in selected_results
    )
    complete = (
        len(selected_results) == len(selected)
        and len(all_pairs) == expected_pairs
        and expected_pairs <= MAXIMUM_TOTAL_PAIRS
        and primitive_populations_valid
    )
    classification = classify_pilot(
        qualified_examples=len(selected),
        rank_zero_width_one_gains=rank_zero_width_one,
        non_rank_zero_width_one_gains=other_width_one,
        rank_zero_width_two_gains=rank_zero_width_two,
        non_rank_zero_width_two_gains=other_width_two,
        behavioural_change=behavioural_change,
        noop_exact=noop_exact,
        complete=complete,
    )

    def above(values: list[float]) -> int:
        return sum(value > SWAP_GAIN_THRESHOLD_NATS for value in values)

    return attach_canonical_hash(
        {
            "artifact_schema_version": SUMMARY_SCHEMA_VERSION,
            "qualified_example_count": len(selected),
            "selected_candidate_ordinals": sorted(selected_ordinals),
            "selected_unique_ids": [row["unique_id"] for row in selected],
            "completed_width_one_pairs": len(width_one_records),
            "completed_width_two_pairs": len(width_two_records),
            "completed_restore_pairs": len(restore_records),
            "completed_noop_pairs": len(noop_records),
            "expected_total_pairs": expected_pairs,
            "completed_total_pairs": len(all_pairs),
            "maximum_pairs_per_selected_example": MAXIMUM_PAIRS_PER_SELECTED_EXAMPLE,
            "maximum_total_pairs": MAXIMUM_TOTAL_PAIRS,
            "noop_exact": noop_exact,
            "noop_maximum_absolute_difference": noop_maximum_absolute_difference,
            "primitive_populations_reconstructed": primitive_populations_valid,
            "per_example_gain_matrix": matrices,
            "bounded_local_candidate_maxima": bounded_maxima_by_example,
            "rank_zero_width_one_gains": rank_zero_width_one,
            "rank_zero_width_two_gains": rank_zero_width_two,
            "non_rank_zero_width_one_gains": other_width_one,
            "non_rank_zero_width_two_gains": other_width_two,
            "width_increments": increments,
            "best_rank_zero_width_one_gain": max(rank_zero_width_one, default=None),
            "best_rank_zero_width_two_gain": max(rank_zero_width_two, default=None),
            "best_non_rank_zero_width_one_gain": max(other_width_one, default=None),
            "best_non_rank_zero_width_two_gain": max(other_width_two, default=None),
            "largest_gain_over_bounded_grid": max(all_gains, default=None),
            "rank_zero_width_one_gains_above_0_01": above(rank_zero_width_one),
            "rank_zero_width_two_gains_above_0_01": above(rank_zero_width_two),
            "non_rank_zero_width_one_gains_above_0_01": above(other_width_one),
            "non_rank_zero_width_two_gains_above_0_01": above(other_width_two),
            "total_gains_above_0_01": above(all_gains),
            "bounded_local_candidate_maxima_above_0_01": above(bounded_maxima_by_example),
            "swap_gain_threshold_nats": SWAP_GAIN_THRESHOLD_NATS,
            "behavioural_change": behavioural_change,
            "answer_margin_sign_changes": sum(
                pair["answer_margin_sign_change"] for pair in restore_records
            ),
            "fixed_trace_answer_changes": sum(
                pair["fixed_trace_extracted_answer_change"] for pair in restore_records
            ),
            "correctness_changes": sum(
                pair["fixed_trace_correctness_change"] for pair in restore_records
            ),
            "classification": classification.value,
            "classification_letter": CLASSIFICATION_LETTER[classification],
            "classification_text": CLASSIFICATION_TEXT[classification],
            "scale_limitation": (
                "A 1.5B result does not settle the 8B operating point; it neither "
                "confirms nor refutes the immutable 8B R2 null."
            ),
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
    generation = generation_for_authorization_document(authorization.document_path)
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
                "command": generation.exact_execution_command,
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
                "generation": generation.label,
                "protocol_document_path": generation.protocol_document_path,
                "protocol_document_sha256": runtime["protocol_document_sha256"],
                "maximum_pairs_per_selected_example": generation.maximum_pairs_per_selected_example,
                "maximum_total_pairs": generation.maximum_total_pairs,
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
        claim_generation = generation_for_authorization_document(
            claim["authorization_document_path"]
        )
        expected_binding = {
            "generation": claim_generation.label,
            "protocol_document_path": claim_generation.protocol_document_path,
            "maximum_pairs_per_selected_example": (
                claim_generation.maximum_pairs_per_selected_example
            ),
            "maximum_total_pairs": claim_generation.maximum_total_pairs,
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
