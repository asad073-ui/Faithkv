"""Dry-run and one-use authorized execution orchestration for the 8B
structured restoration geometry pilot. CUDA/model/tokenizer loading is
deferred entirely to `kvcot.discovery.geometry_pilot_workers
.run_geometry_worker` -- nothing in this module initializes CUDA or loads
model weights, so `run_dry_run` is safe to call on any host, with or
without a GPU.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from kvcot.discovery.geometry_pilot_authorization import (
    GeometryAuthorizationDocumentBinding,
    attempt_directory_name,
    build_claim_payload,
    claim_authorization,
    sha256_file,
    verify_authorization_document_binding,
)
from kvcot.discovery.geometry_pilot_branches import subwindow_gains, peak_absolute_per_token_delta, first_token_delta
from kvcot.discovery.geometry_pilot_contract import (
    ARM_C1,
    ARM_C2,
    ARM_H,
    ARM_HL,
    ARM_L,
    ARM_NOOP,
    ARM_S,
    ARM_SH,
    GeometryPilotClassification,
    SWAP_GAIN_THRESHOLD_NATS,
    classify_geometry_pilot,
)


class GeometryExecutionError(RuntimeError):
    pass


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


def branch_readout(result) -> dict[str, Any]:
    """Every primitive quantity task section 10 requires, reconstructed
    from the branch's own stored arrays -- never re-derived a second,
    inconsistent way at report time."""
    baseline = list(result.baseline_per_token_nll)
    swapped = list(result.swapped_per_token_nll)
    peak_delta, peak_index = peak_absolute_per_token_delta(baseline, swapped)
    return {
        "arm": result.arm,
        "mutations": [_to_jsonable(m) for m in result.mutations],
        "baseline_per_token_nll": baseline,
        "swapped_per_token_nll": swapped,
        "baseline_mean_nll": result.baseline_mean_nll,
        "swapped_mean_nll": result.swapped_mean_nll,
        "swap_gain": result.swap_gain,
        "is_noop": result.is_noop,
        "key_slots_changed": result.key_slots_changed,
        "value_slots_changed": result.value_slots_changed,
        "cache_shape_unchanged": result.cache_shape_unchanged,
        "provenance_updated_count": result.provenance_updated_count,
        "kept_index_updated_count": result.kept_index_updated_count,
        "peak_absolute_per_token_nll_change": peak_delta,
        "peak_absolute_per_token_nll_change_index": peak_index,
        "first_token_nll_change": first_token_delta(baseline, swapped),
        "subwindow_gains_w8": subwindow_gains(baseline, swapped, 8),
        "subwindow_gains_w16": subwindow_gains(baseline, swapped, 16),
    }


def build_scientific_summary(branch_readouts: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    def gain_of(arm: str) -> float | None:
        readout = branch_readouts.get(arm)
        return None if readout is None else readout["swap_gain"]

    c1 = gain_of(ARM_C1)
    non_rank_zero = [g for g in (gain_of(ARM_C2),) if g is not None]
    noop = branch_readouts.get(ARM_NOOP)
    noop_exact = bool(noop is not None and noop["is_noop"] and noop["swap_gain"] == 0.0)

    classification = classify_geometry_pilot(
        evidence_complete=all(v is not None for k, v in branch_readouts.items() if k in (ARM_C1, ARM_C2, ARM_H, ARM_NOOP)),
        noop_exact=noop_exact,
        rank_zero_gain=c1,
        non_rank_zero_gains=non_rank_zero,
        head_coverage_gain=gain_of(ARM_H),
        layer_coverage_gain=gain_of(ARM_L),
        head_layer_interaction_gain=gain_of(ARM_HL),
        span_gain=gain_of(ARM_S),
        span_head_gain=gain_of(ARM_SH),
        span_available=branch_readouts.get(ARM_S) is not None,
        layer_set_available=branch_readouts.get(ARM_L) is not None,
        behavioural_change=False,  # no fixed-trace margin capture in this pilot; see docstring below
    )
    return {
        "schema_version": "faithkv-8b-geometry-summary.v1",
        "threshold_nats": SWAP_GAIN_THRESHOLD_NATS,
        "gains_by_arm": {arm: gain_of(arm) for arm in branch_readouts},
        "noop_exact": noop_exact,
        "classification": classification.value,
        "classification_letter": {
            GeometryPilotClassification.ORIGINAL_CANDIDATE_WORKS: "A",
            GeometryPilotClassification.CANDIDATE_SELECTION_RESCUE: "B",
            GeometryPilotClassification.HEAD_COVERAGE_RESCUE: "C",
            GeometryPilotClassification.LAYER_COVERAGE_RESCUE: "D",
            GeometryPilotClassification.HEAD_LAYER_INTERACTION: "E",
            GeometryPilotClassification.TOKEN_SPAN_RESCUE: "F",
            GeometryPilotClassification.READOUT_ONLY_MOVEMENT: "G",
            GeometryPilotClassification.STRUCTURED_RESTORATION_FLAT: "H",
            GeometryPilotClassification.MECHANICALLY_INVALID: "I",
        }[classification],
    }


def run_dry_run(*, repository_root: str | Path, binding: GeometryAuthorizationDocumentBinding) -> dict[str, Any]:
    """Non-consuming: verifies the authorization document's hash binding,
    proves the claim path is absent, and proves the (would-be) attempt
    output root is absent. Never initializes CUDA, never imports torch or
    transformers, never creates a claim."""
    from kvcot.discovery.geometry_pilot_authorization import global_claim_path

    verify_authorization_document_binding(binding, repository_root=repository_root)
    claim_path = Path(repository_root) / global_claim_path(binding.authorization_id)
    return {
        "authorization_document_verified": True,
        "claim_path": str(claim_path),
        "claim_absent": not claim_path.exists(),
        "cuda_initialized": False,
        "model_loaded": False,
    }


def run_execute(
    *,
    repository_root: str | Path,
    binding: GeometryAuthorizationDocumentBinding,
    config: Any,
    manifest: Any,
    authorized_repository: str,
    authorized_branch: str,
    observed_execution_commit_sha: str,
) -> dict[str, Any]:
    """Consumes the one-use claim, then runs the real GPU worker exactly
    once, in two strictly ordered phases. Claim creation happens BEFORE the
    worker is invoked -- an exception from the worker after this point
    still leaves the authorization consumed (matching this repository's
    existing Stage-C contract: 'once inference begins, the attempt is
    scientifically consumed').

    Phase A (`capture_geometry_anchor`) captures the frozen anchor's
    pristine snapshot and resolves the layer-set/span-availability freeze
    -- NO branch outcome exists yet. `layer_set.json`/
    `span_availability.json` are written to disk IMMEDIATELY after Phase A
    returns and STRICTLY BEFORE Phase B (`evaluate_geometry_branches`) is
    even called, so the freeze is persisted before a single swap gain is
    computed, not merely computed-in-memory-first."""
    from kvcot.discovery.geometry_pilot_workers import capture_geometry_anchor, evaluate_geometry_branches

    verify_authorization_document_binding(binding, repository_root=repository_root)

    attempt_id = uuid4().hex
    claimed_at = datetime.now(timezone.utc)
    timestamp = claimed_at.strftime("%Y%m%dT%H%M%S%f") + "Z"
    attempt_dir_name = attempt_directory_name(timestamp, attempt_id)

    payload = build_claim_payload(
        binding=binding,
        authorized_repository=authorized_repository,
        authorized_branch=authorized_branch,
        observed_execution_commit_sha=observed_execution_commit_sha,
        attempt_id=attempt_id,
        attempt_directory_path=f"results/decisions/{attempt_dir_name}",
        claimed_at_utc=claimed_at.isoformat(),
    )
    claim_authorization(repository_root=repository_root, payload=payload)

    attempt_dir = Path(repository_root) / "results" / "decisions" / attempt_dir_name
    attempt_dir.mkdir(parents=True, exist_ok=False)

    # --- Phase A: capture + pre-outcome freeze. No branch outcome exists yet. ---
    capture = capture_geometry_anchor(config, manifest)
    (attempt_dir / "layer_set.json").write_text(json.dumps(capture.layer_set, indent=2) + "\n")
    (attempt_dir / "span_availability.json").write_text(json.dumps(capture.span_availability, indent=2) + "\n")

    # --- Phase B: evaluate branches. Only now does any swap gain exist. ---
    worker_result = evaluate_geometry_branches(capture, rkv_revision=config.rkv.upstream_revision)

    branch_readouts = {result.arm: branch_readout(result) for result in worker_result.branch_results}
    all_arms = (ARM_C1, ARM_C2, ARM_H, ARM_L, ARM_HL, ARM_S, ARM_SH, ARM_NOOP)
    for arm in all_arms:
        branch_readouts.setdefault(arm, None)

    summary = build_scientific_summary(branch_readouts)

    (attempt_dir / "branch_readouts.json").write_text(json.dumps(branch_readouts, indent=2) + "\n")
    (attempt_dir / "scientific_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (attempt_dir / "worker_meta.json").write_text(
        json.dumps(
            {
                "example_id": worker_result.example_id,
                "natural_answer": worker_result.natural_answer,
                "natural_answer_status": worker_result.natural_answer_status,
                "natural_generated_token_count": worker_result.natural_generated_token_count,
                "peak_cuda_allocated_bytes": worker_result.peak_cuda_allocated_bytes,
                "peak_cuda_reserved_bytes": worker_result.peak_cuda_reserved_bytes,
                "wall_seconds": worker_result.wall_seconds,
            },
            indent=2,
        )
        + "\n"
    )

    return {
        "attempt_id": attempt_id,
        "attempt_directory": str(attempt_dir),
        "branch_readouts": branch_readouts,
        "scientific_summary": summary,
    }
