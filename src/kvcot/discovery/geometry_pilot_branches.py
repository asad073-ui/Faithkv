"""Structured branch construction for the 8B geometry pilot.

Generalizes `kvcot.discovery.pipeline.build_swap_pair_record` (exactly one
candidate/donor pair) to an arbitrary flat list of KV mutations, reusing the
same baseline-then-swapped SEQUENTIAL evaluation discipline (baseline is
cloned, scored, and released BEFORE the swapped clone is even created --
`docs/B2A_R3_STAGE_C_R2_RESULT_ACCEPTANCE_2026-07-25.md`'s ancestor
B1B-R4.1 §17 repair) and the same model-agnostic
`kvcot.discovery.branch_eval.evaluate_branch_compact` scorer -- never a
second, independently-invented evaluation loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from kvcot.discovery.branch_eval import CompactBranchScore, StepFn, evaluate_branch_compact
from kvcot.discovery.geometry_pilot_contract import SCORED_HORIZON
from kvcot.discovery.geometry_pilot_restore import KVMutationSpec, apply_structured_kv_restore


@dataclass(frozen=True)
class GeometryBranchResult:
    arm: str
    mutations: tuple[KVMutationSpec, ...]
    baseline_per_token_nll: tuple[float, ...]
    swapped_per_token_nll: tuple[float, ...]
    baseline_mean_nll: float
    swapped_mean_nll: float
    swap_gain: float
    is_noop: bool
    key_slots_changed: int
    value_slots_changed: int
    cache_shape_unchanged: bool
    provenance_updated_count: int
    kept_index_updated_count: int


def peak_absolute_per_token_delta(baseline: Sequence[float], swapped: Sequence[float]) -> tuple[float, int]:
    deltas = [b - s for b, s in zip(baseline, swapped)]
    index = max(range(len(deltas)), key=lambda i: abs(deltas[i]))
    return deltas[index], index


def first_token_delta(baseline: Sequence[float], swapped: Sequence[float]) -> float:
    return baseline[0] - swapped[0]


def subwindow_gains(baseline: Sequence[float], swapped: Sequence[float], window: int) -> list[float]:
    n = len(baseline)
    if window > n:
        raise ValueError(f"window={window} exceeds horizon length {n}")
    out = []
    for start in range(0, n - window + 1):
        b = baseline[start : start + window]
        s = swapped[start : start + window]
        out.append(sum(b) / window - sum(s) / window)
    return out


def build_geometry_branch_record(
    *,
    arm: str,
    pristine_snapshot: Any,
    mutations: Sequence[KVMutationSpec],
    bridge_token_id: int,
    reference_token_ids: Sequence[int],
    branch_step_fn: StepFn,
    scored_horizon: int = SCORED_HORIZON,
) -> GeometryBranchResult:
    """Build and score one geometry branch from ONE shared pristine,
    post-event `ModelStateSnapshot` -- the caller supplies `mutations`
    already resolved (replacement key/value tensors sourced from this same
    `pristine_snapshot`, never a different one). The no-op arm is the same
    code path as every other arm: pass mutations whose replacement content
    equals the pre-write content and whose donor equals its own candidate.
    """
    if len(reference_token_ids) != scored_horizon:
        raise ValueError(
            f"reference_token_ids must have exactly {scored_horizon} entries, got {len(reference_token_ids)}"
        )

    baseline_snapshot = pristine_snapshot.clone()
    baseline_score = evaluate_branch_compact(branch_step_fn, baseline_snapshot, bridge_token_id, reference_token_ids)
    del baseline_snapshot

    swapped_snapshot = pristine_snapshot.clone()
    restore_result = apply_structured_kv_restore(swapped_snapshot, mutations)
    swapped_score = evaluate_branch_compact(branch_step_fn, swapped_snapshot, bridge_token_id, reference_token_ids)
    del swapped_snapshot

    swap_gain = baseline_score.mean_nll - swapped_score.mean_nll

    return GeometryBranchResult(
        arm=arm,
        mutations=tuple(mutations),
        baseline_per_token_nll=baseline_score.per_token_nll,
        swapped_per_token_nll=swapped_score.per_token_nll,
        baseline_mean_nll=baseline_score.mean_nll,
        swapped_mean_nll=swapped_score.mean_nll,
        swap_gain=swap_gain,
        is_noop=restore_result.is_noop,
        key_slots_changed=restore_result.key_slots_changed,
        value_slots_changed=restore_result.value_slots_changed,
        cache_shape_unchanged=restore_result.cache_shape_unchanged,
        provenance_updated_count=restore_result.provenance_updated_count,
        kept_index_updated_count=restore_result.kept_index_updated_count,
    )
