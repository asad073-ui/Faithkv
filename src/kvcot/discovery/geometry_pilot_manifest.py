"""Frozen branch manifest and pre-outcome freezing helpers for the 8B
structured restoration geometry pilot
(`docs/FAITHKV_8B_STRUCTURED_RESTORATION_GEOMETRY_PROTOCOL_2026-07-25.md`
sections 3-5). Imports neither Torch nor Transformers at module scope --
the `pristine_snapshot`-touching functions import torch locally, exactly
matching this repository's existing `kvcot.discovery.swap`/
`kvcot.discovery.geometry_pilot_restore` convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kvcot.discovery.geometry_pilot_contract import (
    ANCHOR_BRIDGE_TOKEN_ABSOLUTE_POSITION,
    ANCHOR_EVENT_TOKEN_ABSOLUTE_POSITION,
    ANCHOR_FIRST_SCORED_ABSOLUTE_POSITION,
    ANCHOR_KV_HEAD_INDEX,
    ANCHOR_LAYER_INDEX,
    ARM_C1,
    ARM_C2,
    ARM_H,
    ARM_HL,
    ARM_L,
    ARM_NOOP,
    ARM_S,
    ARM_SH,
    CANDIDATE_POOL,
    PRIMARY_DONOR_ABSOLUTE_POSITION,
    RANK_ONE_CANDIDATE_ABSOLUTE_POSITION,
    RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION,
    SPAN_OFFSETS,
    EXPECTED_KV_HEADS,
)


class GeometryManifestError(ValueError):
    pass


@dataclass(frozen=True)
class BranchIntent:
    """One frozen arm's structural definition -- never the concrete
    per-run mutation list, which additionally requires the resolved valid-
    layer set / span availability (§5, computed once per execution and
    persisted before any outcome exists)."""

    arm: str
    candidate_absolute_positions: tuple[int, ...]
    uses_all_kv_heads: bool
    uses_valid_layer_set: bool
    donor_absolute_position: int


#: The 8 branches this pilot actually runs (2 candidate + H + L + HL + S +
#: SH + no-op), out of the 10-branch ceiling the protocol allows.
FROZEN_BRANCH_INTENTS: tuple[BranchIntent, ...] = (
    BranchIntent(ARM_C1, (RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION,), False, False, PRIMARY_DONOR_ABSOLUTE_POSITION),
    BranchIntent(ARM_C2, (RANK_ONE_CANDIDATE_ABSOLUTE_POSITION,), False, False, PRIMARY_DONOR_ABSOLUTE_POSITION),
    BranchIntent(ARM_H, (RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION,), True, False, PRIMARY_DONOR_ABSOLUTE_POSITION),
    BranchIntent(ARM_L, (RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION,), False, True, PRIMARY_DONOR_ABSOLUTE_POSITION),
    BranchIntent(ARM_HL, (RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION,), True, True, PRIMARY_DONOR_ABSOLUTE_POSITION),
    BranchIntent(
        ARM_S,
        tuple(RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION + offset for offset in SPAN_OFFSETS),
        False, False, PRIMARY_DONOR_ABSOLUTE_POSITION,
    ),
    BranchIntent(
        ARM_SH,
        tuple(RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION + offset for offset in SPAN_OFFSETS),
        True, False, PRIMARY_DONOR_ABSOLUTE_POSITION,
    ),
    BranchIntent(ARM_NOOP, (PRIMARY_DONOR_ABSOLUTE_POSITION,), False, False, PRIMARY_DONOR_ABSOLUTE_POSITION),
)


def candidate_pool_manifest() -> dict[str, Any]:
    """The bounded score-prioritized candidate pool, frozen at protocol-
    authoring time from already-finalized R2 evidence -- persisted before
    any new intervention outcome exists. Never called an oracle pool."""
    return {
        "schema_version": "faithkv-8b-geometry-candidate-pool.v1",
        "anchor_compaction_event_id": 5,
        "anchor_layer_index": ANCHOR_LAYER_INDEX,
        "anchor_kv_head_index": ANCHOR_KV_HEAD_INDEX,
        "anchor_event_token_absolute_position": ANCHOR_EVENT_TOKEN_ABSOLUTE_POSITION,
        "anchor_bridge_token_absolute_position": ANCHOR_BRIDGE_TOKEN_ABSOLUTE_POSITION,
        "anchor_first_scored_absolute_position": ANCHOR_FIRST_SCORED_ABSOLUTE_POSITION,
        "pool": [
            {"rank": rank, "candidate_absolute_position": position, "score_e": score_e}
            for rank, position, score_e in CANDIDATE_POOL
        ],
        "primary_donor_absolute_position": PRIMARY_DONOR_ABSOLUTE_POSITION,
        "pool_kind": "bounded_score_prioritized_candidate_pool",
    }


def freeze_layer_set(pristine_snapshot: Any, candidate_layers: range | list[int] | None = None) -> dict[str, Any]:
    """Resolve and persist (as a plain dict, ready for `json.dump`) the
    valid-layer set for Arm L/HL -- every layer in `candidate_layers`
    (default: every model layer) for which BOTH the rank-zero candidate and
    the primary donor resolve in `ANCHOR_KV_HEAD_INDEX`'s provenance. Must
    be called, and its result persisted, before any branch outcome is
    computed -- this function itself never reads a swap gain or any other
    outcome, by construction of its signature."""
    from kvcot.discovery.geometry_pilot_restore import resolve_valid_layers

    if candidate_layers is None:
        candidate_layers = range(len(pristine_snapshot.key_cache))
    valid = resolve_valid_layers(
        pristine_snapshot,
        kv_head_index=ANCHOR_KV_HEAD_INDEX,
        required_absolute_positions=[RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION, PRIMARY_DONOR_ABSOLUTE_POSITION],
        candidate_layers=candidate_layers,
    )
    layer_indices = sorted(valid)
    return {
        "schema_version": "faithkv-8b-geometry-layer-set.v1",
        "kv_head_index": ANCHOR_KV_HEAD_INDEX,
        "required_absolute_positions": [RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION, PRIMARY_DONOR_ABSOLUTE_POSITION],
        "valid_layer_indices": layer_indices,
        "physical_slots_by_layer": {
            str(layer): {str(pos): slot for pos, slot in valid[layer].items()} for layer in layer_indices
        },
        "available": len(layer_indices) >= 2,
        "unavailable_reason": None if len(layer_indices) >= 2 else "fewer_than_two_valid_layers",
    }


def freeze_span_availability(pristine_snapshot: Any, prompt_length: int, total_length: int) -> dict[str, Any]:
    """Resolve and persist span availability for Arm S/SH -- the exact
    positions `[t-1, t, t+1]` around the rank-zero candidate, using only
    `ANCHOR_LAYER_INDEX`/`ANCHOR_KV_HEAD_INDEX`'s own provenance. No
    alternative span is ever substituted; if any position is missing, out
    of the eligible evicted region, or the middle position is not itself
    the frozen rank-zero candidate, the span is recorded unavailable."""
    from kvcot.discovery.geometry_pilot_restore import resolve_physical_position

    positions = tuple(RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION + offset for offset in SPAN_OFFSETS)
    if pristine_snapshot.provenance is None:
        return {
            "schema_version": "faithkv-8b-geometry-span-availability.v1",
            "span_absolute_positions": list(positions),
            "available": False,
            "unavailable_reason": "no_provenance_on_snapshot",
            "physical_slots": None,
        }
    layer_provenance = pristine_snapshot.provenance.layers.get(ANCHOR_LAYER_INDEX)
    if layer_provenance is None:
        return {
            "schema_version": "faithkv-8b-geometry-span-availability.v1",
            "span_absolute_positions": list(positions),
            "available": False,
            "unavailable_reason": "anchor_layer_not_in_provenance",
            "physical_slots": None,
        }
    slots: dict[int, int] = {}
    for position in positions:
        if not (prompt_length <= position < total_length):
            return {
                "schema_version": "faithkv-8b-geometry-span-availability.v1",
                "span_absolute_positions": list(positions),
                "available": False,
                "unavailable_reason": f"position_{position}_outside_eligible_region",
                "physical_slots": None,
            }
        slot = resolve_physical_position(layer_provenance.positions, ANCHOR_KV_HEAD_INDEX, position)
        if slot is None:
            return {
                "schema_version": "faithkv-8b-geometry-span-availability.v1",
                "span_absolute_positions": list(positions),
                "available": False,
                "unavailable_reason": f"position_{position}_has_no_valid_snapshot",
                "physical_slots": None,
            }
        slots[position] = slot
    return {
        "schema_version": "faithkv-8b-geometry-span-availability.v1",
        "span_absolute_positions": list(positions),
        "available": True,
        "unavailable_reason": None,
        "physical_slots": {str(pos): slot for pos, slot in slots.items()},
    }


def resolve_all_kv_head_indices(num_key_value_heads: int) -> tuple[int, ...]:
    if num_key_value_heads != EXPECTED_KV_HEADS:
        raise GeometryManifestError(
            f"protocol requires exactly {EXPECTED_KV_HEADS} KV heads, got {num_key_value_heads}"
        )
    return tuple(range(num_key_value_heads))
