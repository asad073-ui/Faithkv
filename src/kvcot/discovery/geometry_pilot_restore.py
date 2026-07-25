"""Generalized structured KV restore primitive for the 8B geometry pilot.

Generalizes this repository's existing, already-audited swap primitives --
`kvcot.discovery.swap.apply_within_head_swap_owned` (exactly one
(layer, kv_head, slot) write) and
`kvcot.discovery.diagnostic_pilot_swap.apply_diagnostic_kv_restore` (one
layer, many KV heads, one token position per head) -- to an arbitrary flat
list of `(layer_index, kv_head_index, token_position)` mutations spanning
any number of layers and heads, which the closed 1.5B track never needed
(2 KV heads, 1 layer per arm) but the real 8B arms (all-KV-head, multi-
layer, and token-span arms) require. No shape/dtype/device/aliasing
invariant is re-derived here -- every check below mirrors the two existing
primitives exactly, generalized only to iterate over a list instead of one
slot or one layer's heads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class GeometryRestoreError(ValueError):
    pass


@dataclass(frozen=True)
class KVMutationSpec:
    """One requested (layer, kv_head, token_position) restore, with the
    replacement key/value ALREADY resolved from the pristine snapshot's own
    captured tensors -- never from a different snapshot, layer, or head."""

    layer_index: int
    kv_head_index: int
    token_position: int
    replacement_key: Any  # torch.Tensor, shape (head_dim,)
    replacement_value: Any  # torch.Tensor, shape (head_dim,)
    donor_absolute_position: int
    candidate_absolute_position: int


@dataclass(frozen=True)
class StructuredRestoreResult:
    mutations: tuple[KVMutationSpec, ...]
    key_slots_changed: int
    value_slots_changed: int
    is_noop: bool
    cache_shape_unchanged: bool
    provenance_updated_count: int
    kept_index_updated_count: int


def _mutation_key(mutation: KVMutationSpec) -> tuple[int, int, int]:
    return (mutation.layer_index, mutation.kv_head_index, mutation.token_position)


def apply_structured_kv_restore(
    snapshot: Any,  # kvcot.generation.state.ModelStateSnapshot
    mutations: Sequence[KVMutationSpec],
) -> StructuredRestoreResult:
    """Apply every mutation in `mutations` to one caller-owned snapshot
    clone (matching this repository's existing "clone the pristine
    snapshot, then mutate the clone" convention -- the caller, never this
    function, is responsible for calling `snapshot.clone()` first).

    Requirements enforced (never silently repaired):

    - `mutations` must be non-empty.
    - No duplicate `(layer_index, kv_head_index, token_position)` triple.
    - Every `(layer_index, kv_head_index, token_position)` must be in range
      for `snapshot.key_cache`/`value_cache`'s actual shapes.
    - Every replacement key/value must be a contiguous tensor of shape
      `(head_dim,)`, matching cache dtype/device, and must not alias the
      cache's own storage.
    - Every layer/head/slot NOT named in `mutations` is left byte-identical
      (verified by CPU tests via full-tensor diffing, not asserted here).
    - The no-op case (`donor_absolute_position == candidate_absolute_position`
      for every mutation, and every write reproduces the pre-write value) is
      the same code path as every other mutation, never a special case.
    """
    import torch

    if not mutations:
        raise GeometryRestoreError("mutations must not be empty")

    seen: set[tuple[int, int, int]] = set()
    for mutation in mutations:
        key = _mutation_key(mutation)
        if key in seen:
            raise GeometryRestoreError(f"duplicate (layer, kv_head, token_position) triple: {key}")
        seen.add(key)

    if len(snapshot.key_cache) != len(snapshot.value_cache):
        raise GeometryRestoreError("key_cache and value_cache have different layer counts")
    original_shapes = tuple(tuple(t.shape) for t in snapshot.key_cache) + tuple(
        tuple(t.shape) for t in snapshot.value_cache
    )

    key_changes = 0
    value_changes = 0
    provenance_updated_count = 0
    kept_index_updated_count = 0

    for mutation in mutations:
        layer_index = mutation.layer_index
        if not 0 <= layer_index < len(snapshot.key_cache):
            raise GeometryRestoreError(f"layer_index={layer_index} out of range")
        layer_k = snapshot.key_cache[layer_index]
        layer_v = snapshot.value_cache[layer_index]
        if layer_k.shape != layer_v.shape or layer_k.dim() != 4 or layer_k.shape[0] != 1:
            raise GeometryRestoreError(f"expected matching 4-D batch-one key/value tensors at layer {layer_index}")
        _, num_kv_heads, seq_len, head_dim = layer_k.shape

        head = mutation.kv_head_index
        if not (isinstance(head, int) and 0 <= head < num_kv_heads):
            raise GeometryRestoreError(f"kv_head_index={head} out of range for {num_kv_heads} heads at layer {layer_index}")
        position = mutation.token_position
        if not (isinstance(position, int) and 0 <= position < seq_len):
            raise GeometryRestoreError(f"token_position={position} out of range for seq_len={seq_len} at layer {layer_index}")

        key = mutation.replacement_key
        value = mutation.replacement_value
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise GeometryRestoreError("replacement key/value must be tensors")
        if tuple(key.shape) != (head_dim,) or tuple(value.shape) != (head_dim,):
            raise GeometryRestoreError(f"replacement vector has the wrong head dimension at layer {layer_index}")
        if not key.is_contiguous() or not value.is_contiguous():
            raise GeometryRestoreError("replacement vectors must be contiguous")
        if key.dtype != layer_k.dtype or value.dtype != layer_v.dtype:
            raise GeometryRestoreError(f"replacement dtype differs from cache dtype at layer {layer_index}")
        if key.device != layer_k.device or value.device != layer_v.device:
            raise GeometryRestoreError(f"replacement device differs from cache device at layer {layer_index}")

        cache_storage_ids = {layer_k.untyped_storage().data_ptr(), layer_v.untyped_storage().data_ptr()}
        if key.untyped_storage().data_ptr() in cache_storage_ids or value.untyped_storage().data_ptr() in cache_storage_ids:
            raise GeometryRestoreError(f"replacement vector aliases target cache storage at layer {layer_index}")

        before_k = layer_k[0, head, position, :].clone()
        before_v = layer_v[0, head, position, :].clone()
        key_changes += int(not torch.equal(before_k, key))
        value_changes += int(not torch.equal(before_v, value))
        layer_k[0, head, position, :] = key
        layer_v[0, head, position, :] = value

        if snapshot.provenance is not None:
            layer_provenance = snapshot.provenance.layers.get(layer_index)
            if layer_provenance is not None:
                layer_provenance.positions[head, position] = mutation.candidate_absolute_position
                provenance_updated_count += 1
        if snapshot.kv_cluster_bookkeeping_per_layer:
            bookkeeping = snapshot.kv_cluster_bookkeeping_per_layer[layer_index]
            kept = bookkeeping.get("kept_token_indices") if bookkeeping else None
            if kept:
                kept[-1][head, position] = mutation.candidate_absolute_position
                kept_index_updated_count += 1

    final_shapes = tuple(tuple(t.shape) for t in snapshot.key_cache) + tuple(
        tuple(t.shape) for t in snapshot.value_cache
    )
    is_noop = (
        key_changes == 0
        and value_changes == 0
        and all(m.donor_absolute_position == m.candidate_absolute_position for m in mutations)
    )
    return StructuredRestoreResult(
        mutations=tuple(mutations),
        key_slots_changed=key_changes,
        value_slots_changed=value_changes,
        is_noop=is_noop,
        cache_shape_unchanged=original_shapes == final_shapes,
        provenance_updated_count=provenance_updated_count,
        kept_index_updated_count=kept_index_updated_count,
    )


def resolve_physical_position(
    provenance_layer_positions: Any,  # torch.Tensor, shape (num_kv_heads, cache_len)
    kv_head_index: int,
    absolute_position: int,
) -> int | None:
    """Find the physical cache slot currently holding `absolute_position`
    at `kv_head_index`, or `None` if that layer/head has already evicted
    it. Generalizes `compact_target._find_position` to an arbitrary
    (layer, head) rather than assuming the one event-selected layer."""
    row = provenance_layer_positions[kv_head_index]
    matches = (row == absolute_position).nonzero(as_tuple=True)[0]
    return None if matches.numel() == 0 else int(matches[0].item())


def resolve_valid_layers(
    pristine_snapshot: Any,
    *,
    kv_head_index: int,
    required_absolute_positions: Sequence[int],
    candidate_layers: Sequence[int],
) -> dict[int, dict[int, int]]:
    """For every layer in `candidate_layers`, determine whether EVERY
    position in `required_absolute_positions` is still resolvable (not yet
    evicted) at `kv_head_index` in that layer's own provenance. Returns
    `{layer_index: {absolute_position: physical_slot}}` for exactly the
    layers where all required positions resolve -- layers where any
    position is missing are simply absent from the returned mapping, never
    included with a partial/best-effort slot mapping. Reads only the
    already-captured `pristine_snapshot.provenance` (no new forward pass,
    no future information relative to the snapshot's own step)."""
    if pristine_snapshot.provenance is None:
        return {}
    valid: dict[int, dict[int, int]] = {}
    for layer_index in candidate_layers:
        layer_provenance = pristine_snapshot.provenance.layers.get(layer_index)
        if layer_provenance is None:
            continue
        positions = layer_provenance.positions
        if kv_head_index >= positions.shape[0]:
            continue
        resolved: dict[int, int] = {}
        ok = True
        for absolute_position in required_absolute_positions:
            slot = resolve_physical_position(positions, kv_head_index, absolute_position)
            if slot is None:
                ok = False
                break
            resolved[absolute_position] = slot
        if ok:
            valid[layer_index] = resolved
    return valid
