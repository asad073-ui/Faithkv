"""Diagnostic-only same-layer multi-KV-head restore primitive."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class DiagnosticRestoreError(ValueError):
    pass


@dataclass(frozen=True)
class DiagnosticRestoreResult:
    layer_index: int
    kv_head_indices: tuple[int, ...]
    token_position: int | None
    token_positions_by_head: tuple[tuple[int, int], ...]
    key_slots_changed: int
    value_slots_changed: int
    is_noop: bool
    cache_shape_unchanged: bool
    absolute_position_unchanged: bool
    provenance_valid: bool


def apply_diagnostic_kv_restore(
    snapshot: Any,
    *,
    layer_index: int,
    kv_head_indices: tuple[int, ...] | list[int],
    token_position: int | Mapping[int, int],
    replacement_keys: Mapping[int, Any],
    replacement_values: Mapping[int, Any],
    candidate_absolute_position: int,
    donor_absolute_position: int,
) -> DiagnosticRestoreResult:
    """Restore K and V for exactly the requested KV heads in one layer.

    ``snapshot`` is caller-owned and is mutated in place. Replacement vectors
    must not alias cache storage. No layer/head/token shape can change.
    """
    import torch

    heads = tuple(kv_head_indices)
    if not heads:
        raise DiagnosticRestoreError("kv_head_indices must not be empty")
    if len(set(heads)) != len(heads):
        raise DiagnosticRestoreError("duplicate KV head indices are prohibited")
    if len(snapshot.key_cache) != len(snapshot.value_cache):
        raise DiagnosticRestoreError("key/value layer counts disagree")
    if not 0 <= layer_index < len(snapshot.key_cache):
        raise DiagnosticRestoreError("layer_index is out of range")
    layer_k = snapshot.key_cache[layer_index]
    layer_v = snapshot.value_cache[layer_index]
    if layer_k.shape != layer_v.shape or layer_k.ndim != 4 or layer_k.shape[0] != 1:
        raise DiagnosticRestoreError("expected matching 4-D batch-one key/value tensors")
    _, num_kv_heads, seq_len, head_dim = layer_k.shape
    if any(type(head) is not int or not 0 <= head < num_kv_heads for head in heads):
        raise DiagnosticRestoreError("KV head index is out of range")
    if isinstance(token_position, Mapping):
        if set(token_position) != set(heads):
            raise DiagnosticRestoreError("token-position mapping must match the exact selected head set")
        positions = {head: token_position[head] for head in heads}
    else:
        positions = {head: token_position for head in heads}
    if any(type(position) is not int or not 0 <= position < seq_len for position in positions.values()):
        raise DiagnosticRestoreError("token_position is out of range")
    if set(replacement_keys) != set(heads) or set(replacement_values) != set(heads):
        raise DiagnosticRestoreError("replacement mappings must match the exact selected head set")

    original_shapes = tuple(tuple(t.shape) for t in snapshot.key_cache + snapshot.value_cache)
    key_changes = 0
    value_changes = 0
    for head in heads:
        head_position = positions[head]
        key = replacement_keys[head]
        value = replacement_values[head]
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise DiagnosticRestoreError("replacement key/value must be tensors")
        if tuple(key.shape) != (head_dim,) or tuple(value.shape) != (head_dim,):
            raise DiagnosticRestoreError("replacement vector has the wrong head dimension")
        if not key.is_contiguous() or not value.is_contiguous():
            raise DiagnosticRestoreError("replacement vectors must be contiguous")
        if key.dtype != layer_k.dtype or value.dtype != layer_v.dtype:
            raise DiagnosticRestoreError("replacement dtype differs from cache dtype")
        if key.device != layer_k.device or value.device != layer_v.device:
            raise DiagnosticRestoreError("replacement device differs from cache device")
        cache_storage = {layer_k.untyped_storage().data_ptr(), layer_v.untyped_storage().data_ptr()}
        if key.untyped_storage().data_ptr() in cache_storage or value.untyped_storage().data_ptr() in cache_storage:
            raise DiagnosticRestoreError("replacement vector aliases target cache storage")
        before_k = layer_k[0, head, head_position, :].clone()
        before_v = layer_v[0, head, head_position, :].clone()
        key_changes += int(not torch.equal(before_k, key))
        value_changes += int(not torch.equal(before_v, value))
        layer_k[0, head, head_position, :] = key
        layer_v[0, head, head_position, :] = value

    provenance_valid = True
    if snapshot.provenance is not None:
        layer_provenance = snapshot.provenance.layers.get(layer_index)
        if layer_provenance is None:
            provenance_valid = False
        else:
            for head in heads:
                layer_provenance.positions[head, positions[head]] = candidate_absolute_position
    if snapshot.kv_cluster_bookkeeping_per_layer:
        bookkeeping = snapshot.kv_cluster_bookkeeping_per_layer[layer_index]
        kept = bookkeeping.get("kept_token_indices", [])
        if kept:
            for head in heads:
                kept[-1][head, positions[head]] = candidate_absolute_position

    final_shapes = tuple(tuple(t.shape) for t in snapshot.key_cache + snapshot.value_cache)
    is_noop = key_changes == 0 and value_changes == 0 and candidate_absolute_position == donor_absolute_position
    return DiagnosticRestoreResult(
        layer_index=layer_index,
        kv_head_indices=heads,
        token_position=(next(iter(positions.values())) if len(set(positions.values())) == 1 else None),
        token_positions_by_head=tuple((head, positions[head]) for head in heads),
        key_slots_changed=key_changes,
        value_slots_changed=value_changes,
        is_noop=is_noop,
        cache_shape_unchanged=original_shapes == final_shapes,
        absolute_position_unchanged=True,
        provenance_valid=provenance_valid,
    )
