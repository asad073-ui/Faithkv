import copy

import pytest
import torch

from kvcot.discovery.diagnostic_pilot_swap import DiagnosticRestoreError, apply_diagnostic_kv_restore
from kvcot.generation.state import ModelStateSnapshot


def snapshot():
    keys = [torch.arange(1 * 2 * 5 * 3, dtype=torch.float32).reshape(1, 2, 5, 3) + 100 * i for i in range(3)]
    values = [tensor.clone() + 0.5 for tensor in keys]
    kept = torch.tensor([[10, 11, 12, 13, 14], [10, 11, 12, 13, 14]])
    bookkeeping = [
        {"kept_token_indices": [kept.clone()], "evicted_token_num": 0}
        for _ in range(3)
    ]
    return ModelStateSnapshot(
        key_cache=keys,
        value_cache=values,
        query_cache={},
        compression_flags_per_layer=["none"] * 3,
        model_length=5,
        after_think=None,
        absolute_position=5,
        kv_cluster_bookkeeping_per_layer=bookkeeping,
    )


def test_width_one_changes_one_key_and_value_only():
    state = snapshot()
    before = state.clone()
    result = apply_diagnostic_kv_restore(
        state,
        layer_index=1,
        kv_head_indices=[1],
        token_position=2,
        replacement_keys={1: torch.full((3,), 900.0)},
        replacement_values={1: torch.full((3,), -900.0)},
        candidate_absolute_position=99,
        donor_absolute_position=12,
    )
    assert result.key_slots_changed == result.value_slots_changed == 1
    assert torch.equal(state.key_cache[1][0, 1, 2], torch.full((3,), 900.0))
    assert torch.equal(state.value_cache[1][0, 1, 2], torch.full((3,), -900.0))
    for layer in (0, 2):
        assert torch.equal(state.key_cache[layer], before.key_cache[layer])
        assert torch.equal(state.value_cache[layer], before.value_cache[layer])
    mask = torch.ones_like(state.key_cache[1], dtype=torch.bool)
    mask[0, 1, 2] = False
    assert torch.equal(state.key_cache[1][mask], before.key_cache[1][mask])
    assert torch.equal(state.value_cache[1][mask], before.value_cache[1][mask])


def test_width_two_changes_both_kv_heads_shape_and_position_unchanged():
    state = snapshot()
    original_shapes = [tensor.shape for tensor in state.key_cache + state.value_cache]
    result = apply_diagnostic_kv_restore(
        state,
        layer_index=0,
        kv_head_indices=[0, 1],
        token_position=4,
        replacement_keys={0: torch.full((3,), 8.0), 1: torch.full((3,), 9.0)},
        replacement_values={0: torch.full((3,), -8.0), 1: torch.full((3,), -9.0)},
        candidate_absolute_position=77,
        donor_absolute_position=14,
    )
    assert result.key_slots_changed == result.value_slots_changed == 2
    assert result.cache_shape_unchanged and result.absolute_position_unchanged
    assert [tensor.shape for tensor in state.key_cache + state.value_cache] == original_shapes
    assert state.absolute_position == 5
    assert state.kv_cluster_bookkeeping_per_layer[0]["kept_token_indices"][-1][:, 4].tolist() == [77, 77]


def test_width_two_uses_each_heads_corresponding_donor_slot():
    state = snapshot()
    before = state.clone()
    result = apply_diagnostic_kv_restore(
        state,
        layer_index=0,
        kv_head_indices=[0, 1],
        token_position={0: 1, 1: 3},
        replacement_keys={0: torch.full((3,), 8.0), 1: torch.full((3,), 9.0)},
        replacement_values={0: torch.full((3,), -8.0), 1: torch.full((3,), -9.0)},
        candidate_absolute_position=77,
        donor_absolute_position=11,
    )
    assert result.token_position is None
    assert result.token_positions_by_head == ((0, 1), (1, 3))
    assert torch.equal(state.key_cache[0][0, 0, 1], torch.full((3,), 8.0))
    assert torch.equal(state.key_cache[0][0, 1, 3], torch.full((3,), 9.0))
    mask = torch.ones_like(state.key_cache[0], dtype=torch.bool)
    mask[0, 0, 1] = False
    mask[0, 1, 3] = False
    assert torch.equal(state.key_cache[0][mask], before.key_cache[0][mask])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"layer_index": 3, "kv_head_indices": [0]}, "layer_index"),
        ({"layer_index": 0, "kv_head_indices": [2]}, "head index"),
        ({"layer_index": 0, "kv_head_indices": [0, 0]}, "duplicate"),
        ({"layer_index": 0, "kv_head_indices": []}, "empty"),
    ],
)
def test_invalid_layer_head_duplicate_and_empty_rejected(kwargs, match):
    state = snapshot()
    heads = kwargs["kv_head_indices"]
    mapping_heads = set(heads)
    with pytest.raises(DiagnosticRestoreError, match=match):
        apply_diagnostic_kv_restore(
            state,
            token_position=0,
            replacement_keys={head: torch.zeros(3) for head in mapping_heads},
            replacement_values={head: torch.zeros(3) for head in mapping_heads},
            candidate_absolute_position=1,
            donor_absolute_position=2,
            **kwargs,
        )


def test_noop_is_bit_exact():
    state = snapshot()
    before = state.clone()
    key = state.key_cache[0][0, 1, 3].clone().contiguous()
    value = state.value_cache[0][0, 1, 3].clone().contiguous()
    result = apply_diagnostic_kv_restore(
        state,
        layer_index=0,
        kv_head_indices=[1],
        token_position=3,
        replacement_keys={1: key},
        replacement_values={1: value},
        candidate_absolute_position=13,
        donor_absolute_position=13,
    )
    assert result.is_noop
    assert result.key_slots_changed == result.value_slots_changed == 0
    for actual, expected in zip(state.key_cache + state.value_cache, before.key_cache + before.value_cache):
        assert torch.equal(actual, expected)
