import pytest
import torch

from kvcot.discovery.geometry_pilot_restore import (
    GeometryRestoreError,
    KVMutationSpec,
    apply_structured_kv_restore,
    resolve_physical_position,
    resolve_valid_layers,
)
from kvcot.generation.provenance import LayerProvenance, ModelProvenance
from kvcot.generation.state import ModelStateSnapshot

NUM_LAYERS = 4
NUM_KV_HEADS = 3
SEQ_LEN = 6
HEAD_DIM = 5


def _snapshot(with_provenance: bool = True, with_bookkeeping: bool = True) -> ModelStateSnapshot:
    keys = [
        torch.arange(NUM_KV_HEADS * SEQ_LEN * HEAD_DIM, dtype=torch.float32).reshape(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        + 1000 * layer
        for layer in range(NUM_LAYERS)
    ]
    values = [tensor.clone() + 0.5 for tensor in keys]
    provenance = None
    if with_provenance:
        # absolute position == physical slot for every layer except layer 2,
        # where head 1 has "evicted" absolute position 3 (replaced by 99).
        layers = {}
        for layer in range(NUM_LAYERS):
            positions = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0).expand(NUM_KV_HEADS, -1).clone()
            if layer == 2:
                positions[1, 3] = 99  # absolute position 3 no longer present at (layer 2, head 1)
            layers[layer] = LayerProvenance(positions=positions)
        provenance = ModelProvenance(layers=layers, prompt_length=0)
    bookkeeping = None
    if with_bookkeeping:
        kept = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0).expand(NUM_KV_HEADS, -1).clone()
        bookkeeping = [{"kept_token_indices": [kept.clone()], "evicted_token_num": 0} for _ in range(NUM_LAYERS)]
    return ModelStateSnapshot(
        key_cache=keys,
        value_cache=values,
        query_cache={},
        compression_flags_per_layer=["none"] * NUM_LAYERS,
        model_length=SEQ_LEN,
        after_think=None,
        absolute_position=SEQ_LEN,
        provenance=provenance,
        kv_cluster_bookkeeping_per_layer=bookkeeping,
    )


def _mutation(layer, head, position, value=900.0, donor=None, candidate=None) -> KVMutationSpec:
    donor = position if donor is None else donor
    candidate = position if candidate is None else candidate
    return KVMutationSpec(
        layer_index=layer,
        kv_head_index=head,
        token_position=position,
        replacement_key=torch.full((HEAD_DIM,), value),
        replacement_value=torch.full((HEAD_DIM,), -value),
        donor_absolute_position=donor,
        candidate_absolute_position=candidate,
    )


# --- single-mutation (equivalent to Arm C1-C4) ---

def test_single_mutation_changes_exactly_one_slot():
    state = _snapshot()
    before = state.clone()
    result = apply_structured_kv_restore(state, [_mutation(1, 1, 2, donor=2, candidate=99)])
    assert result.key_slots_changed == 1
    assert result.value_slots_changed == 1
    assert not result.is_noop
    assert torch.equal(state.key_cache[1][0, 1, 2], torch.full((HEAD_DIM,), 900.0))
    assert torch.equal(state.value_cache[1][0, 1, 2], torch.full((HEAD_DIM,), -900.0))
    for layer in range(NUM_LAYERS):
        mask = torch.ones_like(state.key_cache[layer], dtype=torch.bool)
        if layer == 1:
            mask[0, 1, 2] = False
        assert torch.equal(state.key_cache[layer][mask], before.key_cache[layer][mask])
        assert torch.equal(state.value_cache[layer][mask], before.value_cache[layer][mask])
    assert state.provenance.layers[1].positions[1, 2].item() == 99


# --- all-KV-heads-at-one-layer (Arm H) ---

def test_all_kv_heads_same_layer_changes_exactly_that_layers_named_slots():
    state = _snapshot()
    before = state.clone()
    mutations = [_mutation(2, head, 4, value=100.0 + head, donor=4, candidate=77) for head in range(NUM_KV_HEADS)]
    result = apply_structured_kv_restore(state, mutations)
    assert result.key_slots_changed == NUM_KV_HEADS
    assert result.value_slots_changed == NUM_KV_HEADS
    for head in range(NUM_KV_HEADS):
        assert torch.equal(state.key_cache[2][0, head, 4], torch.full((HEAD_DIM,), 100.0 + head))
    mask = torch.ones_like(state.key_cache[2], dtype=torch.bool)
    mask[:, :, 4, :] = False
    assert torch.equal(state.key_cache[2][mask], before.key_cache[2][mask])
    for layer in (0, 1, 3):
        assert torch.equal(state.key_cache[layer], before.key_cache[layer])
        assert torch.equal(state.value_cache[layer], before.value_cache[layer])
    # num_key_value_heads used, never a query-head count
    assert result.key_slots_changed == state.key_cache[2].shape[1]


# --- one KV head across multiple layers (Arm L) ---

def test_single_head_across_layers_each_layer_uses_own_values_no_cross_layer_reuse():
    state = _snapshot()
    before = state.clone()
    mutations = [_mutation(layer, 0, 3, value=500.0 + layer, donor=3, candidate=42) for layer in (0, 1, 3)]
    result = apply_structured_kv_restore(state, mutations)
    assert result.key_slots_changed == 3
    for layer in (0, 1, 3):
        assert torch.equal(state.key_cache[layer][0, 0, 3], torch.full((HEAD_DIM,), 500.0 + layer))
        # unselected heads at this layer/position unchanged
        for head in (1, 2):
            assert torch.equal(state.key_cache[layer][0, head, 3], before.key_cache[layer][0, head, 3])
    assert torch.equal(state.key_cache[2], before.key_cache[2])


# --- all heads across multiple layers (Arm HL) ---

def test_all_heads_across_layers_reconstructs_exact_mutation_count():
    state = _snapshot()
    layers = (0, 1, 3)
    mutations = [
        _mutation(layer, head, 5, value=10.0 * layer + head, donor=5, candidate=88)
        for layer in layers
        for head in range(NUM_KV_HEADS)
    ]
    result = apply_structured_kv_restore(state, mutations)
    assert result.key_slots_changed == len(layers) * NUM_KV_HEADS
    assert result.value_slots_changed == len(layers) * NUM_KV_HEADS
    for layer in layers:
        for head in range(NUM_KV_HEADS):
            assert torch.equal(state.key_cache[layer][0, head, 5], torch.full((HEAD_DIM,), 10.0 * layer + head))


# --- three-token span (Arm S / SH) ---

def test_span_changes_exactly_three_positions_neighbors_unchanged():
    state = _snapshot()
    before = state.clone()
    mutations = [_mutation(1, 1, pos, value=200.0 + pos, donor=pos, candidate=300 + pos) for pos in (1, 2, 3)]
    result = apply_structured_kv_restore(state, mutations)
    assert result.key_slots_changed == 3
    for pos in (1, 2, 3):
        assert torch.equal(state.key_cache[1][0, 1, pos], torch.full((HEAD_DIM,), 200.0 + pos))
    for pos in (0, 4, 5):
        assert torch.equal(state.key_cache[1][0, 1, pos], before.key_cache[1][0, 1, pos])


# --- no-op ---

def test_noop_is_bit_exact_single_mutation():
    state = _snapshot()
    before = state.clone()
    key = state.key_cache[0][0, 0, 0].clone().contiguous()
    value = state.value_cache[0][0, 0, 0].clone().contiguous()
    mutation = KVMutationSpec(
        layer_index=0, kv_head_index=0, token_position=0,
        replacement_key=key, replacement_value=value,
        donor_absolute_position=13, candidate_absolute_position=13,
    )
    result = apply_structured_kv_restore(state, [mutation])
    assert result.is_noop
    assert result.key_slots_changed == 0
    assert result.value_slots_changed == 0
    for layer in range(NUM_LAYERS):
        assert torch.equal(state.key_cache[layer], before.key_cache[layer])
        assert torch.equal(state.value_cache[layer], before.value_cache[layer])


def test_noop_group_requires_every_mutation_to_be_individually_noop():
    state = _snapshot()
    key0 = state.key_cache[0][0, 0, 0].clone().contiguous()
    value0 = state.value_cache[0][0, 0, 0].clone().contiguous()
    noop_mutation = KVMutationSpec(
        layer_index=0, kv_head_index=0, token_position=0,
        replacement_key=key0, replacement_value=value0,
        donor_absolute_position=1, candidate_absolute_position=1,
    )
    real_mutation = _mutation(1, 0, 0, value=42.0, donor=0, candidate=55)
    result = apply_structured_kv_restore(state, [noop_mutation, real_mutation])
    assert not result.is_noop


# --- validation / fail-closed ---

def test_empty_mutations_rejected():
    state = _snapshot()
    with pytest.raises(GeometryRestoreError, match="empty"):
        apply_structured_kv_restore(state, [])


def test_duplicate_triple_rejected():
    state = _snapshot()
    m1 = _mutation(0, 0, 0)
    m2 = _mutation(0, 0, 0, value=1.0)
    with pytest.raises(GeometryRestoreError, match="duplicate"):
        apply_structured_kv_restore(state, [m1, m2])


def test_out_of_range_layer_rejected():
    state = _snapshot()
    with pytest.raises(GeometryRestoreError, match="layer_index"):
        apply_structured_kv_restore(state, [_mutation(NUM_LAYERS, 0, 0)])


def test_out_of_range_head_rejected():
    state = _snapshot()
    with pytest.raises(GeometryRestoreError, match="kv_head_index"):
        apply_structured_kv_restore(state, [_mutation(0, NUM_KV_HEADS, 0)])


def test_out_of_range_position_rejected():
    state = _snapshot()
    with pytest.raises(GeometryRestoreError, match="token_position"):
        apply_structured_kv_restore(state, [_mutation(0, 0, SEQ_LEN)])


def test_wrong_shape_replacement_rejected():
    state = _snapshot()
    bad = KVMutationSpec(
        layer_index=0, kv_head_index=0, token_position=0,
        replacement_key=torch.zeros(HEAD_DIM + 1), replacement_value=torch.zeros(HEAD_DIM),
        donor_absolute_position=0, candidate_absolute_position=0,
    )
    with pytest.raises(GeometryRestoreError, match="head dimension"):
        apply_structured_kv_restore(state, [bad])


def test_aliased_replacement_rejected():
    state = _snapshot()
    aliased_key = state.key_cache[0][0, 0, 0, :]  # a view into the cache itself
    bad = KVMutationSpec(
        layer_index=0, kv_head_index=0, token_position=1,
        replacement_key=aliased_key, replacement_value=torch.zeros(HEAD_DIM),
        donor_absolute_position=0, candidate_absolute_position=1,
    )
    with pytest.raises(GeometryRestoreError, match="alias"):
        apply_structured_kv_restore(state, [bad])


# --- physical-position resolution / valid-layer discovery ---

def test_resolve_physical_position_finds_slot():
    state = _snapshot()
    layer0_positions = state.provenance.layers[0].positions
    assert resolve_physical_position(layer0_positions, 1, 3) == 3


def test_resolve_physical_position_returns_none_when_absent():
    state = _snapshot()
    layer2_positions = state.provenance.layers[2].positions
    assert resolve_physical_position(layer2_positions, 1, 3) is None


def test_resolve_valid_layers_excludes_layer_missing_a_required_position():
    state = _snapshot()
    valid = resolve_valid_layers(
        state,
        kv_head_index=1,
        required_absolute_positions=[3, 4],
        candidate_layers=[0, 1, 2, 3],
    )
    assert 2 not in valid  # layer 2/head 1 has evicted absolute position 3
    assert set(valid) == {0, 1, 3}
    assert valid[0] == {3: 3, 4: 4}


def test_resolve_valid_layers_empty_without_provenance():
    state = _snapshot(with_provenance=False)
    valid = resolve_valid_layers(state, kv_head_index=0, required_absolute_positions=[0], candidate_layers=[0, 1])
    assert valid == {}
