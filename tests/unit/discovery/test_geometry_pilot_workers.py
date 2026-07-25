"""Direct CPU tests for the per-arm mutation-construction wiring in
`kvcot.discovery.geometry_pilot_workers` -- the production code that
resolves each arm's real `(layer, kv_head, position)` mutations from a
pristine snapshot. Added after an independent audit found this wiring
(`build_frozen_branch_mutations`/`_mutation_for`/`_resolve_candidate_vector`)
had no direct test coverage of its own; the pre-existing geometry-pilot
tests only exercised the generic restore primitive and the contract/
manifest/authorization layers.
"""
import pytest
import torch

from kvcot.discovery.geometry_pilot_contract import ARM_C1, ARM_C2, ARM_H, ARM_HL, ARM_L, ARM_NOOP, ARM_S, ARM_SH
from kvcot.discovery.geometry_pilot_manifest import freeze_layer_set, freeze_span_availability
from kvcot.discovery.geometry_pilot_workers import (
    GeometryWorkerError,
    _mutation_for,
    _resolve_candidate_vector,
    build_frozen_branch_mutations,
)
from kvcot.generation.provenance import LayerProvenance, ModelProvenance
from kvcot.generation.state import ModelStateSnapshot

NUM_LAYERS = 4
NUM_KV_HEADS = 8
SEQ_LEN = 10
HEAD_DIM = 2

ANCHOR_LAYER = 1
ANCHOR_HEAD = 3
RANK_ZERO = 4
RANK_ONE = 5
DONOR = 2


@pytest.fixture(autouse=True)
def _patch_frozen_identities(monkeypatch):
    import kvcot.discovery.geometry_pilot_workers as mod
    import kvcot.discovery.geometry_pilot_manifest as manifest_mod

    for target in (mod, manifest_mod):
        monkeypatch.setattr(target, "ANCHOR_LAYER_INDEX", ANCHOR_LAYER, raising=False)
        monkeypatch.setattr(target, "ANCHOR_KV_HEAD_INDEX", ANCHOR_HEAD, raising=False)
        monkeypatch.setattr(target, "RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION", RANK_ZERO, raising=False)
        monkeypatch.setattr(target, "PRIMARY_DONOR_ABSOLUTE_POSITION", DONOR, raising=False)
    monkeypatch.setattr(mod, "RANK_ONE_CANDIDATE_ABSOLUTE_POSITION", RANK_ONE, raising=False)


def _snapshot(evict_position_at_layer0=None):
    keys = [
        torch.arange(NUM_KV_HEADS * SEQ_LEN * HEAD_DIM, dtype=torch.float32).reshape(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        + 1000 * layer
        for layer in range(NUM_LAYERS)
    ]
    values = [tensor.clone() + 0.5 for tensor in keys]
    layers = {}
    for layer in range(NUM_LAYERS):
        positions = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0).expand(NUM_KV_HEADS, -1).clone()
        layers[layer] = LayerProvenance(positions=positions)
    if evict_position_at_layer0 is not None:
        layers[0].positions[ANCHOR_HEAD, evict_position_at_layer0] = 9999
    return ModelStateSnapshot(
        key_cache=keys, value_cache=values, query_cache={},
        compression_flags_per_layer=["none"] * NUM_LAYERS,
        model_length=SEQ_LEN, after_think=None, absolute_position=SEQ_LEN,
        provenance=ModelProvenance(layers=layers),
    )


def test_resolve_candidate_vector_matches_direct_indexing():
    snapshot = _snapshot()
    key, value = _resolve_candidate_vector(
        snapshot, layer_index=ANCHOR_LAYER, kv_head_index=ANCHOR_HEAD, absolute_position=RANK_ZERO,
    )
    assert torch.equal(key, snapshot.key_cache[ANCHOR_LAYER][0, ANCHOR_HEAD, RANK_ZERO, :])
    assert torch.equal(value, snapshot.value_cache[ANCHOR_LAYER][0, ANCHOR_HEAD, RANK_ZERO, :])


def test_resolve_candidate_vector_none_when_position_absent():
    snapshot = _snapshot(evict_position_at_layer0=RANK_ZERO)
    key, value = _resolve_candidate_vector(
        snapshot, layer_index=0, kv_head_index=ANCHOR_HEAD, absolute_position=RANK_ZERO,
    )
    assert key is None and value is None


def test_mutation_for_raises_when_candidate_unresolvable():
    snapshot = _snapshot(evict_position_at_layer0=RANK_ZERO)
    with pytest.raises(GeometryWorkerError, match="not resolvable"):
        _mutation_for(
            snapshot, layer_index=0, kv_head_index=ANCHOR_HEAD,
            candidate_absolute_position=RANK_ZERO, donor_absolute_position=DONOR,
        )


def test_mutation_for_produces_correct_donor_slot_and_content():
    snapshot = _snapshot()
    mutation = _mutation_for(
        snapshot, layer_index=ANCHOR_LAYER, kv_head_index=ANCHOR_HEAD,
        candidate_absolute_position=RANK_ZERO, donor_absolute_position=DONOR,
    )
    assert mutation.layer_index == ANCHOR_LAYER
    assert mutation.kv_head_index == ANCHOR_HEAD
    assert mutation.token_position == DONOR  # donor's physical slot == DONOR (identity provenance in fixture)
    assert torch.equal(mutation.replacement_key, snapshot.key_cache[ANCHOR_LAYER][0, ANCHOR_HEAD, RANK_ZERO, :])
    assert mutation.donor_absolute_position == DONOR
    assert mutation.candidate_absolute_position == RANK_ZERO


def test_build_frozen_branch_mutations_c1_c2_single_triple_each():
    snapshot = _snapshot()
    layer_set = freeze_layer_set(snapshot, candidate_layers=range(NUM_LAYERS))
    span_availability = freeze_span_availability(snapshot, prompt_length=0, total_length=SEQ_LEN)
    mutations = build_frozen_branch_mutations(
        snapshot, num_key_value_heads=NUM_KV_HEADS, layer_set=layer_set, span_availability=span_availability,
    )
    assert len(mutations[ARM_C1]) == 1
    assert mutations[ARM_C1][0].candidate_absolute_position == RANK_ZERO
    assert len(mutations[ARM_C2]) == 1
    assert mutations[ARM_C2][0].candidate_absolute_position == RANK_ONE


def test_build_frozen_branch_mutations_h_covers_all_kv_heads_one_layer():
    snapshot = _snapshot()
    layer_set = freeze_layer_set(snapshot, candidate_layers=range(NUM_LAYERS))
    span_availability = freeze_span_availability(snapshot, prompt_length=0, total_length=SEQ_LEN)
    mutations = build_frozen_branch_mutations(
        snapshot, num_key_value_heads=NUM_KV_HEADS, layer_set=layer_set, span_availability=span_availability,
    )
    h_mutations = mutations[ARM_H]
    assert len(h_mutations) == NUM_KV_HEADS
    assert {m.kv_head_index for m in h_mutations} == set(range(NUM_KV_HEADS))
    assert {m.layer_index for m in h_mutations} == {ANCHOR_LAYER}


def test_build_frozen_branch_mutations_l_and_hl_none_when_layer_set_unavailable():
    snapshot = _snapshot()
    unavailable_layer_set = {"available": False, "unavailable_reason": "fewer_than_two_valid_layers", "valid_layer_indices": []}
    span_availability = freeze_span_availability(snapshot, prompt_length=0, total_length=SEQ_LEN)
    mutations = build_frozen_branch_mutations(
        snapshot, num_key_value_heads=NUM_KV_HEADS, layer_set=unavailable_layer_set, span_availability=span_availability,
    )
    assert mutations[ARM_L] is None
    assert mutations[ARM_HL] is None


def test_build_frozen_branch_mutations_l_uses_each_valid_layers_own_content():
    snapshot = _snapshot()
    layer_set = freeze_layer_set(snapshot, candidate_layers=range(NUM_LAYERS))
    assert layer_set["available"]
    span_availability = freeze_span_availability(snapshot, prompt_length=0, total_length=SEQ_LEN)
    mutations = build_frozen_branch_mutations(
        snapshot, num_key_value_heads=NUM_KV_HEADS, layer_set=layer_set, span_availability=span_availability,
    )
    l_mutations = {m.layer_index: m for m in mutations[ARM_L]}
    assert set(l_mutations) == set(layer_set["valid_layer_indices"])
    for layer, mutation in l_mutations.items():
        assert torch.equal(mutation.replacement_key, snapshot.key_cache[layer][0, ANCHOR_HEAD, RANK_ZERO, :])
        assert mutation.kv_head_index == ANCHOR_HEAD


def test_build_frozen_branch_mutations_s_sh_none_when_span_unavailable():
    snapshot = _snapshot(evict_position_at_layer0=None)
    layer_set = freeze_layer_set(snapshot, candidate_layers=range(NUM_LAYERS))
    unavailable_span = {"available": False, "unavailable_reason": "position_x_outside_eligible_region", "span_absolute_positions": [RANK_ZERO - 1, RANK_ZERO, RANK_ZERO + 1]}
    mutations = build_frozen_branch_mutations(
        snapshot, num_key_value_heads=NUM_KV_HEADS, layer_set=layer_set, span_availability=unavailable_span,
    )
    assert mutations[ARM_S] is None
    assert mutations[ARM_SH] is None


def test_build_frozen_branch_mutations_s_covers_exactly_three_positions():
    snapshot = _snapshot()
    layer_set = freeze_layer_set(snapshot, candidate_layers=range(NUM_LAYERS))
    span_availability = freeze_span_availability(snapshot, prompt_length=0, total_length=SEQ_LEN)
    assert span_availability["available"]
    mutations = build_frozen_branch_mutations(
        snapshot, num_key_value_heads=NUM_KV_HEADS, layer_set=layer_set, span_availability=span_availability,
    )
    positions = {m.candidate_absolute_position for m in mutations[ARM_S]}
    assert positions == {RANK_ZERO - 1, RANK_ZERO, RANK_ZERO + 1}
    assert all(m.layer_index == ANCHOR_LAYER and m.kv_head_index == ANCHOR_HEAD for m in mutations[ARM_S])


def test_build_frozen_branch_mutations_sh_covers_span_times_all_heads():
    snapshot = _snapshot()
    layer_set = freeze_layer_set(snapshot, candidate_layers=range(NUM_LAYERS))
    span_availability = freeze_span_availability(snapshot, prompt_length=0, total_length=SEQ_LEN)
    mutations = build_frozen_branch_mutations(
        snapshot, num_key_value_heads=NUM_KV_HEADS, layer_set=layer_set, span_availability=span_availability,
    )
    assert len(mutations[ARM_SH]) == 3 * NUM_KV_HEADS


def test_build_frozen_branch_mutations_noop_candidate_equals_donor():
    snapshot = _snapshot()
    layer_set = freeze_layer_set(snapshot, candidate_layers=range(NUM_LAYERS))
    span_availability = freeze_span_availability(snapshot, prompt_length=0, total_length=SEQ_LEN)
    mutations = build_frozen_branch_mutations(
        snapshot, num_key_value_heads=NUM_KV_HEADS, layer_set=layer_set, span_availability=span_availability,
    )
    noop = mutations[ARM_NOOP][0]
    assert noop.candidate_absolute_position == noop.donor_absolute_position == DONOR
