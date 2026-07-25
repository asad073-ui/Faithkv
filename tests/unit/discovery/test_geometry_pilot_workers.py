"""Direct CPU tests for the per-arm mutation-construction wiring in
`kvcot.discovery.geometry_pilot_workers` -- the production code that
resolves each arm's real `(layer, kv_head, position)` mutations from a
pristine snapshot AND the anchor event's pre-event capture record.

This fixture models the REALISTIC eviction scenario a live GPU run
actually produces (discovered only by running against the real model,
after CPU tests using an unrealistic "candidate still present everywhere
post-event" fixture had missed it): the rank-zero candidate is evicted BY
the anchor event at the anchor (layer, head) itself, so it is only
resolvable via the PRE-event captured state there -- never via the
POST-event snapshot, which no longer holds it at that exact (layer, head)
by construction of the eviction that just happened. At every OTHER layer
(and, in this fixture, at the anchor layer's OTHER head), the candidate
was never evicted and remains resolvable via the POST-event snapshot.
"""
import pytest
import torch

from kvcot.discovery.geometry_pilot_contract import ARM_C1, ARM_C2, ARM_H, ARM_HL, ARM_L, ARM_NOOP, ARM_S, ARM_SH
from kvcot.discovery.geometry_pilot_manifest import freeze_layer_set, freeze_span_availability
from kvcot.discovery.geometry_pilot_workers import (
    GeometryWorkerError,
    _mutation_for,
    _resolve_post_event_vector,
    _resolve_pre_event_vector,
    build_frozen_branch_mutations,
)
from kvcot.discovery.capture import UpdateKvCaptureRecord
from kvcot.generation.provenance import LayerProvenance, ModelProvenance
from kvcot.generation.state import ModelStateSnapshot

NUM_LAYERS = 3
NUM_KV_HEADS = 2
HEAD_DIM = 2
POST_SEQ_LEN = 6
PRE_SEQ_LEN = 7

ANCHOR_LAYER = 1
ANCHOR_HEAD = 1
MIDDLE = 5   # RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION
RANK_ONE = 3
DONOR = 2    # PRIMARY_DONOR_ABSOLUTE_POSITION
LEFT_NEIGHBOR = MIDDLE - 1   # 4
RIGHT_NEIGHBOR = MIDDLE + 1  # 6


def _pre_event_value(head: int, slot: int) -> float:
    return 100.0 * head + 10.0 * slot


def _post_event_value(layer: int, head: int, slot: int) -> float:
    # Deliberately a DIFFERENT formula from pre-event so tests can prove
    # which source a mutation's content actually came from.
    return -1000.0 * layer - 100.0 * head - 10.0 * slot


def _capture_record() -> UpdateKvCaptureRecord:
    # Pre-event: positions occupy slots identically to their absolute
    # index (no eviction has happened yet at this exact call).
    pre_position_map = torch.arange(PRE_SEQ_LEN, dtype=torch.long).unsqueeze(0).expand(NUM_KV_HEADS, -1).clone()
    pre_key = torch.zeros(1, NUM_KV_HEADS, PRE_SEQ_LEN, HEAD_DIM)
    pre_value = torch.zeros(1, NUM_KV_HEADS, PRE_SEQ_LEN, HEAD_DIM)
    for head in range(NUM_KV_HEADS):
        for slot in range(PRE_SEQ_LEN):
            pre_key[0, head, slot, :] = _pre_event_value(head, slot)
            pre_value[0, head, slot, :] = _pre_event_value(head, slot) + 0.5
    return UpdateKvCaptureRecord(
        had_compaction=True,
        pre_call_key_states=pre_key,
        pre_call_value_states=pre_value,
        pre_call_key_shape=tuple(pre_key.shape),
        pre_call_value_shape=tuple(pre_value.shape),
        pre_call_dtype="float32",
        pre_call_device="cpu",
        recomputed_final_score=None,
        recomputed_attention_component=None,
        recomputed_similarity_component=None,
        recomputed_topk_indices=None,
        window_size=1,
        returned_key_states=pre_key,
        returned_value_states=pre_value,
        gather_parity_passed=True,
        pre_event_absolute_position_map=pre_position_map,
        recomputed_kept_absolute_positions=None,
        observed_kept_absolute_positions=None,
        observed_kept_indices_parity_passed=True,
        parity_check_passed=True,
        parity_failure_reason=None,
    )


def _snapshot() -> ModelStateSnapshot:
    keys = [torch.zeros(1, NUM_KV_HEADS, POST_SEQ_LEN, HEAD_DIM) for _ in range(NUM_LAYERS)]
    values = [torch.zeros(1, NUM_KV_HEADS, POST_SEQ_LEN, HEAD_DIM) for _ in range(NUM_LAYERS)]
    layers = {}

    for layer in range(NUM_LAYERS):
        if layer == ANCHOR_LAYER:
            # head ANCHOR_HEAD: MIDDLE (5) was evicted BY this event --
            # absent post-event. Survivors: 0,1,2,3,4,6.
            head_anchor_positions = [0, 1, 2, 3, 4, 6]
            # the OTHER head at the anchor layer: some OTHER position (not
            # MIDDLE) was evicted instead -- MIDDLE remains present.
            head_other_positions = [0, 1, 2, 4, 5, 6]
            positions = torch.zeros(NUM_KV_HEADS, POST_SEQ_LEN, dtype=torch.long)
            for h in range(NUM_KV_HEADS):
                positions[h] = torch.tensor(head_anchor_positions if h == ANCHOR_HEAD else head_other_positions)
        else:
            # Non-anchor layers: nothing evicted recently here -- MIDDLE
            # and DONOR are both still present (first POST_SEQ_LEN
            # absolute positions, unchanged).
            positions = torch.arange(POST_SEQ_LEN, dtype=torch.long).unsqueeze(0).expand(NUM_KV_HEADS, -1).clone()
        layers[layer] = LayerProvenance(positions=positions)

        for h in range(NUM_KV_HEADS):
            for slot in range(POST_SEQ_LEN):
                keys[layer][0, h, slot, :] = _post_event_value(layer, h, slot)
                values[layer][0, h, slot, :] = _post_event_value(layer, h, slot) + 0.5

    return ModelStateSnapshot(
        key_cache=keys, value_cache=values, query_cache={},
        compression_flags_per_layer=["none"] * NUM_LAYERS,
        model_length=POST_SEQ_LEN, after_think=None, absolute_position=POST_SEQ_LEN,
        provenance=ModelProvenance(layers=layers),
    )


@pytest.fixture(autouse=True)
def _patch_frozen_identities(monkeypatch):
    import kvcot.discovery.geometry_pilot_workers as mod
    import kvcot.discovery.geometry_pilot_manifest as manifest_mod

    for target in (mod, manifest_mod):
        monkeypatch.setattr(target, "ANCHOR_LAYER_INDEX", ANCHOR_LAYER, raising=False)
        monkeypatch.setattr(target, "ANCHOR_KV_HEAD_INDEX", ANCHOR_HEAD, raising=False)
        monkeypatch.setattr(target, "RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION", MIDDLE, raising=False)
        monkeypatch.setattr(target, "PRIMARY_DONOR_ABSOLUTE_POSITION", DONOR, raising=False)
    monkeypatch.setattr(mod, "RANK_ONE_CANDIDATE_ABSOLUTE_POSITION", RANK_ONE, raising=False)
    monkeypatch.setattr(manifest_mod, "EXPECTED_KV_HEADS", NUM_KV_HEADS, raising=False)


# --- low-level resolvers ---

def test_resolve_pre_event_vector_matches_pre_event_formula():
    record = _capture_record()
    key, value = _resolve_pre_event_vector(record, kv_head_index=ANCHOR_HEAD, absolute_position=MIDDLE)
    assert torch.equal(key, torch.full((HEAD_DIM,), _pre_event_value(ANCHOR_HEAD, MIDDLE)))


def test_resolve_pre_event_vector_none_when_absent_from_pre_event_map():
    record = _capture_record()
    key, value = _resolve_pre_event_vector(record, kv_head_index=ANCHOR_HEAD, absolute_position=999)
    assert key is None and value is None


def test_resolve_post_event_vector_matches_post_event_formula():
    snapshot = _snapshot()
    key, value = _resolve_post_event_vector(snapshot, layer_index=0, kv_head_index=0, absolute_position=DONOR)
    assert torch.equal(key, torch.full((HEAD_DIM,), _post_event_value(0, 0, DONOR)))


def test_resolve_post_event_vector_none_for_evicted_candidate_at_anchor():
    snapshot = _snapshot()
    key, value = _resolve_post_event_vector(
        snapshot, layer_index=ANCHOR_LAYER, kv_head_index=ANCHOR_HEAD, absolute_position=MIDDLE
    )
    assert key is None and value is None  # confirms the realistic-eviction premise of this fixture


# --- _mutation_for: the critical pre/post routing ---

def test_mutation_for_candidate_at_anchor_uses_pre_event_content():
    snapshot = _snapshot()
    record = _capture_record()
    mutation = _mutation_for(
        snapshot, capture_record=record, anchor_layer_index=ANCHOR_LAYER,
        layer_index=ANCHOR_LAYER, kv_head_index=ANCHOR_HEAD,
        candidate_absolute_position=MIDDLE, donor_absolute_position=DONOR,
    )
    assert torch.equal(mutation.replacement_key, torch.full((HEAD_DIM,), _pre_event_value(ANCHOR_HEAD, MIDDLE)))
    # donor's WRITE-TARGET physical slot resolves from the post-event layout
    donor_positions = snapshot.provenance.layers[ANCHOR_LAYER].positions[ANCHOR_HEAD]
    expected_slot = (donor_positions == DONOR).nonzero(as_tuple=True)[0].item()
    assert mutation.token_position == expected_slot


def test_mutation_for_raises_a_clear_error_if_candidate_only_checked_post_event():
    # Regression guard: using _resolve_post_event_vector directly for the
    # evicted candidate at the anchor (layer, head) must fail (this is
    # exactly the live-GPU failure this fixture was built to catch).
    snapshot = _snapshot()
    key, value = _resolve_post_event_vector(
        snapshot, layer_index=ANCHOR_LAYER, kv_head_index=ANCHOR_HEAD, absolute_position=MIDDLE
    )
    assert key is None


def test_mutation_for_candidate_at_other_layer_uses_post_event_content():
    snapshot = _snapshot()
    record = _capture_record()
    mutation = _mutation_for(
        snapshot, capture_record=record, anchor_layer_index=ANCHOR_LAYER,
        layer_index=0, kv_head_index=ANCHOR_HEAD,
        candidate_absolute_position=MIDDLE, donor_absolute_position=DONOR,
    )
    assert torch.equal(mutation.replacement_key, torch.full((HEAD_DIM,), _post_event_value(0, ANCHOR_HEAD, MIDDLE)))


def test_mutation_for_noop_candidate_equals_donor_uses_post_event_content():
    snapshot = _snapshot()
    record = _capture_record()
    mutation = _mutation_for(
        snapshot, capture_record=record, anchor_layer_index=ANCHOR_LAYER,
        layer_index=ANCHOR_LAYER, kv_head_index=ANCHOR_HEAD,
        candidate_absolute_position=DONOR, donor_absolute_position=DONOR,
    )
    assert torch.equal(mutation.replacement_key, torch.full((HEAD_DIM,), _post_event_value(ANCHOR_LAYER, ANCHOR_HEAD, DONOR)))


def test_mutation_for_raises_when_candidate_unresolvable_anywhere():
    snapshot = _snapshot()
    record = _capture_record()
    with pytest.raises(GeometryWorkerError, match="not resolvable"):
        _mutation_for(
            snapshot, capture_record=record, anchor_layer_index=ANCHOR_LAYER,
            layer_index=ANCHOR_LAYER, kv_head_index=ANCHOR_HEAD,
            candidate_absolute_position=12345, donor_absolute_position=DONOR,
        )


def test_mutation_for_raises_when_donor_unresolvable():
    snapshot = _snapshot()
    record = _capture_record()
    with pytest.raises(GeometryWorkerError, match="donor"):
        _mutation_for(
            snapshot, capture_record=record, anchor_layer_index=ANCHOR_LAYER,
            layer_index=ANCHOR_LAYER, kv_head_index=ANCHOR_HEAD,
            candidate_absolute_position=MIDDLE, donor_absolute_position=99999,
        )


# --- build_frozen_branch_mutations: full arm construction ---

def test_c1_and_c2_use_pre_event_candidate_content():
    snapshot = _snapshot()
    record = _capture_record()
    layer_set = freeze_layer_set(snapshot, candidate_layers=range(NUM_LAYERS))
    span_availability = freeze_span_availability(snapshot, record, prompt_length=0, total_length=PRE_SEQ_LEN)
    mutations = build_frozen_branch_mutations(
        snapshot, record, num_key_value_heads=NUM_KV_HEADS, layer_set=layer_set, span_availability=span_availability,
    )
    c1 = mutations[ARM_C1][0]
    assert torch.equal(c1.replacement_key, torch.full((HEAD_DIM,), _pre_event_value(ANCHOR_HEAD, MIDDLE)))
    c2 = mutations[ARM_C2][0]
    assert torch.equal(c2.replacement_key, torch.full((HEAD_DIM,), _pre_event_value(ANCHOR_HEAD, RANK_ONE)))


def test_arm_h_all_heads_use_pre_event_content_at_anchor_layer():
    snapshot = _snapshot()
    record = _capture_record()
    layer_set = freeze_layer_set(snapshot, candidate_layers=range(NUM_LAYERS))
    span_availability = freeze_span_availability(snapshot, record, prompt_length=0, total_length=PRE_SEQ_LEN)
    mutations = build_frozen_branch_mutations(
        snapshot, record, num_key_value_heads=NUM_KV_HEADS, layer_set=layer_set, span_availability=span_availability,
    )
    by_head = {m.kv_head_index: m for m in mutations[ARM_H]}
    assert set(by_head) == set(range(NUM_KV_HEADS))
    for head, m in by_head.items():
        assert torch.equal(m.replacement_key, torch.full((HEAD_DIM,), _pre_event_value(head, MIDDLE)))


def test_arm_l_other_layers_use_post_event_content():
    snapshot = _snapshot()
    record = _capture_record()
    layer_set = freeze_layer_set(snapshot, candidate_layers=range(NUM_LAYERS))
    assert layer_set["available"]
    span_availability = freeze_span_availability(snapshot, record, prompt_length=0, total_length=PRE_SEQ_LEN)
    mutations = build_frozen_branch_mutations(
        snapshot, record, num_key_value_heads=NUM_KV_HEADS, layer_set=layer_set, span_availability=span_availability,
    )
    by_layer = {m.layer_index: m for m in mutations[ARM_L]}
    for layer, m in by_layer.items():
        assert torch.equal(m.replacement_key, torch.full((HEAD_DIM,), _post_event_value(layer, ANCHOR_HEAD, MIDDLE)))


def test_arm_hl_reconstructs_exact_count_and_sources():
    snapshot = _snapshot()
    record = _capture_record()
    layer_set = freeze_layer_set(snapshot, candidate_layers=range(NUM_LAYERS))
    span_availability = freeze_span_availability(snapshot, record, prompt_length=0, total_length=PRE_SEQ_LEN)
    mutations = build_frozen_branch_mutations(
        snapshot, record, num_key_value_heads=NUM_KV_HEADS, layer_set=layer_set, span_availability=span_availability,
    )
    assert len(mutations[ARM_HL]) == len(layer_set["valid_layer_indices"]) * NUM_KV_HEADS
    for m in mutations[ARM_HL]:
        assert torch.equal(m.replacement_key, torch.full((HEAD_DIM,), _post_event_value(m.layer_index, m.kv_head_index, MIDDLE)))


def test_arm_s_available_middle_pre_event_neighbors_post_event_self():
    snapshot = _snapshot()
    record = _capture_record()
    span_availability = freeze_span_availability(snapshot, record, prompt_length=0, total_length=PRE_SEQ_LEN)
    assert span_availability["available"]
    layer_set = freeze_layer_set(snapshot, candidate_layers=range(NUM_LAYERS))
    mutations = build_frozen_branch_mutations(
        snapshot, record, num_key_value_heads=NUM_KV_HEADS, layer_set=layer_set, span_availability=span_availability,
    )
    s_mutations = mutations[ARM_S]
    assert len(s_mutations) == 3
    by_candidate = {m.candidate_absolute_position: m for m in s_mutations}
    assert set(by_candidate) == {MIDDLE, LEFT_NEIGHBOR, RIGHT_NEIGHBOR}

    middle_mutation = by_candidate[MIDDLE]
    assert middle_mutation.donor_absolute_position == DONOR
    assert torch.equal(middle_mutation.replacement_key, torch.full((HEAD_DIM,), _pre_event_value(ANCHOR_HEAD, MIDDLE)))

    anchor_positions = snapshot.provenance.layers[ANCHOR_LAYER].positions[ANCHOR_HEAD]
    for neighbor in (LEFT_NEIGHBOR, RIGHT_NEIGHBOR):
        m = by_candidate[neighbor]
        assert m.donor_absolute_position == neighbor  # self-referencing, honest no-op
        expected_slot = (anchor_positions == neighbor).nonzero(as_tuple=True)[0].item()
        assert torch.equal(m.replacement_key, torch.full((HEAD_DIM,), _post_event_value(ANCHOR_LAYER, ANCHOR_HEAD, expected_slot)))

    # exactly three DISTINCT physical write-target slots -- no duplicate triple
    token_positions = {m.token_position for m in s_mutations}
    assert len(token_positions) == 3


def test_arm_s_unavailable_when_a_neighbor_was_also_evicted(monkeypatch):
    snapshot = _snapshot()
    # Force the right neighbor to ALSO be absent post-event at the anchor
    # (layer, head) -- no third external donor is invented; span must be
    # recorded unavailable instead.
    positions = snapshot.provenance.layers[ANCHOR_LAYER].positions
    positions[ANCHOR_HEAD] = torch.tensor([0, 1, 2, 3, 4, 99999])
    record = _capture_record()
    span_availability = freeze_span_availability(snapshot, record, prompt_length=0, total_length=PRE_SEQ_LEN)
    assert span_availability["available"] is False
    assert f"position_{RIGHT_NEIGHBOR}_has_no_valid_post_event_snapshot" == span_availability["unavailable_reason"]

    layer_set = freeze_layer_set(snapshot, candidate_layers=range(NUM_LAYERS))
    mutations = build_frozen_branch_mutations(
        snapshot, record, num_key_value_heads=NUM_KV_HEADS, layer_set=layer_set, span_availability=span_availability,
    )
    assert mutations[ARM_S] is None
    assert mutations[ARM_SH] is None


def test_arm_s_unavailable_when_middle_has_no_pre_event_capture():
    snapshot = _snapshot()
    record = _capture_record()
    # Corrupt the pre-event map so MIDDLE cannot be found at all.
    object.__setattr__(record, "pre_event_absolute_position_map", torch.full((NUM_KV_HEADS, PRE_SEQ_LEN), -1, dtype=torch.long))
    span_availability = freeze_span_availability(snapshot, record, prompt_length=0, total_length=PRE_SEQ_LEN)
    assert span_availability["available"] is False
    assert span_availability["unavailable_reason"] == f"position_{MIDDLE}_has_no_valid_pre_event_snapshot"


def test_noop_arm_candidate_equals_donor_and_is_content_no_op():
    snapshot = _snapshot()
    record = _capture_record()
    layer_set = freeze_layer_set(snapshot, candidate_layers=range(NUM_LAYERS))
    span_availability = freeze_span_availability(snapshot, record, prompt_length=0, total_length=PRE_SEQ_LEN)
    mutations = build_frozen_branch_mutations(
        snapshot, record, num_key_value_heads=NUM_KV_HEADS, layer_set=layer_set, span_availability=span_availability,
    )
    noop = mutations[ARM_NOOP][0]
    assert noop.candidate_absolute_position == noop.donor_absolute_position == DONOR
    # content must be identical to what's already at the donor's own slot
    assert torch.equal(noop.replacement_key, torch.full((HEAD_DIM,), _post_event_value(ANCHOR_LAYER, ANCHOR_HEAD, DONOR)))
