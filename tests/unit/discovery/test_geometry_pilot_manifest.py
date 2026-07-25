import pytest
import torch

from kvcot.discovery.geometry_pilot_contract import (
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
    PRIMARY_DONOR_ABSOLUTE_POSITION,
    RANK_ONE_CANDIDATE_ABSOLUTE_POSITION,
    RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION,
)
from kvcot.discovery.geometry_pilot_manifest import (
    FROZEN_BRANCH_INTENTS,
    GeometryManifestError,
    candidate_pool_manifest,
    freeze_layer_set,
    freeze_span_availability,
    resolve_all_kv_head_indices,
)
from kvcot.generation.provenance import LayerProvenance, ModelProvenance
from kvcot.generation.state import ModelStateSnapshot


def test_frozen_branch_intents_cover_exactly_eight_arms_within_ten_ceiling():
    arms = [intent.arm for intent in FROZEN_BRANCH_INTENTS]
    assert arms == [ARM_C1, ARM_C2, ARM_H, ARM_L, ARM_HL, ARM_S, ARM_SH, ARM_NOOP]
    assert len(arms) == 8 <= 10


def test_candidate_arms_use_rank_zero_and_rank_one_respectively():
    by_arm = {intent.arm: intent for intent in FROZEN_BRANCH_INTENTS}
    assert by_arm[ARM_C1].candidate_absolute_positions == (RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION,)
    assert by_arm[ARM_C2].candidate_absolute_positions == (RANK_ONE_CANDIDATE_ABSOLUTE_POSITION,)
    assert not by_arm[ARM_C1].uses_all_kv_heads and not by_arm[ARM_C1].uses_valid_layer_set
    assert not by_arm[ARM_C2].uses_all_kv_heads and not by_arm[ARM_C2].uses_valid_layer_set


def test_head_arm_flags_all_kv_heads_single_layer():
    by_arm = {intent.arm: intent for intent in FROZEN_BRANCH_INTENTS}
    assert by_arm[ARM_H].uses_all_kv_heads
    assert not by_arm[ARM_H].uses_valid_layer_set


def test_layer_arm_flags_valid_layer_set_single_head():
    by_arm = {intent.arm: intent for intent in FROZEN_BRANCH_INTENTS}
    assert by_arm[ARM_L].uses_valid_layer_set
    assert not by_arm[ARM_L].uses_all_kv_heads


def test_head_layer_arm_flags_both():
    by_arm = {intent.arm: intent for intent in FROZEN_BRANCH_INTENTS}
    assert by_arm[ARM_HL].uses_all_kv_heads and by_arm[ARM_HL].uses_valid_layer_set


def test_span_arms_cover_exactly_three_contiguous_positions():
    by_arm = {intent.arm: intent for intent in FROZEN_BRANCH_INTENTS}
    expected = (
        RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION - 1,
        RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION,
        RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION + 1,
    )
    assert by_arm[ARM_S].candidate_absolute_positions == expected
    assert by_arm[ARM_SH].candidate_absolute_positions == expected
    assert by_arm[ARM_SH].uses_all_kv_heads
    assert not by_arm[ARM_S].uses_all_kv_heads


def test_noop_arm_candidate_equals_donor():
    by_arm = {intent.arm: intent for intent in FROZEN_BRANCH_INTENTS}
    assert by_arm[ARM_NOOP].candidate_absolute_positions == (PRIMARY_DONOR_ABSOLUTE_POSITION,)
    assert by_arm[ARM_NOOP].donor_absolute_position == PRIMARY_DONOR_ABSOLUTE_POSITION


def test_candidate_pool_manifest_is_deterministic_and_pool_kind_is_not_oracle():
    manifest = candidate_pool_manifest()
    assert manifest == candidate_pool_manifest()
    assert manifest["pool_kind"] == "bounded_score_prioritized_candidate_pool"
    ranks = [entry["rank"] for entry in manifest["pool"]]
    assert ranks == sorted(ranks)
    assert len(manifest["pool"]) <= 4


def test_resolve_all_kv_head_indices_rejects_query_head_count():
    with pytest.raises(GeometryManifestError):
        resolve_all_kv_head_indices(32)
    assert resolve_all_kv_head_indices(8) == tuple(range(8))


# --- layer-set / span-availability freezing ---

def _fake_capture_record(num_kv_heads=8, pre_seq_len=10):
    from kvcot.discovery.capture import UpdateKvCaptureRecord

    position_map = torch.arange(pre_seq_len, dtype=torch.long).unsqueeze(0).expand(num_kv_heads, -1).clone()
    key = torch.zeros(1, num_kv_heads, pre_seq_len, 2)
    value = torch.zeros(1, num_kv_heads, pre_seq_len, 2)
    return UpdateKvCaptureRecord(
        had_compaction=True,
        pre_call_key_states=key, pre_call_value_states=value,
        pre_call_key_shape=tuple(key.shape), pre_call_value_shape=tuple(value.shape),
        pre_call_dtype="float32", pre_call_device="cpu",
        recomputed_final_score=None, recomputed_attention_component=None,
        recomputed_similarity_component=None, recomputed_topk_indices=None,
        window_size=1, returned_key_states=key, returned_value_states=value,
        gather_parity_passed=True, pre_event_absolute_position_map=position_map,
        recomputed_kept_absolute_positions=None, observed_kept_absolute_positions=None,
        observed_kept_indices_parity_passed=True, parity_check_passed=True, parity_failure_reason=None,
    )


def _snapshot_with_layers(num_layers=4, num_kv_heads=8, seq_len=10):
    keys = [torch.zeros(1, num_kv_heads, seq_len, 2) for _ in range(num_layers)]
    values = [torch.zeros(1, num_kv_heads, seq_len, 2) for _ in range(num_layers)]
    layers = {}
    for layer in range(num_layers):
        positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(num_kv_heads, -1).clone()
        layers[layer] = LayerProvenance(positions=positions)
    # Layer 2 has evicted RANK_ZERO_CANDIDATE-equivalent test position 3 at the anchor head.
    layers[2].positions[ANCHOR_KV_HEAD_INDEX, 3] = 999
    return ModelStateSnapshot(
        key_cache=keys, value_cache=values, query_cache={},
        compression_flags_per_layer=["none"] * num_layers,
        model_length=seq_len, after_think=None, absolute_position=seq_len,
        provenance=ModelProvenance(layers=layers),
    )


def test_freeze_layer_set_excludes_layer_missing_required_position(monkeypatch):
    import kvcot.discovery.geometry_pilot_manifest as mod
    monkeypatch.setattr(mod, "RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION", 3)
    monkeypatch.setattr(mod, "PRIMARY_DONOR_ABSOLUTE_POSITION", 5)
    snapshot = _snapshot_with_layers()
    result = freeze_layer_set(snapshot, candidate_layers=[0, 1, 2, 3])
    assert 2 not in result["valid_layer_indices"]
    assert set(result["valid_layer_indices"]) == {0, 1, 3}
    assert result["available"] is True


def test_freeze_layer_set_unavailable_when_fewer_than_two_valid(monkeypatch):
    import kvcot.discovery.geometry_pilot_manifest as mod
    monkeypatch.setattr(mod, "RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION", 3)
    monkeypatch.setattr(mod, "PRIMARY_DONOR_ABSOLUTE_POSITION", 5)
    snapshot = _snapshot_with_layers()
    result = freeze_layer_set(snapshot, candidate_layers=[2])
    assert result["available"] is False
    assert result["unavailable_reason"] == "fewer_than_two_valid_layers"


def test_freeze_span_availability_all_present(monkeypatch):
    import kvcot.discovery.geometry_pilot_manifest as mod
    monkeypatch.setattr(mod, "ANCHOR_LAYER_INDEX", 1)
    monkeypatch.setattr(mod, "RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION", 4)
    snapshot = _snapshot_with_layers()
    result = freeze_span_availability(snapshot, _fake_capture_record(), prompt_length=0, total_length=10)
    assert result["available"] is True
    assert len(result["physical_slots"]) == 3


def test_freeze_span_availability_missing_neighbor(monkeypatch):
    import kvcot.discovery.geometry_pilot_manifest as mod
    monkeypatch.setattr(mod, "ANCHOR_LAYER_INDEX", 1)
    monkeypatch.setattr(mod, "RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION", 4)
    snapshot = _snapshot_with_layers()
    # Force one span neighbor to be evicted at the anchor (layer, head).
    snapshot.provenance.layers[1].positions[ANCHOR_KV_HEAD_INDEX, 5] = 12345
    result = freeze_span_availability(snapshot, _fake_capture_record(), prompt_length=0, total_length=10)
    assert result["available"] is False
    assert "has_no_valid_post_event_snapshot" in result["unavailable_reason"]


def test_freeze_span_availability_outside_eligible_region(monkeypatch):
    import kvcot.discovery.geometry_pilot_manifest as mod
    monkeypatch.setattr(mod, "ANCHOR_LAYER_INDEX", 1)
    monkeypatch.setattr(mod, "RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION", 0)
    snapshot = _snapshot_with_layers(seq_len=10)
    result = freeze_span_availability(snapshot, _fake_capture_record(), prompt_length=0, total_length=10)
    assert result["available"] is False
    assert "outside_eligible_region" in result["unavailable_reason"]


def test_freeze_span_availability_distinct_from_zero_effect_case(monkeypatch):
    import kvcot.discovery.geometry_pilot_manifest as mod
    monkeypatch.setattr(mod, "ANCHOR_LAYER_INDEX", 1)
    monkeypatch.setattr(mod, "RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION", 4)
    # Unavailable (structural) is a different code path/result shape than
    # "available but happened to produce zero gain" -- callers must never
    # conflate the two; this asserts the unavailable-reason field exists
    # only in the unavailable case.
    snapshot = _snapshot_with_layers()
    available = freeze_span_availability(snapshot, _fake_capture_record(), prompt_length=0, total_length=10)
    assert available["unavailable_reason"] is None
    snapshot.provenance = None
    unavailable = freeze_span_availability(snapshot, _fake_capture_record(), prompt_length=0, total_length=10)
    assert unavailable["unavailable_reason"] is not None
