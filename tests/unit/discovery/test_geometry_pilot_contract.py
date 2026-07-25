import pytest

from kvcot.discovery.geometry_pilot_contract import (
    EXPECTED_KV_HEADS,
    GeometryPilotClassification as C,
    attach_canonical_hash,
    canonical_payload_hash,
    classify_geometry_pilot,
    resolve_all_kv_heads,
    verify_canonical_hash,
)


def _base_kwargs(**overrides):
    kwargs = dict(
        evidence_complete=True,
        noop_exact=True,
        rank_zero_gain=0.0,
        non_rank_zero_gains=[0.0, 0.0, 0.0],
        head_coverage_gain=0.0,
        layer_coverage_gain=0.0,
        head_layer_interaction_gain=0.0,
        span_gain=0.0,
        span_head_gain=0.0,
        span_available=True,
        layer_set_available=True,
        behavioural_change=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_classification_i_mechanically_invalid_on_incomplete_evidence():
    assert classify_geometry_pilot(**_base_kwargs(evidence_complete=False)) == C.MECHANICALLY_INVALID


def test_classification_i_mechanically_invalid_on_noop_mismatch():
    assert classify_geometry_pilot(**_base_kwargs(noop_exact=False)) == C.MECHANICALLY_INVALID


def test_classification_a_original_candidate_works():
    assert classify_geometry_pilot(**_base_kwargs(rank_zero_gain=0.02)) == C.ORIGINAL_CANDIDATE_WORKS


def test_classification_b_candidate_selection_rescue():
    result = classify_geometry_pilot(**_base_kwargs(rank_zero_gain=0.001, non_rank_zero_gains=[0.02]))
    assert result == C.CANDIDATE_SELECTION_RESCUE


def test_classification_c_head_coverage_rescue():
    result = classify_geometry_pilot(**_base_kwargs(head_coverage_gain=0.02))
    assert result == C.HEAD_COVERAGE_RESCUE


def test_classification_d_layer_coverage_rescue():
    result = classify_geometry_pilot(**_base_kwargs(layer_coverage_gain=0.02))
    assert result == C.LAYER_COVERAGE_RESCUE


def test_classification_d_ignored_when_layer_set_unavailable():
    result = classify_geometry_pilot(**_base_kwargs(layer_coverage_gain=0.02, layer_set_available=False))
    assert result != C.LAYER_COVERAGE_RESCUE


def test_classification_e_head_layer_interaction():
    result = classify_geometry_pilot(**_base_kwargs(head_layer_interaction_gain=0.02))
    assert result == C.HEAD_LAYER_INTERACTION


def test_classification_f_token_span_rescue():
    result = classify_geometry_pilot(**_base_kwargs(span_gain=0.02))
    assert result == C.TOKEN_SPAN_RESCUE


def test_classification_f_ignored_when_span_unavailable():
    result = classify_geometry_pilot(**_base_kwargs(span_gain=0.02, span_available=False))
    assert result != C.TOKEN_SPAN_RESCUE


def test_classification_g_readout_only_movement():
    result = classify_geometry_pilot(**_base_kwargs(behavioural_change=True))
    assert result == C.READOUT_ONLY_MOVEMENT


def test_classification_h_structured_restoration_flat():
    assert classify_geometry_pilot(**_base_kwargs()) == C.STRUCTURED_RESTORATION_FLAT


def test_precedence_a_beats_everything_else():
    result = classify_geometry_pilot(
        **_base_kwargs(
            rank_zero_gain=0.02,
            non_rank_zero_gains=[0.02],
            head_coverage_gain=0.02,
            layer_coverage_gain=0.02,
            behavioural_change=True,
        )
    )
    assert result == C.ORIGINAL_CANDIDATE_WORKS


def test_precedence_c_beats_d_e_f_g():
    result = classify_geometry_pilot(
        **_base_kwargs(
            head_coverage_gain=0.02,
            layer_coverage_gain=0.02,
            head_layer_interaction_gain=0.02,
            span_gain=0.02,
            behavioural_change=True,
        )
    )
    assert result == C.HEAD_COVERAGE_RESCUE


def test_missing_branch_does_not_count_as_flat():
    # None values for a not-yet-run branch must never be treated as "moved";
    # only a genuinely completed and available branch can rescue the null.
    result = classify_geometry_pilot(
        **_base_kwargs(
            head_coverage_gain=None,
            layer_coverage_gain=None,
            head_layer_interaction_gain=None,
            span_gain=None,
            span_head_gain=None,
        )
    )
    assert result == C.STRUCTURED_RESTORATION_FLAT


def test_unavailable_span_distinct_from_zero_effect_span():
    zero_effect = classify_geometry_pilot(**_base_kwargs(span_gain=0.0, span_available=True))
    unavailable = classify_geometry_pilot(**_base_kwargs(span_gain=None, span_available=False))
    assert zero_effect == unavailable == C.STRUCTURED_RESTORATION_FLAT
    # but a positive gain behaves differently depending on availability
    assert classify_geometry_pilot(**_base_kwargs(span_gain=0.5, span_available=True)) == C.TOKEN_SPAN_RESCUE
    assert classify_geometry_pilot(**_base_kwargs(span_gain=0.5, span_available=False)) != C.TOKEN_SPAN_RESCUE


def test_resolve_all_kv_heads_uses_kv_head_count_not_query_heads():
    assert resolve_all_kv_heads(EXPECTED_KV_HEADS) == tuple(range(EXPECTED_KV_HEADS))


def test_resolve_all_kv_heads_rejects_wrong_count():
    with pytest.raises(ValueError):
        resolve_all_kv_heads(32)  # query-head count must never be accepted


def test_canonical_hash_roundtrip():
    payload = {"a": 1, "b": [1, 2, 3]}
    stamped = attach_canonical_hash(payload)
    verify_canonical_hash(stamped)
    assert canonical_payload_hash(payload) == stamped["canonical_sha256"]


def test_canonical_hash_detects_tamper():
    stamped = attach_canonical_hash({"a": 1})
    stamped["a"] = 2
    with pytest.raises(ValueError, match="mismatch"):
        verify_canonical_hash(stamped)
