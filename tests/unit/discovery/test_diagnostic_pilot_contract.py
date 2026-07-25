import pytest

from kvcot.discovery.diagnostic_pilot_contract import (
    MODEL_REVISION,
    TOKENIZER_REVISION,
    ModelArchitecture,
    PilotClassification,
    attach_canonical_hash,
    classify_pilot,
    material_margin_change,
    resolve_restore_heads,
    verify_canonical_hash,
)
from kvcot.discovery.diagnostic_pilot_workers import DiagnosticCapturePlan


def architecture(**updates):
    payload = {
        "model_type": "qwen2",
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "torch_dtype": "torch.bfloat16",
        "parameter_placement": {"every_parameter_on_cuda": True},
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
    }
    payload.update(updates)
    return payload


def test_two_kv_head_model_accepts_widths_one_and_two():
    ModelArchitecture.model_validate(architecture())
    assert resolve_restore_heads(1, selected_head=1, num_key_value_heads=2) == (1,)
    assert resolve_restore_heads(2, selected_head=1, num_key_value_heads=2) == (0, 1)


def test_width_four_rejected_and_query_heads_not_used_as_kv_heads():
    with pytest.raises(ValueError, match="prohibited"):
        resolve_restore_heads(4, selected_head=0, num_key_value_heads=2)
    with pytest.raises(ValueError, match="exactly 2 KV heads"):
        resolve_restore_heads(2, selected_head=0, num_key_value_heads=12)


def test_unpinned_revision_and_wrong_kv_count_rejected():
    with pytest.raises(ValueError, match="exact 40-character"):
        ModelArchitecture.model_validate(architecture(model_revision="main"))
    with pytest.raises(ValueError, match="num_key_value_heads"):
        ModelArchitecture.model_validate(architecture(num_key_value_heads=4))


def test_protocol_hash_is_stable_and_self_verifying():
    left = attach_canonical_hash({"b": [2, 3], "a": 1})
    right = attach_canonical_hash({"a": 1, "b": [2, 3]})
    assert left["canonical_sha256"] == right["canonical_sha256"]
    verify_canonical_hash(left)
    left["a"] = 2
    with pytest.raises(ValueError, match="mismatch"):
        verify_canonical_hash(left)


@pytest.mark.parametrize(
    ("a", "b", "behaviour", "expected"),
    [
        ([0.02], [0.0], False, PilotClassification.CANDIDATE),
        ([0.0], [0.02], False, PilotClassification.WIDTH),
        ([0.02], [0.02], False, PilotClassification.BOTH),
        ([0.0], [0.0], True, PilotClassification.READOUT),
        ([0.0], [0.0], False, PilotClassification.KILLED_15B),
    ],
)
def test_result_classifications(a, b, behaviour, expected):
    assert classify_pilot(
        qualified_examples=3,
        arm_a_gains=a,
        arm_b_gains=b,
        behavioural_change=behaviour,
        noop_exact=True,
        complete=True,
    ) == expected


def test_void_and_mechanically_unqualified_fail_closed():
    assert classify_pilot(
        qualified_examples=3, arm_a_gains=[], arm_b_gains=[], behavioural_change=False,
        noop_exact=False, complete=True,
    ) == PilotClassification.VOID
    assert classify_pilot(
        qualified_examples=2, arm_a_gains=[], arm_b_gains=[], behavioural_change=False,
        noop_exact=True, complete=True,
    ) == PilotClassification.MECHANICALLY_UNQUALIFIED
    assert classify_pilot(
        qualified_examples=3, arm_a_gains=[], arm_b_gains=[], behavioural_change=False,
        noop_exact=True, complete=False,
    ) == PilotClassification.VOID


def test_arm_c_uses_sign_change_not_small_delta():
    assert material_margin_change(-0.1, 0.1)
    assert not material_margin_change(0.1, 0.1000000001)
    assert not material_margin_change(0.0, 0.1)


def test_diagnostic_capture_plan_has_no_historical_three_event_constraint():
    trace = object()
    events = tuple(object() for _ in range(7))
    plan = DiagnosticCapturePlan(trace=trace, events=events)
    assert plan.trace is trace
    assert plan.events == events
