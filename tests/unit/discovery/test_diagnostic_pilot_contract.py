import pytest

from kvcot.discovery.diagnostic_pilot_contract import (
    CLASSIFICATION_LETTER,
    CLASSIFICATION_TEXT,
    EXECUTING_GENERATION,
    MODEL_REVISION,
    R1_GENERATION,
    R2_GENERATION,
    SWAP_GAIN_THRESHOLD_NATS,
    TOKENIZER_REVISION,
    ModelArchitecture,
    PilotClassification,
    attach_canonical_hash,
    classify_pilot,
    execution_command_document_argument,
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


def classify(**updates):
    payload = {
        "qualified_examples": 3,
        "rank_zero_width_one_gains": [0.0],
        "non_rank_zero_width_one_gains": [0.0],
        "rank_zero_width_two_gains": [0.0],
        "non_rank_zero_width_two_gains": [0.0],
        "behavioural_change": False,
        "noop_exact": True,
        "complete": True,
    }
    payload.update(updates)
    return classify_pilot(**payload)


@pytest.mark.parametrize(
    ("updates", "expected", "letter"),
    [
        # A -- the current deployable candidate at width one already works.
        ({"rank_zero_width_one_gains": [0.02]}, PilotClassification.CANDIDATE_WORKS, "A"),
        # A wins even when other cells also moved.
        (
            {
                "rank_zero_width_one_gains": [0.02],
                "non_rank_zero_width_one_gains": [0.05],
                "rank_zero_width_two_gains": [0.05],
                "non_rank_zero_width_two_gains": [0.05],
            },
            PilotClassification.CANDIDATE_WORKS,
            "A",
        ),
        # B -- candidate-selection rescue at width one.
        (
            {"non_rank_zero_width_one_gains": [0.02]},
            PilotClassification.CANDIDATE_RESCUE,
            "B",
        ),
        # C -- same candidate, width two rescues width one.
        ({"rank_zero_width_two_gains": [0.02]}, PilotClassification.WIDTH_RESCUE, "C"),
        # D -- only a non-rank-zero width-two cell moved.
        (
            {"non_rank_zero_width_two_gains": [0.02]},
            PilotClassification.INTERACTION,
            "D",
        ),
        # E -- flat NLL, moving behavioural readout.
        ({"behavioural_change": True}, PilotClassification.READOUT_ONLY, "E"),
        # F -- flat bounded diagnostic.
        ({}, PilotClassification.FLAT, "F"),
    ],
)
def test_result_classifications(updates, expected, letter):
    assert classify(**updates) == expected
    assert CLASSIFICATION_LETTER[expected] == letter


def test_threshold_is_strictly_above_0_01_and_unchanged():
    assert SWAP_GAIN_THRESHOLD_NATS == 0.01
    assert classify(rank_zero_width_one_gains=[0.01]) == PilotClassification.FLAT
    assert (
        classify(rank_zero_width_one_gains=[0.0100001])
        == PilotClassification.CANDIDATE_WORKS
    )


def test_void_and_mechanically_unqualified_fail_closed():
    assert classify(noop_exact=False) == PilotClassification.VOID
    assert classify(complete=False) == PilotClassification.VOID
    assert (
        classify(qualified_examples=2) == PilotClassification.MECHANICALLY_UNQUALIFIED
    )
    # H outranks G, and both outrank any gain-based category.
    assert (
        classify(qualified_examples=2, noop_exact=False) == PilotClassification.VOID
    )
    assert (
        classify(qualified_examples=2, rank_zero_width_one_gains=[9.0])
        == PilotClassification.MECHANICALLY_UNQUALIFIED
    )


def test_flat_classification_never_claims_a_global_kill_or_an_8b_result():
    text = CLASSIFICATION_TEXT[PilotClassification.FLAT]
    assert "DID NOT MOVE UNDER THE FROZEN" in text
    assert "DOES NOT GENERALIZE TO THE 8B OPERATING POINT" in text
    for banned in ("KILLED", "ORACLE", "GLOBAL UPPER BOUND"):
        assert banned not in text.upper()
    for classification, body in CLASSIFICATION_TEXT.items():
        assert "B2B" not in body or "NO B2B" in body, classification


def test_r2_generation_grid_budget_and_command_are_frozen():
    assert R2_GENERATION.maximum_pairs_per_selected_example == 9
    assert R2_GENERATION.maximum_total_pairs == 27
    assert R1_GENERATION.maximum_pairs_per_selected_example == 6
    assert R1_GENERATION.maximum_total_pairs == 18
    assert EXECUTING_GENERATION is R2_GENERATION
    assert (
        execution_command_document_argument(R2_GENERATION.exact_execution_command)
        == R2_GENERATION.authorization_document_path
    )
    with pytest.raises(ValueError, match="frozen execution command shape"):
        execution_command_document_argument("kvcot run-something-else --execute")


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
