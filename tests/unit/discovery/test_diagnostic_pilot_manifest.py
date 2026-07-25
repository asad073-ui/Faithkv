import pytest

from kvcot.discovery.diagnostic_pilot_contract import (
    MAXIMUM_QUALIFICATION_CANDIDATES,
    R1_GENERATION,
    R2_GENERATION,
    TOKENIZER_REPOSITORY,
    TOKENIZER_REVISION,
)
from kvcot.discovery.diagnostic_pilot_manifest import (
    CandidateScore,
    bounded_local_candidate_maximum,
    choose_event_by_deployable_score,
    freeze_candidate_pool,
    mechanically_qualifies,
    select_first_three_qualified,
)
from kvcot.discovery.diagnostic_pilot_prepare import (
    DiagnosticPreparationRefused,
    prepare_runtime_inputs,
    _implementation_sha_fields,
    _pair_budget_fields,
    _verify_candidate_prompt_bindings,
)


def qualified_row(ordinal, **updates):
    row = {
        "candidate_ordinal": ordinal,
        "unique_id": f"row-{ordinal}",
        "fullkv_execution_valid": True,
        "rkv_replay_mechanically_valid": True,
        "correctness_status_matched": True,
        "meaningful_compression": True,
        "eligible_event_exists": True,
        "selected_event_has_two_candidates": True,
        "selected_event_count": 1,
        "intervention_not_evaluated": True,
    }
    row.update(updates)
    return row


def zero_event_row(ordinal, **updates):
    """The exact shape of R1's candidate 1: legitimately unqualified."""
    row = qualified_row(
        ordinal,
        eligible_event_exists=False,
        selected_event_has_two_candidates=False,
        selected_event_count=0,
    )
    row.update(updates)
    return row


def test_production_zero_event_row_is_an_ordinary_non_qualifier():
    # This is byte-for-byte the row that raised
    # `ValueError: qualification requires exactly one selected event`
    # in the consumed R1 run and voided it before any intervention.
    row = {
        "candidate_ordinal": 1,
        "fullkv_execution_valid": True,
        "rkv_replay_mechanically_valid": False,
        "correctness_status_matched": True,
        "meaningful_compression": True,
        "eligible_event_exists": False,
        "selected_event_has_two_candidates": False,
        "selected_event_count": 0,
        "intervention_not_evaluated": True,
    }
    assert mechanically_qualifies(row) is False


def test_zero_event_row_does_not_stop_the_selection_scan():
    rows = [
        qualified_row(0, meaningful_compression=False),
        zero_event_row(1),
        qualified_row(2),
        qualified_row(3),
        qualified_row(4),
    ]
    assert [row["candidate_ordinal"] for row in select_first_three_qualified(rows)] == [
        2,
        3,
        4,
    ]


def test_eight_zero_event_rows_complete_the_scan_with_zero_selections():
    rows = [zero_event_row(ordinal) for ordinal in range(8)]
    assert select_first_three_qualified(rows) == ()


def test_one_event_with_a_single_candidate_is_unqualified_without_raising():
    row = qualified_row(0, selected_event_has_two_candidates=False)
    assert mechanically_qualifies(row) is False


@pytest.mark.parametrize("value", [-1, 2, 3, True, False, "0", 0.0, None])
def test_malformed_selected_event_counts_are_rejected(value):
    row = qualified_row(0, selected_event_count=value)
    with pytest.raises(ValueError, match="selected_event_count must be"):
        mechanically_qualifies(row)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"selected_event_count": 0, "eligible_event_exists": True},
            "conflicts with eligible_event_exists",
        ),
        (
            {
                "selected_event_count": 0,
                "eligible_event_exists": False,
                "selected_event_has_two_candidates": True,
            },
            "conflicts with candidate availability",
        ),
        (
            {"selected_event_count": 1, "eligible_event_exists": False},
            "requires eligible_event_exists",
        ),
    ],
)
def test_contradictory_qualification_rows_are_structural_errors(updates, message):
    row = qualified_row(0, **updates)
    with pytest.raises(ValueError, match=message):
        mechanically_qualifies(row)


def test_malformed_rows_are_never_silently_converted_to_false():
    with pytest.raises(ValueError, match="strict bool"):
        mechanically_qualifies(qualified_row(0, meaningful_compression="yes"))
    with pytest.raises(ValueError, match="candidate_ordinal must be a strict int"):
        mechanically_qualifies(qualified_row("0"))


def test_candidate_pool_order_maximum_four_and_tie_breaking():
    pool = freeze_candidate_pool(
        [
            CandidateScore(8, 0.2),
            CandidateScore(3, 0.2),
            CandidateScore(5, 0.9),
            CandidateScore(1, -0.1),
            CandidateScore(7, 0.1),
        ]
    )
    assert [row.absolute_token_position for row in pool.candidates] == [5, 3, 8, 7]
    assert pool.diagnostic_only is True
    assert pool.deployable_performance is False


def test_bounded_local_candidate_maximum_reconstructs_from_primitive_gains():
    records = [
        {"arm": "restore", "diagnostic_only": True, "swap_gain": value}
        for value in (-0.2, 0.004, 0.03)
    ]
    assert bounded_local_candidate_maximum(records) == 0.03
    records[0]["diagnostic_only"] = False
    with pytest.raises(ValueError, match="diagnostic"):
        bounded_local_candidate_maximum(records)
    with pytest.raises(ValueError, match="at least one primitive pair"):
        bounded_local_candidate_maximum([])


def test_event_selection_is_score_then_index_and_outcome_blind():
    events = [
        {"event_index": 4, "deployable_event_score": 2.0},
        {"event_index": 2, "deployable_event_score": 2.0},
        {"event_index": 1, "deployable_event_score": 1.0},
    ]
    assert choose_event_by_deployable_score(events)["event_index"] == 2
    events[0]["swap_gain"] = 99.0
    with pytest.raises(ValueError, match="forbidden outcome"):
        choose_event_by_deployable_score(events)


def test_first_three_mechanically_qualified_in_canonical_order():
    rows = [
        qualified_row(0, meaningful_compression=False),
        qualified_row(1),
        qualified_row(2, correctness_status_matched=False),
        qualified_row(3),
        qualified_row(4),
        qualified_row(5),
    ]
    selected = select_first_three_qualified(rows)
    assert [row["candidate_ordinal"] for row in selected] == [1, 3, 4]


def test_fewer_than_three_returns_short_and_maximum_enforced():
    assert len(select_first_three_qualified([qualified_row(0), qualified_row(1)])) == 2
    with pytest.raises(ValueError, match="maximum"):
        select_first_three_qualified([qualified_row(i) for i in range(9)])


def test_gain_fields_cannot_influence_qualification():
    row = qualified_row(0)
    row["candidate_outcomes"] = [{"swap_gain": 1.0}]
    with pytest.raises(ValueError, match="forbidden outcome"):
        select_first_three_qualified([row])


def test_one_event_per_example_is_structurally_represented_by_selected_event():
    row = qualified_row(0)
    row["selected_event"] = {"event_index": 7}
    selected = select_first_three_qualified([row])
    assert selected[0]["selected_event"] == {"event_index": 7}
    row["selected_event_count"] = 2
    with pytest.raises(ValueError, match="must be zero or one"):
        select_first_three_qualified([row])


def test_external_prompt_artifact_must_equal_frozen_manifest_reconstruction():
    class SyntheticManifest:
        def __init__(self, ordinal):
            self.ordinal = ordinal

        def model_dump(self, *, mode):
            assert mode == "json"
            return {"candidate_ordinal": self.ordinal, "prompt_token_ids": [self.ordinal]}

    manifests = [SyntheticManifest(index) for index in range(MAXIMUM_QUALIFICATION_CANDIDATES)]
    prompts = {
        "artifact_schema_version": "faithkv-post-stage-c-diagnostic-prompts-v1",
        "tokenizer_repository": TOKENIZER_REPOSITORY,
        "tokenizer_revision": TOKENIZER_REVISION,
        "qualification_order": list(range(MAXIMUM_QUALIFICATION_CANDIDATES)),
        "manifests": [manifest.model_dump(mode="json") for manifest in manifests],
    }
    _verify_candidate_prompt_bindings(prompts, manifests)
    prompts["manifests"][0]["prompt_token_ids"] = [999]
    with pytest.raises(DiagnosticPreparationRefused, match="frozen manifest/tokenizer"):
        _verify_candidate_prompt_bindings(prompts, manifests)


def test_runtime_preparation_binds_generation_specific_identity():
    """R2 must bind an exact implementation SHA; R1 must not gain one."""
    assert _implementation_sha_fields(R2_GENERATION, "a" * 40) == {
        "implementation_sha": "a" * 40
    }
    assert _implementation_sha_fields(R1_GENERATION, None) == {}
    with pytest.raises(DiagnosticPreparationRefused, match="does not bind"):
        _implementation_sha_fields(R1_GENERATION, "a" * 40)
    for bad in (None, "", "A" * 40, "a" * 39, "main", 40):
        with pytest.raises(DiagnosticPreparationRefused, match="implementation SHA"):
            _implementation_sha_fields(R2_GENERATION, bad)


def test_runtime_pair_budget_vocabulary_is_generation_specific():
    assert _pair_budget_fields(R1_GENERATION) == {
        "maximum_interventions_per_selected_example": 6,
        "maximum_total_interventions": 18,
    }
    assert _pair_budget_fields(R2_GENERATION) == {
        "maximum_pairs_per_selected_example": 9,
        "maximum_total_pairs": 27,
    }


def test_preparation_refuses_roots_belonging_to_another_generation(tmp_path):
    """R2 cannot be prepared into R1's immutable roots, and vice versa."""
    for bad_root in (
        R1_GENERATION.default_output_root,
        R1_GENERATION.default_runtime_root,
    ):
        with pytest.raises(DiagnosticPreparationRefused, match="collides with the r1"):
            prepare_runtime_inputs(
                repository_root=tmp_path,
                runtime_root=bad_root,
                output_root=R2_GENERATION.default_output_root,
                generation="r2",
                implementation_sha="a" * 40,
            )
        with pytest.raises(DiagnosticPreparationRefused, match="collides with the r1"):
            prepare_runtime_inputs(
                repository_root=tmp_path,
                runtime_root=R2_GENERATION.default_runtime_root,
                output_root=bad_root,
                generation="r2",
                implementation_sha="a" * 40,
            )


def test_preparation_requires_an_absolute_canonical_output_root(tmp_path):
    with pytest.raises(DiagnosticPreparationRefused, match="absolute"):
        prepare_runtime_inputs(
            repository_root=tmp_path,
            runtime_root=R2_GENERATION.default_runtime_root,
            output_root="relative/output",
            generation="r2",
            implementation_sha="a" * 40,
        )
    with pytest.raises(DiagnosticPreparationRefused, match="must differ"):
        prepare_runtime_inputs(
            repository_root=tmp_path,
            runtime_root=R2_GENERATION.default_output_root,
            output_root=R2_GENERATION.default_output_root,
            generation="r2",
            implementation_sha="a" * 40,
        )
