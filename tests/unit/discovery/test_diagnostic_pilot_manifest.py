import pytest

from kvcot.discovery.diagnostic_pilot_manifest import (
    CandidateScore,
    bounded_candidate_upper_bound,
    choose_event_by_deployable_score,
    freeze_candidate_pool,
    select_first_three_qualified,
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


def test_candidate_upper_bound_reconstructs_from_primitive_gains():
    records = [
        {"arm": "candidate_upper_bound", "diagnostic_only": True, "swap_gain": value}
        for value in (-0.2, 0.004, 0.03)
    ]
    assert bounded_candidate_upper_bound(records) == 0.03
    records[0]["diagnostic_only"] = False
    with pytest.raises(ValueError, match="diagnostic"):
        bounded_candidate_upper_bound(records)


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
    with pytest.raises(ValueError, match="exactly one"):
        select_first_three_qualified([row])
