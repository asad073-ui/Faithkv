"""Outcome-blind qualification, event scoring, and candidate-pool helpers."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

from kvcot.discovery.diagnostic_pilot_contract import (
    MAXIMUM_CANDIDATE_POOL_SIZE,
    MAXIMUM_QUALIFICATION_CANDIDATES,
    MAXIMUM_SELECTED_EXAMPLES,
    RESTORE_ARM,
)

FORBIDDEN_QUALIFICATION_FIELDS = frozenset(
    {
        "swap_gain",
        "gain",
        "candidate_gain",
        "answer_margin_after",
        "intervention_answer",
        "intervention_correct",
        "candidate_outcomes",
    }
)


@dataclass(frozen=True)
class CandidateScore:
    absolute_token_position: int
    deployable_score: float


@dataclass(frozen=True)
class FrozenCandidatePool:
    candidates: tuple[CandidateScore, ...]
    diagnostic_only: bool = True
    deployable_performance: bool = False


def freeze_candidate_pool(candidates: Iterable[CandidateScore]) -> FrozenCandidatePool:
    unique: dict[int, CandidateScore] = {}
    for candidate in candidates:
        if not math.isfinite(candidate.deployable_score):
            raise ValueError("candidate deployable score must be finite")
        previous = unique.get(candidate.absolute_token_position)
        if previous is not None and previous != candidate:
            raise ValueError("one token position has conflicting deployable scores")
        unique[candidate.absolute_token_position] = candidate
    ordered = sorted(unique.values(), key=lambda row: (-row.deployable_score, row.absolute_token_position))
    return FrozenCandidatePool(candidates=tuple(ordered[:MAXIMUM_CANDIDATE_POOL_SIZE]))


def bounded_local_candidate_maximum(pair_records: Sequence[dict[str, Any]]) -> float:
    """Maximum gain over a bounded, score-prioritized candidate pool.

    This is a bounded local candidate maximum over at most four frozen
    candidates at one event in one layer.  It is not a causal oracle, not a
    global upper bound, and not a deployable performance number.
    """
    if not pair_records:
        raise ValueError("a bounded candidate maximum requires at least one primitive pair")
    values = []
    for record in pair_records:
        if record.get("arm") != RESTORE_ARM or record.get("diagnostic_only") is not True:
            raise ValueError("bounded candidate records must be explicitly diagnostic restores")
        value = record.get("swap_gain")
        if type(value) not in (float, int) or not math.isfinite(float(value)):
            raise ValueError("candidate pair gain must be finite")
        values.append(float(value))
    return max(values)


def _assert_no_outcome_fields(value: Any, *, path: str = "row") -> None:
    if isinstance(value, dict):
        present = FORBIDDEN_QUALIFICATION_FIELDS & set(value)
        if present:
            raise ValueError(f"qualification input {path} contains forbidden outcome fields: {sorted(present)}")
        for key, child in value.items():
            _assert_no_outcome_fields(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_outcome_fields(child, path=f"{path}[{index}]")


def mechanically_qualifies(row: dict[str, Any]) -> bool:
    """Decide whether one candidate row mechanically qualifies.

    Zero selected events is a legitimate, ordinary non-qualifying outcome —
    an example whose compaction events produced no eligible event plan is
    simply not usable, not malformed.  A structural error (a wrong type, an
    impossible count, or a self-contradictory row) still raises, because
    silently converting malformed evidence to ``False`` would let a broken
    worker record masquerade as a clean scientific non-qualifier.
    """
    _assert_no_outcome_fields(row)
    required_bools = (
        "fullkv_execution_valid",
        "rkv_replay_mechanically_valid",
        "correctness_status_matched",
        "meaningful_compression",
        "eligible_event_exists",
        "selected_event_has_two_candidates",
        "intervention_not_evaluated",
    )
    for field in required_bools:
        if type(row.get(field)) is not bool:
            raise ValueError(f"qualification field {field!r} must be a strict bool")
    if type(row.get("candidate_ordinal")) is not int:
        raise ValueError("candidate_ordinal must be a strict int")

    selected_event_count = row.get("selected_event_count")
    # ``bool`` is a subclass of ``int``; an identity check keeps ``True``
    # from being accepted as the integer one.
    if type(selected_event_count) is not int:
        raise ValueError("selected_event_count must be a strict int")
    if selected_event_count not in (0, 1):
        raise ValueError("selected_event_count must be zero or one")

    if selected_event_count == 0:
        if row["eligible_event_exists"]:
            raise ValueError("zero selected events conflicts with eligible_event_exists")
        if row["selected_event_has_two_candidates"]:
            raise ValueError("zero selected events conflicts with candidate availability")
        return False

    if not row["eligible_event_exists"]:
        raise ValueError("one selected event requires eligible_event_exists")

    return all(row[field] for field in required_bools)


def select_first_three_qualified(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    if len(rows) > MAXIMUM_QUALIFICATION_CANDIDATES:
        raise ValueError("qualification input exceeds the frozen maximum candidate count")
    ordinals = [row.get("candidate_ordinal") for row in rows]
    if ordinals != list(range(len(rows))):
        raise ValueError("qualification rows must remain in canonical contiguous manifest order")
    selected: list[dict[str, Any]] = []
    for row in rows:
        if mechanically_qualifies(row):
            selected.append(row)
            if len(selected) == MAXIMUM_SELECTED_EXAMPLES:
                break
    return tuple(selected)


def choose_event_by_deployable_score(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Choose maximum score, then the lower event index.

    The function accepts no intervention outcome.  Callers persist its input
    before evaluating any branch.
    """
    if not events:
        raise ValueError("at least one eligible event is required")
    for event in events:
        _assert_no_outcome_fields(event, path="event")
        if type(event.get("event_index")) is not int:
            raise ValueError("event_index must be a strict int")
        score = event.get("deployable_event_score")
        if type(score) not in (float, int) or not math.isfinite(float(score)):
            raise ValueError("deployable_event_score must be finite")
    return min(events, key=lambda event: (-float(event["deployable_event_score"]), event["event_index"]))
