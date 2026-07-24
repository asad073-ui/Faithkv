"""CPU regression tests for the B2A-R3 Stage-C R1 provenance-collector
defect: `collect_execution_provenance` must never hard-index the legacy
`LEGACY_STARTING_ANCESTOR_SHA` into an `ancestry` dict built from a
caller-supplied `required_ancestor_shas` tuple that does not contain it
(the exact shape of Stage-C's real call, and the exact failure that
consumed authorization `stage-c-2026-07-24-r1`:
`KeyError: '3c853cff34e52d792cd0e5a96d1a5369f17f8047'`)."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from kvcot.discovery import attempt_artifacts
from kvcot.discovery.attempt_artifacts import (
    B1_REPAIR_ROUND4_STARTING_COMMIT,
    B1_REQUIRED_ANCESTOR_SHAS,
    LEGACY_STARTING_ANCESTOR_SHA,
    collect_execution_provenance,
)

AUTHORIZED_IMPLEMENTATION_SHA = "a" * 40


def _run_git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(tmp_path: Path) -> tuple[Path, str]:
    """A real, tiny, throwaway Git repository -- never the actual FaithKV
    history, so the historical legacy commit is genuinely absent from it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(["init", "-q"], repo)
    _run_git(["config", "user.email", "test@example.com"], repo)
    _run_git(["config", "user.name", "Test"], repo)
    (repo / "third_party" / "R-KV").mkdir(parents=True)
    (repo / "third_party" / "R-KV" / ".keep").write_text("", encoding="utf-8")
    (repo / "f.txt").write_text("1", encoding="utf-8")
    _run_git(["add", "-A"], repo)
    _run_git(["commit", "-q", "-m", "c1"], repo)
    first_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    (repo / "f.txt").write_text("2", encoding="utf-8")
    _run_git(["add", "-A"], repo)
    _run_git(["commit", "-q", "-m", "c2"], repo)
    return repo, first_sha


def _attempt_root(tmp_path: Path) -> Path:
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir(exist_ok=True)
    return attempt_root


# ---------------------------------------------------------------------------
# Exact R1 regression
# ---------------------------------------------------------------------------


def test_custom_ancestor_tuple_without_legacy_sha_does_not_raise(tmp_path):
    repo, _first_sha = _init_repo(tmp_path)

    provenance = collect_execution_provenance(
        repository=repo,
        expected_rkv_sha="not-the-real-sha",
        artifact_root=_attempt_root(tmp_path),
        required_ancestor_shas=(AUTHORIZED_IMPLEMENTATION_SHA,),
    )

    git = provenance["git"]
    assert LEGACY_STARTING_ANCESTOR_SHA not in git["required_ancestry"]
    assert set(git["required_ancestry"]) == {AUTHORIZED_IMPLEMENTATION_SHA}
    assert git["all_required_ancestry_verified"] == all(git["required_ancestry"].values())
    assert git["starting_ancestor"] == LEGACY_STARTING_ANCESTOR_SHA
    assert isinstance(git["starting_ancestor_verified"], bool)


# ---------------------------------------------------------------------------
# Custom tuple with legacy ancestor absent -- independent computation
# ---------------------------------------------------------------------------


def test_legacy_field_computed_independently_without_mutating_required_ancestry(tmp_path):
    repo, _first_sha = _init_repo(tmp_path)

    provenance = collect_execution_provenance(
        repository=repo,
        expected_rkv_sha="x",
        artifact_root=_attempt_root(tmp_path),
        required_ancestor_shas=(AUTHORIZED_IMPLEMENTATION_SHA,),
    )

    git = provenance["git"]
    # The legacy compatibility field must never leak into required_ancestry.
    assert LEGACY_STARTING_ANCESTOR_SHA not in git["required_ancestry"]
    assert set(git["required_ancestry"]) == {AUTHORIZED_IMPLEMENTATION_SHA}
    # The legacy commit genuinely is not an ancestor of this throwaway repo.
    assert git["starting_ancestor_verified"] is False


# ---------------------------------------------------------------------------
# Custom tuple containing legacy ancestor -- reuse, not a second check
# ---------------------------------------------------------------------------


def test_legacy_ancestry_reused_when_caller_includes_it(tmp_path, monkeypatch):
    repo, first_sha = _init_repo(tmp_path)
    merge_base_calls_for_legacy: list[tuple[str, ...]] = []
    real_run = subprocess.run

    def spy_run(args, *rest, **kwargs):
        if isinstance(args, list) and "merge-base" in args and LEGACY_STARTING_ANCESTOR_SHA in args:
            merge_base_calls_for_legacy.append(tuple(args))
        return real_run(args, *rest, **kwargs)

    monkeypatch.setattr(attempt_artifacts.subprocess, "run", spy_run)

    provenance = collect_execution_provenance(
        repository=repo,
        expected_rkv_sha="x",
        artifact_root=_attempt_root(tmp_path),
        required_ancestor_shas=(first_sha, LEGACY_STARTING_ANCESTOR_SHA),
    )

    git = provenance["git"]
    assert LEGACY_STARTING_ANCESTOR_SHA in git["required_ancestry"]
    assert git["starting_ancestor_verified"] == git["required_ancestry"][LEGACY_STARTING_ANCESTOR_SHA]
    # Exactly one is_ancestor check for the legacy SHA -- reused, not repeated.
    assert len(merge_base_calls_for_legacy) == 1


# ---------------------------------------------------------------------------
# Missing legacy commit -- real check, not fabricated success
# ---------------------------------------------------------------------------


def test_missing_legacy_commit_reports_false_not_exception(tmp_path):
    repo, _first_sha = _init_repo(tmp_path)

    provenance = collect_execution_provenance(
        repository=repo,
        expected_rkv_sha="x",
        artifact_root=_attempt_root(tmp_path),
        required_ancestor_shas=("b" * 40,),
    )

    assert provenance["git"]["starting_ancestor_verified"] is False


# ---------------------------------------------------------------------------
# Default behavior (`required_ancestor_shas=None`) preserves B1 defaults
# ---------------------------------------------------------------------------


def test_none_preserves_historical_b1_defaults(tmp_path):
    repo, _first_sha = _init_repo(tmp_path)

    provenance = collect_execution_provenance(
        repository=repo,
        expected_rkv_sha="x",
        artifact_root=_attempt_root(tmp_path),
    )

    git = provenance["git"]
    assert set(git["required_ancestry"]) == set(B1_REQUIRED_ANCESTOR_SHAS)
    assert git["starting_commit"] == B1_REPAIR_ROUND4_STARTING_COMMIT
    assert git["starting_ancestor_verified"] == git["required_ancestry"][LEGACY_STARTING_ANCESTOR_SHA]


# ---------------------------------------------------------------------------
# Explicit empty tuple means "explicitly require no ancestors"
# ---------------------------------------------------------------------------


def test_explicit_empty_tuple_requires_no_ancestors(tmp_path):
    repo, _first_sha = _init_repo(tmp_path)

    provenance = collect_execution_provenance(
        repository=repo,
        expected_rkv_sha="x",
        artifact_root=_attempt_root(tmp_path),
        required_ancestor_shas=(),
    )

    git = provenance["git"]
    assert git["required_ancestry"] == {}
    assert git["all_required_ancestry_verified"] is True
    # Legacy compatibility field is still independently, honestly computed.
    assert git["starting_ancestor_verified"] is False


# ---------------------------------------------------------------------------
# No GPU: the provenance collector never imports torch or initializes CUDA
# ---------------------------------------------------------------------------


def test_provenance_collection_does_not_import_torch_as_a_side_effect(tmp_path):
    repo, _first_sha = _init_repo(tmp_path)
    was_already_imported = "torch" in sys.modules

    collect_execution_provenance(
        repository=repo,
        expected_rkv_sha="x",
        artifact_root=_attempt_root(tmp_path),
        required_ancestor_shas=(AUTHORIZED_IMPLEMENTATION_SHA,),
    )

    if not was_already_imported:
        assert "torch" not in sys.modules
