"""CPU tests for the fixed-path B2A-R3 Stage-C execution entry point."""
from __future__ import annotations

from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from kvcot.discovery import b2a_r3_contract as c
from kvcot.discovery.attempt_artifacts import (
    AttemptDirectory,
    LEGACY_STARTING_ANCESTOR_SHA,
    atomic_write_json,
    build_attempt_references,
    collect_execution_provenance,
)
from kvcot.discovery.b2a_r3_stage_c import (
    REQUIRED_BRANCH,
    REQUIRED_SELECTED_UNIQUE_ID,
    StageCExecutionRefused,
    _build_plan_internal,
    _run_b2a_r3_stage_c_execution_internal,
    plan_b2a_r3_stage_c_execution,
    run_b2a_r3_stage_c_execution,
)
from kvcot.utils.hashing import sha256_file
from tests.unit.discovery.test_b2a_r3_authorization import _document_payload, _write_document
from tests.unit.discovery.test_b2a_r3_provenance import FakeGitState

IMPLEMENTATION_SHA = "a" * 40
EXECUTION_SHA = "d" * 40
ATTEMPT_ID = "feedface"
AUTH_ID = "stage-c-2026-08-05"
AUTH_DOC = "docs/B2A_R3_STAGE_C_EXECUTION_AUTHORIZATION_2026-08-05.md"


def _copy(path: str, root: Path) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(path), target)


def _stage_c_repo(tmp_path: Path) -> tuple[Path, FakeGitState]:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    for path in (
        c.CONFIG_PATH,
        c.CANDIDATE_MANIFEST_PATH,
        c.QUALIFICATION_ARTIFACT_PATH,
        c.SELECTED_MANIFEST_PATH,
        c.SELECTION_PROVENANCE_PATH,
        "results/decisions/b2a_r3_authorization_claims/stage-b-2026-07-24-r2-final.json",
        "docs/B2A_R3_STAGE_B_QUALIFICATION_AUTHORIZATION_2026-07-24.md",
    ):
        _copy(path, root)

    candidate = json.loads((root / c.CANDIDATE_MANIFEST_PATH).read_text(encoding="utf-8"))
    qualification = json.loads((root / c.QUALIFICATION_ARTIFACT_PATH).read_text(encoding="utf-8"))
    selected = json.loads((root / c.SELECTED_MANIFEST_PATH).read_text(encoding="utf-8"))
    from kvcot.discovery.manifest import B2AOneExampleManifest

    selected_hash = B2AOneExampleManifest.model_validate(selected).manifest_hash()
    document_path = root / AUTH_DOC
    _write_document(
        document_path,
        _document_payload(
            authorization_id=AUTH_ID,
            authorization_stage="b2a_r3_execution",
            authorized_branch=REQUIRED_BRANCH,
            authorized_code_commit_sha=IMPLEMENTATION_SHA,
            required_ancestor_shas=(),
            required_rkv_sha="45eaa7d69d20b7388321f077020a610d9afb65bd",
            candidate_manifest_canonical_sha256=candidate["canonical_sha256"],
            qualification_artifact_canonical_sha256=qualification["canonical_sha256"],
            selected_manifest_sha256=selected_hash,
            selected_manifest_hash_algorithm=c.SELECTED_MANIFEST_HASH_ALGORITHM,
        ),
    )

    stage_b_claim = json.loads(
        (root / "results/decisions/b2a_r3_authorization_claims/stage-b-2026-07-24-r2-final.json").read_text(
            encoding="utf-8"
        )
    )
    ancestors = frozenset({
        IMPLEMENTATION_SHA,
        stage_b_claim["authorized_code_commit_sha"],
        stage_b_claim["observed_execution_commit_sha"],
    })
    git_state = FakeGitState(
        commit_sha=EXECUTION_SHA,
        ancestors=ancestors,
        repository_root=str(root),
        changed_paths=(AUTH_DOC,),
        changed_paths_by_head={
            EXECUTION_SHA: (AUTH_DOC,),
            stage_b_claim["observed_execution_commit_sha"]: (stage_b_claim["authorization_document_path"],),
        },
    )
    return root, git_state


def _run_git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _stage_c_repo_with_real_git(tmp_path: Path) -> tuple[Path, FakeGitState]:
    """Same fixture as `_stage_c_repo`, but `root` is a real Git repository
    (with a tracked `third_party/R-KV` path, no legacy B1 commit anywhere in
    its history) instead of a bare `.git` placeholder -- required to exercise
    the *real* `collect_execution_provenance`, which shells out to `git`."""
    root, git_state = _stage_c_repo(tmp_path)
    shutil.rmtree(root / ".git")
    (root / "third_party" / "R-KV").mkdir(parents=True, exist_ok=True)
    (root / "third_party" / "R-KV" / ".keep").write_text("", encoding="utf-8")
    _run_git(["init", "-q"], root)
    _run_git(["config", "user.email", "test@example.com"], root)
    _run_git(["config", "user.name", "Test"], root)
    _run_git(["add", "-A"], root)
    _run_git(["commit", "-q", "-m", "stage-c fixture"], root)
    return root, git_state


def _now() -> datetime:
    return datetime(2026, 8, 5, 0, 0, 0, tzinfo=timezone.utc)


def _fake_provenance(**kwargs):
    return {
        "git": {
            "branch": REQUIRED_BRANCH,
            "head": EXECUTION_SHA,
            "origin_branch_sha": EXECUTION_SHA,
            "starting_commit": IMPLEMENTATION_SHA,
            "required_ancestry": {IMPLEMENTATION_SHA: True},
            "all_required_ancestry_verified": True,
            "expected_rkv_sha": "45eaa7d69d20b7388321f077020a610d9afb65bd",
            "rkv_submodule_sha": "45eaa7d69d20b7388321f077020a610d9afb65bd",
            "rkv_submodule_match": True,
            "dirty": False,
            "staged_paths": [],
            "unstaged_paths": [],
            "untracked_paths": [],
            "claim_path_filtered": kwargs["allowed_dirty_paths"][0],
        },
        "system": {
            "os": "Linux",
            "platform": "test",
            "kernel_release": "test",
            "architecture": "x86_64",
            "cpu": "test",
        },
        "software": {"python": "test"},
        "gpu_evidence_cross_references": [
            "preflight.json:device",
            "fullkv/result.json:device_evidence",
            "rkv/result.json:device_evidence",
        ],
    }


def _fake_device() -> dict:
    return {
        "verified": True,
        "visible_gpu_count": 1,
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "device_index": 0,
        "requested_device": "cuda:0",
        "total_vram_bytes": 24 * 1024**3,
        "compute_capability": (8, 6),
        "driver_version": "test",
        "cuda_runtime": "test",
        "cudnn_version": "test",
        "policy_satisfied": True,
    }


def test_public_api_surface_has_no_scientific_overrides():
    for fn in (plan_b2a_r3_stage_c_execution, run_b2a_r3_stage_c_execution):
        signature = inspect.signature(fn)
        assert set(signature.parameters) == {"authorization_document_path", "repository_root"}
        assert not {
            "config_path",
            "candidate_manifest_path",
            "qualification_path",
            "selected_manifest_path",
            "selection_provenance_path",
            "output_path",
            "attempt_directory",
            "model_revision",
            "tokenizer_revision",
            "runner",
            "device_preflight",
            "thresholds",
        } & set(signature.parameters)


def test_plan_uses_fixed_paths_and_does_not_write(tmp_path):
    root, git_state = _stage_c_repo(tmp_path)
    plan, _inputs, _git = _build_plan_internal(
        authorization_document_path=AUTH_DOC,
        repository_root=root,
        git_state=git_state,
        now=_now,
        attempt_id_factory=lambda: ATTEMPT_ID,
    )

    assert plan.authorization_stage == "b2a_r3_execution"
    assert plan.selected_unique_id == REQUIRED_SELECTED_UNIQUE_ID
    assert plan.config_path == c.CONFIG_PATH
    assert plan.global_claim_path == c.global_claim_path(AUTH_ID)
    assert plan.would_consume_authorization is False
    assert plan.would_initialize_cuda is False
    assert not (root / plan.global_claim_path).exists()
    assert not (root / plan.attempt_directory_path).exists()


@pytest.mark.parametrize(
    "path",
    [
        "/abs/docs/B2A_R3_STAGE_C_EXECUTION_AUTHORIZATION_2026-08-05.md",
        "../docs/B2A_R3_STAGE_C_EXECUTION_AUTHORIZATION_2026-08-05.md",
        "docs/not-stage-c.md",
    ],
)
def test_plan_rejects_unsafe_document_paths(tmp_path, path):
    root, git_state = _stage_c_repo(tmp_path)
    with pytest.raises(StageCExecutionRefused):
        _build_plan_internal(
            authorization_document_path=path,
            repository_root=root,
            git_state=git_state,
            now=_now,
            attempt_id_factory=lambda: ATTEMPT_ID,
        )


def test_execute_consumes_claim_before_device_and_coordinator(tmp_path):
    root, git_state = _stage_c_repo(tmp_path)
    calls: list[str] = []

    def device():
        calls.append("device")
        assert (root / c.global_claim_path(AUTH_ID)).is_file()
        return _fake_device()

    def coordinator(_config, _manifest, **kwargs):
        calls.append("coordinator")
        attempt_directory = Path(kwargs["attempt_directory"])
        assert (root / c.global_claim_path(AUTH_ID)).is_file()
        assert (attempt_directory / "stage_c_binding.json").is_file()
        atomic_write_json(
            attempt_directory / "process_outcome.json",
            {
                "attempt_id": ATTEMPT_ID,
                "return_codes": {"fullkv": 0, "rkv": 0},
                "timeout_state": {"fullkv": False, "rkv": False},
                "partial_success": False,
                "coordinator_observed_process_seconds": {"fullkv": 1.0, "rkv": 1.0},
            },
        )
        atomic_write_json(
            attempt_directory / "completion.json",
            {
                "attempt_id": ATTEMPT_ID,
                "finished_at": _now().isoformat(),
                "outcome": "gate_passed",
                "exit_code": 0,
                "gate_passed": True,
                "intended_final_relative_path": "final.json",
                "artifact_path": str(attempt_directory / "final.json"),
                "config_hash": "not-used-by-final-reference-test",
                "manifest_hash": "not-used-by-final-reference-test",
            },
        )
        attempt = AttemptDirectory(attempt_id=ATTEMPT_ID, path=attempt_directory)
        atomic_write_json(
            attempt_directory / "final.json",
            {"passed": True, "attempt_artifacts": build_attempt_references(attempt, exclude=("final.json",))},
        )
        return SimpleNamespace(overall_passed=True)

    result = _run_b2a_r3_stage_c_execution_internal(
        authorization_document_path=AUTH_DOC,
        repository_root=root,
        git_state=git_state,
        now=_now,
        attempt_id_factory=lambda: ATTEMPT_ID,
        device_preflight_fn=device,
        coordinator_fn=coordinator,
        provenance_collector=_fake_provenance,
    )

    assert calls == ["device", "coordinator"]
    assert result.authorization_consumed is True
    assert result.retry_allowed is False
    assert result.device_preflight_passed is True
    assert result.fullkv_process_outcome == 0
    assert result.rkv_process_outcome == 0
    assert result.overall_gate_passed is True
    assert result.final_verification_passed is True
    assert sha256_file(root / c.global_claim_path(AUTH_ID)) == sha256_file(result.authorization_claim_path)


def test_stage_c_invocation_argv_is_sanitized_for_attempt_verifier(tmp_path, monkeypatch):
    root, git_state = _stage_c_repo(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "kvcot",
            "run-b2a-r3-stage-c",
            "--authorization-document",
            "docs/B2A_R3_STAGE_C_EXECUTION_AUTHORIZATION_2026-08-05.md",
            "--execute",
        ],
    )

    def coordinator(_config, _manifest, **kwargs):
        attempt_directory = Path(kwargs["attempt_directory"])
        invocation = json.loads((attempt_directory / "invocation.json").read_text(encoding="utf-8"))
        assert "--authorization-document" not in invocation["argv"]
        assert "docs/B2A_R3_STAGE_C_EXECUTION_AUTHORIZATION_2026-08-05.md" not in invocation["argv"]
        assert "--stage-c-document" in invocation["argv"]
        assert "<stage-c-document-path-recorded-separately>" in invocation["argv"]
        assert invocation["authorization_document_path"] == AUTH_DOC
        for item in invocation["argv"]:
            lowered = item.lower()
            assert "authorization" not in lowered
            assert "token" not in lowered
            assert "secret" not in lowered
            assert "password" not in lowered
        raise RuntimeError("stop after invocation assertion")

    with pytest.raises(StageCExecutionRefused, match="retry is prohibited"):
        _run_b2a_r3_stage_c_execution_internal(
            authorization_document_path=AUTH_DOC,
            repository_root=root,
            git_state=git_state,
            now=_now,
            attempt_id_factory=lambda: ATTEMPT_ID,
            device_preflight_fn=_fake_device,
            coordinator_fn=coordinator,
            provenance_collector=_fake_provenance,
        )


def test_second_invocation_refuses_before_device_or_coordinator(tmp_path):
    root, git_state = _stage_c_repo(tmp_path)
    claim_path = root / c.global_claim_path(AUTH_ID)
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.write_text("already consumed\n", encoding="utf-8")

    with pytest.raises(Exception):
        _run_b2a_r3_stage_c_execution_internal(
            authorization_document_path=AUTH_DOC,
            repository_root=root,
            git_state=git_state,
            now=_now,
            attempt_id_factory=lambda: ATTEMPT_ID,
            device_preflight_fn=lambda: pytest.fail("device preflight must not run"),
            coordinator_fn=lambda *_args, **_kwargs: pytest.fail("coordinator must not run"),
            provenance_collector=_fake_provenance,
        )


def test_corrupt_selected_row_is_rejected(tmp_path):
    root, git_state = _stage_c_repo(tmp_path)
    manifest_path = root / c.SELECTED_MANIFEST_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unique_id"] = "test/number_theory/820.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(StageCExecutionRefused, match="fixed Stage-C input verification failed"):
        _build_plan_internal(
            authorization_document_path=AUTH_DOC,
            repository_root=root,
            git_state=git_state,
            now=_now,
            attempt_id_factory=lambda: ATTEMPT_ID,
        )


def test_real_provenance_collector_no_longer_blocks_before_device_preflight(tmp_path):
    """B2A-R3 Stage-C R1 regression: authorization `stage-c-2026-07-24-r1`
    was consumed and then failed with
    `KeyError: '3c853cff34e52d792cd0e5a96d1a5369f17f8047'` inside the real
    `collect_execution_provenance`, strictly before device preflight, CUDA
    initialization, or worker launch. This test exercises the exact
    production call shape -- the real collector, a custom
    `required_ancestor_shas` tuple built the same way Stage-C really builds
    it (`(authorized_code_commit_sha, *claim required_ancestor_shas)`, here
    `(IMPLEMENTATION_SHA,)`, which does not contain the legacy B1 SHA) --
    against a real (throwaway) Git repository, and asserts execution reaches
    the mocked device preflight and coordinator instead of raising."""
    root, git_state = _stage_c_repo_with_real_git(tmp_path)
    calls: list[str] = []
    was_torch_already_imported = "torch" in sys.modules

    def device():
        calls.append("device")
        assert "torch" not in sys.modules or was_torch_already_imported
        return _fake_device()

    def coordinator(_config, _manifest, **kwargs):
        calls.append("coordinator")
        attempt_directory = Path(kwargs["attempt_directory"])
        atomic_write_json(
            attempt_directory / "process_outcome.json",
            {
                "attempt_id": ATTEMPT_ID,
                "return_codes": {"fullkv": 0, "rkv": 0},
                "timeout_state": {"fullkv": False, "rkv": False},
                "partial_success": False,
                "coordinator_observed_process_seconds": {"fullkv": 1.0, "rkv": 1.0},
            },
        )
        atomic_write_json(
            attempt_directory / "completion.json",
            {
                "attempt_id": ATTEMPT_ID,
                "finished_at": _now().isoformat(),
                "outcome": "gate_passed",
                "exit_code": 0,
                "gate_passed": True,
                "intended_final_relative_path": "final.json",
                "artifact_path": str(attempt_directory / "final.json"),
                "config_hash": "not-used-by-final-reference-test",
                "manifest_hash": "not-used-by-final-reference-test",
            },
        )
        attempt = AttemptDirectory(attempt_id=ATTEMPT_ID, path=attempt_directory)
        atomic_write_json(
            attempt_directory / "final.json",
            {"passed": True, "attempt_artifacts": build_attempt_references(attempt, exclude=("final.json",))},
        )
        return SimpleNamespace(overall_passed=True)

    result = _run_b2a_r3_stage_c_execution_internal(
        authorization_document_path=AUTH_DOC,
        repository_root=root,
        git_state=git_state,
        now=_now,
        attempt_id_factory=lambda: ATTEMPT_ID,
        device_preflight_fn=device,
        coordinator_fn=coordinator,
        provenance_collector=collect_execution_provenance,
    )

    assert calls == ["device", "coordinator"]
    assert result.authorization_consumed is True

    provenance_path = result.attempt_directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    git_evidence = provenance["git"]
    assert LEGACY_STARTING_ANCESTOR_SHA not in git_evidence["required_ancestry"]
    assert IMPLEMENTATION_SHA in git_evidence["required_ancestry"]
    assert isinstance(git_evidence["starting_ancestor_verified"], bool)
