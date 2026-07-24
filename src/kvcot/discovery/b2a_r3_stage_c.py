"""Fixed-path B2A-R3 Stage-C execution entry point.

This module is orchestration only. It verifies the frozen B2A-R3 evidence
chain, constructs the Stage-C claim internally, consumes the one-use claim,
writes Stage-C binding evidence, performs device preflight only after
claim consumption, then delegates to the existing B2A scientific
coordinator. No public API accepts alternate config, manifest, output,
attempt, model, tokenizer, runner, threshold, or device metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import uuid
from typing import Any, Callable

from kvcot.config import config_identity
from kvcot.discovery import b2a_r3_contract as c
from kvcot.discovery.attempt_artifacts import (
    atomic_write_json,
    collect_execution_provenance,
    sha256_file,
)
from kvcot.discovery.b2a_r3_artifacts import verify_qualification_artifact
from kvcot.discovery.b2a_r3_authorization import (
    AUTHORIZATION_STAGE_B2A_R3_EXECUTION,
    AuthorizationAlreadyConsumed,
    AuthorizationClaimRefused,
    ConsumedAuthorizationContext,
    claim_authorization,
    verify_authorization_preconditions,
    verify_persisted_stage_b_authorization_binding,
)
from kvcot.discovery.b2a_r3_authorization_document import (
    AuthorizationDocumentR3,
    parse_authorization_document,
)
from kvcot.discovery.b2a_r3_candidates import verify_candidate_manifest_structure
from kvcot.discovery.b2a_r3_freeze import verify_selection_provenance
from kvcot.discovery.b2a_r3_provenance import (
    GitStateProvider,
    SubprocessGitStateProvider,
    verify_attempt_provenance,
)
from kvcot.discovery.discovery_config import (
    PINNED_RKV_UPSTREAM_REVISION,
    canonical_config_hash,
    load_discovery_config,
)
from kvcot.discovery.manifest import B2AOneExampleManifest
from kvcot.utils.hashing import sha256_json

__all__ = [
    "StageCExecutionRefused",
    "StageCExecutionPlan",
    "StageCExecutionResult",
    "plan_b2a_r3_stage_c_execution",
    "run_b2a_r3_stage_c_execution",
]

REQUIRED_BRANCH = "research/b2a-r3-runtime-qualified-calibration"
REQUIRED_SELECTED_UNIQUE_ID = "test/number_theory/631.json"
AUTHORIZATION_DOCUMENT_RE = re.compile(
    r"^docs/B2A_R3_STAGE_C_EXECUTION_AUTHORIZATION_[0-9]{4}-[0-9]{2}-[0-9]{2}\.md$"
)


class StageCExecutionRefused(RuntimeError):
    """Fail-closed Stage-C orchestration refusal."""


@dataclass(frozen=True)
class _StageCInputs:
    config: Any
    config_byte_sha256: str
    config_canonical_sha256: str
    candidate_manifest: dict[str, Any]
    qualification_artifact: dict[str, Any]
    selected_manifest: B2AOneExampleManifest
    selection_provenance: dict[str, Any]
    selection_provenance_canonical_sha256: str


@dataclass(frozen=True)
class StageCExecutionPlan:
    authorization_id: str
    authorization_stage: str
    authorized_code_commit_sha: str
    observed_execution_commit_sha: str
    authorization_document_path: str
    authorization_document_sha256: str
    config_path: str
    config_sha256: str
    candidate_manifest_sha256: str
    qualification_artifact_sha256: str
    selected_manifest_sha256: str
    selection_provenance_sha256: str
    selected_unique_id: str
    model_revision: str
    tokenizer_revision: str
    required_rkv_sha: str
    global_claim_path: str
    attempt_id: str
    attempt_directory_path: str
    claim_payload: dict[str, Any]
    would_consume_authorization: bool = False
    would_initialize_cuda: bool = False
    would_load_tokenizer: bool = False
    would_load_model: bool = False
    would_import_rkv: bool = False
    would_run_workers: bool = False


@dataclass(frozen=True)
class StageCExecutionResult:
    authorization_id: str
    authorization_claim_path: Path
    authorization_claim_canonical_sha256: str
    attempt_id: str
    attempt_directory: Path
    selected_unique_id: str
    config_sha256: str
    selection_provenance_sha256: str
    device_preflight_passed: bool
    fullkv_process_outcome: Any
    rkv_process_outcome: Any
    final_verification_passed: bool
    overall_gate_passed: bool
    authorization_consumed: bool = True
    retry_allowed: bool = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _compact_timestamp(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise StageCExecutionRefused(f"{path} must contain a JSON object")
    return payload


def _resolve_repository_root(repository_root: str | Path) -> Path:
    root = Path(repository_root).resolve()
    if not (root / ".git").exists():
        raise StageCExecutionRefused(f"repository root is not a Git checkout: {root}")
    return root


def _resolve_authorization_document_path(repository_root: Path, authorization_document_path: str | Path) -> tuple[str, Path]:
    supplied = Path(authorization_document_path)
    if supplied.is_absolute():
        raise StageCExecutionRefused("authorization_document_path must be repository-relative")
    relative = supplied.as_posix()
    if relative != str(Path(relative).as_posix()) or ".." in Path(relative).parts:
        raise StageCExecutionRefused("authorization_document_path must be normalized and repository-relative")
    if AUTHORIZATION_DOCUMENT_RE.fullmatch(relative) is None:
        raise StageCExecutionRefused("authorization_document_path does not match the Stage-C date pattern")
    absolute = (repository_root / relative).resolve()
    try:
        absolute.relative_to(repository_root)
    except ValueError as exc:
        raise StageCExecutionRefused("authorization document resolves outside the repository root") from exc
    return relative, absolute


def _load_fixed_inputs(repository_root: Path, git_state: GitStateProvider) -> _StageCInputs:
    try:
        config_path = repository_root / c.CONFIG_PATH
        config = load_discovery_config(config_path)
        config_byte_sha256 = config_identity(config_path)
        config_canonical_sha256 = canonical_config_hash(config)

        candidate_manifest = _load_json(repository_root / c.CANDIDATE_MANIFEST_PATH)
        candidate = verify_candidate_manifest_structure(
            candidate_manifest, expected_config_sha256=config_byte_sha256
        )
        qualification_artifact = _load_json(repository_root / c.QUALIFICATION_ARTIFACT_PATH)
        stage_b_binding = verify_persisted_stage_b_authorization_binding(
            authorization_id=qualification_artifact["stage_b_authorization_id"],
            repository_root=repository_root,
            git_state=git_state,
            candidate_manifest=candidate_manifest,
            expected_config_sha256=config_byte_sha256,
        )
        qualification = verify_qualification_artifact(
            qualification_artifact,
            candidate_manifest=candidate_manifest,
            expected_config_sha256=config_byte_sha256,
            stage_b_authorization_context=stage_b_binding,
        )

        selected_manifest = B2AOneExampleManifest.model_validate_json(
            (repository_root / c.SELECTED_MANIFEST_PATH).read_text(encoding="utf-8")
        )
        selection_provenance = _load_json(repository_root / c.SELECTION_PROVENANCE_PATH)
        selected = verify_selection_provenance(
            selection_provenance,
            selected_manifest=selected_manifest,
            candidate_manifest=candidate_manifest,
            qualification_artifact=qualification_artifact,
            expected_config_sha256=config_byte_sha256,
            stage_b_authorization_context=stage_b_binding,
        )
    except StageCExecutionRefused:
        raise
    except Exception as exc:
        raise StageCExecutionRefused(f"fixed Stage-C input verification failed: {exc}") from exc

    if selected_manifest.unique_id != REQUIRED_SELECTED_UNIQUE_ID:
        raise StageCExecutionRefused(
            f"selected row {selected_manifest.unique_id!r} != {REQUIRED_SELECTED_UNIQUE_ID!r}"
        )
    if selected.selected_unique_id != REQUIRED_SELECTED_UNIQUE_ID:
        raise StageCExecutionRefused("selection provenance does not verify the accepted selected row")
    if qualification.selected_unique_id != REQUIRED_SELECTED_UNIQUE_ID:
        raise StageCExecutionRefused("qualification artifact does not bind the accepted selected row")
    if candidate.dataset_revision != config.dataset.revision:
        raise StageCExecutionRefused("candidate/config dataset revisions disagree")
    if qualification.model_revision != config.model.revision:
        raise StageCExecutionRefused("qualification/config model revisions disagree")
    if qualification.tokenizer_revision != config.model.tokenizer_revision:
        raise StageCExecutionRefused("qualification/config tokenizer revisions disagree")
    if selected_manifest.dataset_revision != config.dataset.revision:
        raise StageCExecutionRefused("selected-manifest/config dataset revisions disagree")

    return _StageCInputs(
        config=config,
        config_byte_sha256=config_byte_sha256,
        config_canonical_sha256=config_canonical_sha256,
        candidate_manifest=candidate_manifest,
        qualification_artifact=qualification_artifact,
        selected_manifest=selected_manifest,
        selection_provenance=selection_provenance,
        selection_provenance_canonical_sha256=selection_provenance["canonical_sha256"],
    )


def _build_claim_payload(
    *,
    document: AuthorizationDocumentR3,
    document_relative_path: str,
    authorization_document_sha256: str,
    git_state: GitStateProvider,
    inputs: _StageCInputs,
    claimed_at: datetime,
    attempt_id: str,
) -> dict[str, Any]:
    observed_commit = git_state.current_commit_sha()
    timestamp = _compact_timestamp(claimed_at)
    attempt_directory_path = f"results/decisions/{c.attempt_directory_name(timestamp, attempt_id)}"
    payload: dict[str, Any] = {
        "artifact_schema_version": c.AUTHORIZATION_CLAIM_ARTIFACT_SCHEMA_VERSION,
        "authorization_id": document.authorization_id,
        "authorization_stage": document.authorization_stage,
        "authorization_document_path": document_relative_path,
        "authorization_document_sha256": authorization_document_sha256,
        "authorized_repository": document.authorized_repository,
        "authorized_branch": document.authorized_branch,
        "authorized_code_commit_sha": document.authorized_code_commit_sha,
        "observed_repository": git_state.current_repository(),
        "observed_branch": git_state.current_branch(),
        "observed_execution_commit_sha": observed_commit,
        "required_ancestor_shas": list(document.required_ancestor_shas),
        "required_rkv_sha": document.required_rkv_sha,
        "observed_rkv_sha": git_state.rkv_submodule_sha(),
        "candidate_manifest_canonical_sha256": inputs.candidate_manifest["canonical_sha256"],
        "qualification_artifact_canonical_sha256": inputs.qualification_artifact["canonical_sha256"],
        "selected_manifest_sha256": inputs.selected_manifest.manifest_hash(),
        "selected_manifest_hash_algorithm": c.SELECTED_MANIFEST_HASH_ALGORITHM,
        "attempt_id": attempt_id,
        "global_claim_path": c.global_claim_path(document.authorization_id),
        "attempt_directory_path": attempt_directory_path,
        "claimed_at_utc": claimed_at.astimezone(timezone.utc).isoformat(),
    }
    payload["canonical_sha256"] = c.compute_canonical_sha256(payload)
    return payload


def _create_claimed_attempt_directory(repository_root: Path, relative_path: str, attempt_id: str) -> Path:
    attempt_path = (repository_root / relative_path).resolve()
    try:
        attempt_path.relative_to(repository_root)
    except ValueError as exc:
        raise StageCExecutionRefused("claimed attempt directory resolves outside the repository root") from exc
    try:
        attempt_path.mkdir(parents=True, exist_ok=False)
        (attempt_path / "fullkv").mkdir(exist_ok=False)
        (attempt_path / "rkv").mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise StageCExecutionRefused(
            f"claimed attempt directory already exists; authorization remains consumed: {attempt_path}"
        ) from exc
    if not attempt_path.name.endswith(f"_{attempt_id}"):
        raise StageCExecutionRefused("claimed attempt directory does not end with the claimed attempt ID")
    return attempt_path


def _write_stage_c_failure(attempt_directory: Path | None, attempt_id: str | None, exc: BaseException) -> None:
    if attempt_directory is None or attempt_id is None:
        return
    try:
        atomic_write_json(
            attempt_directory / "stage_c_execution_failure.json",
            {
                "attempt_id": attempt_id,
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "authorization_consumed": True,
                "retry_allowed": False,
            },
        )
    except Exception:
        pass
    try:
        if not (attempt_directory / "completion.json").exists():
            atomic_write_json(
                attempt_directory / "completion.json",
                {
                    "attempt_id": attempt_id,
                    "finished_at": _utc_now().isoformat(),
                    "outcome": "exception",
                    "exit_code": 2,
                    "artifact_path": None,
                    "gate_passed": None,
                    "authorization_consumed": True,
                    "retry_allowed": False,
                },
            )
    except Exception:
        pass


def _write_invocation(attempt_directory: Path, plan: StageCExecutionPlan) -> None:
    safe_argv = [arg for arg in sys.argv if "token" not in arg.lower() and "secret" not in arg.lower()]
    atomic_write_json(
        attempt_directory / "invocation.json",
        {
            "attempt_id": plan.attempt_id,
            "started_at": _utc_now().isoformat(),
            "argv": safe_argv,
            "working_directory": str(Path.cwd()),
            "pid": os.getpid(),
            "command": "kvcot run-b2a-r3-stage-c",
            "authorization_document_path": plan.authorization_document_path,
            "config_path": c.CONFIG_PATH,
            "manifest_path": c.SELECTED_MANIFEST_PATH,
            "environment": {
                "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
            },
        },
    )


def _write_stage_c_binding(
    *,
    attempt_directory: Path,
    plan: StageCExecutionPlan,
    consumed: ConsumedAuthorizationContext,
    inputs: _StageCInputs,
) -> Path:
    payload: dict[str, Any] = {
        "artifact_schema_version": "faithkv-b2a-r3-stage-c-binding-v1",
        "authorization_id": plan.authorization_id,
        "authorization_stage": plan.authorization_stage,
        "authorization_document_path": plan.authorization_document_path,
        "authorization_document_sha256": plan.authorization_document_sha256,
        "authorization_claim_canonical_sha256": consumed.authorization_claim_canonical_sha256,
        "claim_global_path": plan.global_claim_path,
        "authorized_code_sha": plan.authorized_code_commit_sha,
        "execution_authorization_commit_sha": plan.observed_execution_commit_sha,
        "repository": c.REQUIRED_REPOSITORY,
        "branch": REQUIRED_BRANCH,
        "rkv_sha": plan.required_rkv_sha,
        "config_path": c.CONFIG_PATH,
        "config_sha256": plan.config_sha256,
        "candidate_manifest_path": c.CANDIDATE_MANIFEST_PATH,
        "candidate_manifest_canonical_sha256": plan.candidate_manifest_sha256,
        "qualification_artifact_path": c.QUALIFICATION_ARTIFACT_PATH,
        "qualification_artifact_canonical_sha256": plan.qualification_artifact_sha256,
        "selected_manifest_path": c.SELECTED_MANIFEST_PATH,
        "selected_manifest_sha256": plan.selected_manifest_sha256,
        "selection_provenance_path": c.SELECTION_PROVENANCE_PATH,
        "selection_provenance_canonical_sha256": plan.selection_provenance_sha256,
        "selected_unique_id": plan.selected_unique_id,
        "model_name": inputs.config.model.name,
        "model_revision": inputs.config.model.revision,
        "tokenizer_name": inputs.config.model.tokenizer_name,
        "tokenizer_revision": inputs.config.model.tokenizer_revision,
        "dataset_name": inputs.config.dataset.name,
        "dataset_revision": inputs.config.dataset.revision,
        "attempt_id": plan.attempt_id,
        "attempt_directory": plan.attempt_directory_path,
        "exact_command_identity": (
            "kvcot run-b2a-r3-stage-c --authorization-document "
            f"{plan.authorization_document_path} --execute"
        ),
        "invocation_limit": 1,
        "runtime_limit_gpu_hours": 4,
        "vram_limit_gib": 22,
        "cpu_offload_prohibited": True,
    }
    payload["canonical_sha256"] = c.compute_canonical_sha256(payload)
    return atomic_write_json(attempt_directory / "stage_c_binding.json", payload)


def _extract_process_outcome(attempt_directory: Path) -> tuple[Any, Any]:
    path = attempt_directory / "process_outcome.json"
    if not path.is_file():
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("return_codes", {}).get("fullkv"), payload.get("return_codes", {}).get("rkv")


def _build_plan_internal(
    *,
    authorization_document_path: str | Path,
    repository_root: str | Path = ".",
    git_state: GitStateProvider | None = None,
    now: Callable[[], datetime] = _utc_now,
    attempt_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
) -> tuple[StageCExecutionPlan, _StageCInputs, GitStateProvider]:
    root = _resolve_repository_root(repository_root)
    git_state = git_state or SubprocessGitStateProvider(str(root))
    document_relative, document_path = _resolve_authorization_document_path(root, authorization_document_path)
    inputs = _load_fixed_inputs(root, git_state)
    document = parse_authorization_document(document_path)
    if document.authorization_stage != AUTHORIZATION_STAGE_B2A_R3_EXECUTION:
        raise StageCExecutionRefused("authorization document is not a Stage-C execution document")
    if document.authorized_branch != REQUIRED_BRANCH:
        raise StageCExecutionRefused("authorization document does not authorize the B2A-R3 branch")
    if document.required_rkv_sha != PINNED_RKV_UPSTREAM_REVISION:
        raise StageCExecutionRefused("authorization document does not bind the required R-KV SHA")

    document_sha256 = sha256_file(document_path)
    claimed_at = now()
    attempt_id = attempt_id_factory()
    payload = _build_claim_payload(
        document=document,
        document_relative_path=document_relative,
        authorization_document_sha256=document_sha256,
        git_state=git_state,
        inputs=inputs,
        claimed_at=claimed_at,
        attempt_id=attempt_id,
    )
    verify_authorization_preconditions(
        payload,
        git_state=git_state,
        authorization_document_path=document_path,
        candidate_manifest=inputs.candidate_manifest,
        expected_config_sha256=inputs.config_byte_sha256,
        qualification_artifact=inputs.qualification_artifact,
        selected_manifest=inputs.selected_manifest,
        selection_provenance=inputs.selection_provenance,
        repository_root=root,
    )
    claim_path = root / payload["global_claim_path"]
    if claim_path.exists():
        raise AuthorizationAlreadyConsumed(
            f"a filesystem entry already exists at {claim_path}; authorization is permanently consumed"
        )
    plan = StageCExecutionPlan(
        authorization_id=payload["authorization_id"],
        authorization_stage=payload["authorization_stage"],
        authorized_code_commit_sha=payload["authorized_code_commit_sha"],
        observed_execution_commit_sha=payload["observed_execution_commit_sha"],
        authorization_document_path=document_relative,
        authorization_document_sha256=document_sha256,
        config_path=c.CONFIG_PATH,
        config_sha256=inputs.config_byte_sha256,
        candidate_manifest_sha256=inputs.candidate_manifest["canonical_sha256"],
        qualification_artifact_sha256=inputs.qualification_artifact["canonical_sha256"],
        selected_manifest_sha256=inputs.selected_manifest.manifest_hash(),
        selection_provenance_sha256=inputs.selection_provenance_canonical_sha256,
        selected_unique_id=inputs.selected_manifest.unique_id,
        model_revision=inputs.config.model.revision,
        tokenizer_revision=inputs.config.model.tokenizer_revision,
        required_rkv_sha=payload["required_rkv_sha"],
        global_claim_path=payload["global_claim_path"],
        attempt_id=payload["attempt_id"],
        attempt_directory_path=payload["attempt_directory_path"],
        claim_payload=payload,
    )
    return plan, inputs, git_state


def plan_b2a_r3_stage_c_execution(
    *,
    authorization_document_path: str | Path,
    repository_root: str | Path = ".",
) -> StageCExecutionPlan:
    """Plan a fixed-path Stage-C execution without writing any file."""
    plan, _inputs, _git_state = _build_plan_internal(
        authorization_document_path=authorization_document_path,
        repository_root=repository_root,
    )
    return plan


def _run_b2a_r3_stage_c_execution_internal(
    *,
    authorization_document_path: str | Path,
    repository_root: str | Path = ".",
    git_state: GitStateProvider | None = None,
    now: Callable[[], datetime] = _utc_now,
    attempt_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    device_preflight_fn: Callable[[], Any] | None = None,
    coordinator_fn: Callable[..., Any] | None = None,
    provenance_collector: Callable[..., dict[str, Any]] = collect_execution_provenance,
) -> StageCExecutionResult:
    root = _resolve_repository_root(repository_root)
    plan, inputs, git_state = _build_plan_internal(
        authorization_document_path=authorization_document_path,
        repository_root=root,
        git_state=git_state,
        now=now,
        attempt_id_factory=attempt_id_factory,
    )
    document_path = root / plan.authorization_document_path
    attempt_directory: Path | None = None
    consumed: ConsumedAuthorizationContext | None = None
    try:
        verified = verify_authorization_preconditions(
            plan.claim_payload,
            git_state=git_state,
            authorization_document_path=document_path,
            candidate_manifest=inputs.candidate_manifest,
            expected_config_sha256=inputs.config_byte_sha256,
            qualification_artifact=inputs.qualification_artifact,
            selected_manifest=inputs.selected_manifest,
            selection_provenance=inputs.selection_provenance,
            repository_root=root,
        )
        consumed = claim_authorization(
            plan.claim_payload,
            repository_root=root,
            verified_context=verified,
            git_state=git_state,
        )
        if not consumed.claim_path.is_file():
            raise StageCExecutionRefused("authorization claim was not present after consumption")

        attempt_directory = _create_claimed_attempt_directory(root, plan.attempt_directory_path, plan.attempt_id)
        _write_invocation(attempt_directory, plan)
        _write_stage_c_binding(attempt_directory=attempt_directory, plan=plan, consumed=consumed, inputs=inputs)
        provenance = provenance_collector(
            repository=root,
            expected_rkv_sha=plan.required_rkv_sha,
            artifact_root=attempt_directory,
            allowed_dirty_paths=(plan.global_claim_path,),
            required_ancestor_shas=(plan.authorized_code_commit_sha, *plan.claim_payload["required_ancestor_shas"]),
        )
        atomic_write_json(attempt_directory / "provenance.json", provenance)

        ok, reasons = verify_attempt_provenance(
            verified.policy,
            git_state,
            active_authorization_paths=verified.active_paths,
        )
        if not ok:
            raise StageCExecutionRefused(f"post-claim provenance verification failed: {reasons}")

        if device_preflight_fn is None:
            import torch
            from kvcot.discovery.strict_device import verify_single_rtx3090

            if not torch.cuda.is_available():
                raise StageCExecutionRefused("Stage-C execution requires CUDA; none is available")
            device_preflight = verify_single_rtx3090(torch.cuda, torch_module=torch)
            device_payload = {"verified": True, **device_preflight.__dict__}
        else:
            device_preflight = device_preflight_fn()
            device_payload = dict(device_preflight)

        atomic_write_json(
            attempt_directory / "preflight.json",
            {
                "passed": True,
                "blockers": [],
                "config_hash": inputs.config_canonical_sha256,
                "manifest_hash": inputs.selected_manifest.manifest_hash(),
                "device": device_payload,
            },
        )

        if coordinator_fn is None:
            from kvcot.discovery.b2a_execute import run_b2a_calibration

            coordinator_fn = run_b2a_calibration
        artifact = coordinator_fn(
            inputs.config,
            inputs.selected_manifest,
            config_path=str(c.CONFIG_PATH),
            manifest_path=str(c.SELECTED_MANIFEST_PATH),
            attempt_directory=attempt_directory,
            cli_device_preflight=device_payload,
            expected_provenance_branch=REQUIRED_BRANCH,
            expected_provenance_rkv_sha=plan.required_rkv_sha,
            expected_provenance_ancestor_shas=(
                plan.authorized_code_commit_sha,
                *tuple(plan.claim_payload["required_ancestor_shas"]),
            ),
        )
        from kvcot.discovery.attempt_verification import verify_final_reference_manifest

        final_verified, _final_reasons = verify_final_reference_manifest(attempt_directory)
        fullkv_outcome, rkv_outcome = _extract_process_outcome(attempt_directory)
        return StageCExecutionResult(
            authorization_id=plan.authorization_id,
            authorization_claim_path=consumed.claim_path,
            authorization_claim_canonical_sha256=consumed.authorization_claim_canonical_sha256,
            attempt_id=plan.attempt_id,
            attempt_directory=attempt_directory,
            selected_unique_id=plan.selected_unique_id,
            config_sha256=plan.config_sha256,
            selection_provenance_sha256=plan.selection_provenance_sha256,
            device_preflight_passed=True,
            fullkv_process_outcome=fullkv_outcome,
            rkv_process_outcome=rkv_outcome,
            final_verification_passed=final_verified,
            overall_gate_passed=bool(getattr(artifact, "overall_passed", False)),
        )
    except Exception as exc:
        _write_stage_c_failure(
            attempt_directory,
            plan.attempt_id if "plan" in locals() else None,
            exc,
        )
        if consumed is not None:
            raise StageCExecutionRefused(
                f"Stage-C execution failed after authorization consumption; retry is prohibited: {exc}"
            ) from exc
        raise


def run_b2a_r3_stage_c_execution(
    *,
    authorization_document_path: str | Path,
    repository_root: str | Path = ".",
) -> StageCExecutionResult:
    """Consume the one-use Stage-C authorization and run the fixed attempt."""
    return _run_b2a_r3_stage_c_execution_internal(
        authorization_document_path=authorization_document_path,
        repository_root=repository_root,
    )
