"""One-use authorization document and atomic claim handling."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from kvcot.discovery.attempt_artifacts import atomic_write_json, sha256_file
from kvcot.discovery.diagnostic_pilot_contract import (
    AUTOMATIC_RETRIES,
    BRANCH,
    CLAIM_SCHEMA_VERSION,
    CANDIDATE_MANIFEST_CANONICAL_SHA256,
    EXACT_EXECUTION_COMMAND,
    MAXIMUM_CANDIDATE_POOL_SIZE,
    MAXIMUM_INVOCATIONS,
    MAXIMUM_QUALIFICATION_CANDIDATES,
    MAXIMUM_SELECTED_EXAMPLES,
    MODEL_REVISION,
    REPOSITORY,
    RESTORE_WIDTHS,
    RKV_REVISION,
    RUNTIME_LIMIT_SECONDS,
    TOKENIZER_REVISION,
    VRAM_LIMIT_BYTES,
    attach_canonical_hash,
    verify_canonical_hash,
)

AUTHORIZATION_JSON_BEGIN = "<!-- DIAGNOSTIC_PILOT_AUTHORIZATION_JSON_BEGIN -->"
AUTHORIZATION_JSON_END = "<!-- DIAGNOSTIC_PILOT_AUTHORIZATION_JSON_END -->"


class DiagnosticAuthorizationRefused(RuntimeError):
    pass


class DiagnosticAuthorizationConsumed(DiagnosticAuthorizationRefused):
    pass


@dataclass(frozen=True)
class VerifiedDiagnosticAuthorization:
    document_path: Path
    document_sha256: str
    payload: dict[str, Any]


def parse_authorization_document(path: str | Path) -> VerifiedDiagnosticAuthorization:
    document_path = Path(path).resolve()
    text = document_path.read_text(encoding="utf-8")
    if text.count(AUTHORIZATION_JSON_BEGIN) != 1 or text.count(AUTHORIZATION_JSON_END) != 1:
        raise DiagnosticAuthorizationRefused("authorization document must contain exactly one bounded JSON payload")
    raw = text.split(AUTHORIZATION_JSON_BEGIN, 1)[1].split(AUTHORIZATION_JSON_END, 1)[0].strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DiagnosticAuthorizationRefused("authorization JSON is malformed") from exc
    if not isinstance(payload, dict):
        raise DiagnosticAuthorizationRefused("authorization payload must be an object")
    verify_canonical_hash(payload)
    required = {
        "repository": REPOSITORY,
        "authorized_branch": BRANCH,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "rkv_revision": RKV_REVISION,
        "maximum_invocations": MAXIMUM_INVOCATIONS,
        "automatic_retries": AUTOMATIC_RETRIES,
        "maximum_qualification_candidates": MAXIMUM_QUALIFICATION_CANDIDATES,
        "maximum_selected_examples": MAXIMUM_SELECTED_EXAMPLES,
        "maximum_selected_events": MAXIMUM_SELECTED_EXAMPLES,
        "events_per_example": 1,
        "maximum_candidate_pool_size": MAXIMUM_CANDIDATE_POOL_SIZE,
        "maximum_interventions_per_selected_example": MAXIMUM_CANDIDATE_POOL_SIZE + 2,
        "kv_restore_widths": list(RESTORE_WIDTHS),
        "single_rtx3090": True,
        "cpu_offload": False,
        "vram_limit_bytes": VRAM_LIMIT_BYTES,
        "runtime_limit_seconds": RUNTIME_LIMIT_SECONDS,
        "exact_command": EXACT_EXECUTION_COMMAND,
        "candidate_manifest_canonical_sha256": CANDIDATE_MANIFEST_CANONICAL_SHA256,
        "implementation_audit_verdict": "PASS",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise DiagnosticAuthorizationRefused(f"authorization {key} does not match the frozen value")
    for key in (
        "authorized_implementation_sha",
        "protocol_document_sha256",
        "runtime_config_canonical_sha256",
        "implementation_audit_sha256",
    ):
        value = payload.get(key)
        length = 40 if key == "authorized_implementation_sha" else 64
        if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
            raise DiagnosticAuthorizationRefused(f"authorization {key} is not an exact lowercase hash")
    for key in (
        "authorization_id",
        "claim_path",
        "runtime_config_path",
        "output_root",
        "implementation_audit_path",
    ):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise DiagnosticAuthorizationRefused(f"authorization {key} must be a non-empty string")
    for key in ("claim_path", "runtime_config_path", "output_root", "implementation_audit_path"):
        if not Path(payload[key]).is_absolute():
            raise DiagnosticAuthorizationRefused(f"authorization {key} must be an absolute path")
    audit_path = Path(payload["implementation_audit_path"])
    if not audit_path.is_file():
        raise DiagnosticAuthorizationRefused("bound implementation audit file is absent")
    if sha256_file(audit_path) != payload["implementation_audit_sha256"]:
        raise DiagnosticAuthorizationRefused("bound implementation audit file hash mismatch")
    return VerifiedDiagnosticAuthorization(document_path, sha256_file(document_path), payload)


def claim_authorization_once(
    authorization: VerifiedDiagnosticAuthorization,
    *,
    attempt_id: str,
    attempt_directory: str,
) -> tuple[Path, dict[str, Any]]:
    claim_path = Path(authorization.payload["claim_path"]).resolve()
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim = attach_canonical_hash(
        {
            "artifact_schema_version": CLAIM_SCHEMA_VERSION,
            "authorization_id": authorization.payload["authorization_id"],
            "authorization_document_path": str(authorization.document_path),
            "authorization_document_sha256": authorization.document_sha256,
            "authorized_implementation_sha": authorization.payload["authorized_implementation_sha"],
            "attempt_id": attempt_id,
            "attempt_directory": attempt_directory,
            "maximum_invocations": MAXIMUM_INVOCATIONS,
            "automatic_retries": AUTOMATIC_RETRIES,
            "retry_allowed": False,
        }
    )
    data = (json.dumps(claim, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{claim_path.name}.", suffix=".claim.tmp", dir=str(claim_path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
        try:
            # A same-filesystem hard link atomically exposes complete bytes
            # and fails if another invocation already owns the claim path.
            os.link(temporary_path, claim_path)
        except FileExistsError as exc:
            raise DiagnosticAuthorizationConsumed(
                f"authorization already consumed at {claim_path}"
            ) from exc
        directory_descriptor = os.open(claim_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return claim_path, claim


def verify_claim(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_canonical_hash(payload)
    if payload.get("artifact_schema_version") != CLAIM_SCHEMA_VERSION:
        raise DiagnosticAuthorizationRefused("claim schema version mismatch")
    if payload.get("retry_allowed") is not False:
        raise DiagnosticAuthorizationRefused("claim must permanently prohibit retry")
    return payload
