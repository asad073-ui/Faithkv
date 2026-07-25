"""One-use authorization-claim mechanism for the 8B geometry pilot.

Deliberately reuses this repository's proven building blocks --
`kvcot.discovery.b2a_r3_contract.validate_authorization_id` (generic
authorization-ID syntax, not B2A-specific) and the canonical-hash
convention already defined in `kvcot.discovery.geometry_pilot_contract` --
rather than inventing a fourth hashing/ID scheme. The claims directory is
new and disjoint from every existing claims root
(`results/decisions/b2a_r3_authorization_claims/`,
`results/decisions/b2a_r3_attempt_*`), so this pilot's claims can never
collide with, overwrite, or be mistaken for B2A/Stage-C or the closed 1.5B
diagnostic pilot's claims/attempts.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kvcot.discovery.b2a_r3_contract import validate_authorization_id
from kvcot.discovery.geometry_pilot_contract import (
    CLAIM_SCHEMA_VERSION,
    canonical_payload_hash,
    verify_canonical_hash,
)

GEOMETRY_CLAIMS_DIR = "results/decisions/geometry_pilot_authorization_claims"
GEOMETRY_ATTEMPTS_PREFIX = "geometry_pilot_attempt_"


class GeometryAuthorizationError(ValueError):
    pass


class GeometryAuthorizationAlreadyConsumed(RuntimeError):
    """Raised whenever a claim already exists at the deterministic global
    path -- complete, partial, empty, or corrupt, it is ALWAYS treated as
    consumed. Never repaired, deleted, or treated as retryable."""


def global_claim_path(authorization_id: str) -> str:
    validate_authorization_id(authorization_id)
    return f"{GEOMETRY_CLAIMS_DIR}/{authorization_id}.json"


def attempt_directory_name(utc_compact_timestamp: str, attempt_id: str) -> str:
    return f"{GEOMETRY_ATTEMPTS_PREFIX}{utc_compact_timestamp}_{attempt_id}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GeometryAuthorizationDocumentBinding:
    """What a fixed-path execution command reads out of the committed,
    dated authorization document before ever consuming a claim."""

    authorization_id: str
    authorization_document_path: str
    authorization_document_sha256: str
    protocol_document_path: str
    protocol_document_sha256: str
    implementation_commit_sha: str
    operating_point_sha256: str
    candidate_manifest_canonical_sha256: str
    model_revision: str
    tokenizer_revision: str
    rkv_revision: str
    maximum_branches: int
    maximum_invocations: int
    runtime_limit_seconds: int
    vram_limit_bytes: int


def verify_authorization_document_binding(
    binding: GeometryAuthorizationDocumentBinding,
    *,
    repository_root: str | Path,
) -> None:
    """Fail closed if the on-disk authorization document does not
    byte-hash to `authorization_document_sha256`, or if any bound field is
    missing/malformed. Never repairs a mismatch."""
    doc_path = Path(repository_root) / binding.authorization_document_path
    if not doc_path.is_file():
        raise GeometryAuthorizationError(f"authorization document not found at {doc_path}")
    observed = sha256_file(doc_path)
    if observed != binding.authorization_document_sha256:
        raise GeometryAuthorizationError(
            f"authorization document hash mismatch: bound={binding.authorization_document_sha256} "
            f"observed={observed}"
        )
    if binding.maximum_invocations != 1:
        raise GeometryAuthorizationError("this pilot authorizes exactly one invocation")
    if binding.maximum_branches > 10:
        raise GeometryAuthorizationError("this pilot authorizes at most 10 branches")


def build_claim_payload(
    *,
    binding: GeometryAuthorizationDocumentBinding,
    authorized_repository: str,
    authorized_branch: str,
    observed_execution_commit_sha: str,
    attempt_id: str,
    attempt_directory_path: str,
    claimed_at_utc: str,
) -> dict[str, Any]:
    payload = {
        "artifact_schema_version": CLAIM_SCHEMA_VERSION,
        "authorization_id": binding.authorization_id,
        "authorization_document_path": binding.authorization_document_path,
        "authorization_document_sha256": binding.authorization_document_sha256,
        "protocol_document_path": binding.protocol_document_path,
        "protocol_document_sha256": binding.protocol_document_sha256,
        "implementation_commit_sha": binding.implementation_commit_sha,
        "operating_point_sha256": binding.operating_point_sha256,
        "candidate_manifest_canonical_sha256": binding.candidate_manifest_canonical_sha256,
        "authorized_repository": authorized_repository,
        "authorized_branch": authorized_branch,
        "observed_execution_commit_sha": observed_execution_commit_sha,
        "model_revision": binding.model_revision,
        "tokenizer_revision": binding.tokenizer_revision,
        "rkv_revision": binding.rkv_revision,
        "attempt_id": attempt_id,
        "attempt_directory_path": attempt_directory_path,
        "global_claim_path": global_claim_path(binding.authorization_id),
        "claimed_at_utc": claimed_at_utc,
    }
    return {**payload, "canonical_sha256": canonical_payload_hash(payload)}


def claim_authorization(*, repository_root: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically create the claim file at its one deterministic path.
    Creation of this exclusively-created filesystem entry IS the
    consumption event -- never a later successful GPU step. Raises
    `GeometryAuthorizationAlreadyConsumed` if the path already exists in
    any state (never overwritten, never treated as retryable)."""
    verify_canonical_hash(payload)
    claim_path = Path(repository_root) / global_claim_path(payload["authorization_id"])
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise GeometryAuthorizationAlreadyConsumed(
            f"a claim already exists at {claim_path} -- this authorization is consumed, no retry"
        ) from exc
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
    except Exception:
        # Best-effort cleanup of a partially-written claim we JUST created
        # in this same call; a claim created by ANY other process/attempt
        # is never touched.
        try:
            claim_path.unlink()
        except OSError:
            pass
        raise
    return claim_path


def verify_claim_exists_and_matches(*, repository_root: str | Path, authorization_id: str) -> dict[str, Any]:
    claim_path = Path(repository_root) / global_claim_path(authorization_id)
    if not claim_path.is_file():
        raise GeometryAuthorizationError(f"no claim found at {claim_path}")
    payload = json.loads(claim_path.read_text())
    verify_canonical_hash(payload)
    return payload


def load_binding_from_json(path: str | Path) -> GeometryAuthorizationDocumentBinding:
    """Load a `GeometryAuthorizationDocumentBinding` from a committed JSON
    sidecar next to the human-readable markdown authorization document --
    this repository already pairs several `.md` authorization/protocol
    documents with `.json`/`.sha256` sidecars (e.g.
    `configs/discovery/b2a_r3_candidate_manifest.json` alongside its
    protocol prose); this is the same pattern, applied to authorization
    binding fields rather than free-form markdown parsing."""
    raw = json.loads(Path(path).read_text())
    return GeometryAuthorizationDocumentBinding(**raw)
