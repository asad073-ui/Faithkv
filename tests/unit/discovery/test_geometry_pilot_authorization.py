import json

import pytest

from kvcot.discovery.geometry_pilot_authorization import (
    GEOMETRY_CLAIMS_DIR,
    GeometryAuthorizationAlreadyConsumed,
    GeometryAuthorizationDocumentBinding,
    GeometryAuthorizationError,
    attempt_directory_name,
    build_claim_payload,
    claim_authorization,
    global_claim_path,
    load_binding_from_json,
    sha256_file,
    verify_authorization_document_binding,
    verify_claim_exists_and_matches,
)


def _binding(**overrides) -> GeometryAuthorizationDocumentBinding:
    kwargs = dict(
        authorization_id="geometry-pilot-2026-07-25",
        authorization_document_path="docs/FAKE_AUTH.md",
        authorization_document_sha256="",  # filled by caller after writing the fixture file
        protocol_document_path="docs/FAKE_PROTOCOL.md",
        protocol_document_sha256="a" * 64,
        implementation_commit_sha="b" * 40,
        operating_point_sha256="c" * 64,
        candidate_manifest_canonical_sha256="d" * 64,
        model_revision="e" * 40,
        tokenizer_revision="e" * 40,
        rkv_revision="f" * 40,
        maximum_branches=8,
        maximum_invocations=1,
        runtime_limit_seconds=14_400,
        vram_limit_bytes=22 * 1024**3,
    )
    kwargs.update(overrides)
    return GeometryAuthorizationDocumentBinding(**kwargs)


def test_global_claim_path_is_deterministic_and_disjoint_from_b2a_r3():
    path = global_claim_path("geometry-pilot-2026-07-25")
    assert path == f"{GEOMETRY_CLAIMS_DIR}/geometry-pilot-2026-07-25.json"
    assert "b2a_r3" not in path


def test_attempt_directory_name_disjoint_prefix():
    name = attempt_directory_name("20260725T000000000000Z", "abc123")
    assert name.startswith("geometry_pilot_attempt_")
    assert "b2a_r3" not in name


def test_verify_authorization_document_binding_passes_on_matching_hash(tmp_path):
    doc = tmp_path / "docs" / "FAKE_AUTH.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("frozen authorization content\n")
    binding = _binding(authorization_document_sha256=sha256_file(doc))
    verify_authorization_document_binding(binding, repository_root=tmp_path)


def test_verify_authorization_document_binding_fails_on_tamper(tmp_path):
    doc = tmp_path / "docs" / "FAKE_AUTH.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("original content\n")
    binding = _binding(authorization_document_sha256=sha256_file(doc))
    doc.write_text("tampered content\n")
    with pytest.raises(GeometryAuthorizationError, match="mismatch"):
        verify_authorization_document_binding(binding, repository_root=tmp_path)


def test_verify_authorization_document_binding_rejects_missing_file(tmp_path):
    binding = _binding(authorization_document_sha256="0" * 64)
    with pytest.raises(GeometryAuthorizationError, match="not found"):
        verify_authorization_document_binding(binding, repository_root=tmp_path)


def test_verify_authorization_document_binding_rejects_more_than_one_invocation(tmp_path):
    doc = tmp_path / "docs" / "FAKE_AUTH.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("x\n")
    binding = _binding(authorization_document_sha256=sha256_file(doc), maximum_invocations=2)
    with pytest.raises(GeometryAuthorizationError, match="one invocation"):
        verify_authorization_document_binding(binding, repository_root=tmp_path)


def test_verify_authorization_document_binding_rejects_over_ten_branches(tmp_path):
    doc = tmp_path / "docs" / "FAKE_AUTH.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("x\n")
    binding = _binding(authorization_document_sha256=sha256_file(doc), maximum_branches=11)
    with pytest.raises(GeometryAuthorizationError, match="10 branches"):
        verify_authorization_document_binding(binding, repository_root=tmp_path)


def test_claim_authorization_creates_exactly_one_file_and_is_reloadable(tmp_path):
    binding = _binding(authorization_document_sha256="9" * 64)
    payload = build_claim_payload(
        binding=binding,
        authorized_repository="asad073-ui/Faithkv",
        authorized_branch="research/8b-structured-restoration-geometry",
        observed_execution_commit_sha="1" * 40,
        attempt_id="deadbeef",
        attempt_directory_path=attempt_directory_name("20260725T000000000000Z", "deadbeef"),
        claimed_at_utc="2026-07-25T00:00:00+00:00",
    )
    claim_path = claim_authorization(repository_root=tmp_path, payload=payload)
    assert claim_path.is_file()
    reloaded = verify_claim_exists_and_matches(repository_root=tmp_path, authorization_id=binding.authorization_id)
    assert reloaded["attempt_id"] == "deadbeef"


def test_claim_authorization_second_attempt_is_refused_never_overwritten(tmp_path):
    binding = _binding(authorization_document_sha256="9" * 64)
    payload = build_claim_payload(
        binding=binding,
        authorized_repository="asad073-ui/Faithkv",
        authorized_branch="research/8b-structured-restoration-geometry",
        observed_execution_commit_sha="1" * 40,
        attempt_id="first",
        attempt_directory_path=attempt_directory_name("20260725T000000000000Z", "first"),
        claimed_at_utc="2026-07-25T00:00:00+00:00",
    )
    claim_authorization(repository_root=tmp_path, payload=payload)

    second_payload = dict(payload)
    second_payload["attempt_id"] = "second"
    del second_payload["canonical_sha256"]
    from kvcot.discovery.geometry_pilot_contract import attach_canonical_hash

    second_payload = attach_canonical_hash(second_payload)

    with pytest.raises(GeometryAuthorizationAlreadyConsumed):
        claim_authorization(repository_root=tmp_path, payload=second_payload)

    # the original claim's content is untouched
    reloaded = verify_claim_exists_and_matches(repository_root=tmp_path, authorization_id=binding.authorization_id)
    assert reloaded["attempt_id"] == "first"


def test_claim_authorization_rejects_tampered_payload_before_writing(tmp_path):
    payload = {"authorization_id": "x", "canonical_sha256": "0" * 64}
    with pytest.raises(ValueError, match="mismatch"):
        claim_authorization(repository_root=tmp_path, payload=payload)
    assert not (tmp_path / GEOMETRY_CLAIMS_DIR).exists()


def test_verify_claim_exists_and_matches_missing_claim_raises(tmp_path):
    with pytest.raises(GeometryAuthorizationError, match="no claim found"):
        verify_claim_exists_and_matches(repository_root=tmp_path, authorization_id="nope")


def test_load_binding_from_json_roundtrip(tmp_path):
    binding = _binding(authorization_document_sha256="9" * 64)
    sidecar = tmp_path / "binding.json"
    sidecar.write_text(json.dumps(binding.__dict__))
    loaded = load_binding_from_json(sidecar)
    assert loaded == binding
