import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kvcot.discovery.attempt_artifacts import sha256_file
from kvcot.discovery.diagnostic_pilot_authorization import (
    AUTHORIZATION_JSON_BEGIN,
    AUTHORIZATION_JSON_END,
    DiagnosticAuthorizationConsumed,
    DiagnosticAuthorizationRefused,
    claim_authorization_once,
    parse_authorization_document,
    verify_claim,
)
from kvcot.discovery.diagnostic_pilot_contract import (
    BRANCH,
    CANDIDATE_MANIFEST_CANONICAL_SHA256,
    GENERATIONS,
    MODEL_REVISION,
    R1_GENERATION,
    R2_GENERATION,
    REPOSITORY,
    RKV_REVISION,
    TOKENIZER_REVISION,
    attach_canonical_hash,
    execution_command_document_argument,
    generation_for_authorization_document,
    verify_canonical_hash,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def build_authorization(tmp_path, *, generation=R2_GENERATION, document_path=None, **overrides):
    """Write a synthetic authorization document and return its path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    claim = tmp_path / "claim.json"
    audit = tmp_path / "implementation-audit.md"
    audit.write_text("PASS synthetic audit\n", encoding="utf-8")
    pair_budget = (
        {"maximum_interventions_per_selected_example": 6}
        if generation is R1_GENERATION
        else {"maximum_pairs_per_selected_example": 9, "maximum_total_pairs": 27}
    )
    payload = {
        "authorization_id": "synthetic-one-use",
        "repository": REPOSITORY,
        "authorized_branch": BRANCH,
        "authorized_implementation_sha": "a" * 40,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "rkv_revision": RKV_REVISION,
        "maximum_invocations": 1,
        "automatic_retries": 0,
        "claim_path": str(claim),
        "runtime_config_path": str(tmp_path / "runtime.json"),
        "output_root": str(tmp_path / "execution"),
        "protocol_document_sha256": "b" * 64,
        "runtime_config_canonical_sha256": "c" * 64,
        "implementation_audit_path": str(audit),
        "implementation_audit_sha256": sha256_file(audit),
        "implementation_audit_verdict": "PASS",
        "candidate_manifest_canonical_sha256": CANDIDATE_MANIFEST_CANONICAL_SHA256,
        "maximum_qualification_candidates": 8,
        "maximum_selected_examples": 3,
        "maximum_selected_events": 3,
        "events_per_example": 1,
        "maximum_candidate_pool_size": 4,
        **pair_budget,
        "kv_restore_widths": [1, 2],
        "single_rtx3090": True,
        "cpu_offload": False,
        "vram_limit_bytes": 22 * 1024**3,
        "runtime_limit_seconds": 5_400,
        "exact_command": generation.exact_execution_command,
    }
    payload.update(overrides)
    document = tmp_path / (document_path or generation.authorization_document_path)
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(
        "# Synthetic\n\n"
        f"{AUTHORIZATION_JSON_BEGIN}\n{json.dumps(attach_canonical_hash(payload))}\n"
        f"{AUTHORIZATION_JSON_END}\n",
        encoding="utf-8",
    )
    return document, claim


def authorization(tmp_path):
    document, claim = build_authorization(tmp_path)
    return parse_authorization_document(document), claim


def committed_payload(relative_path):
    """The JSON payload of a committed authorization document."""
    text = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
    raw = text.split(AUTHORIZATION_JSON_BEGIN, 1)[1].split(AUTHORIZATION_JSON_END, 1)[0]
    return json.loads(raw.strip())


def test_consumed_r1_authorization_document_is_unchanged_and_still_resolves():
    """The consumed R1 authorization stays intact and R1-resolvable.

    Its bound implementation audit lives outside the repository, so the
    full audit-binding parse is a host gate; everything the repair could
    have broken is checked here from the committed bytes alone.
    """
    payload = committed_payload(R1_GENERATION.authorization_document_path)
    verify_canonical_hash(payload)
    assert payload["canonical_sha256"] == (
        "780ba2db1d7d77ce7d7c940b43c0813add5df8585effbc4d4d1ca061a6445bf8"
    )
    assert payload["authorization_id"] == (
        "post-stage-c-diagnostic-pilot-2026-07-25-one-use"
    )
    assert payload["exact_command"] == R1_GENERATION.exact_execution_command
    assert payload["maximum_interventions_per_selected_example"] == 6
    assert payload["output_root"] == R1_GENERATION.default_output_root
    # The document resolves to R1, and its own command names it.
    assert (
        generation_for_authorization_document(
            REPOSITORY_ROOT / R1_GENERATION.authorization_document_path
        )
        is R1_GENERATION
    )
    assert (
        execution_command_document_argument(payload["exact_command"])
        == R1_GENERATION.authorization_document_path
    )


def test_committed_r1_authorization_payload_still_parses_end_to_end(tmp_path):
    """Re-parse the committed R1 payload against a local audit file.

    Only the two host-path bindings are redirected into `tmp_path`; every
    frozen scientific and governance field is the committed one.
    """
    payload = committed_payload(R1_GENERATION.authorization_document_path)
    audit = tmp_path / "implementation-audit.md"
    audit.write_text("PASS synthetic stand-in for the host audit\n", encoding="utf-8")
    payload["implementation_audit_path"] = str(audit)
    payload["implementation_audit_sha256"] = sha256_file(audit)
    payload["claim_path"] = str(tmp_path / "claim.json")
    payload["output_root"] = str(tmp_path / "execution")
    payload["runtime_config_path"] = str(tmp_path / "runtime.json")
    document = tmp_path / R1_GENERATION.authorization_document_path
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(
        f"{AUTHORIZATION_JSON_BEGIN}\n{json.dumps(attach_canonical_hash(payload))}\n"
        f"{AUTHORIZATION_JSON_END}\n",
        encoding="utf-8",
    )
    parsed = parse_authorization_document(document)
    assert parsed.generation is R1_GENERATION
    assert parsed.payload["maximum_interventions_per_selected_example"] == 6
    assert parsed.payload["exact_command"] == R1_GENERATION.exact_execution_command


def test_r2_authorization_shape_parses(tmp_path):
    document, _claim = build_authorization(tmp_path, generation=R2_GENERATION)
    parsed = parse_authorization_document(document)
    assert parsed.generation is R2_GENERATION
    assert parsed.payload["maximum_pairs_per_selected_example"] == 9
    assert parsed.payload["maximum_total_pairs"] == 27


def test_r1_and_r2_pair_budgets_are_not_interchangeable(tmp_path):
    r1_document, _claim = build_authorization(
        tmp_path / "a",
        generation=R1_GENERATION,
        maximum_interventions_per_selected_example=9,
    )
    with pytest.raises(
        DiagnosticAuthorizationRefused,
        match="maximum_interventions_per_selected_example",
    ):
        parse_authorization_document(r1_document)
    r2_document, _claim = build_authorization(
        tmp_path / "b", generation=R2_GENERATION, maximum_pairs_per_selected_example=6
    )
    with pytest.raises(
        DiagnosticAuthorizationRefused, match="maximum_pairs_per_selected_example"
    ):
        parse_authorization_document(r2_document)


def test_document_not_named_by_its_own_command_is_rejected(tmp_path):
    # An R2 payload lifted verbatim into a differently-located document.
    document, _claim = build_authorization(
        tmp_path, generation=R2_GENERATION, document_path="docs/somewhere-else.md"
    )
    with pytest.raises(DiagnosticAuthorizationRefused, match="not a frozen"):
        parse_authorization_document(document)
    # An R2 document carrying the R1 command.
    document, _claim = build_authorization(
        tmp_path / "swapped",
        generation=R2_GENERATION,
        exact_command=R1_GENERATION.exact_execution_command,
    )
    with pytest.raises(DiagnosticAuthorizationRefused, match="exact_command"):
        parse_authorization_document(document)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("output_root", R1_GENERATION.default_output_root, "output root collides"),
        (
            "output_root",
            f"{R1_GENERATION.default_output_root}/nested",
            "output root collides",
        ),
        (
            "claim_path",
            f"{R1_GENERATION.default_output_root}/authorization-claim.json",
            "claim path collides",
        ),
        (
            "runtime_config_path",
            f"{R1_GENERATION.default_runtime_root}/runtime_config.json",
            "runtime config path collides",
        ),
    ],
)
def test_r2_authorization_cannot_collide_with_consumed_r1_paths(
    tmp_path, field, value, message
):
    document, _claim = build_authorization(
        tmp_path, generation=R2_GENERATION, **{field: value}
    )
    with pytest.raises(DiagnosticAuthorizationRefused, match=message):
        parse_authorization_document(document)


def test_r2_default_paths_are_disjoint_from_r1():
    assert R2_GENERATION.default_output_root != R1_GENERATION.default_output_root
    assert R2_GENERATION.default_runtime_root != R1_GENERATION.default_runtime_root
    for r2, r1 in (
        (R2_GENERATION.default_output_root, R1_GENERATION.default_output_root),
        (R2_GENERATION.default_runtime_root, R1_GENERATION.default_runtime_root),
    ):
        assert not Path(r2).is_relative_to(Path(r1))
        assert not Path(r1).is_relative_to(Path(r2))


def test_r1_generation_attempt_verification_remains_reachable():
    """Verification must stay generation-agnostic.

    The consumed R1 attempt itself lives outside the repository, so it is
    verified as an explicit host gate rather than here; what this test
    pins is that nothing in the repair made R1 unverifiable by
    construction.
    """
    from kvcot.discovery.diagnostic_pilot_execute import EXECUTING_GENERATION

    payload = committed_payload(R1_GENERATION.authorization_document_path)
    assert EXECUTING_GENERATION is not R1_GENERATION
    assert Path(payload["output_root"]) == Path(R1_GENERATION.default_output_root)
    # R1 evidence is verifiable but never executable again.
    assert R1_GENERATION in GENERATIONS


def test_claim_consumption_is_atomic_and_permanently_prohibits_retry(tmp_path):
    auth, claim_path = authorization(tmp_path)
    written, claim = claim_authorization_once(auth, attempt_id="attempt", attempt_directory="/tmp/attempt")
    assert written == claim_path
    assert verify_claim(claim_path)["retry_allowed"] is False
    with pytest.raises(DiagnosticAuthorizationConsumed):
        claim_authorization_once(auth, attempt_id="again", attempt_directory="/tmp/again")


def test_authorization_binds_the_actual_implementation_audit_bytes(tmp_path):
    auth, _claim_path = authorization(tmp_path)
    Path(auth.payload["implementation_audit_path"]).write_text(
        "FAIL mutated audit\n", encoding="utf-8"
    )
    with pytest.raises(Exception, match="audit file hash mismatch"):
        parse_authorization_document(auth.document_path)


def test_worker_and_execute_module_imports_do_not_import_torch_or_transformers():
    code = """
import sys
import kvcot.discovery.diagnostic_pilot_workers
import kvcot.discovery.diagnostic_pilot_execute
assert not any(x == 'torch' or x.startswith('torch.') for x in sys.modules)
assert not any(x == 'transformers' or x.startswith('transformers.') for x in sys.modules)
"""
    env = dict(os.environ)
    env["USE_TORCH"] = "0"
    completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, env=env)
    assert completed.returncode == 0, completed.stderr


def test_worker_entry_preserves_failure_artifact_and_forbids_retry(tmp_path):
    output = tmp_path / "worker.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "kvcot.discovery.diagnostic_pilot_worker_entry",
            "--role",
            "fullkv",
            "--config",
            str(tmp_path / "missing-config.yaml"),
            "--prompts",
            str(tmp_path / "missing-prompts.json"),
            "--candidate-ordinal",
            "0",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    failure_path = Path(f"{output}.failure.json")
    assert completed.returncode == 1
    assert not output.exists()
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["role"] == "fullkv"
    assert failure["candidate_ordinal"] == 0
    assert failure["retry_allowed"] is False
