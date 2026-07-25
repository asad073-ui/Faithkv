import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kvcot.discovery.diagnostic_pilot_authorization import (
    AUTHORIZATION_JSON_BEGIN,
    AUTHORIZATION_JSON_END,
    DiagnosticAuthorizationConsumed,
    claim_authorization_once,
    parse_authorization_document,
    verify_claim,
)
from kvcot.discovery.diagnostic_pilot_contract import (
    BRANCH,
    CANDIDATE_MANIFEST_CANONICAL_SHA256,
    EXACT_EXECUTION_COMMAND,
    MODEL_REVISION,
    REPOSITORY,
    RKV_REVISION,
    TOKENIZER_REVISION,
    attach_canonical_hash,
)


def authorization(tmp_path):
    claim = tmp_path / "claim.json"
    payload = attach_canonical_hash(
        {
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
            "implementation_audit_sha256": "d" * 64,
            "candidate_manifest_canonical_sha256": CANDIDATE_MANIFEST_CANONICAL_SHA256,
            "maximum_qualification_candidates": 8,
            "maximum_selected_examples": 3,
            "maximum_selected_events": 3,
            "events_per_example": 1,
            "maximum_candidate_pool_size": 4,
            "maximum_interventions_per_selected_example": 6,
            "kv_restore_widths": [1, 2],
            "single_rtx3090": True,
            "cpu_offload": False,
            "vram_limit_bytes": 22 * 1024**3,
            "runtime_limit_seconds": 5_400,
            "exact_command": EXACT_EXECUTION_COMMAND,
        }
    )
    document = tmp_path / "authorization.md"
    document.write_text(
        f"# Synthetic\n\n{AUTHORIZATION_JSON_BEGIN}\n{json.dumps(payload)}\n{AUTHORIZATION_JSON_END}\n",
        encoding="utf-8",
    )
    return parse_authorization_document(document), claim


def test_claim_consumption_is_atomic_and_permanently_prohibits_retry(tmp_path):
    auth, claim_path = authorization(tmp_path)
    written, claim = claim_authorization_once(auth, attempt_id="attempt", attempt_directory="/tmp/attempt")
    assert written == claim_path
    assert verify_claim(claim_path)["retry_allowed"] is False
    with pytest.raises(DiagnosticAuthorizationConsumed):
        claim_authorization_once(auth, attempt_id="again", attempt_directory="/tmp/again")


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
