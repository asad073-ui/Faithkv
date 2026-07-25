import json
from pathlib import Path

import pytest

from kvcot.discovery.attempt_artifacts import atomic_write_json
from kvcot.discovery.diagnostic_pilot_authorization import (
    AUTHORIZATION_JSON_BEGIN,
    AUTHORIZATION_JSON_END,
)
from kvcot.discovery.diagnostic_pilot_contract import (
    BRANCH,
    EXACT_EXECUTION_COMMAND,
    MODEL_REVISION,
    REPOSITORY,
    RKV_REVISION,
    TOKENIZER_REVISION,
    attach_canonical_hash,
)
from kvcot.discovery.diagnostic_pilot_execute import (
    _summarize,
    dry_run_diagnostic_pilot,
    run_diagnostic_pilot,
    verify_attempt,
)


def qualification(ordinal):
    return {
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


def pair(arm, gain, *, margin=False, candidate=1, donor=9):
    baseline = [1.0] * 48
    intervention = [1.0 - gain * 48] + [1.0] * 47
    baseline_mean = sum(baseline) / len(baseline)
    intervention_mean = sum(intervention) / len(intervention)
    before_margin = -0.5 if margin else 0.5
    after_margin = 0.5
    return {
        "arm": arm,
        "diagnostic_only": arm == "candidate_upper_bound",
        "deployable_performance": False,
        "restore_width": 2 if arm == "restore_width" else 1,
        "kv_head_indices": [0, 1] if arm == "restore_width" else [0],
        "candidate_absolute_position": candidate,
        "donor_absolute_position": donor,
        "baseline_scored_token_ids": list(range(48)),
        "intervention_scored_token_ids": list(range(48)),
        "baseline_per_token_nll": baseline,
        "intervention_per_token_nll": intervention,
        "baseline_mean_nll": baseline_mean,
        "intervention_mean_nll": intervention_mean,
        "swap_gain": baseline_mean - intervention_mean,
        "baseline_answer_token_margins": [{"absolute_position": 5, "margin": before_margin}],
        "intervention_answer_token_margins": [{"absolute_position": 5, "margin": after_margin}],
        "answer_margin_sign_change": margin,
        "baseline_fixed_trace_extracted_answer": "1",
        "intervention_fixed_trace_extracted_answer": "1",
        "baseline_fixed_trace_correctness_status": "correct",
        "intervention_fixed_trace_correctness_status": "correct",
        "fixed_trace_extracted_answer_change": False,
        "fixed_trace_correctness_change": False,
        "baseline_final_state_sha256": "a" * 64,
        "intervention_final_state_sha256": "a" * 64,
        "mutation": {"is_noop": arm == "no_op"},
    }


def result(ordinal, *, a=0.0, b=0.0, margin=False, noop=True):
    return {
        "candidate_ordinal": ordinal,
        "unique_id": f"row-{ordinal}",
        "selected_event": {
            "donor_absolute_position": 9,
            "candidate_pool": [
                {"absolute_token_position": 1},
                {"absolute_token_position": 2},
            ]
        },
        "pair_records": [
            pair("candidate_upper_bound", a, margin=margin, candidate=1),
            pair("candidate_upper_bound", a / 2, candidate=2),
            pair("restore_width", b, candidate=1),
            pair("no_op", 0.0, candidate=9),
        ],
        "noop_exact": noop,
    }


@pytest.mark.parametrize(
    ("a", "b", "margin", "classification"),
    [
        (0.02, 0.0, False, "candidate_selection_implicated"),
        (0.0, 0.02, False, "restore_dilution_implicated"),
        (0.02, 0.02, False, "candidate_and_width_implicated"),
        (0.0, 0.0, True, "readout_implicated"),
        (0.0, 0.0, False, "mechanism_killed_at_1_5b"),
    ],
)
def test_summary_reconstructs_every_scientific_outcome(a, b, margin, classification):
    qualifications = [qualification(index) for index in range(3)]
    results = [result(index, a=a, b=b, margin=margin) for index in range(3)]
    summary = _summarize(qualifications, results)
    assert summary["classification"] == classification
    assert summary["completed_arm_a_pairs"] == 6
    assert summary["completed_arm_b_pairs"] == 3
    assert summary["completed_noop_pairs"] == 3
    assert summary["completed_total_pairs"] == summary["expected_total_pairs"] == 12
    assert summary["b2b_status"] == "blocked"
    assert "does not confirm or refute" in summary["scale_limitation"]


def test_noop_mismatch_voids_result():
    summary = _summarize(
        [qualification(index) for index in range(3)],
        [result(0), result(1, noop=False), result(2)],
    )
    assert summary["classification"] == "void"


def test_incomplete_pair_population_voids_result():
    results = [result(index) for index in range(3)]
    results[0]["pair_records"].pop()
    summary = _summarize([qualification(index) for index in range(3)], results)
    assert summary["classification"] == "void"


def test_fewer_than_three_qualified_is_mechanically_unqualified():
    summary = _summarize(
        [qualification(0), qualification(1)],
        [result(0), result(1)],
    )
    assert summary["classification"] == "mechanically_unqualified"


def write_authorization(tmp_path):
    repository = tmp_path / "repository"
    document = repository / "docs" / "authorization.md"
    document.parent.mkdir(parents=True)
    claim = tmp_path / "claim.json"
    output_root = tmp_path / "execution"
    payload = attach_canonical_hash(
        {
            "authorization_id": "synthetic-coordinator",
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
            "runtime_config_canonical_sha256": "b" * 64,
            "protocol_document_sha256": "c" * 64,
            "candidate_manifest_canonical_sha256": (
                "b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42"
            ),
            "output_root": str(output_root),
            "implementation_audit_sha256": "e" * 64,
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
    document.write_text(
        f"{AUTHORIZATION_JSON_BEGIN}\n{json.dumps(payload)}\n{AUTHORIZATION_JSON_END}\n",
        encoding="utf-8",
    )
    return repository, document, claim, output_root


def test_dry_run_is_non_consuming_and_requests_no_cuda_or_weights(tmp_path, monkeypatch):
    repository, document, claim, output_root = write_authorization(tmp_path)
    monkeypatch.setattr(
        "kvcot.discovery.diagnostic_pilot_execute.verify_runtime_inputs",
        lambda _path: {
            "canonical_sha256": "b" * 64,
            "protocol_document_sha256": "c" * 64,
            "model_snapshot_path": "/exact/model",
            "tokenizer_snapshot_path": "/exact/tokenizer",
        },
    )

    def fake_git(root, *args):
        command = tuple(args)
        if Path(root).name == "R-KV" and command == ("rev-parse", "HEAD"):
            return RKV_REVISION
        if command == ("rev-parse", "HEAD") or command == ("rev-parse", f"origin/{BRANCH}"):
            return "d" * 40
        if command == ("rev-parse", "HEAD^"):
            return "a" * 40
        if command == ("branch", "--show-current"):
            return BRANCH
        if command == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        if command == ("diff", "--name-only", "HEAD^", "HEAD"):
            return "docs/authorization.md"
        raise AssertionError(command)

    monkeypatch.setattr("kvcot.discovery.diagnostic_pilot_execute._git", fake_git)
    report = dry_run_diagnostic_pilot(
        repository_root=repository, authorization_document=document
    )
    assert report["would_initialize_cuda"] is False
    assert report["would_load_model_weights"] is False
    assert report["would_consume_claim"] is False
    assert not claim.exists()
    assert not output_root.exists()


def test_fake_worker_coordinator_writes_reconstructable_immutable_attempt(tmp_path, monkeypatch):
    repository, document, claim, _output_root = write_authorization(tmp_path)
    monkeypatch.setattr(
        "kvcot.discovery.diagnostic_pilot_execute._preflight",
        lambda *_args: {"passed": True, "would_initialize_cuda": False},
    )
    monkeypatch.setattr(
        "kvcot.discovery.diagnostic_pilot_execute.verify_runtime_inputs",
        lambda _path: {
            "protocol_document_sha256": "c" * 64,
            "canonical_sha256": "b" * 64,
            "candidate_prompts_path": str(tmp_path / "prompts.json"),
        },
    )

    def fake_launch(*, role, ordinal, output, **_kwargs):
        if role == "fullkv":
            payload = {
                "candidate_ordinal": ordinal,
                "peak_cuda_allocated_bytes": 10,
                "peak_cuda_reserved_bytes": 20,
            }
        else:
            payload = result(ordinal)
            payload.update(
                {
                    "qualification": qualification(ordinal),
                    "fixed_trace_extracted_answer": "1",
                    "fixed_trace_correctness_status": "correct",
                    "peak_cuda_allocated_bytes": 10,
                    "peak_cuda_reserved_bytes": 20,
                }
            )
        atomic_write_json(output, payload)
        return payload

    monkeypatch.setattr("kvcot.discovery.diagnostic_pilot_execute._launch_worker", fake_launch)
    report = run_diagnostic_pilot(
        repository_root=repository, authorization_document=document
    )
    attempt = Path(report["attempt_directory"])
    assert claim.exists()
    assert report["retry_allowed"] is False
    assert verify_attempt(attempt)["verified"] is True
    assert json.loads((attempt / "scientific_summary.json").read_text())["classification"] == (
        "mechanism_killed_at_1_5b"
    )
