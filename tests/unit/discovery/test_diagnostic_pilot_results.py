import json
from pathlib import Path
import time

import pytest

from kvcot.discovery.attempt_artifacts import atomic_write_json, sha256_file
from kvcot.discovery.diagnostic_pilot_authorization import (
    AUTHORIZATION_JSON_BEGIN,
    AUTHORIZATION_JSON_END,
)
from kvcot.discovery.diagnostic_pilot_contract import (
    BRANCH,
    CANDIDATE_MANIFEST_PATH,
    CONFIG_PATH,
    EXACT_EXECUTION_COMMAND,
    MODEL_REVISION,
    PAIR_SCHEMA_VERSION,
    PROTOCOL_DOCUMENT_PATH,
    REPOSITORY,
    RKV_REVISION,
    TOKENIZER_REVISION,
    attach_canonical_hash,
)
from kvcot.discovery.diagnostic_pilot_execute import (
    DiagnosticExecutionRefused,
    _launch_worker,
    _preserve_consumed_failure,
    _raw_worker_binding_valid,
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
        "fullkv_correctness_status": "correct",
        "rkv_correctness_status": "correct",
        "rkv_natural_cap_hit": False,
        "observed_compaction_event_count": 1,
        "eligible_scored_event_count": 1,
    }


def placement_evidence():
    return {
        "requested_device": "cuda:0",
        "every_parameter_on_cuda": True,
        "no_offload_verified": True,
        "parameter_count": 100,
        "unique_device_types": ["cuda"],
        "unique_devices": ["cuda:0"],
        "hf_device_map": None,
    }


def device_evidence():
    return {
        "verified": True,
        "visible_gpu_count": 1,
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "device_index": 0,
        "requested_device": "cuda:0",
        "total_vram_bytes": 24 * 1024**3,
        "compute_capability": [8, 6],
        "driver_version": "synthetic-driver",
        "cuda_runtime": "synthetic-cuda",
        "cudnn_version": "synthetic-cudnn",
    }


def pair(arm, gain, *, margin=False, candidate=1, donor=9):
    baseline = [1.0] * 48
    intervention = [1.0 - gain * 48] + [1.0] * 47
    baseline_mean = sum(baseline) / len(baseline)
    intervention_mean = sum(intervention) / len(intervention)
    before_margin = -0.5 if margin else 0.5
    after_margin = 0.5
    kv_heads = [0, 1] if arm == "restore_width" else [0]
    donor_slots = {"0": 3, **({"1": 4} if arm == "restore_width" else {})}

    def margin_row(value):
        alternative_logp = -2.0
        reference_logp = alternative_logp + value
        return {
            "absolute_position": 5,
            "reference_token_id": 7,
            "strongest_alternative_token_id": 8,
            "reference_log_probability": reference_logp,
            "correct_answer_log_probability": reference_logp,
            "strongest_alternative_log_probability": alternative_logp,
            "margin": value,
            "is_correct_answer_token": True,
        }

    return {
        "artifact_schema_version": PAIR_SCHEMA_VERSION,
        "arm": arm,
        "diagnostic_only": arm == "candidate_upper_bound",
        "deployable_performance": False,
        "restore_width": 2 if arm == "restore_width" else 1,
        "compaction_event_id": 2,
        "layer_index": 4,
        "selected_kv_head": 0,
        "kv_head_indices": kv_heads,
        "candidate_absolute_position": candidate,
        "donor_absolute_position": donor,
        "donor_post_storage_positions_by_head": donor_slots,
        "baseline_scored_token_ids": list(range(48)),
        "intervention_scored_token_ids": list(range(48)),
        "baseline_per_token_nll": baseline,
        "intervention_per_token_nll": intervention,
        "baseline_mean_nll": baseline_mean,
        "intervention_mean_nll": intervention_mean,
        "swap_gain": baseline_mean - intervention_mean,
        "baseline_answer_token_margins": [margin_row(before_margin)],
        "intervention_answer_token_margins": [margin_row(after_margin)],
        "answer_margin_sign_change": margin,
        "baseline_fixed_trace_extracted_answer": "1",
        "intervention_fixed_trace_extracted_answer": "1",
        "baseline_fixed_trace_correctness_status": "correct",
        "intervention_fixed_trace_correctness_status": "correct",
        "fixed_trace_extracted_answer_change": False,
        "fixed_trace_correctness_change": False,
        "baseline_final_state_sha256": "a" * 64,
        "intervention_final_state_sha256": "a" * 64,
        "mutation": {
            "layer_index": 4,
            "kv_head_indices": kv_heads,
            "token_positions_by_head": [
                [head, donor_slots[str(head)]] for head in kv_heads
            ],
            "cache_shape_unchanged": True,
            "absolute_position_unchanged": True,
            "provenance_valid": True,
            "key_slots_changed": 0 if arm == "no_op" else len(kv_heads),
            "value_slots_changed": 0 if arm == "no_op" else len(kv_heads),
            "is_noop": arm == "no_op",
        },
    }


def result(ordinal, *, a=0.0, b=0.0, margin=False, noop=True):
    payload = {
        "role": "diagnostic_rkv",
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "rkv_revision": RKV_REVISION,
        "dataset_repo": "synthetic-dataset",
        "dataset_revision": "synthetic-revision",
        "manifest_hash": "1" * 64,
        "prompt_token_ids_sha256": "2" * 64,
        "prompt_token_count": 10,
        "architecture": {"num_attention_heads": 12, "num_key_value_heads": 2},
        "parameter_placement": placement_evidence(),
        "device_evidence": device_evidence(),
        "candidate_ordinal": ordinal,
        "unique_id": f"row-{ordinal}",
        "selected_event": {
            "event_index": 2,
            "absolute_event_position": 100,
            "layer_index": 4,
            "selected_kv_head": 0,
            "deployable_event_score": 0.9,
            "donor_absolute_position": 9,
            "candidate_pool": [
                {"absolute_token_position": 1, "deployable_score": 0.9},
                {"absolute_token_position": 2, "deployable_score": 0.8},
            ],
            "candidate_pool_frozen_before_intervention": True,
            "candidate_pool_diagnostic_only": True,
        },
        "eligible_event_score_evidence": [
            {
                "event_index": 2,
                "absolute_event_position": 100,
                "layer_index": 4,
                "selected_kv_head": 0,
                "deployable_event_score": 0.9,
                "candidate_pool": [
                    {"absolute_token_position": 1, "deployable_score": 0.9},
                    {"absolute_token_position": 2, "deployable_score": 0.8},
                ],
                "captured_before_intervention": True,
                "rank_zero_candidate_available_in_all_kv_heads": True,
            }
        ],
        "score_replay_retained_full_snapshots": 0,
        "selected_snapshot_count": 1,
        "pair_records": [
            pair("candidate_upper_bound", a, margin=margin, candidate=1),
            pair("candidate_upper_bound", a / 2, candidate=2),
            pair("restore_width", b, candidate=1),
            pair("no_op", 0.0, candidate=9),
        ],
        "noop_exact": noop,
        "qualified": True,
        "selected_event_replay_valid": True,
        "rkv_natural_cap_hit": False,
        "natural_full_token_count": 100,
        "final_cache_length_per_layer": {"0": 80},
        "fixed_trace_extracted_answer": "1",
        "fixed_trace_correctness_status": "correct",
        "free_running_answer_flip_available": False,
    }
    for rank, candidate_pair in enumerate(payload["pair_records"][:2]):
        candidate_pair["candidate_pool_rank"] = rank
        candidate_pair["is_control_candidate"] = rank == 0
    return payload


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


def test_arm_a_threshold_count_is_over_primitive_candidates_not_only_maxima():
    summary = _summarize(
        [qualification(index) for index in range(3)],
        [result(index, a=0.03) for index in range(3)],
    )
    assert summary["arm_a_gains_above_0_01"] == 6
    assert summary["arm_a_bounded_maxima_above_0_01"] == 3


def test_incomplete_pair_population_voids_result():
    results = [result(index) for index in range(3)]
    results[0]["pair_records"].pop()
    summary = _summarize([qualification(index) for index in range(3)], results)
    assert summary["classification"] == "void"


@pytest.mark.parametrize("mutation", ("pool_order", "event_choice", "pair_identity", "slot_map"))
def test_strict_primitive_reconstruction_voids_semantic_tampering(mutation):
    results = [result(index) for index in range(3)]
    target = results[0]
    if mutation == "pool_order":
        target["selected_event"]["candidate_pool"].reverse()
    elif mutation == "event_choice":
        target["eligible_event_score_evidence"].append(
            {
                "event_index": 1,
                "absolute_event_position": 90,
                "layer_index": 3,
                "selected_kv_head": 1,
                "deployable_event_score": 1.1,
                "candidate_pool": [
                    {"absolute_token_position": 4, "deployable_score": 1.1},
                    {"absolute_token_position": 5, "deployable_score": 1.0},
                ],
                "captured_before_intervention": True,
                "rank_zero_candidate_available_in_all_kv_heads": True,
            }
        )
    elif mutation == "pair_identity":
        target["pair_records"][0]["layer_index"] = 5
    else:
        target["pair_records"][0]["mutation"]["token_positions_by_head"] = [[0, 2]]
    summary = _summarize([qualification(index) for index in range(3)], results)
    assert summary["classification"] == "void"


def test_incorrect_answer_token_margin_sign_change_is_not_arm_c_movement():
    results = [result(index, margin=True) for index in range(3)]
    for row in results:
        for candidate_pair in row["pair_records"]:
            candidate_pair["answer_margin_sign_change"] = False
            for key in (
                "baseline_answer_token_margins",
                "intervention_answer_token_margins",
            ):
                for margin_row in candidate_pair[key]:
                    margin_row["is_correct_answer_token"] = False
                    margin_row["correct_answer_log_probability"] = None
    summary = _summarize([qualification(index) for index in range(3)], results)
    assert summary["primitive_populations_reconstructed"] is True
    assert summary["behavioural_change"] is False
    assert summary["classification"] == "mechanism_killed_at_1_5b"


def test_fullkv_rkv_raw_binding_is_reconstructed_from_primitives():
    rkv = result(0)
    rkv["qualification"] = qualification(0)
    fullkv = {
        "candidate_ordinal": 0,
        "role": "fullkv",
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "dataset_repo": "synthetic-dataset",
        "dataset_revision": "synthetic-revision",
        "manifest_hash": "1" * 64,
        "prompt_token_ids_sha256": "2" * 64,
        "prompt_token_count": 10,
        "dataset_row_identity": {"unique_id": "row-0"},
        "natural_answer_status": "correct",
        "cap_hit": False,
        "actual_batch_size_verified": True,
        "parameter_placement": placement_evidence(),
        "device_evidence": device_evidence(),
    }
    assert _raw_worker_binding_valid(fullkv, rkv)
    fullkv["dataset_row_identity"]["unique_id"] = "different-row"
    assert not _raw_worker_binding_valid(fullkv, rkv)
    fullkv["dataset_row_identity"]["unique_id"] = "row-0"
    rkv["rkv_natural_cap_hit"] = True
    assert not _raw_worker_binding_valid(fullkv, rkv)


def test_exhausted_budget_never_launches_a_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kvcot.discovery.diagnostic_pilot_execute.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not launch")),
    )
    with pytest.raises(DiagnosticExecutionRefused, match="was not launched"):
        _launch_worker(
            role="diagnostic-rkv",
            ordinal=0,
            output=tmp_path / "result.json",
            config_path=tmp_path / "config.yaml",
            prompts_path=tmp_path / "prompts.json",
            attempt_id="attempt",
            timeout=0.0,
        )


def test_late_exception_adds_failure_marker_without_rewriting_success_files(tmp_path):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    claim_path = tmp_path / "claim.json"
    claim_path.write_text("{}\n", encoding="utf-8")
    completion = {"completed": True, "retry_allowed": False}
    final = {"attempt_id": "attempt", "scientific_summary": {"classification": "synthetic"}}
    atomic_write_json(attempt / "completion.json", completion)
    atomic_write_json(attempt / "final.json", final)
    _preserve_consumed_failure(
        attempt=attempt,
        attempt_id="attempt",
        claim_path=claim_path,
        started=time.perf_counter(),
        exc=RuntimeError("late finalization failure"),
    )
    assert json.loads((attempt / "completion.json").read_text()) == completion
    assert json.loads((attempt / "final.json").read_text()) == final
    failure = json.loads((attempt / "failure.json").read_text())
    assert failure["failure_message"] == "late finalization failure"
    assert failure["retry_allowed"] is False


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
    audit = tmp_path / "implementation-audit.md"
    audit.write_text("PASS synthetic audit\n", encoding="utf-8")
    source_root = Path(__file__).resolve().parents[3]
    for relative_path in (PROTOCOL_DOCUMENT_PATH, CONFIG_PATH, CANDIDATE_MANIFEST_PATH):
        source = source_root / relative_path
        destination = repository / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    protocol_sha256 = sha256_file(repository / PROTOCOL_DOCUMENT_PATH)
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
            "protocol_document_sha256": protocol_sha256,
            "candidate_manifest_canonical_sha256": (
                "b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42"
            ),
            "output_root": str(output_root),
            "implementation_audit_path": str(audit),
            "implementation_audit_sha256": sha256_file(audit),
            "implementation_audit_verdict": "PASS",
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
        lambda _path, **_kwargs: {
            "canonical_sha256": "b" * 64,
            "protocol_document_sha256": sha256_file(repository / PROTOCOL_DOCUMENT_PATH),
            "config_byte_sha256": sha256_file(repository / CONFIG_PATH),
            "model_snapshot_path": "/exact/model",
            "tokenizer_snapshot_path": "/exact/tokenizer",
            "output_root": str(output_root),
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
        lambda _path, **_kwargs: {
            "protocol_document_sha256": sha256_file(repository / PROTOCOL_DOCUMENT_PATH),
            "canonical_sha256": "b" * 64,
            "candidate_prompts_path": str(tmp_path / "prompts.json"),
        },
    )

    def fake_launch(*, role, ordinal, output, **_kwargs):
        if role == "fullkv":
            payload = {
                "candidate_ordinal": ordinal,
                "role": "fullkv",
                "model_revision": MODEL_REVISION,
                "tokenizer_revision": TOKENIZER_REVISION,
                "dataset_repo": "synthetic-dataset",
                "dataset_revision": "synthetic-revision",
                "manifest_hash": "1" * 64,
                "prompt_token_ids_sha256": "2" * 64,
                "prompt_token_count": 10,
                "dataset_row_identity": {"unique_id": f"row-{ordinal}"},
                "natural_answer_status": "correct",
                "cap_hit": False,
                "actual_batch_size_verified": True,
                "parameter_placement": placement_evidence(),
                "device_evidence": device_evidence(),
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
    completion = json.loads((attempt / "completion.json").read_text())
    assert completion["peak_cuda_allocated_bytes"] == 10
    assert completion["peak_cuda_reserved_bytes"] == 20
    assert completion["peak_tracked_cuda_bytes"] == 20
    final_path = attempt / "final.json"
    original_final_text = final_path.read_text(encoding="utf-8")
    final = json.loads(original_final_text)
    final["scientific_summary"]["classification"] = "void"
    final_path.write_text(json.dumps(final, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="embedded scientific summary"):
        verify_attempt(attempt)
    final_path.write_text(original_final_text, encoding="utf-8")
    final = json.loads(original_final_text)
    final["claim_canonical_sha256"] = "0" * 64
    final_path.write_text(json.dumps(final, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="claim canonical hash mismatch"):
        verify_attempt(attempt)


def test_post_claim_setup_failure_is_preserved_inside_attempt(tmp_path, monkeypatch):
    repository, document, claim_path, _output_root = write_authorization(tmp_path)
    monkeypatch.setattr(
        "kvcot.discovery.diagnostic_pilot_execute._preflight",
        lambda *_args: {"passed": True, "would_initialize_cuda": False},
    )
    monkeypatch.setattr(
        "kvcot.discovery.diagnostic_pilot_execute.verify_runtime_inputs",
        lambda _path, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic runtime setup failure")
        ),
    )
    with pytest.raises(RuntimeError, match="synthetic runtime setup failure"):
        run_diagnostic_pilot(repository_root=repository, authorization_document=document)
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    attempt = Path(claim["attempt_directory"])
    completion = json.loads((attempt / "completion.json").read_text(encoding="utf-8"))
    final = json.loads((attempt / "final.json").read_text(encoding="utf-8"))
    assert completion["authorization_consumed"] is True
    assert completion["retry_allowed"] is False
    assert final["classification"] == "void"
    assert verify_attempt(attempt)["verified"] is True


@pytest.mark.parametrize(
    ("kind", "target"),
    [
        ("directory", "fullkv"),
        ("directory", "rkv"),
        ("artifact", "invocation.json"),
        ("artifact", "authorization_claim.json"),
        ("artifact", "preflight.json"),
        ("artifact", "protocol_binding.json"),
        ("artifact", "environment.json"),
    ],
)
def test_every_post_claim_setup_boundary_preserves_failure(
    tmp_path, monkeypatch, kind, target
):
    repository, document, claim_path, _output_root = write_authorization(tmp_path)
    monkeypatch.setattr(
        "kvcot.discovery.diagnostic_pilot_execute._preflight",
        lambda *_args: {"passed": True, "would_initialize_cuda": False},
    )
    monkeypatch.setattr(
        "kvcot.discovery.diagnostic_pilot_execute.verify_runtime_inputs",
        lambda _path, **_kwargs: {
            "protocol_document_sha256": "c" * 64,
            "canonical_sha256": "b" * 64,
            "candidate_prompts_path": str(tmp_path / "prompts.json"),
        },
    )
    if kind == "directory":
        original_mkdir = Path.mkdir

        def injected_mkdir(path, *args, **kwargs):
            if path.name == target:
                raise OSError(f"synthetic {target} setup failure")
            return original_mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", injected_mkdir)
    else:
        original_write = atomic_write_json
        failed = {"value": False}

        def injected_write(path, payload):
            if path.name == target and not failed["value"]:
                failed["value"] = True
                raise OSError(f"synthetic {target} setup failure")
            return original_write(path, payload)

        monkeypatch.setattr(
            "kvcot.discovery.diagnostic_pilot_execute.atomic_write_json",
            injected_write,
        )
    with pytest.raises(OSError, match="setup failure"):
        run_diagnostic_pilot(repository_root=repository, authorization_document=document)
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    attempt = Path(claim["attempt_directory"])
    assert json.loads((attempt / "completion.json").read_text())["retry_allowed"] is False
    assert json.loads((attempt / "final.json").read_text())["classification"] == "void"


def test_post_claim_attempt_creation_failure_uses_claim_directory_fallback(tmp_path, monkeypatch):
    repository, document, claim_path, _output_root = write_authorization(tmp_path)
    monkeypatch.setattr(
        "kvcot.discovery.diagnostic_pilot_execute._preflight",
        lambda *_args: {"passed": True, "would_initialize_cuda": False},
    )
    original_mkdir = Path.mkdir

    def fail_attempt_mkdir(path, *args, **kwargs):
        if path.name.startswith("diagnostic-pilot-attempt-"):
            raise OSError("synthetic attempt mkdir failure")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_attempt_mkdir)
    with pytest.raises(OSError, match="synthetic attempt mkdir failure"):
        run_diagnostic_pilot(repository_root=repository, authorization_document=document)
    fallback = claim_path.with_name(f"{claim_path.name}.failure.json")
    failure = json.loads(fallback.read_text(encoding="utf-8"))
    assert failure["authorization_consumed"] is True
    assert failure["retry_allowed"] is False
    assert failure["failure_type"] == "OSError"


def test_claim_write_failure_after_creation_is_preserved_without_retry(tmp_path, monkeypatch):
    repository, document, claim_path, _output_root = write_authorization(tmp_path)
    monkeypatch.setattr(
        "kvcot.discovery.diagnostic_pilot_execute._preflight",
        lambda *_args: {"passed": True, "would_initialize_cuda": False},
    )

    def consume_then_fail(_authorization, *, attempt_id, attempt_directory):
        claim_path.write_text("{\n", encoding="utf-8")
        raise OSError("synthetic claim write failure after consumption")

    monkeypatch.setattr(
        "kvcot.discovery.diagnostic_pilot_execute.claim_authorization_once",
        consume_then_fail,
    )
    with pytest.raises(OSError, match="after consumption"):
        run_diagnostic_pilot(repository_root=repository, authorization_document=document)
    failure_path = claim_path.with_name(f"{claim_path.name}.failure.json")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["authorization_consumed"] is True
    assert failure["retry_allowed"] is False
    assert failure["failure_type"] == "OSError"
