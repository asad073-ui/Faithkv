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
    MODEL_REVISION,
    NOOP_ARM,
    PAIR_SCHEMA_VERSION,
    R1_GENERATION,
    R2_GENERATION,
    REPOSITORY,
    RESTORE_ARM,
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


def pair(arm, gain, *, width=1, rank=None, margin=False, candidate=1, donor=9):
    baseline = [1.0] * 48
    intervention = [1.0 - gain * 48] + [1.0] * 47
    baseline_mean = sum(baseline) / len(baseline)
    intervention_mean = sum(intervention) / len(intervention)
    before_margin = -0.5 if margin else 0.5
    after_margin = 0.5
    kv_heads = [0, 1] if width == 2 else [0]
    donor_slots = {"0": 3, **({"1": 4} if width == 2 else {})}

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
        "diagnostic_only": arm == RESTORE_ARM,
        "deployable_performance": False,
        "restore_width": width,
        "candidate_pool_rank": rank,
        "is_rank_zero_candidate": rank == 0,
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
            "key_slots_changed": 0 if arm == NOOP_ARM else len(kv_heads),
            "value_slots_changed": 0 if arm == NOOP_ARM else len(kv_heads),
            "is_noop": arm == NOOP_ARM,
        },
    }


#: Two-candidate frozen pool used by every synthetic example.
POOL = [
    {"absolute_token_position": 1, "deployable_score": 0.9},
    {"absolute_token_position": 2, "deployable_score": 0.8},
]


def grid_pairs(gains, *, margin=False, donor=9):
    """The complete rank-major candidate x width grid, then one no-op."""
    records = []
    for rank, candidate in enumerate(POOL):
        for width in (1, 2):
            records.append(
                pair(
                    RESTORE_ARM,
                    gains.get((rank, width), 0.0),
                    width=width,
                    rank=rank,
                    margin=margin and rank == 0 and width == 1,
                    candidate=candidate["absolute_token_position"],
                    donor=donor,
                )
            )
    records.append(pair(NOOP_ARM, 0.0, width=1, rank=None, candidate=donor, donor=donor))
    return records


def result(ordinal, *, gains=None, margin=False, noop=True):
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
            "candidate_pool": [dict(row) for row in POOL],
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
                "candidate_pool": [dict(row) for row in POOL],
                "captured_before_intervention": True,
                "rank_zero_candidate_available_in_all_kv_heads": True,
            }
        ],
        "score_replay_retained_full_snapshots": 0,
        "selected_snapshot_count": 1,
        "pair_records": grid_pairs(gains or {}, margin=margin),
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
    return payload


@pytest.mark.parametrize(
    ("gains", "margin", "classification"),
    [
        # A: the rank-zero width-one (current deployable) cell moved.
        ({(0, 1): 0.02}, False, "current_candidate_works"),
        # B: only a non-rank-zero width-one candidate moved.
        ({(1, 1): 0.02}, False, "candidate_selection_rescue"),
        # C: same candidate, width two moved where width one did not.
        ({(0, 2): 0.02}, False, "restore_width_rescue"),
        # D: only a non-rank-zero width-two cell moved.
        ({(1, 2): 0.02}, False, "candidate_width_interaction"),
        # E: flat NLL, moving behavioural readout.
        ({}, True, "readout_only_movement"),
        # F: flat bounded diagnostic.
        ({}, False, "flat_bounded_diagnostic"),
    ],
)
def test_summary_reconstructs_every_scientific_outcome(gains, margin, classification):
    qualifications = [qualification(index) for index in range(3)]
    results = [result(index, gains=gains, margin=margin) for index in range(3)]
    summary = _summarize(qualifications, results)
    assert summary["classification"] == classification
    # Two pooled candidates x two widths x three examples, plus one no-op each.
    assert summary["completed_width_one_pairs"] == 6
    assert summary["completed_width_two_pairs"] == 6
    assert summary["completed_restore_pairs"] == 12
    assert summary["completed_noop_pairs"] == 3
    assert summary["completed_total_pairs"] == summary["expected_total_pairs"] == 15
    assert summary["maximum_pairs_per_selected_example"] == 9
    assert summary["maximum_total_pairs"] == 27
    assert summary["swap_gain_threshold_nats"] == 0.01
    assert summary["b2b_status"] == "blocked"
    assert "does not settle the 8B operating point" in summary["scale_limitation"]


def test_summary_reports_the_complete_per_example_gain_matrix():
    gains = {(0, 1): 0.05, (0, 2): 0.07, (1, 1): -0.01, (1, 2): 0.02}
    results = [result(index, gains=gains) for index in range(3)]
    summary = _summarize([qualification(index) for index in range(3)], results)
    matrices = summary["per_example_gain_matrix"]
    assert len(matrices) == 3
    rows = matrices[0]["rows"]
    assert [row["candidate_pool_rank"] for row in rows] == [0, 1]
    assert [row["is_rank_zero_candidate"] for row in rows] == [True, False]
    assert [row["candidate_absolute_position"] for row in rows] == [1, 2]
    assert rows[0]["width_1_gain"] == pytest.approx(0.05)
    assert rows[0]["width_2_gain"] == pytest.approx(0.07)
    assert rows[0]["width_increment"] == pytest.approx(0.02)
    assert rows[1]["width_increment"] == pytest.approx(0.03)
    assert summary["best_rank_zero_width_one_gain"] == pytest.approx(0.05)
    assert summary["best_rank_zero_width_two_gain"] == pytest.approx(0.07)
    assert summary["best_non_rank_zero_width_one_gain"] == pytest.approx(-0.01)
    assert summary["best_non_rank_zero_width_two_gain"] == pytest.approx(0.02)
    assert summary["largest_gain_over_bounded_grid"] == pytest.approx(0.07)
    assert summary["rank_zero_width_one_gains_above_0_01"] == 3
    assert summary["rank_zero_width_two_gains_above_0_01"] == 3
    assert summary["non_rank_zero_width_one_gains_above_0_01"] == 0
    assert summary["non_rank_zero_width_two_gains_above_0_01"] == 3
    assert summary["total_gains_above_0_01"] == 9
    assert summary["noop_maximum_absolute_difference"] == 0.0
    assert summary["classification"] == "current_candidate_works"
    assert summary["classification_letter"] == "A"


def test_noop_mismatch_voids_result():
    summary = _summarize(
        [qualification(index) for index in range(3)],
        [result(0), result(1, noop=False), result(2)],
    )
    assert summary["classification"] == "void"


def test_threshold_counts_are_over_primitive_cells_not_only_bounded_maxima():
    summary = _summarize(
        [qualification(index) for index in range(3)],
        [result(index, gains={(0, 1): 0.03, (1, 1): 0.02}) for index in range(3)],
    )
    # Six primitive width-one cells cross the threshold; the bounded local
    # candidate maximum is one number per example.
    assert summary["rank_zero_width_one_gains_above_0_01"] == 3
    assert summary["non_rank_zero_width_one_gains_above_0_01"] == 3
    assert summary["bounded_local_candidate_maxima_above_0_01"] == 3
    assert len(summary["bounded_local_candidate_maxima"]) == 3


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
    assert summary["classification"] == "flat_bounded_diagnostic"
    assert summary["classification_letter"] == "F"


def zero_event_result(ordinal):
    """A structurally valid worker record for a legitimate zero-event row."""
    payload = result(ordinal)
    payload["selected_event"] = None
    payload["eligible_event_score_evidence"] = []
    payload["selected_snapshot_count"] = 0
    payload["pair_records"] = []
    payload["noop_exact"] = None
    payload["qualified"] = False
    payload["qualification"] = zero_event_qualification(ordinal)
    return payload


def cap_hit_result(ordinal):
    """R1's candidate 0 shape: eligible events existed but the cap was hit."""
    payload = result(ordinal)
    payload["rkv_natural_cap_hit"] = True
    payload["pair_records"] = []
    payload["noop_exact"] = None
    payload["qualified"] = False
    row = qualification(ordinal)
    row["rkv_replay_mechanically_valid"] = False
    row["rkv_natural_cap_hit"] = True
    payload["qualification"] = row
    return payload


def zero_event_qualification(ordinal):
    row = qualification(ordinal)
    row.update(
        {
            "eligible_event_exists": False,
            "selected_event_has_two_candidates": False,
            "selected_event_count": 0,
            "eligible_scored_event_count": 0,
        }
    )
    return row


def test_raw_binding_accepts_a_valid_unqualified_zero_event_worker_record():
    rkv = zero_event_result(0)
    fullkv = fullkv_result(0)
    assert _raw_worker_binding_valid(fullkv, rkv) is True
    # A zero-event record that nevertheless carries interventions is not a
    # clean non-qualifier and must fail the binding.
    rkv["pair_records"] = [pair(RESTORE_ARM, 0.0, width=1, rank=0)]
    assert _raw_worker_binding_valid(fullkv, rkv) is False


def fullkv_result(ordinal):
    return {
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


def write_authorization(tmp_path, *, generation=R2_GENERATION, **overrides):
    """A synthetic authorization living at its generation's frozen path."""
    repository = tmp_path / "repository"
    document = repository / generation.authorization_document_path
    document.parent.mkdir(parents=True, exist_ok=True)
    claim = tmp_path / "claim.json"
    output_root = tmp_path / "execution"
    audit = tmp_path / "implementation-audit.md"
    audit.write_text("PASS synthetic audit\n", encoding="utf-8")
    source_root = Path(__file__).resolve().parents[3]
    for relative_path in (
        generation.protocol_document_path,
        CONFIG_PATH,
        CANDIDATE_MANIFEST_PATH,
    ):
        source = source_root / relative_path
        destination = repository / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    protocol_sha256 = sha256_file(repository / generation.protocol_document_path)
    pair_budget = (
        {"maximum_interventions_per_selected_example": 6}
        if generation is R1_GENERATION
        else {
            "maximum_pairs_per_selected_example": 9,
            "maximum_total_pairs": 27,
        }
    )
    payload = {
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
        **pair_budget,
        "kv_restore_widths": [1, 2],
        "single_rtx3090": True,
        "cpu_offload": False,
        "vram_limit_bytes": 22 * 1024**3,
        "runtime_limit_seconds": 5_400,
        "exact_command": generation.exact_execution_command,
    }
    payload.update(overrides)
    document.write_text(
        f"{AUTHORIZATION_JSON_BEGIN}\n{json.dumps(attach_canonical_hash(payload))}\n"
        f"{AUTHORIZATION_JSON_END}\n",
        encoding="utf-8",
    )
    return repository, document, claim, output_root


def test_dry_run_is_non_consuming_and_requests_no_cuda_or_weights(tmp_path, monkeypatch):
    repository, document, claim, output_root = write_authorization(tmp_path)
    monkeypatch.setattr(
        "kvcot.discovery.diagnostic_pilot_execute.verify_runtime_inputs",
        lambda _path, **_kwargs: {
            "canonical_sha256": "b" * 64,
            "protocol_document_path": R2_GENERATION.protocol_document_path,
            "protocol_document_sha256": sha256_file(
                repository / R2_GENERATION.protocol_document_path
            ),
            "implementation_sha": "a" * 40,
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
            return R2_GENERATION.authorization_document_path
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
            "protocol_document_path": R2_GENERATION.protocol_document_path,
            "protocol_document_sha256": sha256_file(
                repository / R2_GENERATION.protocol_document_path
            ),
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
    summary = json.loads((attempt / "scientific_summary.json").read_text())
    assert summary["classification"] == "flat_bounded_diagnostic"
    assert summary["completed_width_one_pairs"] == 6
    assert summary["completed_width_two_pairs"] == 6
    assert summary["completed_noop_pairs"] == 3
    binding = json.loads((attempt / "protocol_binding.json").read_text())
    assert binding["generation"] == "r2"
    assert binding["maximum_pairs_per_selected_example"] == 9
    assert binding["maximum_total_pairs"] == 27
    invocation = json.loads((attempt / "invocation.json").read_text())
    assert invocation["command"] == R2_GENERATION.exact_execution_command
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


def coordinator_harness(tmp_path, monkeypatch, repository, rkv_for_ordinal):
    """Wire a fake-worker coordinator run over the first-eight scan."""
    monkeypatch.setattr(
        "kvcot.discovery.diagnostic_pilot_execute._preflight",
        lambda *_args: {"passed": True, "would_initialize_cuda": False},
    )
    monkeypatch.setattr(
        "kvcot.discovery.diagnostic_pilot_execute.verify_runtime_inputs",
        lambda _path, **_kwargs: {
            "protocol_document_path": R2_GENERATION.protocol_document_path,
            "protocol_document_sha256": sha256_file(
                repository / R2_GENERATION.protocol_document_path
            ),
            "canonical_sha256": "b" * 64,
            "candidate_prompts_path": str(tmp_path / "prompts.json"),
        },
    )
    launched = []

    def fake_launch(*, role, ordinal, output, **_kwargs):
        launched.append((role, ordinal))
        if role == "fullkv":
            payload = fullkv_result(ordinal)
        else:
            payload = rkv_for_ordinal(ordinal)
            payload.setdefault("qualification", qualification(ordinal))
            payload.update(
                {
                    "fixed_trace_extracted_answer": "1",
                    "fixed_trace_correctness_status": "correct",
                    "peak_cuda_allocated_bytes": 10,
                    "peak_cuda_reserved_bytes": 20,
                }
            )
        atomic_write_json(output, payload)
        return payload

    monkeypatch.setattr("kvcot.discovery.diagnostic_pilot_execute._launch_worker", fake_launch)
    return launched


def test_coordinator_continues_past_a_zero_event_row_and_selects_later_candidates(
    tmp_path, monkeypatch
):
    """The exact R1 production failure, end to end.

    R1 raised `qualification requires exactly one selected event` on
    candidate 1 and never reached candidate 2.
    """
    repository, document, _claim, _output_root = write_authorization(tmp_path)

    def rkv_for_ordinal(ordinal):
        if ordinal == 0:
            return cap_hit_result(ordinal)
        if ordinal == 1:
            return zero_event_result(ordinal)
        return result(ordinal)

    launched = coordinator_harness(tmp_path, monkeypatch, repository, rkv_for_ordinal)
    report = run_diagnostic_pilot(
        repository_root=repository, authorization_document=document
    )
    attempt = Path(report["attempt_directory"])
    summary = json.loads((attempt / "scientific_summary.json").read_text())
    assert summary["selected_candidate_ordinals"] == [2, 3, 4]
    assert summary["qualified_example_count"] == 3
    assert summary["classification"] == "flat_bounded_diagnostic"
    # The scan visited candidates 0 through 4 and stopped once three
    # examples qualified -- it never aborted on the zero-event row.
    assert [ordinal for role, ordinal in launched if role == "fullkv"] == [0, 1, 2, 3, 4]
    assert verify_attempt(attempt)["verified"] is True


def test_eight_zero_event_rows_write_a_mechanically_unqualified_result(
    tmp_path, monkeypatch
):
    repository, document, _claim, _output_root = write_authorization(tmp_path)
    launched = coordinator_harness(tmp_path, monkeypatch, repository, zero_event_result)
    report = run_diagnostic_pilot(
        repository_root=repository, authorization_document=document
    )
    attempt = Path(report["attempt_directory"])
    # A valid mechanically-unqualified result, not a failure artifact.
    assert not (attempt / "failure.json").exists()
    summary = json.loads((attempt / "scientific_summary.json").read_text())
    assert summary["classification"] == "mechanically_unqualified"
    assert summary["classification_letter"] == "G"
    assert summary["qualified_example_count"] == 0
    assert summary["completed_total_pairs"] == 0
    assert "NO SCIENTIFIC INTERPRETATION" in summary["classification_text"]
    assert [ordinal for role, ordinal in launched if role == "fullkv"] == list(range(8))
    assert json.loads((attempt / "completion.json").read_text())["completed"] is True
    assert verify_attempt(attempt)["verified"] is True


def test_fewer_than_three_qualified_rows_are_mechanically_unqualified_not_a_failure(
    tmp_path, monkeypatch
):
    repository, document, _claim, _output_root = write_authorization(tmp_path)

    def rkv_for_ordinal(ordinal):
        return result(ordinal) if ordinal < 2 else zero_event_result(ordinal)

    coordinator_harness(tmp_path, monkeypatch, repository, rkv_for_ordinal)
    report = run_diagnostic_pilot(
        repository_root=repository, authorization_document=document
    )
    attempt = Path(report["attempt_directory"])
    assert not (attempt / "failure.json").exists()
    summary = json.loads((attempt / "scientific_summary.json").read_text())
    assert summary["classification"] == "mechanically_unqualified"
    assert summary["qualified_example_count"] == 2
    assert verify_attempt(attempt)["verified"] is True


def test_consumed_r1_authorization_can_never_be_executed_again(tmp_path, monkeypatch):
    repository, document, claim, _output_root = write_authorization(
        tmp_path, generation=R1_GENERATION
    )
    monkeypatch.setattr(
        "kvcot.discovery.diagnostic_pilot_execute.verify_runtime_inputs",
        lambda _path, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must refuse before touching runtime inputs")
        ),
    )
    with pytest.raises(DiagnosticExecutionRefused, match="not executable by this"):
        dry_run_diagnostic_pilot(
            repository_root=repository, authorization_document=document
        )
    with pytest.raises(DiagnosticExecutionRefused, match="not executable by this"):
        run_diagnostic_pilot(
            repository_root=repository, authorization_document=document
        )
    assert not claim.exists()


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
            "protocol_document_path": R2_GENERATION.protocol_document_path,
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
