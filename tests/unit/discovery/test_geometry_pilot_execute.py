import json
import sys

import pytest

from kvcot.discovery.geometry_pilot_authorization import (
    GeometryAuthorizationAlreadyConsumed,
    GeometryAuthorizationDocumentBinding,
    sha256_file,
)
from kvcot.discovery.geometry_pilot_branches import GeometryBranchResult
from kvcot.discovery.geometry_pilot_contract import (
    ARM_C1,
    ARM_C2,
    ARM_H,
    ARM_HL,
    ARM_L,
    ARM_NOOP,
    ARM_S,
    ARM_SH,
    GeometryPilotClassification,
)
from kvcot.discovery.geometry_pilot_execute import (
    branch_readout,
    build_scientific_summary,
    run_dry_run,
    run_execute,
)


def _branch(arm, gain, is_noop=False):
    n = 48
    baseline = [0.5] * n
    swapped = [0.5 - gain] * n
    return GeometryBranchResult(
        arm=arm, mutations=(), baseline_per_token_nll=tuple(baseline), swapped_per_token_nll=tuple(swapped),
        baseline_mean_nll=sum(baseline) / n, swapped_mean_nll=sum(swapped) / n, swap_gain=gain,
        is_noop=is_noop, key_slots_changed=0 if is_noop else 1, value_slots_changed=0 if is_noop else 1,
        cache_shape_unchanged=True, provenance_updated_count=0, kept_index_updated_count=0,
    )


def test_branch_readout_reconstructs_every_primitive_field():
    result = _branch(ARM_C1, 0.02)
    readout = branch_readout(result)
    for key in (
        "swap_gain", "baseline_mean_nll", "swapped_mean_nll", "peak_absolute_per_token_nll_change",
        "first_token_nll_change", "subwindow_gains_w8", "subwindow_gains_w16",
    ):
        assert key in readout
    assert readout["swap_gain"] == pytest.approx(0.02)
    assert len(readout["subwindow_gains_w8"]) == 48 - 8 + 1
    assert len(readout["subwindow_gains_w16"]) == 48 - 16 + 1


def test_scientific_summary_classification_h_when_all_flat_and_noop_exact():
    readouts = {
        ARM_C1: branch_readout(_branch(ARM_C1, 0.001)),
        ARM_C2: branch_readout(_branch(ARM_C2, 0.001)),
        ARM_H: branch_readout(_branch(ARM_H, 0.001)),
        ARM_L: branch_readout(_branch(ARM_L, 0.001)),
        ARM_HL: branch_readout(_branch(ARM_HL, 0.001)),
        ARM_S: branch_readout(_branch(ARM_S, 0.001)),
        ARM_SH: branch_readout(_branch(ARM_SH, 0.001)),
        ARM_NOOP: branch_readout(_branch(ARM_NOOP, 0.0, is_noop=True)),
    }
    summary = build_scientific_summary(readouts)
    assert summary["classification"] == GeometryPilotClassification.STRUCTURED_RESTORATION_FLAT.value
    assert summary["classification_letter"] == "H"
    assert summary["noop_exact"] is True


def test_scientific_summary_classification_a_when_rank_zero_moves():
    readouts = {
        ARM_C1: branch_readout(_branch(ARM_C1, 0.5)),
        ARM_C2: branch_readout(_branch(ARM_C2, 0.001)),
        ARM_H: branch_readout(_branch(ARM_H, 0.001)),
        ARM_L: None,
        ARM_HL: None,
        ARM_S: None,
        ARM_SH: None,
        ARM_NOOP: branch_readout(_branch(ARM_NOOP, 0.0, is_noop=True)),
    }
    summary = build_scientific_summary(readouts)
    assert summary["classification_letter"] == "A"


def test_scientific_summary_classification_i_when_noop_not_exact():
    readouts = {
        ARM_C1: branch_readout(_branch(ARM_C1, 0.001)),
        ARM_C2: branch_readout(_branch(ARM_C2, 0.001)),
        ARM_H: branch_readout(_branch(ARM_H, 0.001)),
        ARM_L: None, ARM_HL: None, ARM_S: None, ARM_SH: None,
        ARM_NOOP: branch_readout(_branch(ARM_NOOP, 0.002, is_noop=False)),
    }
    summary = build_scientific_summary(readouts)
    assert summary["classification_letter"] == "I"


def test_scientific_summary_missing_branch_not_counted_as_flat_zero():
    # If C2/H/no-op are missing entirely (never executed), evidence is
    # incomplete -> mechanically invalid, never silently treated as flat.
    readouts = {ARM_C1: branch_readout(_branch(ARM_C1, 0.001))}
    for arm in (ARM_C2, ARM_H, ARM_L, ARM_HL, ARM_S, ARM_SH, ARM_NOOP):
        readouts[arm] = None
    summary = build_scientific_summary(readouts)
    assert summary["classification_letter"] == "I"


# --- dry run / execute ---

def _binding(doc_path, doc_hash):
    return GeometryAuthorizationDocumentBinding(
        authorization_id="geometry-pilot-2026-07-25",
        authorization_document_path=doc_path,
        authorization_document_sha256=doc_hash,
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


def test_run_dry_run_never_touches_cuda_or_torch(tmp_path, monkeypatch):
    doc = tmp_path / "docs" / "AUTH.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("frozen\n")
    binding = _binding("docs/AUTH.md", sha256_file(doc))

    # Guarantee no torch/transformers module is imported as a side effect.
    before = set(sys.modules)
    result = run_dry_run(repository_root=tmp_path, binding=binding)
    after = set(sys.modules)
    assert not (after - before) & {"torch", "transformers"}
    assert result["cuda_initialized"] is False
    assert result["model_loaded"] is False
    assert result["claim_absent"] is True


def test_run_execute_creates_claim_and_attempt_dir_then_refuses_retry(tmp_path, monkeypatch):
    doc = tmp_path / "docs" / "AUTH.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("frozen\n")
    binding = _binding("docs/AUTH.md", sha256_file(doc))

    class FakeManifest:
        unique_id = "test/fake/1.json"

    class FakeWorkerResult:
        example_id = "test/fake/1.json"
        natural_answer = "36"
        natural_answer_status = "correct"
        natural_generated_token_count = 10
        layer_set = {"available": False, "unavailable_reason": "fewer_than_two_valid_layers", "valid_layer_indices": []}
        span_availability = {"available": False, "unavailable_reason": "position_x_outside_eligible_region"}
        branch_results = (_branch(ARM_C1, 0.001), _branch(ARM_NOOP, 0.0, is_noop=True))
        peak_cuda_allocated_bytes = 123
        peak_cuda_reserved_bytes = 456
        wall_seconds = 1.23

    import kvcot.discovery.geometry_pilot_workers as workers_mod

    monkeypatch.setattr(workers_mod, "run_geometry_worker", lambda config, manifest: FakeWorkerResult())

    result = run_execute(
        repository_root=tmp_path, binding=binding, config=object(), manifest=FakeManifest(),
        authorized_repository="asad073-ui/Faithkv",
        authorized_branch="research/8b-structured-restoration-geometry",
        observed_execution_commit_sha="1" * 40,
    )
    attempt_dir = tmp_path / "results" / "decisions" / _basename(result["attempt_directory"])
    assert (attempt_dir / "scientific_summary.json").is_file()
    assert (attempt_dir / "layer_set.json").is_file()
    assert (attempt_dir / "span_availability.json").is_file()

    with pytest.raises(GeometryAuthorizationAlreadyConsumed):
        run_execute(
            repository_root=tmp_path, binding=binding, config=object(), manifest=FakeManifest(),
            authorized_repository="asad073-ui/Faithkv",
            authorized_branch="research/8b-structured-restoration-geometry",
            observed_execution_commit_sha="1" * 40,
        )


def _basename(path_str):
    from pathlib import Path

    return Path(path_str).name
