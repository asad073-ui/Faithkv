import hashlib

import pytest
import torch

from kvcot.discovery.geometry_pilot_branches import (
    build_geometry_branch_record,
    first_token_delta,
    peak_absolute_per_token_delta,
    subwindow_gains,
)
from kvcot.discovery.geometry_pilot_restore import KVMutationSpec
from kvcot.generation.provenance import LayerProvenance, ModelProvenance
from kvcot.generation.state import ModelStateSnapshot

NUM_LAYERS = 2
NUM_KV_HEADS = 2
SEQ_LEN = 4
HEAD_DIM = 3
HORIZON = 48


def _seeded(*parts) -> torch.Generator:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") % (2**63))


def _content_fingerprint_step_fn(vocab_size: int = 64):
    """Deterministic, content-seeded step function mirroring
    tests/unit/discovery/_synthetic_harness.py's `branch_step_fn`: identical
    cache content -> identical logits; any real content change anywhere in
    the cache changes the seed and therefore the resulting NLL."""

    def step_fn(snapshot, token_id):
        fingerprint = int(
            hashlib.sha256(
                b"".join(t.detach().contiguous().numpy().tobytes() for t in list(snapshot.key_cache) + list(snapshot.value_cache))
            ).hexdigest()[:8],
            16,
        )
        logits = torch.randn(vocab_size, generator=_seeded("branch", fingerprint, token_id))
        return logits, snapshot

    return step_fn


def _snapshot() -> ModelStateSnapshot:
    keys = [
        torch.arange(NUM_KV_HEADS * SEQ_LEN * HEAD_DIM, dtype=torch.float32).reshape(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        + 1000 * layer
        for layer in range(NUM_LAYERS)
    ]
    values = [tensor.clone() + 0.5 for tensor in keys]
    layers = {}
    for layer in range(NUM_LAYERS):
        positions = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0).expand(NUM_KV_HEADS, -1).clone()
        layers[layer] = LayerProvenance(positions=positions)
    return ModelStateSnapshot(
        key_cache=keys,
        value_cache=values,
        query_cache={},
        compression_flags_per_layer=["none"] * NUM_LAYERS,
        model_length=SEQ_LEN,
        after_think=None,
        absolute_position=SEQ_LEN,
        provenance=ModelProvenance(layers=layers),
        kv_cluster_bookkeeping_per_layer=None,
    )


def _reference_tokens():
    return list(range(HORIZON))


def test_real_mutation_produces_nonzero_gain_and_matches_manual_nll():
    snapshot = _snapshot()
    mutation = KVMutationSpec(
        layer_index=0, kv_head_index=0, token_position=1,
        replacement_key=torch.full((HEAD_DIM,), 555.0),
        replacement_value=torch.full((HEAD_DIM,), -555.0),
        donor_absolute_position=1, candidate_absolute_position=99,
    )
    result = build_geometry_branch_record(
        arm="candidate_rank0",
        pristine_snapshot=snapshot,
        mutations=[mutation],
        bridge_token_id=0,
        reference_token_ids=_reference_tokens(),
        branch_step_fn=_content_fingerprint_step_fn(),
        scored_horizon=HORIZON,
    )
    assert not result.is_noop
    assert result.key_slots_changed == 1
    assert result.swap_gain == pytest.approx(result.baseline_mean_nll - result.swapped_mean_nll)
    # content actually changed -> content-seeded step_fn must diverge somewhere
    assert result.baseline_per_token_nll != result.swapped_per_token_nll


def test_noop_mutation_is_zero_gain_and_identical_nll_arrays():
    snapshot = _snapshot()
    key = snapshot.key_cache[0][0, 0, 2].clone().contiguous()
    value = snapshot.value_cache[0][0, 0, 2].clone().contiguous()
    mutation = KVMutationSpec(
        layer_index=0, kv_head_index=0, token_position=2,
        replacement_key=key, replacement_value=value,
        donor_absolute_position=2, candidate_absolute_position=2,
    )
    result = build_geometry_branch_record(
        arm="no_op",
        pristine_snapshot=snapshot,
        mutations=[mutation],
        bridge_token_id=0,
        reference_token_ids=_reference_tokens(),
        branch_step_fn=_content_fingerprint_step_fn(),
        scored_horizon=HORIZON,
    )
    assert result.is_noop
    assert result.swap_gain == 0.0
    assert result.baseline_per_token_nll == result.swapped_per_token_nll


def test_wrong_horizon_length_rejected():
    snapshot = _snapshot()
    mutation = KVMutationSpec(
        layer_index=0, kv_head_index=0, token_position=0,
        replacement_key=torch.zeros(HEAD_DIM), replacement_value=torch.zeros(HEAD_DIM),
        donor_absolute_position=0, candidate_absolute_position=0,
    )
    with pytest.raises(ValueError, match="entries"):
        build_geometry_branch_record(
            arm="no_op",
            pristine_snapshot=snapshot,
            mutations=[mutation],
            bridge_token_id=0,
            reference_token_ids=[0, 1, 2],
            branch_step_fn=_content_fingerprint_step_fn(),
            scored_horizon=HORIZON,
        )


def test_pristine_snapshot_untouched_by_either_branch():
    snapshot = _snapshot()
    before = snapshot.clone()
    mutation = KVMutationSpec(
        layer_index=1, kv_head_index=1, token_position=3,
        replacement_key=torch.full((HEAD_DIM,), -1.0),
        replacement_value=torch.full((HEAD_DIM,), -2.0),
        donor_absolute_position=3, candidate_absolute_position=101,
    )
    build_geometry_branch_record(
        arm="all_kv_heads_selected_layer",
        pristine_snapshot=snapshot,
        mutations=[mutation],
        bridge_token_id=0,
        reference_token_ids=_reference_tokens(),
        branch_step_fn=_content_fingerprint_step_fn(),
        scored_horizon=HORIZON,
    )
    for layer in range(NUM_LAYERS):
        assert torch.equal(snapshot.key_cache[layer], before.key_cache[layer])
        assert torch.equal(snapshot.value_cache[layer], before.value_cache[layer])


# --- reconstruction helpers ---

def test_peak_absolute_delta_and_first_token_delta_reconstruct():
    baseline = [1.0, 2.0, 0.5, 9.0]
    swapped = [1.0, 1.0, 0.5, 0.0]
    delta, index = peak_absolute_per_token_delta(baseline, swapped)
    assert index == 3
    assert delta == pytest.approx(9.0)
    assert first_token_delta(baseline, swapped) == pytest.approx(0.0)


def test_subwindow_gains_reconstruct_from_arrays():
    baseline = [1.0, 1.0, 1.0, 1.0]
    swapped = [0.0, 0.0, 0.0, 0.0]
    gains = subwindow_gains(baseline, swapped, window=2)
    assert gains == [pytest.approx(1.0), pytest.approx(1.0), pytest.approx(1.0)]


def test_subwindow_window_larger_than_horizon_rejected():
    with pytest.raises(ValueError):
        subwindow_gains([1.0, 2.0], [1.0, 2.0], window=8)
