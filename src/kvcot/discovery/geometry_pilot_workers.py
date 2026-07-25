"""GPU worker for the 8B structured restoration geometry pilot.

Reuses this repository's existing, already-audited real-model plumbing
unchanged: `kvcot.discovery.strict_device.load_rkv_discovery_model`,
`kvcot.discovery.real_model_adapter` (`RealModelState`,
`build_real_prefill_fn`/`build_real_decode_one_fn`/`build_real_snapshot_fn`/
`build_real_branch_step_fn_restore_once`), `kvcot.discovery.pass1
.run_natural_pass1`, and `kvcot.discovery.pass2.run_pass2_capture` --
never a second, independently-written model-loading or replay path.

What is genuinely NEW here is narrow: instead of
`kvcot.discovery.orchestrator.run_example`'s fixed 4-pairs-per-event grid
(candidate x donor cross product, single (layer, head) per event), this
worker (a) reuses the EXACT already-known R2 event/layer/head/candidate/
donor identities rather than re-sampling them via `build_pass1_plan`, and
(b) after Pass-2 captures the frozen anchor event's pristine snapshot,
builds the 8 NEW structured branches (`kvcot.discovery
.geometry_pilot_manifest`/`geometry_pilot_restore`/`geometry_pilot_branches`)
instead of the original 4-pair grid.

This worker's instrumentation (timing, peak CUDA memory, CUDA-availability
guard) is intentionally lighter than `kvcot.discovery.b2a_workers
.run_rkv_worker`'s full execution-state/envelope machinery -- that depth
was built up over many independent-audit rounds specific to the ORIGINAL
grid's evidence shape; this module reuses its PROVEN lower-level primitives
rather than replicating that entire audit history for a narrower, bounded,
one-shot pilot.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from kvcot.discovery.geometry_pilot_branches import GeometryBranchResult, build_geometry_branch_record
from kvcot.discovery.geometry_pilot_contract import (
    ANCHOR_BRIDGE_TOKEN_ABSOLUTE_POSITION,
    ANCHOR_COMPACTION_EVENT_ID,
    ANCHOR_EVENT_TOKEN_ABSOLUTE_POSITION,
    ANCHOR_FIRST_SCORED_ABSOLUTE_POSITION,
    ANCHOR_KV_HEAD_INDEX,
    ANCHOR_LAYER_INDEX,
    ARM_C1,
    ARM_C2,
    ARM_H,
    ARM_HL,
    ARM_L,
    ARM_NOOP,
    ARM_S,
    ARM_SH,
    PRIMARY_DONOR_ABSOLUTE_POSITION,
    RANK_ONE_CANDIDATE_ABSOLUTE_POSITION,
    RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION,
    SCORED_HORIZON,
    SPAN_OFFSETS,
)
from kvcot.discovery.geometry_pilot_manifest import (
    freeze_layer_set,
    freeze_span_availability,
    resolve_all_kv_head_indices,
)
from kvcot.discovery.geometry_pilot_restore import KVMutationSpec, resolve_physical_position


class GeometryWorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeometryWorkerResult:
    example_id: str
    model_revision: str
    tokenizer_revision: str
    rkv_revision: str
    natural_answer: str | None
    natural_answer_status: str
    natural_generated_token_count: int
    layer_set: dict[str, Any]
    span_availability: dict[str, Any]
    branch_results: tuple[GeometryBranchResult, ...]
    peak_cuda_allocated_bytes: int | None
    peak_cuda_reserved_bytes: int | None
    wall_seconds: float


def _anchor_event_plan(*, num_hidden_layers: int):
    """Hand-construct the frozen anchor `EventPlan` from already-known R2
    evidence -- NEVER `build_pass1_plan`'s outcome-blind sampler, since the
    identity is not being newly selected here, it is being exactly
    reproduced. `cross_product` is extended (relative to the original R2
    pair set) to also cover the two span neighbors this protocol's Arm
    S/SH need; every position named here was either already causally
    tested in R2 (1159, 1168, 192) or is a structural neighbor position
    (1158, 1160) whose availability is independently checked before use
    (`geometry_pilot_manifest.freeze_span_availability`)."""
    from kvcot.discovery.pass1 import EventPlan
    from kvcot.discovery.sampling import CandidateDonorSelection

    if not (0 <= ANCHOR_LAYER_INDEX < num_hidden_layers):
        raise GeometryWorkerError(f"anchor layer {ANCHOR_LAYER_INDEX} out of range for {num_hidden_layers} layers")

    selection = CandidateDonorSelection(
        evicted_selected=(RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION, RANK_ONE_CANDIDATE_ABSOLUTE_POSITION),
        donor_selected=(PRIMARY_DONOR_ABSOLUTE_POSITION, PRIMARY_DONOR_ABSOLUTE_POSITION),
        cross_product=(
            (RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION, PRIMARY_DONOR_ABSOLUTE_POSITION),
            (RANK_ONE_CANDIDATE_ABSOLUTE_POSITION, PRIMARY_DONOR_ABSOLUTE_POSITION),
            (RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION + SPAN_OFFSETS[0], PRIMARY_DONOR_ABSOLUTE_POSITION),
            (RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION + SPAN_OFFSETS[2], PRIMARY_DONOR_ABSOLUTE_POSITION),
        ),
    )
    return EventPlan(
        compaction_event_id=ANCHOR_COMPACTION_EVENT_ID,
        absolute_event_position=ANCHOR_EVENT_TOKEN_ABSOLUTE_POSITION,
        chronological_event_ordinal=0,
        depth_stratum=1,
        layer_index=ANCHOR_LAYER_INDEX,
        kv_head_index=ANCHOR_KV_HEAD_INDEX,
        candidate_donor_selection=selection,
    )


def _resolve_candidate_vector(pristine_snapshot: Any, *, layer_index: int, kv_head_index: int, absolute_position: int):
    layer_provenance = None
    if pristine_snapshot.provenance is not None:
        layer_provenance = pristine_snapshot.provenance.layers.get(layer_index)
    if layer_provenance is None:
        return None, None
    slot = resolve_physical_position(layer_provenance.positions, kv_head_index, absolute_position)
    if slot is None:
        return None, None
    key = pristine_snapshot.key_cache[layer_index][0, kv_head_index, slot, :].detach().clone().contiguous()
    value = pristine_snapshot.value_cache[layer_index][0, kv_head_index, slot, :].detach().clone().contiguous()
    return key, value


def _mutation_for(
    pristine_snapshot: Any, *, layer_index: int, kv_head_index: int,
    candidate_absolute_position: int, donor_absolute_position: int,
) -> KVMutationSpec:
    candidate_key, candidate_value = _resolve_candidate_vector(
        pristine_snapshot, layer_index=layer_index, kv_head_index=kv_head_index,
        absolute_position=candidate_absolute_position,
    )
    if candidate_key is None:
        raise GeometryWorkerError(
            f"candidate {candidate_absolute_position} not resolvable at layer {layer_index}, head {kv_head_index}"
        )
    donor_layer_provenance = pristine_snapshot.provenance.layers.get(layer_index)
    donor_slot = resolve_physical_position(donor_layer_provenance.positions, kv_head_index, donor_absolute_position)
    if donor_slot is None:
        raise GeometryWorkerError(
            f"donor {donor_absolute_position} not resolvable at layer {layer_index}, head {kv_head_index}"
        )
    return KVMutationSpec(
        layer_index=layer_index,
        kv_head_index=kv_head_index,
        token_position=donor_slot,
        replacement_key=candidate_key,
        replacement_value=candidate_value,
        donor_absolute_position=donor_absolute_position,
        candidate_absolute_position=candidate_absolute_position,
    )


def build_frozen_branch_mutations(
    pristine_snapshot: Any,
    *,
    num_key_value_heads: int,
    layer_set: dict[str, Any],
    span_availability: dict[str, Any],
) -> dict[str, list[KVMutationSpec] | None]:
    """Build the concrete mutation list for every one of the 8 frozen arms
    from ONE already-captured pristine snapshot. Returns `None` for an arm
    whose structural prerequisite (§5) was not satisfied -- never a
    substitute geometry."""
    all_heads = resolve_all_kv_head_indices(num_key_value_heads)
    mutations: dict[str, list[KVMutationSpec] | None] = {}

    mutations[ARM_C1] = [
        _mutation_for(
            pristine_snapshot, layer_index=ANCHOR_LAYER_INDEX, kv_head_index=ANCHOR_KV_HEAD_INDEX,
            candidate_absolute_position=RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION,
            donor_absolute_position=PRIMARY_DONOR_ABSOLUTE_POSITION,
        )
    ]
    mutations[ARM_C2] = [
        _mutation_for(
            pristine_snapshot, layer_index=ANCHOR_LAYER_INDEX, kv_head_index=ANCHOR_KV_HEAD_INDEX,
            candidate_absolute_position=RANK_ONE_CANDIDATE_ABSOLUTE_POSITION,
            donor_absolute_position=PRIMARY_DONOR_ABSOLUTE_POSITION,
        )
    ]
    mutations[ARM_H] = [
        _mutation_for(
            pristine_snapshot, layer_index=ANCHOR_LAYER_INDEX, kv_head_index=head,
            candidate_absolute_position=RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION,
            donor_absolute_position=PRIMARY_DONOR_ABSOLUTE_POSITION,
        )
        for head in all_heads
    ]

    if layer_set["available"]:
        mutations[ARM_L] = [
            _mutation_for(
                pristine_snapshot, layer_index=layer, kv_head_index=ANCHOR_KV_HEAD_INDEX,
                candidate_absolute_position=RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION,
                donor_absolute_position=PRIMARY_DONOR_ABSOLUTE_POSITION,
            )
            for layer in layer_set["valid_layer_indices"]
        ]
        mutations[ARM_HL] = [
            _mutation_for(
                pristine_snapshot, layer_index=layer, kv_head_index=head,
                candidate_absolute_position=RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION,
                donor_absolute_position=PRIMARY_DONOR_ABSOLUTE_POSITION,
            )
            for layer in layer_set["valid_layer_indices"]
            for head in all_heads
        ]
    else:
        mutations[ARM_L] = None
        mutations[ARM_HL] = None

    if span_availability["available"]:
        span_positions = span_availability["span_absolute_positions"]
        mutations[ARM_S] = [
            _mutation_for(
                pristine_snapshot, layer_index=ANCHOR_LAYER_INDEX, kv_head_index=ANCHOR_KV_HEAD_INDEX,
                candidate_absolute_position=position, donor_absolute_position=position,
            )
            for position in span_positions
        ]
        mutations[ARM_SH] = [
            _mutation_for(
                pristine_snapshot, layer_index=ANCHOR_LAYER_INDEX, kv_head_index=head,
                candidate_absolute_position=position, donor_absolute_position=position,
            )
            for position in span_positions
            for head in all_heads
        ]
    else:
        mutations[ARM_S] = None
        mutations[ARM_SH] = None

    mutations[ARM_NOOP] = [
        _mutation_for(
            pristine_snapshot, layer_index=ANCHOR_LAYER_INDEX, kv_head_index=ANCHOR_KV_HEAD_INDEX,
            candidate_absolute_position=PRIMARY_DONOR_ABSOLUTE_POSITION,
            donor_absolute_position=PRIMARY_DONOR_ABSOLUTE_POSITION,
        )
    ]
    return mutations


def run_geometry_worker(
    config: Any,
    manifest: Any,
    *,
    _load_model: Callable[[], Any] | None = None,
    _load_tokenizer: Callable[[], Any] | None = None,
    _cuda: Any | None = None,
    _device: str = "cuda:0",
    _clock: Callable[[], float] | None = None,
) -> GeometryWorkerResult:
    """Real-model entry point. `_load_model`/`_load_tokenizer`/`_cuda`
    injection follows exactly `kvcot.discovery.b2a_workers.run_rkv_worker`'s
    convention (CPU tests inject fakes; production passes none of them and
    requires real CUDA)."""
    import torch

    from kvcot.discovery.framework_seed import apply_framework_seed
    from kvcot.discovery.math500_verification import build_math500_answer_fn
    from kvcot.discovery.pass1 import NaturalRunProvenance, run_natural_pass1
    from kvcot.discovery.pass2 import run_pass2_capture
    from kvcot.discovery.real_model_adapter import (
        RealModelState,
        build_real_branch_step_fn_restore_once,
        build_real_decode_one_fn,
        build_real_prefill_fn,
        build_real_snapshot_fn,
    )
    from kvcot.generation.provenance import ModelProvenance
    from kvcot.generation.replay import CompactionTracker
    from kvcot.generation.state import reset_patched_state

    cuda = _cuda if _cuda is not None else torch.cuda
    clock = _clock or time.perf_counter

    cuda_available = bool(cuda.is_available())
    if not cuda_available and _load_model is None:
        raise GeometryWorkerError("run_geometry_worker requires CUDA; none is available.")

    apply_framework_seed(config.generation.framework_seed, config.generation.attention_backend, cuda_available=cuda_available)

    started_at = clock()
    if cuda_available:
        cuda.reset_peak_memory_stats()

    if _load_tokenizer is not None:
        tokenizer = _load_tokenizer()
    else:
        from kvcot.discovery.snapshot_boundary import resolve_local_snapshot

        tokenizer_snapshot = resolve_local_snapshot(config.model.tokenizer_name, config.model.tokenizer_revision, "tokenizer")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot.local_path, local_files_only=True, use_fast=True)

    if _load_model is not None:
        model = _load_model()
    else:
        from kvcot.discovery.snapshot_boundary import resolve_local_snapshot
        from kvcot.discovery.strict_device import load_rkv_discovery_model

        model_snapshot = resolve_local_snapshot(config.model.name, config.model.revision, "model")
        tokenizer_snapshot = resolve_local_snapshot(config.model.tokenizer_name, config.model.tokenizer_revision, "tokenizer")
        model = load_rkv_discovery_model(config, model_snapshot.local_path, tokenizer_snapshot.local_path, _device)

    num_hidden_layers = len(model.model.layers) if _load_model is None else config.model.num_hidden_layers
    num_key_value_heads = config.model.num_key_value_heads

    def fresh_state() -> RealModelState:
        cache = reset_patched_state(model)
        return RealModelState(
            model=model, cache=cache, model_provenance=ModelProvenance(),
            compaction=CompactionTracker(), absolute_position=0, device=_device,
        )

    prefill_fn = build_real_prefill_fn(_device)
    decode_one_fn = build_real_decode_one_fn(_device)
    snapshot_fn = build_real_snapshot_fn()
    branch_step_fn = build_real_branch_step_fn_restore_once(_device)

    provenance = NaturalRunProvenance(
        model_name=config.model.name, model_revision=config.model.revision,
        tokenizer_name=config.model.tokenizer_name, tokenizer_revision=config.model.tokenizer_revision,
        rkv_revision=config.rkv.upstream_revision, config_sha256=manifest.config_sha256 if hasattr(manifest, "config_sha256") else "",
        dataset_name=config.dataset.name, example_id=manifest.unique_id,
    )
    answer_fn = build_math500_answer_fn(manifest.gold_answer)

    trace = run_natural_pass1(
        provenance, manifest.prompt_token_ids, fresh_state(), prefill_fn, decode_one_fn,
        config.generation.base_max_new_tokens, tokenizer.eos_token_id, answer_fn,
    )
    if trace.natural_answer_status != "correct":
        raise GeometryWorkerError(f"natural run answer status is {trace.natural_answer_status!r}, expected 'correct'")

    event_plan = _anchor_event_plan(num_hidden_layers=num_hidden_layers)

    class _FrozenPlan:
        def __init__(self, trace_, events_):
            self.trace = trace_
            self.events = events_

    pass1_plan = _FrozenPlan(trace, (event_plan, event_plan, event_plan))

    pass2_result = run_pass2_capture(
        pass1_plan, trace.full_token_ids, fresh_state(), prefill_fn, decode_one_fn, snapshot_fn,
    )
    if not pass2_result.valid:
        raise GeometryWorkerError(f"pass-2 capture invalid: {pass2_result.invalid_reason}")

    target_capture = pass2_result.target_captures[0]
    pristine_snapshot = target_capture.pristine_snapshot

    layer_set = freeze_layer_set(pristine_snapshot, candidate_layers=range(num_hidden_layers))
    span_availability = freeze_span_availability(
        pristine_snapshot, prompt_length=trace.prompt_length, total_length=len(trace.full_token_ids),
    )

    bridge_pos = ANCHOR_BRIDGE_TOKEN_ABSOLUTE_POSITION
    first_scored = ANCHOR_FIRST_SCORED_ABSOLUTE_POSITION
    bridge_token_id = trace.full_token_ids[bridge_pos]
    reference_token_ids = list(trace.full_token_ids[first_scored : first_scored + SCORED_HORIZON])
    if len(reference_token_ids) != SCORED_HORIZON:
        raise GeometryWorkerError("insufficient future tokens for the scored horizon")

    mutations_by_arm = build_frozen_branch_mutations(
        pristine_snapshot, num_key_value_heads=num_key_value_heads,
        layer_set=layer_set, span_availability=span_availability,
    )

    branch_results = []
    for arm, mutations in mutations_by_arm.items():
        if mutations is None:
            continue
        branch_results.append(
            build_geometry_branch_record(
                arm=arm, pristine_snapshot=pristine_snapshot, mutations=mutations,
                bridge_token_id=bridge_token_id, reference_token_ids=reference_token_ids,
                branch_step_fn=branch_step_fn,
            )
        )

    peak_allocated = int(cuda.max_memory_allocated()) if cuda_available else None
    peak_reserved = int(cuda.max_memory_reserved()) if cuda_available else None
    wall_seconds = clock() - started_at

    return GeometryWorkerResult(
        example_id=manifest.unique_id,
        model_revision=config.model.revision,
        tokenizer_revision=config.model.tokenizer_revision,
        rkv_revision=config.rkv.upstream_revision,
        natural_answer=trace.natural_answer,
        natural_answer_status=trace.natural_answer_status,
        natural_generated_token_count=len(trace.generated_token_ids),
        layer_set=layer_set,
        span_availability=span_availability,
        branch_results=tuple(branch_results),
        peak_cuda_allocated_bytes=peak_allocated,
        peak_cuda_reserved_bytes=peak_reserved,
        wall_seconds=wall_seconds,
    )
