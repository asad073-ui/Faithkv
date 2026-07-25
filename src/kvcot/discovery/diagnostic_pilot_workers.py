"""Production workers for the post-Stage-C diagnostic pilot.

Torch, Transformers, and R-KV imports remain inside worker bodies.  The
coordinator launches FullKV and diagnostic R-KV in separate OS processes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import gc
import hashlib
import time
from typing import Any

from kvcot.discovery.diagnostic_pilot_contract import (
    EXPECTED_KV_HEADS,
    EXPECTED_QUERY_HEADS,
    MODEL_REVISION,
    NOOP_ARM,
    PAIR_SCHEMA_VERSION,
    RESTORE_ARM,
    RESTORE_WIDTHS,
    RKV_REVISION,
    SCORED_HORIZON,
    TOKENIZER_REVISION,
    VRAM_LIMIT_BYTES,
    ModelArchitecture,
    material_margin_change,
    resolve_restore_heads,
)
from kvcot.discovery.diagnostic_pilot_manifest import (
    CandidateScore,
    FrozenCandidatePool,
    freeze_candidate_pool,
)


class DiagnosticWorkerRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class DiagnosticCapturePlan:
    """Minimal structural input consumed by ``run_pass2_capture``.

    The diagnostic pilot may capture every mechanically eligible event
    before selecting one outcome-blind target.  It therefore deliberately
    does not reuse ``Pass1Plan``, whose three-event cardinality is part of
    the historical B2A protocol rather than this pilot's contract.
    """

    trace: Any
    events: tuple[Any, ...]


@dataclass(frozen=True)
class DiagnosticScoredEvent:
    """Compact pre-intervention event-ranking evidence.

    No cache tensor or full model snapshot is retained by this object.
    """

    event_plan: Any
    candidate_pool: FrozenCandidatePool


def run_fullkv_diagnostic_worker(config: Any, manifest: Any) -> dict[str, Any]:
    """Delegate to the unchanged canonical FullKV worker."""
    from kvcot.discovery.b2a_workers import run_fullkv_worker

    return run_fullkv_worker(config, manifest)


def _architecture_evidence(model: Any, placement: dict[str, Any]) -> dict[str, Any]:
    hidden_size = int(model.config.hidden_size)
    query_heads = int(model.config.num_attention_heads)
    payload = ModelArchitecture(
        model_type=str(model.config.model_type),
        num_hidden_layers=int(model.config.num_hidden_layers),
        num_attention_heads=query_heads,
        num_key_value_heads=int(model.config.num_key_value_heads),
        head_dim=hidden_size // query_heads,
        torch_dtype=str(next(model.parameters()).dtype),
        parameter_placement=placement,
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
    )
    return payload.model_dump(mode="json")


def _find_position(values: Any, absolute_position: int) -> int | None:
    matches = (values == absolute_position).nonzero(as_tuple=True)[0]
    return None if matches.numel() == 0 else int(matches[0].item())


def _eligible_event_plans(trace: Any, *, manifest: Any, num_layers: int, num_kv_heads: int) -> list[Any]:
    """Build deterministic pre-intervention capture plans for every eligible event."""
    from kvcot.discovery.pass1 import EventPlan, _pools_for_layer_head, eligible_event_ids
    from kvcot.discovery.sampling import (
        IdentitySeedParts,
        select_candidates_and_donors,
        select_kv_head,
        select_layer,
    )

    identity = IdentitySeedParts(
        global_seed=13,
        dataset_name=manifest.dataset_repo,
        problem_index=manifest.example_index,
        model_revision=MODEL_REVISION,
        rkv_revision=RKV_REVISION,
    )
    event_by_id = {event.compaction_event_id: event for event in trace.compaction_events}
    plans: list[Any] = []
    for ordinal, event_id in enumerate(eligible_event_ids(trace)):
        event = event_by_id[event_id]
        # Existing deterministic selectors are reused.  The stratum is a
        # pure event-index mapping, fixed before score capture.
        depth_stratum = event_id % 3
        layer = select_layer(event_id, depth_stratum, num_layers, identity).layer_index
        head = select_kv_head(event_id, num_kv_heads, identity)
        observation = event.layer_observations.get(layer)
        pools = None if observation is None else _pools_for_layer_head(observation, head)
        if pools is None:
            continue
        evicted_pool, retained_pool = pools
        candidate_donor = select_candidates_and_donors(
            evicted_pool, retained_pool, event_id, layer, head, identity
        )
        if candidate_donor is None:
            continue
        donor = candidate_donor.donor_selected[0]
        if any(
            donor not in set(int(value) for value in observation.observed_kept_absolute_positions[kv_head].tolist())
            for kv_head in range(num_kv_heads)
        ):
            # Arm B requires the same absolute donor identity in every KV
            # head.  Exclude an event mechanically before intervention if
            # that corresponding post-event slot does not exist.
            continue
        plans.append(
            EventPlan(
                compaction_event_id=event_id,
                absolute_event_position=event.absolute_event_position,
                chronological_event_ordinal=ordinal,
                depth_stratum=depth_stratum,
                layer_index=layer,
                kv_head_index=head,
                candidate_donor_selection=candidate_donor,
            )
        )
    return plans


def _candidate_scores_for_record(event: Any, record: Any) -> list[CandidateScore]:
    head = event.kv_head_index
    pre_map = record.pre_event_absolute_position_map[head]
    observed = record.observed_kept_absolute_positions[head]
    retained = set(int(value) for value in observed.tolist())
    non_recent_length = int(pre_map.shape[0]) - int(record.window_size)
    candidates = []
    for physical_index in range(non_recent_length):
        absolute = int(pre_map[physical_index].item())
        if absolute in retained:
            continue
        candidates.append(
            CandidateScore(
                absolute_token_position=absolute,
                deployable_score=float(record.recomputed_final_score[0, head, physical_index].item()),
            )
        )
    return candidates


def _run_event_score_replay(
    *, trace: Any, event_plans: list[Any], fresh_state: Any, prefill_fn: Any, decode_one_fn: Any
) -> tuple[bool, str | None, list[DiagnosticScoredEvent]]:
    """Replay once and retain only compact event-score evidence.

    ``capture_update_kv`` necessarily materializes a record at each targeted
    event.  This loop validates, summarizes, and releases that record before
    advancing to the next token.  It never takes a full-model snapshot.
    """
    import contextlib
    import torch

    from kvcot.discovery.capture import capture_update_kv

    state = fresh_state()
    if not hasattr(state, "projected_pre_event_position_map"):
        return False, "diagnostic_score_replay_requires_authoritative_provenance", []
    target_pairs = {
        (plan.absolute_event_position, plan.layer_index): plan for plan in event_plans
    }
    target_layers = sorted({plan.layer_index for plan in event_plans})
    sinks: dict[int, list[Any]] = {layer: [] for layer in target_layers}
    current_position = {"pos": -1}
    scored: list[DiagnosticScoredEvent] = []
    captured_event_ids: set[int] = set()

    def should_capture(position: int, layer_index: int) -> bool:
        return (position, layer_index) in target_pairs

    with contextlib.ExitStack() as stack:
        for layer_index in target_layers:
            cluster = state.kv_cluster_for_layer(layer_index)

            def position_map(_layer_index=layer_index):
                return state.projected_pre_event_position_map(_layer_index)

            stack.enter_context(
                capture_update_kv(
                    cluster,
                    sinks[layer_index],
                    pre_event_position_map_fn=position_map,
                    layer_idx=layer_index,
                    current_position_fn=lambda: current_position["pos"],
                    should_capture=should_capture,
                )
            )

        prompt_tokens = trace.full_token_ids[: trace.prompt_length]
        prefill_result = prefill_fn(state, prompt_tokens)
        if (
            len(prefill_result.per_position_logits) != trace.prompt_length
            or len(prefill_result.per_position_layer_observations) != trace.prompt_length
        ):
            return False, "diagnostic_score_replay_prefill_shape_mismatch", []
        state = prefill_result.new_state

        for offset, token_id in enumerate(trace.full_token_ids[trace.prompt_length :]):
            position = trace.prompt_length + offset
            current_position["pos"] = position
            before = {layer: len(sinks[layer]) for layer in target_layers}
            step = decode_one_fn(state, token_id)
            state = step.new_state
            for layer_index in target_layers:
                if len(sinks[layer_index]) == before[layer_index]:
                    continue
                if len(sinks[layer_index]) != before[layer_index] + 1:
                    return False, "diagnostic_score_replay_capture_cardinality", []
                record = sinks[layer_index].pop()
                plan = target_pairs.get((position, layer_index))
                if plan is None or not record.had_compaction:
                    return False, "diagnostic_score_replay_unplanned_capture", []
                if not record.parity_check_passed:
                    return False, "diagnostic_score_replay_parity_failed", []
                pass1_observation = trace.compaction_events[
                    plan.compaction_event_id
                ].layer_observations.get(layer_index)
                if (
                    pass1_observation is None
                    or pass1_observation.observed_kept_absolute_positions is None
                ):
                    return False, "diagnostic_score_replay_missing_pass1_survivors", []
                pass1_positions = pass1_observation.observed_kept_absolute_positions[
                    plan.kv_head_index
                ]
                pass2_positions = record.observed_kept_absolute_positions[plan.kv_head_index]
                if pass1_positions.shape != pass2_positions.shape or not torch.equal(
                    pass1_positions, pass2_positions
                ):
                    return False, "diagnostic_score_replay_survivor_mismatch", []
                pool = freeze_candidate_pool(_candidate_scores_for_record(plan, record))
                captured_event_ids.add(plan.compaction_event_id)
                rank_zero_available_all_heads = bool(pool.candidates) and all(
                    _find_position(
                        record.pre_event_absolute_position_map[kv_head],
                        pool.candidates[0].absolute_token_position,
                    )
                    is not None
                    for kv_head in range(EXPECTED_KV_HEADS)
                )
                if len(pool.candidates) >= 2 and rank_zero_available_all_heads:
                    scored.append(DiagnosticScoredEvent(plan, pool))
                del record

    expected_event_ids = {plan.compaction_event_id for plan in event_plans}
    if captured_event_ids != expected_event_ids:
        return False, "diagnostic_score_replay_missing_target", []
    return True, None, scored


def _choose_scored_event(scored: list[DiagnosticScoredEvent]) -> DiagnosticScoredEvent | None:
    if not scored:
        return None
    return min(
        scored,
        key=lambda row: (
            -row.candidate_pool.candidates[0].deployable_score,
            row.event_plan.compaction_event_id,
            row.event_plan.layer_index,
            row.event_plan.kv_head_index,
        ),
    )


def _freeze_selected_target(target: Any, scored: DiagnosticScoredEvent) -> tuple[Any, dict[str, Any]]:
    """Bind the full selected-event snapshot to the prior score-only choice."""
    from kvcot.discovery.pass2 import TargetCapture
    from kvcot.discovery.sampling import CandidateDonorSelection

    pool = freeze_candidate_pool(
        _candidate_scores_for_record(target.event_plan, target.capture_record)
    )
    if pool != scored.candidate_pool:
        raise DiagnosticWorkerRefused(
            "selected-event candidate pool changed between score and snapshot replays"
        )
    donor = target.event_plan.candidate_donor_selection.donor_selected[0]
    candidate_positions = tuple(candidate.absolute_token_position for candidate in pool.candidates)
    candidate_donor = CandidateDonorSelection(
        evicted_selected=candidate_positions,
        donor_selected=(donor,),
        cross_product=tuple((candidate, donor) for candidate in candidate_positions),
    )
    selected_plan = replace(target.event_plan, candidate_donor_selection=candidate_donor)
    selected_target = TargetCapture(
        event_plan=selected_plan,
        capture_record=target.capture_record,
        pristine_snapshot=target.pristine_snapshot,
    )
    evidence = {
        "event_index": selected_plan.compaction_event_id,
        "absolute_event_position": selected_plan.absolute_event_position,
        "layer_index": selected_plan.layer_index,
        "selected_kv_head": selected_plan.kv_head_index,
        "deployable_event_score": pool.candidates[0].deployable_score,
        "donor_absolute_position": donor,
        "candidate_pool": [candidate.__dict__ for candidate in pool.candidates],
        "candidate_pool_frozen_before_intervention": True,
        "candidate_pool_diagnostic_only": True,
    }
    return selected_target, evidence


def _answer_positions(tokenizer: Any, trace: Any) -> set[int]:
    if not trace.natural_answer:
        return set()
    answer_ids = list(tokenizer.encode(trace.natural_answer, add_special_tokens=False))
    generated = list(trace.generated_token_ids)
    if not answer_ids or len(answer_ids) > len(generated):
        return set()
    start = None
    for index in range(len(generated) - len(answer_ids) + 1):
        if generated[index : index + len(answer_ids)] == answer_ids:
            start = index
    if start is None:
        return set()
    return {trace.prompt_length + start + offset for offset in range(len(answer_ids))}


def _hash_live_state(state: Any) -> str:
    import torch

    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            cpu = value.detach().to("cpu").contiguous()
            digest.update(b"tensor")
            digest.update(str(tuple(cpu.shape)).encode("ascii"))
            digest.update(str(cpu.dtype).encode("ascii"))
            digest.update(cpu.view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict")
            for key in sorted(value, key=str):
                update(str(key))
                update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(b"sequence")
            for item in value:
                update(item)
        else:
            digest.update(repr(value).encode("utf-8"))

    update(int(state.absolute_position))
    update(list(state.cache.key_cache))
    update(list(state.cache.value_cache))
    update(getattr(state.cache, "query_cache", {}))
    update(state.compaction.event_steps)
    update(getattr(state.model, "length", 0))
    update(getattr(state.model, "after_think", None))
    for layer_index in sorted(state.model_provenance.layers):
        update(state.model_provenance.layers[layer_index].positions)
        layer = state.model.model.layers[layer_index]
        update(layer.self_attn.config.compression)
        cluster = getattr(layer.self_attn, "kv_cluster", None)
        if cluster is not None:
            update(
                {
                    "evicted_token_num": cluster.evicted_token_num,
                    "kept_token_indices": cluster.kept_token_indices,
                    "kept_attention_scores": cluster.kept_attention_scores,
                    "kept_similarity_scores": getattr(cluster, "kept_similarity_scores", []),
                    "kept_final_scores": getattr(cluster, "kept_final_scores", []),
                }
            )
    return digest.hexdigest()


def _evaluate_branch(
    *, step_fn: Any, snapshot: Any, bridge_token_id: int, reference_ids: list[int],
    first_scored_position: int, answer_positions: set[int], correct_answer: bool,
) -> dict[str, Any]:
    import torch

    logits, state = step_fn(snapshot, bridge_token_id)
    nll: list[float] = []
    margins: list[dict[str, Any]] = []
    scored_ids: list[int] = []
    for offset, target in enumerate(reference_ids):
        absolute_position = first_scored_position + offset
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        target_log_probability = float(log_probs[target].item())
        masked = log_probs.clone()
        masked[target] = -torch.inf
        strongest_value, strongest_id = torch.max(masked, dim=-1)
        nll.append(-target_log_probability)
        scored_ids.append(int(target))
        if absolute_position in answer_positions:
            margins.append(
                {
                    "absolute_position": absolute_position,
                    "reference_token_id": int(target),
                    "strongest_alternative_token_id": int(strongest_id.item()),
                    "reference_log_probability": target_log_probability,
                    "correct_answer_log_probability": target_log_probability if correct_answer else None,
                    "strongest_alternative_log_probability": float(strongest_value.item()),
                    "margin": target_log_probability - float(strongest_value.item()),
                    "is_correct_answer_token": correct_answer,
                }
            )
        logits, state = step_fn(state, target)
    return {
        "scored_token_ids": scored_ids,
        "per_token_nll": nll,
        "mean_nll": sum(nll) / len(nll),
        "answer_token_margins": margins,
        "final_state_sha256": _hash_live_state(state),
    }


def _vectors_for_candidate(record: Any, candidate_position: int, heads: tuple[int, ...]) -> tuple[dict[int, Any], dict[int, Any]]:
    keys: dict[int, Any] = {}
    values: dict[int, Any] = {}
    for head in heads:
        pre_index = _find_position(record.pre_event_absolute_position_map[head], candidate_position)
        if pre_index is None:
            raise DiagnosticWorkerRefused("candidate absolute position missing from a KV head pre-event map")
        keys[head] = record.pre_call_key_states[0, head, pre_index, :].detach().clone().contiguous()
        values[head] = record.pre_call_value_states[0, head, pre_index, :].detach().clone().contiguous()
    return keys, values


def _pair_record(
    *, target: Any, candidate_position: int, donor_position: int, width: int, baseline: dict[str, Any],
    step_fn: Any, bridge_token_id: int, reference_ids: list[int], first_scored_position: int,
    answer_positions: set[int], correct_answer: bool, fixed_trace_answer: str | None,
    fixed_trace_correctness_status: str, arm: str, candidate_pool_rank: int | None,
) -> dict[str, Any]:
    from kvcot.discovery.diagnostic_pilot_swap import apply_diagnostic_kv_restore

    event = target.event_plan
    record = target.capture_record
    heads = resolve_restore_heads(width, selected_head=event.kv_head_index, num_key_value_heads=EXPECTED_KV_HEADS)
    donor_slots = {
        head: _find_position(record.observed_kept_absolute_positions[head], donor_position)
        for head in heads
    }
    if any(position is None for position in donor_slots.values()):
        raise DiagnosticWorkerRefused("donor is absent from a selected head's post-event cache")
    resolved_donor_slots = {head: int(position) for head, position in donor_slots.items()}
    keys, values = _vectors_for_candidate(record, candidate_position, heads)
    swapped_snapshot = target.pristine_snapshot.clone()
    mutation = apply_diagnostic_kv_restore(
        swapped_snapshot,
        layer_index=event.layer_index,
        kv_head_indices=heads,
        token_position=resolved_donor_slots,
        replacement_keys=keys,
        replacement_values=values,
        candidate_absolute_position=candidate_position,
        donor_absolute_position=donor_position,
    )
    intervention = _evaluate_branch(
        step_fn=step_fn,
        snapshot=swapped_snapshot,
        bridge_token_id=bridge_token_id,
        reference_ids=reference_ids,
        first_scored_position=first_scored_position,
        answer_positions=answer_positions,
        correct_answer=correct_answer,
    )
    gain = baseline["mean_nll"] - intervention["mean_nll"]
    baseline_margins = {row["absolute_position"]: row["margin"] for row in baseline["answer_token_margins"]}
    intervention_margins = {
        row["absolute_position"]: row["margin"] for row in intervention["answer_token_margins"]
    }
    margin_sign_change = correct_answer and any(
        material_margin_change(baseline_margins.get(position), intervention_margins.get(position))
        for position in set(baseline_margins) | set(intervention_margins)
    )
    return {
        "artifact_schema_version": PAIR_SCHEMA_VERSION,
        "arm": arm,
        # Every grid cell is a bounded diagnostic probe over a frozen
        # candidate pool.  The no-op is a mechanical control, not a
        # diagnostic claim.  Neither is ever a deployable performance
        # number.
        "diagnostic_only": arm == RESTORE_ARM,
        "deployable_performance": False,
        "compaction_event_id": event.compaction_event_id,
        "layer_index": event.layer_index,
        "selected_kv_head": event.kv_head_index,
        "kv_head_indices": list(heads),
        "restore_width": width,
        "candidate_pool_rank": candidate_pool_rank,
        "is_rank_zero_candidate": candidate_pool_rank == 0,
        "candidate_absolute_position": candidate_position,
        "donor_absolute_position": donor_position,
        "donor_post_storage_positions_by_head": resolved_donor_slots,
        "baseline_scored_token_ids": baseline["scored_token_ids"],
        "intervention_scored_token_ids": intervention["scored_token_ids"],
        "baseline_per_token_nll": baseline["per_token_nll"],
        "intervention_per_token_nll": intervention["per_token_nll"],
        "baseline_mean_nll": baseline["mean_nll"],
        "intervention_mean_nll": intervention["mean_nll"],
        "swap_gain": gain,
        "baseline_answer_token_margins": baseline["answer_token_margins"],
        "intervention_answer_token_margins": intervention["answer_token_margins"],
        "answer_span_absolute_positions": sorted(answer_positions),
        "answer_span_token_ids": [
            int(reference_ids[position - first_scored_position])
            for position in sorted(answer_positions)
            if first_scored_position <= position < first_scored_position + len(reference_ids)
        ],
        "answer_margin_sign_change": margin_sign_change,
        "baseline_fixed_trace_extracted_answer": fixed_trace_answer,
        "intervention_fixed_trace_extracted_answer": fixed_trace_answer,
        "baseline_fixed_trace_correctness_status": fixed_trace_correctness_status,
        "intervention_fixed_trace_correctness_status": fixed_trace_correctness_status,
        "fixed_trace_extracted_answer_change": False,
        "fixed_trace_correctness_change": False,
        "free_running_answer_flip_available": False,
        "mutation": mutation.__dict__,
        "baseline_final_state_sha256": baseline["final_state_sha256"],
        "intervention_final_state_sha256": intervention["final_state_sha256"],
    }


def run_rkv_diagnostic_worker(config: Any, manifest: Any, fullkv_result: dict[str, Any]) -> dict[str, Any]:
    """Run qualification and, only if qualified, the frozen interventions."""
    import torch
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache

    from kvcot.discovery.discovery_config import canonical_config_hash
    from kvcot.discovery.framework_seed import apply_framework_seed
    from kvcot.discovery.math500_verification import build_math500_answer_fn
    from kvcot.discovery.no_offload import assert_no_offloaded_parameters
    from kvcot.discovery.pass1 import NaturalRunProvenance, run_natural_pass1
    from kvcot.discovery.pass2 import run_pass2_capture
    from kvcot.discovery.real_model_adapter import (
        RealModelState,
        build_real_branch_step_fn_restore_once,
        build_real_decode_one_fn,
        build_real_prefill_fn,
        build_real_snapshot_fn,
    )
    from kvcot.discovery.runtime_evidence import derive_parameter_placement, derive_runtime_identity
    from kvcot.discovery.runtime_rkv_verification import verify_runtime_matches_frozen
    from kvcot.discovery.snapshot_boundary import resolve_local_snapshot
    from kvcot.discovery.strict_device import load_rkv_discovery_model, verify_single_rtx3090
    from kvcot.generation.provenance import LayerProvenance, ModelProvenance
    from kvcot.generation.replay import CompactionTracker
    from kvcot.generation.state import reset_patched_state
    from kvcot.utils.hashing import sha256_int_ids

    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise DiagnosticWorkerRefused("diagnostic R-KV worker requires CUDA")
    device = verify_single_rtx3090(torch.cuda, torch_module=torch)
    apply_framework_seed(config.generation.framework_seed, config.generation.attention_backend, cuda_available=True)
    model_snapshot = resolve_local_snapshot(config.model.name, config.model.revision, "model")
    tokenizer_snapshot = resolve_local_snapshot(config.model.tokenizer_name, config.model.tokenizer_revision, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot.local_path, local_files_only=True, use_fast=True)
    rendered_ids = list(manifest.prompt_token_ids)
    if sha256_int_ids(rendered_ids) != manifest.prompt_token_ids_sha256:
        raise DiagnosticWorkerRefused("frozen prompt token-ID hash mismatch")
    model = load_rkv_discovery_model(config, model_snapshot.local_path, tokenizer_snapshot.local_path, "cuda:0")
    assert_no_offloaded_parameters(model)
    runtime_identity_obj = derive_runtime_identity(
        model=model,
        tokenizer=tokenizer,
        requested_model_revision=MODEL_REVISION,
        requested_tokenizer_revision=TOKENIZER_REVISION,
        verified_model_revision=model_snapshot.resolved_revision,
        verified_tokenizer_revision=tokenizer_snapshot.resolved_revision,
    )
    runtime_identity = runtime_identity_obj.__dict__
    if not runtime_identity_obj.model_revision_match or not runtime_identity_obj.tokenizer_revision_match:
        raise DiagnosticWorkerRefused("resolved model/tokenizer identity does not match the frozen revision")
    placement_obj = derive_parameter_placement(model, requested_device="cuda:0")
    placement = placement_obj.__dict__
    architecture = _architecture_evidence(model, placement)
    runtime_rkv = verify_runtime_matches_frozen(config.rkv, model)
    if not runtime_rkv.passed:
        raise DiagnosticWorkerRefused("runtime R-KV configuration does not match the frozen config")
    num_layers = int(model.config.num_hidden_layers)
    num_kv_heads = int(model.config.num_key_value_heads)

    def fresh_state() -> Any:
        cache = reset_patched_state(model, DynamicCache)
        provenance = ModelProvenance(
            layers={index: LayerProvenance.empty(num_kv_heads) for index in range(num_layers)}
        )
        return RealModelState(
            model=model,
            cache=cache,
            model_provenance=provenance,
            compaction=CompactionTracker(),
            absolute_position=0,
            device="cuda:0",
        )

    answer_verifier = build_math500_answer_fn(tokenizer, manifest.gold_answer)
    provenance = NaturalRunProvenance(
        model_name=config.model.name,
        model_revision=config.model.revision,
        tokenizer_name=config.model.tokenizer_name,
        tokenizer_revision=config.model.tokenizer_revision,
        rkv_revision=config.rkv.upstream_revision,
        config_sha256=canonical_config_hash(config),
        dataset_name=manifest.dataset_repo,
        example_id=manifest.unique_id,
    )
    prefill = build_real_prefill_fn("cuda:0")
    decode = build_real_decode_one_fn("cuda:0")
    trace = run_natural_pass1(
        provenance,
        list(manifest.prompt_token_ids),
        fresh_state(),
        prefill,
        decode,
        config.generation.max_new_tokens,
        tokenizer.eos_token_id,
        answer_verifier,
    )
    event_plans = _eligible_event_plans(
        trace, manifest=manifest, num_layers=num_layers, num_kv_heads=num_kv_heads
    )
    pass2_valid = False
    pass2_reason = "no_eligible_event_plan"
    selected_target = None
    selected_event = None
    pass2_result = None
    scored_events: list[DiagnosticScoredEvent] = []
    if event_plans:
        score_valid, score_reason, scored_events = _run_event_score_replay(
            trace=trace,
            event_plans=event_plans,
            fresh_state=fresh_state,
            prefill_fn=prefill,
            decode_one_fn=decode,
        )
        chosen = _choose_scored_event(scored_events) if score_valid else None
        if chosen is not None:
            replay_plan = DiagnosticCapturePlan(trace=trace, events=(chosen.event_plan,))
            pass2_result = run_pass2_capture(
                replay_plan,
                trace.full_token_ids,
                fresh_state(),
                prefill,
                decode,
                build_real_snapshot_fn(),
            )
            pass2_valid = bool(pass2_result.valid)
            pass2_reason = pass2_result.invalid_reason
            if pass2_valid:
                selected_target, selected_event = _freeze_selected_target(
                    pass2_result.target_captures[0], chosen
                )
                del pass2_result
                pass2_result = None
        else:
            pass2_valid = score_valid
            pass2_reason = score_reason or "no_event_with_two_eligible_candidates"

    from kvcot.discovery.strict_device import (
        _single_worker_placement_ok,
        verify_device_gate_from_raw_evidence,
    )

    rkv_device_evidence = {"verified": True, **device.__dict__}
    fullkv_valid = (
        fullkv_result.get("cap_hit") is False
        and fullkv_result.get("actual_batch_size_verified") is True
        and _single_worker_placement_ok(fullkv_result.get("parameter_placement"))
        and verify_device_gate_from_raw_evidence(
            fullkv_result.get("device_evidence", {}), rkv_device_evidence
        )
    )
    rkv_status = trace.natural_answer_status
    fullkv_status = fullkv_result.get("natural_answer_status")
    meaningful_compression = bool(trace.compaction_events) and any(
        length < len(trace.full_token_ids) for length in trace.cache_length_final_per_layer.values()
    )
    qualification = {
        "candidate_ordinal": None,
        "unique_id": manifest.unique_id,
        "fullkv_execution_valid": fullkv_valid,
        "rkv_replay_mechanically_valid": pass2_valid and trace.cap_hit is False,
        "correctness_status_matched": fullkv_status == rkv_status,
        "meaningful_compression": meaningful_compression,
        "eligible_event_exists": bool(scored_events),
        "selected_event_has_two_candidates": bool(
            selected_event and len(selected_event["candidate_pool"]) >= 2
        ),
        "selected_event_count": 1 if selected_event is not None else 0,
        "intervention_not_evaluated": True,
        "fullkv_correctness_status": fullkv_status,
        "rkv_correctness_status": rkv_status,
        "pass2_invalid_reason": pass2_reason,
        "rkv_natural_cap_hit": trace.cap_hit,
        "observed_compaction_event_count": len(trace.compaction_events),
        "eligible_event_plan_count": len(event_plans),
        "eligible_scored_event_count": len(scored_events),
    }
    qualifies = all(
        qualification[field]
        for field in (
            "fullkv_execution_valid",
            "rkv_replay_mechanically_valid",
            "correctness_status_matched",
            "meaningful_compression",
            "eligible_event_exists",
            "selected_event_has_two_candidates",
            "intervention_not_evaluated",
        )
    )
    pairs: list[dict[str, Any]] = []
    noop_exact = False
    if qualifies and selected_target is not None:
        event = selected_target.event_plan
        t = event.absolute_event_position
        bridge_token_id = int(trace.full_token_ids[t + 1])
        reference_ids = [int(value) for value in trace.full_token_ids[t + 2 : t + 2 + SCORED_HORIZON]]
        if len(reference_ids) != SCORED_HORIZON:
            raise DiagnosticWorkerRefused("selected event lacks the frozen 48-token horizon")
        step_fn = build_real_branch_step_fn_restore_once(model, "cuda:0", consume_owned_snapshot=True)
        answer_positions = _answer_positions(tokenizer, trace)
        baseline = _evaluate_branch(
            step_fn=step_fn,
            snapshot=selected_target.pristine_snapshot.clone(),
            bridge_token_id=bridge_token_id,
            reference_ids=reference_ids,
            first_scored_position=t + 2,
            answer_positions=answer_positions,
            correct_answer=rkv_status == "correct",
        )
        donor = int(selected_event["donor_absolute_position"])
        # The complete bounded factorial grid for the one frozen event:
        # every candidate in the frozen pool at every protocol restore
        # width, emitted rank-major so the primitive order is itself
        # reconstructable.  Same event, same layer, same intervention time,
        # same donor, same 48-token readout, same baseline for every cell.
        for candidate_rank, candidate in enumerate(selected_event["candidate_pool"]):
            for width in RESTORE_WIDTHS:
                pairs.append(
                    _pair_record(
                        target=selected_target,
                        candidate_position=int(candidate["absolute_token_position"]),
                        donor_position=donor,
                        width=width,
                        baseline=baseline,
                        step_fn=step_fn,
                        bridge_token_id=bridge_token_id,
                        reference_ids=reference_ids,
                        first_scored_position=t + 2,
                        answer_positions=answer_positions,
                        correct_answer=rkv_status == "correct",
                        fixed_trace_answer=trace.natural_answer,
                        fixed_trace_correctness_status=rkv_status,
                        arm=RESTORE_ARM,
                        candidate_pool_rank=candidate_rank,
                    )
                )
        noop = _pair_record(
            target=selected_target,
            candidate_position=donor,
            donor_position=donor,
            width=1,
            baseline=baseline,
            step_fn=step_fn,
            bridge_token_id=bridge_token_id,
            reference_ids=reference_ids,
            first_scored_position=t + 2,
            answer_positions=answer_positions,
            correct_answer=rkv_status == "correct",
            fixed_trace_answer=trace.natural_answer,
            fixed_trace_correctness_status=rkv_status,
            arm=NOOP_ARM,
            candidate_pool_rank=None,
        )
        pairs.append(noop)
        noop_exact = (
            noop["baseline_scored_token_ids"] == noop["intervention_scored_token_ids"]
            and noop["baseline_per_token_nll"] == noop["intervention_per_token_nll"]
            and noop["swap_gain"] == 0.0
            and noop["baseline_final_state_sha256"] == noop["intervention_final_state_sha256"]
            and noop["mutation"]["is_noop"] is True
        )

    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    if max(peak_allocated, peak_reserved) > VRAM_LIMIT_BYTES:
        raise DiagnosticWorkerRefused("peak allocated/reserved VRAM exceeded 22 GiB")
    result = {
        "role": "diagnostic_rkv",
        "unique_id": manifest.unique_id,
        "dataset_repo": manifest.dataset_repo,
        "dataset_revision": manifest.dataset_revision,
        "manifest_hash": manifest.manifest_hash(),
        "prompt_token_ids_sha256": manifest.prompt_token_ids_sha256,
        "prompt_token_count": len(manifest.prompt_token_ids),
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "rkv_revision": RKV_REVISION,
        "architecture": architecture,
        "device_evidence": rkv_device_evidence,
        "parameter_placement": placement,
        "runtime_identity": runtime_identity,
        "runtime_rkv_config_hash": runtime_rkv.runtime_hash,
        "frozen_rkv_config_hash": runtime_rkv.frozen_hash,
        "qualification": qualification,
        "qualified": qualifies,
        "selected_event_replay_valid": pass2_valid,
        "rkv_natural_cap_hit": trace.cap_hit,
        "natural_full_token_count": len(trace.full_token_ids),
        "final_cache_length_per_layer": {
            str(layer): int(length)
            for layer, length in trace.cache_length_final_per_layer.items()
        },
        "eligible_event_score_evidence": [
            {
                "event_index": row.event_plan.compaction_event_id,
                "absolute_event_position": row.event_plan.absolute_event_position,
                "layer_index": row.event_plan.layer_index,
                "selected_kv_head": row.event_plan.kv_head_index,
                "deployable_event_score": row.candidate_pool.candidates[0].deployable_score,
                "candidate_pool": [candidate.__dict__ for candidate in row.candidate_pool.candidates],
                "captured_before_intervention": True,
                "rank_zero_candidate_available_in_all_kv_heads": True,
            }
            for row in scored_events
        ],
        "score_replay_retained_full_snapshots": 0,
        "selected_snapshot_count": 1 if selected_target is not None else 0,
        "selected_event": selected_event,
        "pair_records": pairs,
        "noop_exact": noop_exact if qualifies else None,
        "fixed_trace_extracted_answer": trace.natural_answer,
        "fixed_trace_correctness_status": rkv_status,
        "free_running_answer_flip_available": False,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result
