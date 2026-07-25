"""CPU-only preparation of canonical diagnostic runtime inputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kvcot.discovery.attempt_artifacts import atomic_write_json, sha256_file
from kvcot.discovery.diagnostic_pilot_contract import (
    BRANCH,
    CANDIDATE_MANIFEST_BYTE_SHA256,
    CANDIDATE_MANIFEST_CANONICAL_SHA256,
    CANDIDATE_MANIFEST_PATH,
    CONFIG_PATH,
    DATASET_REVISION,
    FRAMEWORK_SEED,
    MAXIMUM_CANDIDATE_POOL_SIZE,
    MAXIMUM_EVENTS_PER_EXAMPLE,
    MAXIMUM_QUALIFICATION_CANDIDATES,
    MAXIMUM_SELECTED_EXAMPLES,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    PROTOCOL_DOCUMENT_PATH,
    PROTOCOL_SCHEMA_VERSION,
    RESTORE_WIDTHS,
    RKV_REVISION,
    RUNTIME_LIMIT_SECONDS,
    RUNTIME_SCHEMA_VERSION,
    SCORED_HORIZON,
    SWAP_GAIN_THRESHOLD_NATS,
    TOKENIZER_REPOSITORY,
    TOKENIZER_REVISION,
    VRAM_LIMIT_BYTES,
    attach_canonical_hash,
    verify_canonical_hash,
)
from kvcot.discovery.discovery_config import (
    canonical_config_hash,
    generation_config_hash,
    load_discovery_config,
    rkv_config_hash,
)
from kvcot.discovery.manifest import B2AOneExampleManifest, ChatTemplateRenderingConfig
from kvcot.discovery.manifest_prepare import render_with_loaded_tokenizer
from kvcot.discovery.snapshot_boundary import resolve_local_snapshot
from kvcot.utils.hashing import sha256_int_ids, sha256_json, sha256_text

DEFAULT_RUNTIME_ROOT = Path("/workspace/faithkv-post-stage-c-diagnostic-runtime")
DEFAULT_OUTPUT_ROOT = Path("/workspace/faithkv-post-stage-c-diagnostic-execution")


class DiagnosticPreparationRefused(RuntimeError):
    pass


def _load_candidate_manifest(repository_root: Path) -> dict[str, Any]:
    path = repository_root / CANDIDATE_MANIFEST_PATH
    if sha256_file(path) != CANDIDATE_MANIFEST_BYTE_SHA256:
        raise DiagnosticPreparationRefused("candidate-manifest byte hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("canonical_sha256") != CANDIDATE_MANIFEST_CANONICAL_SHA256:
        raise DiagnosticPreparationRefused("candidate-manifest canonical hash mismatch")
    if payload.get("dataset_revision") != DATASET_REVISION:
        raise DiagnosticPreparationRefused("candidate-manifest dataset revision mismatch")
    if payload.get("qualification_limit") != MAXIMUM_QUALIFICATION_CANDIDATES:
        raise DiagnosticPreparationRefused("candidate-manifest qualification limit mismatch")
    return payload


def _manifest_for_candidate(candidate: dict[str, Any], tokenizer: Any) -> B2AOneExampleManifest:
    row = candidate["row"]
    user_message, messages, token_ids = render_with_loaded_tokenizer(tokenizer, row)
    prompt_config = ChatTemplateRenderingConfig(
        message_roles=("user",),
        add_generation_prompt=True,
        tokenize=True,
        add_special_tokens_note=(
            "special-token behavior is entirely delegated to the tokenizer's own chat_template (Jinja) -- "
            "apply_chat_template was called with no separate add_special_tokens override"
        ),
    )
    return B2AOneExampleManifest(
        dataset_repo="HuggingFaceH4/MATH-500",
        dataset_config="default",
        dataset_split="test",
        dataset_revision=DATASET_REVISION,
        example_index=candidate["source_example_index"],
        unique_id=candidate["unique_id"],
        raw_content_hash=candidate["raw_row_sha256"],
        gold_answer=row["answer"],
        prompt_token_ids_sha256=sha256_int_ids(token_ids),
        tokenizer_revision_used_for_prompt_hash=TOKENIZER_REVISION,
        rendered_user_message_sha256=sha256_text(user_message),
        chat_template_source_sha256=sha256_text(tokenizer.chat_template),
        chat_message_payload_sha256=sha256_json(messages),
        prompt_rendering_config=prompt_config,
        prompt_token_count=len(token_ids),
        prompt_token_ids=tuple(token_ids),
    )


def _write_new_or_identical(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise DiagnosticPreparationRefused(f"refusing to replace non-identical frozen runtime artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def prepare_runtime_inputs(
    *, repository_root: str | Path, runtime_root: str | Path = DEFAULT_RUNTIME_ROOT
) -> dict[str, Any]:
    """Prepare prompt identities and runtime binding without loading weights."""
    repository_root = Path(repository_root).resolve()
    runtime_root = Path(runtime_root).resolve()
    candidate_manifest = _load_candidate_manifest(repository_root)
    config_path = repository_root / CONFIG_PATH
    config = load_discovery_config(config_path)
    if config.model.name != MODEL_REPOSITORY or config.model.revision != MODEL_REVISION:
        raise DiagnosticPreparationRefused("diagnostic config model binding mismatch")
    if config.model.tokenizer_name != TOKENIZER_REPOSITORY or config.model.tokenizer_revision != TOKENIZER_REVISION:
        raise DiagnosticPreparationRefused("diagnostic config tokenizer binding mismatch")
    tokenizer_snapshot = resolve_local_snapshot(TOKENIZER_REPOSITORY, TOKENIZER_REVISION, "tokenizer")
    model_snapshot = resolve_local_snapshot(MODEL_REPOSITORY, MODEL_REVISION, "model")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot.local_path, local_files_only=True, use_fast=True)
    if tokenizer.chat_template is None:
        raise DiagnosticPreparationRefused("frozen tokenizer has no chat template")

    candidate_rows = candidate_manifest["candidates"][:MAXIMUM_QUALIFICATION_CANDIDATES]
    manifests = [_manifest_for_candidate(candidate, tokenizer) for candidate in candidate_rows]
    prompt_payload = attach_canonical_hash(
        {
            "artifact_schema_version": "faithkv-post-stage-c-diagnostic-prompts-v1",
            "tokenizer_repository": TOKENIZER_REPOSITORY,
            "tokenizer_revision": TOKENIZER_REVISION,
            "qualification_order": list(range(MAXIMUM_QUALIFICATION_CANDIDATES)),
            "manifests": [manifest.model_dump(mode="json") for manifest in manifests],
        }
    )
    prompts_path = runtime_root / "candidate_prompts.json"
    _write_new_or_identical(prompts_path, prompt_payload)

    model_config_path = Path(model_snapshot.local_path) / "config.json"
    if not model_config_path.is_file():
        raise DiagnosticPreparationRefused("exact model snapshot has no config.json")
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    if model_config.get("num_key_value_heads") != 2 or model_config.get("num_attention_heads") != 12:
        raise DiagnosticPreparationRefused("exact snapshot architecture does not match the protocol")

    runtime_payload = attach_canonical_hash(
        {
            "artifact_schema_version": RUNTIME_SCHEMA_VERSION,
            "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
            "repository": "asad073-ui/Faithkv",
            "branch": BRANCH,
            "protocol_document_path": PROTOCOL_DOCUMENT_PATH,
            "protocol_document_sha256": sha256_file(repository_root / PROTOCOL_DOCUMENT_PATH),
            "config_path": CONFIG_PATH,
            "config_byte_sha256": sha256_file(config_path),
            "config_canonical_sha256": canonical_config_hash(config),
            "generation_config_canonical_sha256": generation_config_hash(config.generation),
            "rkv_config_canonical_sha256": rkv_config_hash(config.rkv),
            "candidate_manifest_path": CANDIDATE_MANIFEST_PATH,
            "candidate_manifest_canonical_sha256": CANDIDATE_MANIFEST_CANONICAL_SHA256,
            "candidate_manifest_byte_sha256": CANDIDATE_MANIFEST_BYTE_SHA256,
            "candidate_prompts_path": str(prompts_path),
            "candidate_prompts_canonical_sha256": prompt_payload["canonical_sha256"],
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "tokenizer_repository": TOKENIZER_REPOSITORY,
            "tokenizer_revision": TOKENIZER_REVISION,
            "model_snapshot_path": model_snapshot.local_path,
            "tokenizer_snapshot_path": tokenizer_snapshot.local_path,
            "model_config_sha256": sha256_file(model_config_path),
            "model_config_architecture": {
                "model_type": model_config.get("model_type"),
                "num_hidden_layers": model_config.get("num_hidden_layers"),
                "num_attention_heads": model_config.get("num_attention_heads"),
                "num_key_value_heads": model_config.get("num_key_value_heads"),
                "head_dim": model_config.get("hidden_size") // model_config.get("num_attention_heads"),
                "torch_dtype": model_config.get("torch_dtype"),
            },
            "dataset_revision": DATASET_REVISION,
            "rkv_revision": RKV_REVISION,
            "framework_seed": FRAMEWORK_SEED,
            "maximum_qualification_candidates": MAXIMUM_QUALIFICATION_CANDIDATES,
            "maximum_selected_examples": MAXIMUM_SELECTED_EXAMPLES,
            "maximum_selected_events": MAXIMUM_SELECTED_EXAMPLES,
            "events_per_example": MAXIMUM_EVENTS_PER_EXAMPLE,
            "maximum_candidate_pool_size": MAXIMUM_CANDIDATE_POOL_SIZE,
            "maximum_interventions_per_selected_example": MAXIMUM_CANDIDATE_POOL_SIZE + 2,
            "maximum_total_interventions": MAXIMUM_SELECTED_EXAMPLES * (MAXIMUM_CANDIDATE_POOL_SIZE + 2),
            "restore_widths": list(RESTORE_WIDTHS),
            "scored_horizon": SCORED_HORIZON,
            "swap_gain_threshold_nats": SWAP_GAIN_THRESHOLD_NATS,
            "runtime_limit_seconds": RUNTIME_LIMIT_SECONDS,
            "vram_limit_bytes": VRAM_LIMIT_BYTES,
            "single_rtx3090": True,
            "cpu_offload": False,
            "output_root": str(DEFAULT_OUTPUT_ROOT),
        }
    )
    runtime_path = runtime_root / "runtime_config.json"
    _write_new_or_identical(runtime_path, runtime_payload)
    return {
        "runtime_config_path": str(runtime_path),
        "runtime_config_canonical_sha256": runtime_payload["canonical_sha256"],
        "candidate_prompts_path": str(prompts_path),
        "candidate_prompts_canonical_sha256": prompt_payload["canonical_sha256"],
        "model_snapshot_path": model_snapshot.local_path,
        "tokenizer_snapshot_path": tokenizer_snapshot.local_path,
    }


def verify_runtime_inputs(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_canonical_hash(payload)
    if payload.get("artifact_schema_version") != RUNTIME_SCHEMA_VERSION:
        raise DiagnosticPreparationRefused("runtime-config schema mismatch")
    expected = {
        "repository": "asad073-ui/Faithkv",
        "branch": BRANCH,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_document_path": PROTOCOL_DOCUMENT_PATH,
        "config_path": CONFIG_PATH,
        "candidate_manifest_path": CANDIDATE_MANIFEST_PATH,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "tokenizer_repository": TOKENIZER_REPOSITORY,
        "tokenizer_revision": TOKENIZER_REVISION,
        "dataset_revision": DATASET_REVISION,
        "rkv_revision": RKV_REVISION,
        "framework_seed": FRAMEWORK_SEED,
        "maximum_qualification_candidates": MAXIMUM_QUALIFICATION_CANDIDATES,
        "maximum_selected_examples": MAXIMUM_SELECTED_EXAMPLES,
        "maximum_selected_events": MAXIMUM_SELECTED_EXAMPLES,
        "events_per_example": MAXIMUM_EVENTS_PER_EXAMPLE,
        "maximum_candidate_pool_size": MAXIMUM_CANDIDATE_POOL_SIZE,
        "maximum_interventions_per_selected_example": MAXIMUM_CANDIDATE_POOL_SIZE + 2,
        "maximum_total_interventions": MAXIMUM_SELECTED_EXAMPLES * (MAXIMUM_CANDIDATE_POOL_SIZE + 2),
        "restore_widths": list(RESTORE_WIDTHS),
        "scored_horizon": SCORED_HORIZON,
        "swap_gain_threshold_nats": SWAP_GAIN_THRESHOLD_NATS,
        "runtime_limit_seconds": RUNTIME_LIMIT_SECONDS,
        "vram_limit_bytes": VRAM_LIMIT_BYTES,
        "single_rtx3090": True,
        "cpu_offload": False,
        "output_root": str(DEFAULT_OUTPUT_ROOT),
        "candidate_manifest_canonical_sha256": CANDIDATE_MANIFEST_CANONICAL_SHA256,
        "candidate_manifest_byte_sha256": CANDIDATE_MANIFEST_BYTE_SHA256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise DiagnosticPreparationRefused(f"runtime-config field {key!r} does not match the protocol")
    model_snapshot = resolve_local_snapshot(MODEL_REPOSITORY, MODEL_REVISION, "model")
    tokenizer_snapshot = resolve_local_snapshot(TOKENIZER_REPOSITORY, TOKENIZER_REVISION, "tokenizer")
    if payload.get("model_snapshot_path") != model_snapshot.local_path:
        raise DiagnosticPreparationRefused("runtime model snapshot path is not the exact local pin")
    if payload.get("tokenizer_snapshot_path") != tokenizer_snapshot.local_path:
        raise DiagnosticPreparationRefused("runtime tokenizer snapshot path is not the exact local pin")
    if sha256_file(Path(model_snapshot.local_path) / "config.json") != payload.get("model_config_sha256"):
        raise DiagnosticPreparationRefused("runtime model config hash mismatch")
    prompts = json.loads(Path(payload["candidate_prompts_path"]).read_text(encoding="utf-8"))
    verify_canonical_hash(prompts)
    if prompts["canonical_sha256"] != payload["candidate_prompts_canonical_sha256"]:
        raise DiagnosticPreparationRefused("candidate prompt artifact does not match runtime binding")
    if len(prompts["manifests"]) != MAXIMUM_QUALIFICATION_CANDIDATES:
        raise DiagnosticPreparationRefused("candidate prompt count does not match frozen maximum")
    for raw in prompts["manifests"]:
        B2AOneExampleManifest.model_validate(raw)
    return payload
