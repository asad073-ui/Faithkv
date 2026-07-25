"""CPU-only preparation of canonical diagnostic runtime inputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kvcot.discovery.attempt_artifacts import atomic_write_json, sha256_file
from kvcot.discovery.diagnostic_pilot_contract import (
    _HEX40,
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
    PROTOCOL_SCHEMA_VERSION,
    R1_GENERATION,
    RESTORE_WIDTHS,
    RKV_REVISION,
    RUNTIME_LIMIT_SECONDS,
    RUNTIME_SCHEMA_VERSION,
    SCORED_HORIZON,
    SWAP_GAIN_THRESHOLD_NATS,
    TOKENIZER_REPOSITORY,
    TOKENIZER_REVISION,
    VRAM_LIMIT_BYTES,
    DiagnosticGeneration,
    attach_canonical_hash,
    generation_by_label,
    generation_for_protocol_document_path,
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

#: Backward-compatible R1 defaults.  Callers that pass nothing continue to
#: reproduce the consumed R1 runtime artifact byte-for-byte.
DEFAULT_RUNTIME_ROOT = Path(R1_GENERATION.default_runtime_root)
DEFAULT_OUTPUT_ROOT = Path(R1_GENERATION.default_output_root)


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


def _verify_candidate_prompt_bindings(
    prompts: dict[str, Any], expected_manifests: list[B2AOneExampleManifest]
) -> None:
    expected = {
        "artifact_schema_version": "faithkv-post-stage-c-diagnostic-prompts-v1",
        "tokenizer_repository": TOKENIZER_REPOSITORY,
        "tokenizer_revision": TOKENIZER_REVISION,
        "qualification_order": list(range(MAXIMUM_QUALIFICATION_CANDIDATES)),
        "manifests": [manifest.model_dump(mode="json") for manifest in expected_manifests],
    }
    for key, value in expected.items():
        if prompts.get(key) != value:
            raise DiagnosticPreparationRefused(
                f"candidate prompt field {key!r} differs from the frozen manifest/tokenizer reconstruction"
            )


def _pair_budget_fields(generation: DiagnosticGeneration) -> dict[str, Any]:
    """Generation-specific pair-budget vocabulary.

    R1's field names are frozen so its already-written runtime artifact
    keeps verifying byte-for-byte.  R2 records the bounded factorial grid's
    pair budget under its own explicit names.
    """
    if generation is R1_GENERATION:
        return {
            "maximum_interventions_per_selected_example": (
                generation.maximum_pairs_per_selected_example
            ),
            "maximum_total_interventions": generation.maximum_total_pairs,
        }
    return {
        "maximum_pairs_per_selected_example": generation.maximum_pairs_per_selected_example,
        "maximum_total_pairs": generation.maximum_total_pairs,
    }


def _implementation_sha_fields(
    generation: DiagnosticGeneration, implementation_sha: str | None
) -> dict[str, Any]:
    """Bind the audited implementation commit into the runtime artifact.

    R1's artifact predates this binding and is left byte-identical; every
    later generation must name the exact implementation SHA it was
    prepared against so preflight can compare it with the authorization.
    """
    if generation is R1_GENERATION:
        if implementation_sha is not None:
            raise DiagnosticPreparationRefused(
                "the R1 runtime artifact does not bind an implementation SHA"
            )
        return {}
    if not isinstance(implementation_sha, str) or _HEX40.fullmatch(implementation_sha) is None:
        raise DiagnosticPreparationRefused(
            "implementation SHA must be an exact 40-character lowercase commit SHA"
        )
    return {"implementation_sha": implementation_sha}


def prepare_runtime_inputs(
    *,
    repository_root: str | Path,
    runtime_root: str | Path = DEFAULT_RUNTIME_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    generation: str = R1_GENERATION.label,
    implementation_sha: str | None = None,
) -> dict[str, Any]:
    """Prepare prompt identities and runtime binding without loading weights.

    ``output_root`` is explicit so a new generation can bind its own
    execution root, while the existing default keeps every prior caller
    reproducing the R1 artifact unchanged.
    """
    repository_root = Path(repository_root).resolve()
    runtime_root = Path(runtime_root).resolve()
    output_root = Path(output_root)
    if not output_root.is_absolute():
        raise DiagnosticPreparationRefused("output root must be an absolute path")
    output_root = output_root.resolve()
    pilot_generation = generation_by_label(generation)
    implementation_sha_fields = _implementation_sha_fields(pilot_generation, implementation_sha)
    if runtime_root == output_root:
        raise DiagnosticPreparationRefused("runtime root and output root must differ")
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
            "protocol_document_path": pilot_generation.protocol_document_path,
            "protocol_document_sha256": sha256_file(
                repository_root / pilot_generation.protocol_document_path
            ),
            **implementation_sha_fields,
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
            **_pair_budget_fields(pilot_generation),
            "restore_widths": list(RESTORE_WIDTHS),
            "scored_horizon": SCORED_HORIZON,
            "swap_gain_threshold_nats": SWAP_GAIN_THRESHOLD_NATS,
            "runtime_limit_seconds": RUNTIME_LIMIT_SECONDS,
            "vram_limit_bytes": VRAM_LIMIT_BYTES,
            "single_rtx3090": True,
            "cpu_offload": False,
            "output_root": str(output_root),
        }
    )
    runtime_path = runtime_root / "runtime_config.json"
    _write_new_or_identical(runtime_path, runtime_payload)
    return {
        "generation": pilot_generation.label,
        "runtime_config_path": str(runtime_path),
        "runtime_config_canonical_sha256": runtime_payload["canonical_sha256"],
        "candidate_prompts_path": str(prompts_path),
        "candidate_prompts_canonical_sha256": prompt_payload["canonical_sha256"],
        "model_snapshot_path": model_snapshot.local_path,
        "tokenizer_snapshot_path": tokenizer_snapshot.local_path,
        "protocol_document_path": pilot_generation.protocol_document_path,
        "implementation_sha": implementation_sha,
        "output_root": str(output_root),
        "maximum_pairs_per_selected_example": (
            pilot_generation.maximum_pairs_per_selected_example
        ),
        "maximum_total_pairs": pilot_generation.maximum_total_pairs,
    }


def verify_runtime_inputs(
    path: str | Path, *, repository_root: str | Path = "."
) -> dict[str, Any]:
    repository_root = Path(repository_root).resolve()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_canonical_hash(payload)
    if payload.get("artifact_schema_version") != RUNTIME_SCHEMA_VERSION:
        raise DiagnosticPreparationRefused("runtime-config schema mismatch")
    protocol_document_path = payload.get("protocol_document_path")
    if not isinstance(protocol_document_path, str):
        raise DiagnosticPreparationRefused("runtime-config protocol document path is missing")
    try:
        pilot_generation = generation_for_protocol_document_path(protocol_document_path)
    except ValueError as exc:
        raise DiagnosticPreparationRefused(str(exc)) from exc
    # The output root is bound by the artifact's own canonical hash.  It
    # must be absolute so preflight can compare it exactly against the
    # authorization, but it is deliberately not pinned to any single
    # generation's default.
    output_root = payload.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise DiagnosticPreparationRefused("runtime-config output root must be a non-empty string")
    if not Path(output_root).is_absolute():
        raise DiagnosticPreparationRefused("runtime-config output root must be an absolute path")
    if Path(output_root).resolve() != Path(output_root):
        raise DiagnosticPreparationRefused("runtime-config output root must be canonical")
    if pilot_generation is not R1_GENERATION:
        bound_sha = payload.get("implementation_sha")
        if not isinstance(bound_sha, str) or _HEX40.fullmatch(bound_sha) is None:
            raise DiagnosticPreparationRefused(
                "runtime-config implementation SHA must be an exact 40-character "
                "lowercase commit SHA"
            )
    elif "implementation_sha" in payload:
        raise DiagnosticPreparationRefused(
            "the R1 runtime artifact does not bind an implementation SHA"
        )
    expected = {
        "repository": "asad073-ui/Faithkv",
        "branch": BRANCH,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
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
        **_pair_budget_fields(pilot_generation),
        "restore_widths": list(RESTORE_WIDTHS),
        "scored_horizon": SCORED_HORIZON,
        "swap_gain_threshold_nats": SWAP_GAIN_THRESHOLD_NATS,
        "runtime_limit_seconds": RUNTIME_LIMIT_SECONDS,
        "vram_limit_bytes": VRAM_LIMIT_BYTES,
        "single_rtx3090": True,
        "cpu_offload": False,
        "candidate_manifest_canonical_sha256": CANDIDATE_MANIFEST_CANONICAL_SHA256,
        "candidate_manifest_byte_sha256": CANDIDATE_MANIFEST_BYTE_SHA256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise DiagnosticPreparationRefused(f"runtime-config field {key!r} does not match the protocol")
    if sha256_file(repository_root / pilot_generation.protocol_document_path) != payload.get(
        "protocol_document_sha256"
    ):
        raise DiagnosticPreparationRefused("runtime protocol hash differs from the checked-out document")
    config_path = repository_root / CONFIG_PATH
    config = load_discovery_config(config_path)
    derived_config_hashes = {
        "config_byte_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_config_hash(config),
        "generation_config_canonical_sha256": generation_config_hash(config.generation),
        "rkv_config_canonical_sha256": rkv_config_hash(config.rkv),
    }
    for key, value in derived_config_hashes.items():
        if payload.get(key) != value:
            raise DiagnosticPreparationRefused(
                f"runtime {key!r} differs from the checked-out diagnostic config"
            )
    candidate_manifest = _load_candidate_manifest(repository_root)
    model_snapshot = resolve_local_snapshot(MODEL_REPOSITORY, MODEL_REVISION, "model")
    tokenizer_snapshot = resolve_local_snapshot(TOKENIZER_REPOSITORY, TOKENIZER_REVISION, "tokenizer")
    if payload.get("model_snapshot_path") != model_snapshot.local_path:
        raise DiagnosticPreparationRefused("runtime model snapshot path is not the exact local pin")
    if payload.get("tokenizer_snapshot_path") != tokenizer_snapshot.local_path:
        raise DiagnosticPreparationRefused("runtime tokenizer snapshot path is not the exact local pin")
    model_config_path = Path(model_snapshot.local_path) / "config.json"
    if sha256_file(model_config_path) != payload.get("model_config_sha256"):
        raise DiagnosticPreparationRefused("runtime model config hash mismatch")
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    expected_architecture = {
        "model_type": model_config.get("model_type"),
        "num_hidden_layers": model_config.get("num_hidden_layers"),
        "num_attention_heads": model_config.get("num_attention_heads"),
        "num_key_value_heads": model_config.get("num_key_value_heads"),
        "head_dim": model_config.get("hidden_size") // model_config.get("num_attention_heads"),
        "torch_dtype": model_config.get("torch_dtype"),
    }
    if payload.get("model_config_architecture") != expected_architecture:
        raise DiagnosticPreparationRefused("runtime model architecture differs from exact config bytes")
    prompts = json.loads(Path(payload["candidate_prompts_path"]).read_text(encoding="utf-8"))
    verify_canonical_hash(prompts)
    if prompts["canonical_sha256"] != payload["candidate_prompts_canonical_sha256"]:
        raise DiagnosticPreparationRefused("candidate prompt artifact does not match runtime binding")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_snapshot.local_path, local_files_only=True, use_fast=True
    )
    expected_manifests = [
        _manifest_for_candidate(candidate, tokenizer)
        for candidate in candidate_manifest["candidates"][:MAXIMUM_QUALIFICATION_CANDIDATES]
    ]
    _verify_candidate_prompt_bindings(prompts, expected_manifests)
    return payload
