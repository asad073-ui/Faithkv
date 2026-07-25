"""Pure contracts for the 8B structured restoration geometry pilot
(`docs/FAITHKV_8B_STRUCTURED_RESTORATION_GEOMETRY_PROTOCOL_2026-07-25.md`).

This module intentionally imports neither Torch nor Transformers -- it is
the single source of protocol constants, arm identities, and the frozen
classification precedence used by preparation, dry-run, authorization
verification, workers, summaries, and independent reconstruction.

Unlike the closed 1.5B `diagnostic_pilot_contract` (2 KV heads, a 2-axis
candidate/width grid, classifications A-H), this pilot runs against the
real 8B Stage-C operating point (8 KV heads, 32 layers) and adds layer-
coverage and token-span axes the 1.5B track never had, hence the wider
A-I classification set defined here.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from kvcot.utils.hashing import sha256_json

PROTOCOL_SCHEMA_VERSION = "faithkv-8b-geometry-protocol.v1"
RUNTIME_SCHEMA_VERSION = "faithkv-8b-geometry-runtime.v1"
CLAIM_SCHEMA_VERSION = "faithkv-8b-geometry-claim.v1"
BRANCH_SCHEMA_VERSION = "faithkv-8b-geometry-branch.v1"
SUMMARY_SCHEMA_VERSION = "faithkv-8b-geometry-summary.v1"

REPOSITORY = "asad073-ui/Faithkv"
BRANCH = "research/8b-structured-restoration-geometry"
BASE_SHA = "34f3bca2ff3711e5c575e339e8e5cc7059034512"

MODEL_REPOSITORY = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_REVISION = "6a6f4aa4197940add57724a7707d069478df56b1"
TOKENIZER_REPOSITORY = MODEL_REPOSITORY
TOKENIZER_REVISION = MODEL_REVISION
DATASET_REPOSITORY = "HuggingFaceH4/MATH-500"
DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
RKV_REVISION = "45eaa7d69d20b7388321f077020a610d9afb65bd"

SELECTED_MANIFEST_PATH = "configs/discovery/b2a_one_example_manifest.json"
SELECTED_MANIFEST_SHA256 = "dea628339f4b82678fa18bdc86c8dafa11c5ed87714a8b3a79b588884cab02e0"
SELECTED_UNIQUE_ID = "test/number_theory/631.json"
SELECTED_EXAMPLE_INDEX = 287

R2_CLAIM_CANONICAL_SHA256 = "b63b2ec9f392834dd643910741951029846aade9205d5717b922669d49b2fdde"
R2_ATTEMPT_ID = "73cb2b597b83449eb0d990fcab2741b6"
R2_OBSERVED_EXECUTION_COMMIT_SHA = "0673bbeba25a9b7a6d6e0a78b8a391f36d43f216"

PROTOCOL_DOCUMENT_PATH = "docs/FAITHKV_8B_STRUCTURED_RESTORATION_GEOMETRY_PROTOCOL_2026-07-25.md"
EXECUTION_AUTHORIZATION_PATH = (
    "docs/FAITHKV_8B_STRUCTURED_RESTORATION_GEOMETRY_PRIMARY_AUTHORIZATION_2026-07-25.md"
)
EXACT_EXECUTION_COMMAND = (
    "kvcot run-8b-geometry-pilot --authorization-document "
    f"{EXECUTION_AUTHORIZATION_PATH} --execute"
)

EXPECTED_MODEL_TYPE = "llama"
EXPECTED_NUM_HIDDEN_LAYERS = 32
EXPECTED_QUERY_HEADS = 32
EXPECTED_KV_HEADS = 8
EXPECTED_HEAD_DIM = 128

# --- Frozen anchor event (section 2 of the protocol document) ---
ANCHOR_COMPACTION_EVENT_ID = 5
ANCHOR_LAYER_INDEX = 16
ANCHOR_KV_HEAD_INDEX = 1
ANCHOR_EVENT_TOKEN_ABSOLUTE_POSITION = 1664
ANCHOR_BRIDGE_TOKEN_ABSOLUTE_POSITION = 1665
ANCHOR_FIRST_SCORED_ABSOLUTE_POSITION = 1666

# --- Frozen candidate pool (section 3): rank -> (absolute_position, score_e) ---
CANDIDATE_POOL: tuple[tuple[int, int, float], ...] = (
    (0, 1159, -0.0008087158203125),
    (1, 1168, -0.00080108642578125),
)
RANK_ZERO_CANDIDATE_ABSOLUTE_POSITION = 1159
RANK_ONE_CANDIDATE_ABSOLUTE_POSITION = 1168
PRIMARY_DONOR_ABSOLUTE_POSITION = 192
MAXIMUM_CANDIDATE_POOL_SIZE = 4

SPAN_OFFSETS: tuple[int, int, int] = (-1, 0, 1)

SCORED_HORIZON = 48
SWAP_GAIN_THRESHOLD_NATS = 0.01
RUNTIME_LIMIT_SECONDS = 14_400  # 4.00 GPU-hours ceiling shared with replication
VRAM_LIMIT_BYTES = 22 * 1024**3
FRAMEWORK_SEED = 13
MAXIMUM_INVOCATIONS = 1
AUTOMATIC_RETRIES = 0
MAXIMUM_TOTAL_BRANCHES = 10

# --- Arm identities ---
ARM_C1 = "candidate_rank0"
ARM_C2 = "candidate_rank1"
ARM_C3 = "candidate_rank2"
ARM_C4 = "candidate_rank3"
ARM_H = "all_kv_heads_selected_layer"
ARM_L = "selected_kv_head_valid_layers"
ARM_HL = "all_kv_heads_valid_layers"
ARM_S = "three_token_span_selected_layer_head"
ARM_SH = "span_all_kv_heads_selected_layer"
ARM_NOOP = "no_op"

CANDIDATE_ARMS = (ARM_C1, ARM_C2, ARM_C3, ARM_C4)
ALL_ARMS = (ARM_C1, ARM_C2, ARM_C3, ARM_C4, ARM_H, ARM_L, ARM_HL, ARM_S, ARM_SH, ARM_NOOP)


def canonical_payload_hash(payload: dict[str, Any]) -> str:
    """Hash a payload after omitting its self-hash field (matches the
    `b2a_r3_contract`/`diagnostic_pilot_contract` canonical-hash rule --
    never reinvented a third way in this repository)."""
    return sha256_json({key: value for key, value in payload.items() if key != "canonical_sha256"})


def attach_canonical_hash(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["canonical_sha256"] = canonical_payload_hash(result)
    return result


def verify_canonical_hash(payload: dict[str, Any]) -> None:
    stored = payload.get("canonical_sha256")
    if not isinstance(stored, str) or len(stored) != 64:
        raise ValueError("canonical_sha256 must be 64 lowercase hexadecimal characters")
    expected = canonical_payload_hash(payload)
    if stored != expected:
        raise ValueError(f"canonical_sha256 mismatch: stored={stored} expected={expected}")


def resolve_all_kv_heads(num_key_value_heads: int) -> tuple[int, ...]:
    """Resolve the 'all KV heads' arm width against `num_key_value_heads`
    ONLY, never a query-head count."""
    if num_key_value_heads != EXPECTED_KV_HEADS:
        raise ValueError(f"protocol requires exactly {EXPECTED_KV_HEADS} KV heads, got {num_key_value_heads}")
    return tuple(range(num_key_value_heads))


class GeometryPilotClassification(str, Enum):
    """The frozen A-I mechanism categories (task section 11), evaluated in
    a fixed precedence order by `classify_geometry_pilot` and never
    re-derived anywhere else."""

    ORIGINAL_CANDIDATE_WORKS = "original_candidate_works"  # A
    CANDIDATE_SELECTION_RESCUE = "candidate_selection_rescue"  # B
    HEAD_COVERAGE_RESCUE = "head_coverage_rescue"  # C
    LAYER_COVERAGE_RESCUE = "layer_coverage_rescue"  # D
    HEAD_LAYER_INTERACTION = "head_layer_interaction"  # E
    TOKEN_SPAN_RESCUE = "token_span_rescue"  # F
    READOUT_ONLY_MOVEMENT = "readout_only_movement"  # G
    STRUCTURED_RESTORATION_FLAT = "structured_restoration_flat"  # H
    MECHANICALLY_INVALID = "mechanically_invalid"  # I


CLASSIFICATION_LETTER: dict[GeometryPilotClassification, str] = {
    GeometryPilotClassification.ORIGINAL_CANDIDATE_WORKS: "A",
    GeometryPilotClassification.CANDIDATE_SELECTION_RESCUE: "B",
    GeometryPilotClassification.HEAD_COVERAGE_RESCUE: "C",
    GeometryPilotClassification.LAYER_COVERAGE_RESCUE: "D",
    GeometryPilotClassification.HEAD_LAYER_INTERACTION: "E",
    GeometryPilotClassification.TOKEN_SPAN_RESCUE: "F",
    GeometryPilotClassification.READOUT_ONLY_MOVEMENT: "G",
    GeometryPilotClassification.STRUCTURED_RESTORATION_FLAT: "H",
    GeometryPilotClassification.MECHANICALLY_INVALID: "I",
}


def classify_geometry_pilot(
    *,
    evidence_complete: bool,
    noop_exact: bool,
    rank_zero_gain: float | None,
    non_rank_zero_gains: list[float],
    head_coverage_gain: float | None,
    layer_coverage_gain: float | None,
    head_layer_interaction_gain: float | None,
    span_gain: float | None,
    span_head_gain: float | None,
    span_available: bool,
    layer_set_available: bool,
    behavioural_change: bool,
) -> GeometryPilotClassification:
    """Assign exactly one frozen A-I category, precedence I, then A-H in
    the task's own listed order. Every `*_gain` argument is a swap gain in
    nats (mean(baseline NLL) - mean(swapped NLL)) already recomputed from
    primitive per-token arrays -- this function never recomputes a gain
    itself, it only compares already-computed numbers against the frozen
    `SWAP_GAIN_THRESHOLD_NATS`."""
    if not evidence_complete or not noop_exact:
        return GeometryPilotClassification.MECHANICALLY_INVALID

    def moved(value: float | None) -> bool:
        return value is not None and value > SWAP_GAIN_THRESHOLD_NATS

    if moved(rank_zero_gain):
        return GeometryPilotClassification.ORIGINAL_CANDIDATE_WORKS
    if any(moved(g) for g in non_rank_zero_gains):
        return GeometryPilotClassification.CANDIDATE_SELECTION_RESCUE
    if moved(head_coverage_gain):
        return GeometryPilotClassification.HEAD_COVERAGE_RESCUE
    if layer_set_available and moved(layer_coverage_gain):
        return GeometryPilotClassification.LAYER_COVERAGE_RESCUE
    if layer_set_available and moved(head_layer_interaction_gain):
        return GeometryPilotClassification.HEAD_LAYER_INTERACTION
    if span_available and (moved(span_gain) or moved(span_head_gain)):
        return GeometryPilotClassification.TOKEN_SPAN_RESCUE
    if behavioural_change:
        return GeometryPilotClassification.READOUT_ONLY_MOVEMENT
    return GeometryPilotClassification.STRUCTURED_RESTORATION_FLAT


CLASSIFICATION_TEXT: dict[GeometryPilotClassification, str] = {
    GeometryPilotClassification.ORIGINAL_CANDIDATE_WORKS: (
        "8B GEOMETRY PILOT INFORMATIVE (A) —\n"
        "THE ORIGINAL RANK-ZERO SINGLE-HEAD/SINGLE-LAYER RESTORE EXCEEDED 0.01 NATS;\n"
        "THIS REQUIRES RECONCILIATION WITH THE ORIGINAL R2 NULL BEFORE ANY METHOD;\n"
        "NO B2B AUTHORIZATION"
    ),
    GeometryPilotClassification.CANDIDATE_SELECTION_RESCUE: (
        "8B GEOMETRY PILOT INFORMATIVE (B) —\n"
        "A NON-RANK-ZERO CANDIDATE EXCEEDED 0.01 NATS WHERE RANK ZERO DID NOT;\n"
        "CANDIDATE SELECTION IS IMPLICATED;\nNO B2B AUTHORIZATION"
    ),
    GeometryPilotClassification.HEAD_COVERAGE_RESCUE: (
        "8B GEOMETRY PILOT INFORMATIVE (C) —\n"
        "ALL-KV-HEAD RESTORE AT THE SELECTED LAYER EXCEEDED 0.01 NATS WHERE\n"
        "SINGLE-HEAD RESTORE DID NOT; SINGLE-HEAD RESTORATION IS TOO NARROW;\n"
        "NO B2B AUTHORIZATION"
    ),
    GeometryPilotClassification.LAYER_COVERAGE_RESCUE: (
        "8B GEOMETRY PILOT INFORMATIVE (D) —\n"
        "RESTORE ACROSS THE VALID LAYER SET EXCEEDED 0.01 NATS WHERE THE\n"
        "SINGLE-LAYER ARM DID NOT; THE LOAD-BEARING STATE IS LAYER-DISTRIBUTED;\n"
        "NO B2B AUTHORIZATION"
    ),
    GeometryPilotClassification.HEAD_LAYER_INTERACTION: (
        "8B GEOMETRY PILOT INFORMATIVE (E) —\n"
        "ONLY THE COMBINED HEAD-AND-LAYER RESTORE EXCEEDED 0.01 NATS;\n"
        "NEITHER AXIS ALONE IS SUFFICIENT;\nNO B2B AUTHORIZATION"
    ),
    GeometryPilotClassification.TOKEN_SPAN_RESCUE: (
        "8B GEOMETRY PILOT INFORMATIVE (F) —\n"
        "A THREE-TOKEN SPAN RESTORE EXCEEDED 0.01 NATS WHERE THE SINGLE-TOKEN\n"
        "GEOMETRY DID NOT; THE CAUSAL UNIT IS A LOCAL TOKEN SPAN;\n"
        "NO B2B AUTHORIZATION"
    ),
    GeometryPilotClassification.READOUT_ONLY_MOVEMENT: (
        "8B GEOMETRY PILOT INFORMATIVE (G) —\n"
        "NO GAIN EXCEEDED 0.01 NATS BUT A PREDECLARED FIXED-TRACE BEHAVIOURAL\n"
        "READOUT CHANGED; THE MEAN-NLL READOUT IS INSENSITIVE TO A BEHAVIOURAL\n"
        "RESPONSE;\nNO B2B AUTHORIZATION"
    ),
    GeometryPilotClassification.STRUCTURED_RESTORATION_FLAT: (
        "8B GEOMETRY PILOT VALID (H) —\n"
        "EVERY AVAILABLE CAUSAL BRANCH REMAINED AT OR BELOW 0.01 NATS, NO\n"
        "BEHAVIOURAL READOUT CHANGED, AND THE NO-OP WAS EXACT;\n"
        "SINGLE-TOKEN AND LOCAL-SPAN RESTORATION ARE UNSUPPORTED UNDER THE\n"
        "FROZEN PRIMARY-ROW CONDITIONS; THIS DOES NOT GENERALIZE BEYOND THEM;\n"
        "NO B2B AUTHORIZATION"
    ),
    GeometryPilotClassification.MECHANICALLY_INVALID: (
        "8B GEOMETRY PILOT MECHANICALLY INVALID (I) —\n"
        "MUTATION VALIDITY, NO-OP, CACHE SHAPE, MODEL BINDING, PRIMITIVE\n"
        "RECONSTRUCTION, RUNTIME, MEMORY, OR EVIDENCE INTEGRITY FAILED;\n"
        "NO SCIENTIFIC INTERPRETATION;\nNO RETRY UNDER THE CONSUMED AUTHORIZATION"
    ),
}


def margin_sign(value: float) -> Literal[-1, 0, 1]:
    return -1 if value < 0 else (1 if value > 0 else 0)


def material_margin_change(before: float | None, after: float | None) -> bool:
    if before is None or after is None:
        return False
    return margin_sign(before) != 0 and margin_sign(after) != 0 and margin_sign(before) != margin_sign(after)
