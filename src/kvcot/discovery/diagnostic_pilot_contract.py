"""Pure contracts for the post-Stage-C diagnostic pilot.

This module intentionally imports neither Torch nor Transformers.  It is the
single source of protocol constants used by preparation, dry-run,
authorization verification, workers, summaries, and independent
reconstruction.
"""
from __future__ import annotations

from enum import Enum
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kvcot.utils.hashing import sha256_json

PROTOCOL_SCHEMA_VERSION = "faithkv-post-stage-c-diagnostic-pilot-protocol-v1"
RUNTIME_SCHEMA_VERSION = "faithkv-post-stage-c-diagnostic-pilot-runtime-v1"
CLAIM_SCHEMA_VERSION = "faithkv-post-stage-c-diagnostic-pilot-claim-v1"
PAIR_SCHEMA_VERSION = "faithkv-post-stage-c-diagnostic-pilot-pair-v1"
SUMMARY_SCHEMA_VERSION = "faithkv-post-stage-c-diagnostic-pilot-summary-v1"

REPOSITORY = "asad073-ui/Faithkv"
BRANCH = "research/post-stage-c-diagnostic-pilot"
STARTING_SHA = "ffc78bad874952e205032845b401df14516f2e69"
MODEL_REPOSITORY = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
TOKENIZER_REPOSITORY = MODEL_REPOSITORY
TOKENIZER_REVISION = MODEL_REVISION
DATASET_REPOSITORY = "HuggingFaceH4/MATH-500"
DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
RKV_REVISION = "45eaa7d69d20b7388321f077020a610d9afb65bd"
CANDIDATE_MANIFEST_PATH = "configs/discovery/b2a_r3_candidate_manifest.json"
CANDIDATE_MANIFEST_CANONICAL_SHA256 = "b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42"
CANDIDATE_MANIFEST_BYTE_SHA256 = "d76da08e7bafa5aafcab4ff6112165eb14ab08b080c2483527ca4773ef8c0258"
PROTOCOL_DOCUMENT_PATH = "docs/POST_STAGE_C_DIAGNOSTIC_PILOT_PROTOCOL_FREEZE_2026-07-25.md"
CONFIG_PATH = "configs/discovery/qwen15b_math500_b1024_diagnostic.yaml"
EXECUTION_AUTHORIZATION_PATH = "docs/POST_STAGE_C_DIAGNOSTIC_PILOT_EXECUTION_AUTHORIZATION_2026-07-25.md"
EXACT_EXECUTION_COMMAND = (
    "kvcot run-post-stage-c-diagnostic-pilot --authorization-document "
    f"{EXECUTION_AUTHORIZATION_PATH} --execute"
)

MAXIMUM_QUALIFICATION_CANDIDATES = 8
MAXIMUM_SELECTED_EXAMPLES = 3
MAXIMUM_EVENTS_PER_EXAMPLE = 1
MAXIMUM_CANDIDATE_POOL_SIZE = 4
EXPECTED_QUERY_HEADS = 12
EXPECTED_KV_HEADS = 2
RESTORE_WIDTHS = (1, 2)
SCORED_HORIZON = 48
SWAP_GAIN_THRESHOLD_NATS = 0.01
RUNTIME_LIMIT_SECONDS = 5_400
VRAM_LIMIT_BYTES = 22 * 1024**3
FRAMEWORK_SEED = 13
MAXIMUM_INVOCATIONS = 1
AUTOMATIC_RETRIES = 0

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def canonical_payload_hash(payload: dict[str, Any]) -> str:
    """Hash a payload after omitting its self-hash field."""
    return sha256_json({key: value for key, value in payload.items() if key != "canonical_sha256"})


def attach_canonical_hash(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["canonical_sha256"] = canonical_payload_hash(result)
    return result


def verify_canonical_hash(payload: dict[str, Any]) -> None:
    stored = payload.get("canonical_sha256")
    if not isinstance(stored, str) or _HEX64.fullmatch(stored) is None:
        raise ValueError("canonical_sha256 must be 64 lowercase hexadecimal characters")
    expected = canonical_payload_hash(payload)
    if stored != expected:
        raise ValueError(f"canonical_sha256 mismatch: stored={stored} expected={expected}")


class ModelArchitecture(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    model_type: str
    num_hidden_layers: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_key_value_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    torch_dtype: str
    parameter_placement: dict[str, Any]
    model_revision: str
    tokenizer_revision: str

    @field_validator("model_revision", "tokenizer_revision")
    @classmethod
    def revision_is_pinned(cls, value: str) -> str:
        if _HEX40.fullmatch(value) is None:
            raise ValueError("revision must be an exact 40-character lowercase commit SHA")
        return value

    @model_validator(mode="after")
    def protocol_binding(self) -> "ModelArchitecture":
        if self.num_attention_heads != EXPECTED_QUERY_HEADS:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} does not match {EXPECTED_QUERY_HEADS}"
            )
        if self.num_key_value_heads != EXPECTED_KV_HEADS:
            raise ValueError(
                f"num_key_value_heads={self.num_key_value_heads} does not match {EXPECTED_KV_HEADS}"
            )
        if self.model_revision != MODEL_REVISION or self.tokenizer_revision != TOKENIZER_REVISION:
            raise ValueError("model/tokenizer revision does not match the frozen protocol")
        return self


def resolve_restore_heads(width: int, *, selected_head: int, num_key_value_heads: int) -> tuple[int, ...]:
    """Resolve a protocol width against KV heads, never query heads."""
    if num_key_value_heads != EXPECTED_KV_HEADS:
        raise ValueError(f"protocol requires exactly {EXPECTED_KV_HEADS} KV heads")
    if not 0 <= selected_head < num_key_value_heads:
        raise ValueError("selected KV head is out of range")
    if width == 1:
        return (selected_head,)
    if width == num_key_value_heads == EXPECTED_KV_HEADS:
        return tuple(range(num_key_value_heads))
    raise ValueError(f"restore width {width} is prohibited; valid widths are {RESTORE_WIDTHS}")


class PilotClassification(str, Enum):
    CANDIDATE = "candidate_selection_implicated"
    WIDTH = "restore_dilution_implicated"
    BOTH = "candidate_and_width_implicated"
    READOUT = "readout_implicated"
    KILLED_15B = "mechanism_killed_at_1_5b"
    MECHANICALLY_UNQUALIFIED = "mechanically_unqualified"
    VOID = "void"


CLASSIFICATION_TEXT: dict[PilotClassification, str] = {
    PilotClassification.CANDIDATE: (
        "DIAGNOSTIC PILOT INFORMATIVE —\nBOUNDED CANDIDATE UPPER BOUND MOVED;\n"
        "CANDIDATE SELECTION IS A PLAUSIBLE BOTTLENECK;\nNO B2B AUTHORIZATION"
    ),
    PilotClassification.WIDTH: (
        "DIAGNOSTIC PILOT INFORMATIVE —\nALL-KV-HEAD RESTORE MOVED;\n"
        "SINGLE-HEAD RESTORE IS TOO NARROW;\nNO B2B AUTHORIZATION"
    ),
    PilotClassification.BOTH: (
        "DIAGNOSTIC PILOT INFORMATIVE —\nCANDIDATE CHOICE AND RESTORE WIDTH BOTH MOVED;\n"
        "INTERACTION TEST REQUIRED BEFORE METHOD DESIGN;\nNO B2B AUTHORIZATION"
    ),
    PilotClassification.READOUT: (
        "DIAGNOSTIC PILOT INFORMATIVE —\nNLL GATE REMAINED FLAT BUT BEHAVIOURAL READOUT MOVED;\n"
        "READOUT REQUIRES REDEFINITION BEFORE METHOD DESIGN;\nNO B2B AUTHORIZATION"
    ),
    PilotClassification.KILLED_15B: (
        "DIAGNOSTIC PILOT VALID —\nCANDIDATE UPPER BOUND, ALL-HEAD RESTORE, AND BEHAVIOURAL READOUT FLAT;\n"
        "SINGLE-TOKEN KV RESTORE KILLED AT THE 1.5B OPERATING POINT;\nNO B2B"
    ),
    PilotClassification.MECHANICALLY_UNQUALIFIED: (
        "DIAGNOSTIC PILOT MECHANICALLY UNQUALIFIED —\nFEWER THAN THREE VALID EXAMPLES;\n"
        "NO SCIENTIFIC INTERPRETATION;\nNO RETRY UNDER THE CONSUMED AUTHORIZATION"
    ),
    PilotClassification.VOID: (
        "DIAGNOSTIC PILOT VOID —\nNO-OP, EVIDENCE, BINDING, OR EXECUTION FAILURE;\n"
        "NO SCIENTIFIC INTERPRETATION;\nNO RETRY UNDER THE CONSUMED AUTHORIZATION"
    ),
}


def classify_pilot(
    *,
    qualified_examples: int,
    arm_a_gains: list[float],
    arm_b_gains: list[float],
    behavioural_change: bool,
    noop_exact: bool,
    complete: bool,
) -> PilotClassification:
    if not noop_exact or not complete:
        return PilotClassification.VOID
    if qualified_examples < MAXIMUM_SELECTED_EXAMPLES:
        return PilotClassification.MECHANICALLY_UNQUALIFIED
    a_moved = any(value > SWAP_GAIN_THRESHOLD_NATS for value in arm_a_gains)
    b_moved = any(value > SWAP_GAIN_THRESHOLD_NATS for value in arm_b_gains)
    if a_moved and b_moved:
        return PilotClassification.BOTH
    if a_moved:
        return PilotClassification.CANDIDATE
    if b_moved:
        return PilotClassification.WIDTH
    if behavioural_change:
        return PilotClassification.READOUT
    return PilotClassification.KILLED_15B


def margin_sign(value: float) -> Literal[-1, 0, 1]:
    return -1 if value < 0 else (1 if value > 0 else 0)


def material_margin_change(before: float | None, after: float | None) -> bool:
    if before is None or after is None:
        return False
    return margin_sign(before) != 0 and margin_sign(after) != 0 and margin_sign(before) != margin_sign(after)
