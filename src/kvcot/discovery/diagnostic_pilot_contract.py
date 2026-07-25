"""Pure contracts for the post-Stage-C diagnostic pilot.

This module intentionally imports neither Torch nor Transformers.  It is the
single source of protocol constants used by preparation, dry-run,
authorization verification, workers, summaries, and independent
reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
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

R2_PROTOCOL_DOCUMENT_PATH = (
    "docs/POST_STAGE_C_DIAGNOSTIC_PILOT_R2_PROTOCOL_FREEZE_2026-07-25.md"
)
R2_EXECUTION_AUTHORIZATION_PATH = (
    "docs/POST_STAGE_C_DIAGNOSTIC_PILOT_R2_EXECUTION_AUTHORIZATION_2026-07-25.md"
)

#: The two arms of the R2 bounded factorial grid.  ``RESTORE_ARM`` covers
#: every (candidate rank, restore width) cell; ``NOOP_ARM`` is the single
#: exact mechanical control.  The grid is a bounded score-prioritized
#: candidate pool, never a causal oracle or a global upper bound.
RESTORE_ARM = "restore"
NOOP_ARM = "no_op"
PILOT_ARMS = (RESTORE_ARM, NOOP_ARM)

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

_EXECUTION_COMMAND_PATTERN = re.compile(
    r"^kvcot run-post-stage-c-diagnostic-pilot --authorization-document "
    r"(?P<document>\S+) --execute$"
)


@dataclass(frozen=True)
class DiagnosticGeneration:
    """One dated, self-contained diagnostic-pilot generation.

    R1 is the consumed attempt.  It is retained verbatim so its
    authorization, runtime binding, claim, and attempt stay parseable and
    verifiable forever.  R2 is the separately authorized repaired pilot.
    Every path, hash root, and pair budget that could otherwise collide
    between the two lives here rather than in a single mutable constant.
    """

    label: str
    protocol_document_path: str
    authorization_document_path: str
    default_runtime_root: str
    default_output_root: str
    maximum_pairs_per_selected_example: int

    @property
    def maximum_total_pairs(self) -> int:
        return MAXIMUM_SELECTED_EXAMPLES * self.maximum_pairs_per_selected_example

    @property
    def exact_execution_command(self) -> str:
        return (
            "kvcot run-post-stage-c-diagnostic-pilot --authorization-document "
            f"{self.authorization_document_path} --execute"
        )


R1_GENERATION = DiagnosticGeneration(
    label="r1",
    protocol_document_path=PROTOCOL_DOCUMENT_PATH,
    authorization_document_path=EXECUTION_AUTHORIZATION_PATH,
    default_runtime_root="/workspace/faithkv-post-stage-c-diagnostic-runtime",
    default_output_root="/workspace/faithkv-post-stage-c-diagnostic-execution",
    # Historical R1 grid: every pooled candidate at width one, the rank-zero
    # candidate at width two, and one no-op.
    maximum_pairs_per_selected_example=MAXIMUM_CANDIDATE_POOL_SIZE + 2,
)

R2_GENERATION = DiagnosticGeneration(
    label="r2",
    protocol_document_path=R2_PROTOCOL_DOCUMENT_PATH,
    authorization_document_path=R2_EXECUTION_AUTHORIZATION_PATH,
    default_runtime_root="/workspace/faithkv-post-stage-c-diagnostic-runtime-r2",
    default_output_root="/workspace/faithkv-post-stage-c-diagnostic-execution-r2",
    # R2 bounded factorial grid: every pooled candidate at both restore
    # widths, plus one exact no-op.
    maximum_pairs_per_selected_example=MAXIMUM_CANDIDATE_POOL_SIZE * len(RESTORE_WIDTHS) + 1,
)

GENERATIONS: tuple[DiagnosticGeneration, ...] = (R1_GENERATION, R2_GENERATION)

#: The generation whose intervention grid this implementation actually
#: executes.  Parsing and verification remain available for every
#: generation; execution is deliberately restricted to this one.
EXECUTING_GENERATION = R2_GENERATION

MAXIMUM_PAIRS_PER_SELECTED_EXAMPLE = R2_GENERATION.maximum_pairs_per_selected_example
MAXIMUM_TOTAL_PAIRS = R2_GENERATION.maximum_total_pairs


def generation_by_label(label: str) -> DiagnosticGeneration:
    for generation in GENERATIONS:
        if generation.label == label:
            return generation
    raise ValueError(f"unknown diagnostic-pilot generation label {label!r}")


def generation_for_protocol_document_path(path: str) -> DiagnosticGeneration:
    for generation in GENERATIONS:
        if generation.protocol_document_path == path:
            return generation
    raise ValueError(f"{path!r} is not a frozen diagnostic-pilot protocol document")


def generation_for_authorization_document(path: str | Path) -> DiagnosticGeneration:
    """Resolve the generation an authorization document belongs to.

    The document is identified by its repository-relative path suffix, so a
    verified checkout and a test fixture rooted elsewhere resolve the same
    way while an arbitrary unrelated document never resolves at all.
    """
    resolved = Path(path).resolve().as_posix()
    for generation in GENERATIONS:
        if resolved == generation.authorization_document_path or resolved.endswith(
            f"/{generation.authorization_document_path}"
        ):
            return generation
    raise ValueError(
        f"{path} is not a frozen diagnostic-pilot execution authorization document"
    )


def execution_command_document_argument(command: str) -> str:
    """Extract the ``--authorization-document`` argument of an exact command."""
    match = _EXECUTION_COMMAND_PATTERN.fullmatch(command.strip())
    if match is None:
        raise ValueError("exact_command does not match the frozen execution command shape")
    return match.group("document")


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
    """The frozen R2 mechanism categories A through H.

    Every member maps to exactly one lettered category in the R2 protocol
    freeze.  The categories are evaluated in a fixed precedence order by
    ``classify_pilot`` and never re-derived anywhere else.
    """

    CANDIDATE_WORKS = "current_candidate_works"  # A
    CANDIDATE_RESCUE = "candidate_selection_rescue"  # B
    WIDTH_RESCUE = "restore_width_rescue"  # C
    INTERACTION = "candidate_width_interaction"  # D
    READOUT_ONLY = "readout_only_movement"  # E
    FLAT = "flat_bounded_diagnostic"  # F
    MECHANICALLY_UNQUALIFIED = "mechanically_unqualified"  # G
    VOID = "void"  # H


CLASSIFICATION_LETTER: dict[PilotClassification, str] = {
    PilotClassification.CANDIDATE_WORKS: "A",
    PilotClassification.CANDIDATE_RESCUE: "B",
    PilotClassification.WIDTH_RESCUE: "C",
    PilotClassification.INTERACTION: "D",
    PilotClassification.READOUT_ONLY: "E",
    PilotClassification.FLAT: "F",
    PilotClassification.MECHANICALLY_UNQUALIFIED: "G",
    PilotClassification.VOID: "H",
}

CLASSIFICATION_TEXT: dict[PilotClassification, str] = {
    PilotClassification.CANDIDATE_WORKS: (
        "DIAGNOSTIC PILOT R2 INFORMATIVE (A) —\n"
        "A RANK-ZERO WIDTH-ONE RESTORE EXCEEDED 0.01 NATS;\n"
        "THE RESTORATION MECHANISM EXISTS AT 1.5B UNDER THE CURRENT DEPLOYABLE CANDIDATE;\n"
        "THE PREVIOUS 8B NULL MAY BE MODEL-, ROW-, EVENT- OR IMPLEMENTATION-REGIME DEPENDENT;\n"
        "NO B2B AUTHORIZATION"
    ),
    PilotClassification.CANDIDATE_RESCUE: (
        "DIAGNOSTIC PILOT R2 INFORMATIVE (B) —\n"
        "A NON-RANK-ZERO WIDTH-ONE CANDIDATE EXCEEDED 0.01 NATS WHERE RANK ZERO DID NOT;\n"
        "CANDIDATE SELECTION IS IMPLICATED;\nNO B2B AUTHORIZATION"
    ),
    PilotClassification.WIDTH_RESCUE: (
        "DIAGNOSTIC PILOT R2 INFORMATIVE (C) —\n"
        "FOR THE SAME CANDIDATE, WIDTH TWO EXCEEDED 0.01 NATS WHERE WIDTH ONE DID NOT;\n"
        "SINGLE-HEAD RESTORATION IS TOO NARROW;\nNO B2B AUTHORIZATION"
    ),
    PilotClassification.INTERACTION: (
        "DIAGNOSTIC PILOT R2 INFORMATIVE (D) —\n"
        "ONLY A NON-RANK-ZERO WIDTH-TWO RESTORE EXCEEDED 0.01 NATS;\n"
        "CANDIDATE CHOICE AND KV-HEAD WIDTH INTERACT;\nNO B2B AUTHORIZATION"
    ),
    PilotClassification.READOUT_ONLY: (
        "DIAGNOSTIC PILOT R2 INFORMATIVE (E) —\n"
        "NO GAIN EXCEEDED 0.01 NATS BUT A PREDECLARED BEHAVIOURAL READOUT CHANGED;\n"
        "THE MEAN-NLL READOUT IS INSENSITIVE TO A BEHAVIOURAL RESPONSE;\n"
        "NO B2B AUTHORIZATION"
    ),
    PilotClassification.FLAT: (
        "DIAGNOSTIC PILOT R2 VALID (F) —\n"
        "SINGLE-TOKEN, SINGLE-LAYER KV RESTORATION DID NOT MOVE UNDER THE FROZEN\n"
        "CANDIDATE POOL, EVENT SELECTION, INTERVENTION TIME, DONOR CONSTRUCTION,\n"
        "1.5B OPERATING POINT AND FIXED READOUTS;\n"
        "THIS DOES NOT GENERALIZE TO THE 8B OPERATING POINT;\nNO B2B AUTHORIZATION"
    ),
    PilotClassification.MECHANICALLY_UNQUALIFIED: (
        "DIAGNOSTIC PILOT R2 MECHANICALLY UNQUALIFIED (G) —\n"
        "FEWER THAN THREE EXAMPLES QUALIFIED AFTER THE COMPLETE FIRST-EIGHT SCAN;\n"
        "NO SCIENTIFIC INTERPRETATION;\nNO RETRY UNDER THE CONSUMED AUTHORIZATION"
    ),
    PilotClassification.VOID: (
        "DIAGNOSTIC PILOT R2 VOID (H) —\n"
        "EXECUTION, EVIDENCE, NO-OP, BINDING OR RECONSTRUCTION FAILURE;\n"
        "NO SCIENTIFIC INTERPRETATION;\nNO RETRY UNDER THE CONSUMED AUTHORIZATION"
    ),
}


def classify_pilot(
    *,
    qualified_examples: int,
    rank_zero_width_one_gains: list[float],
    non_rank_zero_width_one_gains: list[float],
    rank_zero_width_two_gains: list[float],
    non_rank_zero_width_two_gains: list[float],
    behavioural_change: bool,
    noop_exact: bool,
    complete: bool,
) -> PilotClassification:
    """Assign exactly one frozen R2 mechanism category.

    Precedence is H, G, then A through F.  The width-two categories are
    only reachable once every width-one cell has stayed at or below the
    threshold, which is what makes C a same-candidate width rescue and D a
    genuine candidate-by-width interaction rather than a relabelling of A
    or B.
    """
    if not noop_exact or not complete:
        return PilotClassification.VOID
    if qualified_examples < MAXIMUM_SELECTED_EXAMPLES:
        return PilotClassification.MECHANICALLY_UNQUALIFIED

    def moved(values: list[float]) -> bool:
        return any(value > SWAP_GAIN_THRESHOLD_NATS for value in values)

    if moved(rank_zero_width_one_gains):
        return PilotClassification.CANDIDATE_WORKS
    if moved(non_rank_zero_width_one_gains):
        return PilotClassification.CANDIDATE_RESCUE
    if moved(rank_zero_width_two_gains):
        return PilotClassification.WIDTH_RESCUE
    if moved(non_rank_zero_width_two_gains):
        return PilotClassification.INTERACTION
    if behavioural_change:
        return PilotClassification.READOUT_ONLY
    return PilotClassification.FLAT


def margin_sign(value: float) -> Literal[-1, 0, 1]:
    return -1 if value < 0 else (1 if value > 0 else 0)


def material_margin_change(before: float | None, after: float | None) -> bool:
    if before is None or after is None:
        return False
    return margin_sign(before) != 0 and margin_sign(after) != 0 and margin_sign(before) != margin_sign(after)
