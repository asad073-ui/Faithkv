# FaithKV 8B Structured Restoration Geometry Protocol (2026-07-25)

Schema/version identifier: `faithkv-8b-geometry-protocol.v1`.

This document freezes a bounded mechanism-discovery pilot at the actual 8B
Stage-C operating point. It does not amend the immutable R2 result, does not
rerun Stage C, does not authorize B2B, does not change the frozen
`0.01`-nat mechanism threshold, and does not authorize a deployable method
claim. Its predecessors are `docs/B2A_R3_STAGE_C_R2_RESULT_ACCEPTANCE_2026-07-25.md`
(the immutable 8B null) and the closed 1.5B diagnostic line
(`docs/POST_STAGE_C_DIAGNOSTIC_PILOT_R2_RESULT_2026-07-25.md`, classification
G — mechanically unqualified, not reopened or reused here).

The purpose of this pilot is narrow: determine whether the original 8B null
was caused by candidate choice, KV-head coverage, layer coverage, token-span
coverage, or an insensitive readout — never to establish a method directly.

## 1. Frozen identities (bound to `authoritative-8b-operating-point.json`)

All values in this table were independently recomputed in this session from
primitive stored quantities in a read-only evidence snapshot extracted from
the two required preserved 8B archives, never copied from prose summaries.
Full derivation, per-pair identities, and hash bindings live in
`/workspace/faithkv-8b-geometry/authoritative-8b-operating-point.json`
(sha256 `89b2a866adee2780087457d90d569744f5e2d3a1eb8912b515af83fd4ae93ed0`).

| Field | Frozen value |
|---|---|
| Repository | `asad073-ui/Faithkv` |
| Geometry branch | `research/8b-structured-restoration-geometry` |
| Geometry branch base SHA | `34f3bca2ff3711e5c575e339e8e5cc7059034512` |
| R2 authorized code commit | `b6a71afb0a8538a43775b94f3b821c890f99db74` |
| R2 observed execution commit | `0673bbeba25a9b7a6d6e0a78b8a391f36d43f216` |
| R2 claim canonical SHA-256 | `b63b2ec9f392834dd643910741951029846aade9205d5717b922669d49b2fdde` |
| R2 attempt ID | `73cb2b597b83449eb0d990fcab2741b6` |
| Model repository | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| Model revision | `6a6f4aa4197940add57724a7707d069478df56b1` |
| Tokenizer repository/revision | same as model |
| Model type | `llama`; 32 hidden layers; 32 attention heads; **8 KV heads**; head_dim 128; bf16 |
| Dataset repository/config/split | `HuggingFaceH4/MATH-500`, `default`, `test` |
| Dataset revision | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` |
| Selected row | `test/number_theory/631.json` (`example_index=287`) |
| Selected-manifest SHA-256 | `dea628339f4b82678fa18bdc86c8dafa11c5ed87714a8b3a79b588884cab02e0` |
| R-KV revision | `45eaa7d69d20b7388321f077020a610d9afb65bd` |
| Compaction budget | 1024 |
| Generation | greedy, `do_sample=false`, batch size 1, framework seed 13, `flash_attention_2`, `DynamicCache` |
| Scored readout horizon | 48 tokens, greedy, deterministic |
| Frozen mechanism threshold | `0.01` nats — **unchanged, never re-derived here** |

The model has **8 KV heads**, not 2 (the closed 1.5B track's architecture).
Every head-coverage arm below is defined against this repository's own
`num_key_value_heads=8`, resolved from the live model config, never assumed.

## 2. Anchor event (deterministic, outcome-blind selection)

R2 exercised three compaction events (5, 9, 13), each contributing 4 real
pairs (2 candidates × 2 donors) plus, at event 5 only, the mandatory no-op
control. This protocol's 10 new bounded branches are anchored to **exactly
one** of those three events — chosen by a rule applied to already-finalized
R2 evidence, before any new intervention outcome exists:

1. Event 5 has the lowest `compaction_event_id` and lowest
   `chronological_event_ordinal` (0) — the same lowest-ID / first-drawn
   tie-break convention already used elsewhere in this repository's
   deterministic selection code.
2. Event 5 is the only one of the three whose two tested candidates have a
   genuinely discriminating deployable score (`score_e`): event 9's two
   candidates are exactly score-tied, so no rank-zero/rank-one distinction
   is even possible there.
3. Event 5 is the only one of the three with an existing, already-validated
   no-op control pair in the immutable R2 evidence, at the identical
   (layer, head, donor) locus this protocol reuses.

**Frozen anchor:** compaction event 5, layer index 16, KV head index 1,
event token absolute position 1664, bridge token absolute position 1665,
first scored absolute position 1666.

Events 9 and 13 are not reselected, not rerun, and not substituted for event
5 by this protocol; they remain part of the original R2 evidence and the
null-diagnosis document only.

## 3. Frozen candidate pool (bounded score-prioritized candidate pool)

Ordered only by `score_e` (the deployable, pre-intervention R-KV eviction
score already recomputed and stored on each R2 pair record — **not** the
`oracle_non_deployable` schema marker, which is a static flag on optional
oracle-only fields and does not describe `score_e`/`score_r` themselves).
Lower (more negative) `score_e` is higher R-KV eviction priority.

| Rank | Candidate absolute position | `score_e` |
|---|---|---|
| 0 | 1159 | −0.0008087158203125 |
| 1 | 1168 | −0.00080108642578125 |

Both were already causally tested (against both donors) in R2; no new
scoring computation is performed to reach this pool, and it is not extended
beyond these two (fewer than the maximum of 4 is permitted — inventing two
more untested candidates at protocol-freeze time would not be evidence-
grounded). This is a **bounded score-prioritized candidate pool**, not an
oracle candidate pool, not a global upper bound, not a causal oracle.

**Primary donor:** absolute position 192 (the existing no-op locus).
**Alternate donor** (evidence only, not used by the new arms below): 1629.

## 4. Frozen structured arms (maximum 10 branches)

All arms share: layer 16 anchor unless stated otherwise, KV head 1 anchor
unless stated otherwise, donor 192, event token 1664 / bridge 1665 / first
scored 1666, 48-token readout, identical mutation-time semantics to the
immutable R2 pairs.

| Arm | Candidate(s) | Layer(s) | KV head(s) | Purpose |
|---|---|---|---|---|
| C1 | 1159 (rank 0) | 16 | 1 | Rank-zero control (== original R2 pair) |
| C2 | 1168 (rank 1) | 16 | 1 | Candidate-selection rescue test |
| H | 1159 | 16 | **all 8** | Head-coverage rescue test |
| L | 1159 | **valid-layer set** (§5) | 1 | Layer-coverage rescue test |
| HL | 1159 | valid-layer set (§5) | all 8 | Head × layer interaction test |
| S | 1158, 1159, 1160 | 16 | 1 | Token-span rescue test (only if structurally available, §5) |
| SH | 1158, 1159, 1160 | 16 | all 8 | Span × head interaction test (only if S is available) |
| no-op | 192 | 16 | 1 | Exact mechanical control (candidate == donor == 192) |

Only 2 candidate branches exist (C1, C2) because the frozen pool (§3) has
only 2 members — the protocol permits up to 4 candidate branches, it does
not require exactly 4. Total maximum branches: **8** (2 candidate + 1 H + 1
L + 1 HL + 1 S + 1 SH + 1 no-op), within the 10-branch ceiling.

Arm C1 is byte-for-byte the same intervention as the original R2 pair
(candidate 1159, donor 192, layer 16, head 1) and is expected to reproduce
that pair's swap gain exactly, as a harness-correctness check before the new
arms are trusted.

## 5. Layer-set and span-availability freezing (mechanical, pre-outcome)

Determining which layers still hold both the candidate (1159) and the donor
(192) absolute positions in KV head 1's cache at token step 1664 — and
whether token positions 1158 and 1160 have valid pre-event snapshots — is a
**structural fact about the frozen row**, not a causal outcome. It can only
be read after the pristine multi-layer snapshot for event 5 is captured
(unavoidable — no layer's retained-position set is known before the model
actually runs), but it must be, and is, computed and persisted in a
dedicated pre-outcome artifact (`layer_set.json` /
`span_availability.json` in the attempt directory) **before any branch's
NLL, swap gain, or other outcome is computed**. No branch's outcome is
permitted to change either set after it is written. If fewer than 2 layers
qualify for §4's Arm L, Arm L (and therefore Arm HL) is recorded as
structurally unavailable rather than run against a smaller ad hoc set. If
either neighbor position (1158, 1160) lacks a valid snapshot or falls
outside the eligible evicted region, Arm S (and therefore Arm SH) is
recorded as structurally unavailable — never substituted with a different
span.

Valid-layer resolution and span-availability resolution use only the
already-captured event-5 pristine snapshot's own per-layer provenance
(`snapshot.provenance.layers[L].positions[head, :]`, which this repository's
existing `kvcot.generation.provenance`/`kvcot.discovery.swap` machinery
already maintains for every layer continuously, not only on a layer's own
compaction step) — no new model forward passes, no new capture requests,
and no future information relative to token step 1664.

## 6. Required restore primitive

`kvcot.discovery.geometry_pilot_restore.apply_structured_kv_restore` accepts
a flat, non-empty list of `(layer_index, kv_head_index, token_position,
replacement_key, replacement_value)` mutations against one caller-owned
snapshot clone, generalizing this repository's existing single-slot
primitive (`kvcot.discovery.swap.apply_within_head_swap_owned`) and existing
per-layer multi-head primitive (`kvcot.discovery.diagnostic_pilot_swap
.apply_diagnostic_kv_restore`) rather than reimplementing shape/dtype/
device/aliasing validation a third time. Every replacement key/value comes
from that same pristine snapshot's own captured tensors at the resolved
physical slot for the stated `(layer_index, kv_head_index)` — never a
tensor sourced from a different snapshot, layer, or head than the one
being restored. Duplicate `(layer, head, position)` triples, empty
mutation lists, and out-of-range indices are rejected before any write.
Unselected layers, heads, and token positions are provably unchanged
(verified in CPU tests by shape/content diffing every layer/head/slot not
named in the mutation list). The exact no-op case is the same code path as
every other mutation, not a special case.

## 7. Readouts and classification

Every branch persists the full primitive evidence listed in the task's
required-readouts section (candidate/layer/head/donor identities, key/value
slots changed, baseline and intervention per-token NLL arrays, swap gain,
peak absolute per-token NLL change, first-token NLL change, 8- and 16-token
subwindow gains, runtime, VRAM). No free-running generation is performed by
this pilot; only fixed-trace mean-NLL and (where the row's existing
extraction procedure permits) fixed-trace margin/extracted-answer evidence
are captured — a free-running answer flip is never claimed.

Classification uses exactly the nine categories A–I defined in the task
instructions (§11), evaluated in the fixed precedence order I (mechanically
invalid, checked first — an incomplete or non-exact-no-op result is never
given a scientific interpretation) → A → B → C → D → E → F → G → H,
implemented in `kvcot.discovery.geometry_pilot_contract
.classify_geometry_pilot` and enforced by that module's own test suite. The
`0.01`-nat threshold is unchanged; no new threshold is introduced by this
protocol.

## 8. Bounds

- Maximum branches: 10 (this pilot uses 8, per §4).
- Maximum GPU runtime for discovery + replication combined: 4.00 GPU-hours.
- Hardware: exactly one RTX 3090, no CPU/disk/meta offload, peak tracked
  CUDA memory ≤ 22 GiB.
- Maximum invocations of the primary authorized command: 1. Zero automatic
  retries.
- No B2B execution or authorization of any kind.
- No method claim is authorized from this pilot's primary example alone
  (task §20 replication gate governs whether replication is even
  attempted).

## 9. Non-goals restated

This protocol does not reopen, rerun, or reinterpret the closed 1.5B
diagnostic line. It does not change the `0.01`-nat threshold, the R2 null
classification, or any Stage-C evidence. It does not itself authorize GPU
execution — a separate, later, one-use, dated authorization document is
required (task §17) before any CUDA initialization occurs under this
protocol.
