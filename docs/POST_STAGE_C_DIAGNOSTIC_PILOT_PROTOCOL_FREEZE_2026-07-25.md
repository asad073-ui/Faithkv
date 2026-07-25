# Post-Stage-C Diagnostic Pilot Protocol Freeze (2026-07-25)

Schema/version identifier:
`faithkv-post-stage-c-diagnostic-pilot-protocol-v1`.

This document freezes a diagnostic pilot only. It does not amend the
immutable R2 result, does not rerun Stage C, does not authorize B2B, and does
not authorize a deployable method claim. Its predecessor is the R2 acceptance
commit `ffc78bad874952e205032845b401df14516f2e69` and
`docs/B2A_R3_STAGE_C_R2_RESULT_ACCEPTANCE_2026-07-25.md`.

## Frozen identities

| Field | Frozen value |
|---|---|
| Repository | `asad073-ui/Faithkv` |
| Pilot branch | `research/post-stage-c-diagnostic-pilot` |
| Starting SHA | `ffc78bad874952e205032845b401df14516f2e69` |
| Model repository | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Model revision | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| Tokenizer repository | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Tokenizer revision | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| Dataset repository/config/split | `HuggingFaceH4/MATH-500`, `default`, `test` |
| Dataset revision | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` |
| Candidate manifest | `configs/discovery/b2a_r3_candidate_manifest.json` |
| Candidate-manifest canonical SHA-256 | `b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42` |
| Candidate-manifest byte SHA-256 | `d76da08e7bafa5aafcab4ff6112165eb14ab08b080c2483527ca4773ef8c0258` |
| R-KV revision | `45eaa7d69d20b7388321f077020a610d9afb65bd` |

The model and tokenizer revisions are exact 40-character Hugging Face commit
SHAs. An unpinned revision such as `main` is invalid.

## Frozen bounds and qualification

- Maximum qualification candidates: 8, the existing manifest's frozen
  `qualification_limit`.
- Maximum selected examples: 3.
- Selected compaction events: exactly one per selected example, maximum 3.
- Maximum Arm-A candidate-pool size: 4.
- Maximum distinct intervention branches per selected example: 6 (up to
  four width-one candidates, one width-two candidate, and one no-op); the
  control reuses the first width-one candidate computation.

Iterate the first eight candidates in canonical manifest order. Select the
first three satisfying all of: valid FullKV execution; mechanically valid
R-KV replay; matching FullKV/R-KV correctness status; meaningful
compression; at least one eligible event; at least two eligible candidate
tokens in the selected event; and no intervention result yet evaluated for
that example. For each selected example choose the eligible event with the
greatest existing deployable event score, breaking ties by lower event index.

For every eligible event, derive its layer/head before score capture using
the repository's existing deterministic `select_layer` and `select_kv_head`
functions, seed 13 and the frozen example/model/R-KV identity. The depth
stratum supplied to `select_layer` is `event_index mod 3`. At that frozen
layer/head, the deployable event score is the greatest existing R-KV final
score among eligible evicted tokens. Break a remaining event tie by lower
event index, then lower layer and KV-head index. This rule consumes no
intervention result.

Qualification and event selection must not inspect swap gain, intervention
answer-token margins, intervention extracted answers, intervention
correctness, or candidate intervention outcomes. If fewer than three qualify
within eight rows, the pilot stops as mechanically unqualified. No later row
may be substituted.

## Frozen intervention arms

The validated architecture must report `num_key_value_heads == 2`. The model
has 12 query-attention heads, but those are not independently restorable KV
heads. The only valid restore widths are one and all KV heads in the same
selected layer; all means the exact set `{0,1}`. Width four and cross-layer
restores are prohibited.

Control is the current score-selected candidate, the selected KV head, and
the unchanged 48-token mean-NLL readout.

Arm A constructs and persists its pool before any intervention. It contains
at most four eligible evicted tokens ordered by existing deployable score
descending and then absolute token index ascending. Every candidate receives
the identical width-one intervention using the same selected event/head,
donor rule, bridge, and scoring window. Every primitive result is reported.
The maximum gain is a bounded candidate-selection upper bound and is always
labeled diagnostic, never deployable performance. No scorer is trained or
introduced.

The donor rule is the existing seeded `select_candidates_and_donors` rule at
the selected event/layer/head; the first returned donor is used for every
candidate and width arm. Thus donor identity is frozen before any
intervention and cannot vary with candidate gain. An event is mechanically
ineligible if that donor absolute identity has no corresponding post-event
slot in either KV head, or if its rank-zero score-selected candidate has no
pre-event K/V identity in either KV head. Both checks occur before the row is
declared qualified or any intervention is evaluated.

Arm B uses only the current score-selected candidate and selected layer. It
restores that candidate into both KV heads `{0,1}` at the corresponding donor
slot for each head, changing only KV-head width. Key and value are both
restored; absolute positions and cache length do not change.

Event ranking uses one fixed-token score replay that immediately reduces each
target capture to candidate identities and deployable scores and releases its
cache tensors; it retains no full-model snapshots. After the event is frozen,
one additional fixed-token replay captures exactly one complete post-event
snapshot for branch evaluation. The selected-event replay must reproduce the
frozen candidate pool exactly. This preserves the all-event ranking rule
without retaining an unbounded population of model-state snapshots.

Arm C reuses each branch's teacher-forced computations. For every answer-span
token position available in the evaluated branch it records the reference
token ID, strongest non-reference token ID, their log probabilities, and the
margin `log p(reference) - log p(strongest non-reference)`. When the frozen
extracted answer is correct, this is the correct-answer-token margin. It also
records the fixed-trace extracted answer and correctness before/after. No
free-running intervention generation is performed; free-running answer flips
are explicitly unavailable and never conflated with fixed-trace evidence.

The no-op restores the existing donor K/V content and identity into itself
through the same primitive. NLL arrays and output state must be bit-exact.

## Scientific rule

The scientific threshold remains the strict `swap_gain > 0.01` nats. It is
not modified by this pilot.

Arm C's predeclared material margin criterion is a strict sign change in at
least one correct-versus-strongest-alternative answer-token margin. A value
equal to zero has no sign and touching zero alone is not success. An actual
extracted-answer change or correctness change also qualifies. This preference
boundary is interpretable in model probabilities and avoids inventing a
small post-hoc floating-point delta.

The pilot is informative if at least one Arm-A bounded upper bound exceeds
0.01, at least one Arm-B all-KV-head gain exceeds 0.01, or Arm C meets its
predeclared criterion while A and B stay below 0.01 everywhere.

If every bounded candidate upper bound and every all-KV-head restore remains
below 0.01 on all three examples and the behavioral readout is unchanged,
the single-token KV-restore mechanism is killed at the 1.5B pilot operating
point. The result does not confirm or refute the immutable 8B R2 null.

## Runtime and execution boundary

| Field | Frozen value |
|---|---|
| Total wall-time ceiling | 5,400 seconds (1.5 GPU-hours) |
| Peak tracked CUDA-memory ceiling | 22 GiB |
| Hardware | exactly one visible NVIDIA GeForce RTX 3090 |
| CPU offload | prohibited |
| Multiple GPUs | prohibited |
| Authorized invocations | 1 |
| Automatic retries | 0 |
| Seed | 13 |
| Generation mode | greedy (`do_sample=false`) |
| Maximum new tokens per natural/replay pass | 2,048 |
| NLL horizon | exactly 48 reference tokens after one unscored bridge token |

The 2,048-token cap bounds the worst-case candidate scan to at most 65,536
natural/replay decode selections (eight rows, one FullKV pass and at most
three diagnostic R-KV passes per row), plus at most 1,029 branch feed steps
(three selected examples, one baseline plus six intervention branches, 49
feeds each). This workload reduction preserves all three arms, the no-op,
and three examples where they mechanically qualify; it is frozen so the
1.5-hour authorization ceiling is not silently enlarged.

The one invocation requires a separate later execution authorization. Claim
consumption permanently spends it, including on CUDA failure, timeout,
disconnect, verification failure, or any other exception. Partial evidence
must be preserved and a retry must never be suggested. No B2B or method claim
is authorized.
