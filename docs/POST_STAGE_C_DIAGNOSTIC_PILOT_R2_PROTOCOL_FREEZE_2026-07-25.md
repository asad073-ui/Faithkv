# Post-Stage-C Diagnostic Pilot R2 — Protocol Freeze (2026-07-25)

```text
POST-STAGE-C DIAGNOSTIC PILOT R2 PROTOCOL FROZEN

R1 WAS VOID BEFORE ANY INTERVENTION RAN
NO GAIN OR BEHAVIOUR OUTCOME HAS EVER BEEN OBSERVED
R2 IS A SEPARATELY REPAIRED, TESTED, AUDITED AND AUTHORIZED DIAGNOSTIC EXECUTION

NOT STAGE C
NOT B2B
NOT A RETRY UNDER THE CONSUMED R1 AUTHORIZATION
NO METHOD CLAIM
```

This document freezes the R2 protocol **before** the R2 implementation is
declared complete and **before** any R2 intervention outcome exists. It does
not modify, reinterpret, or supersede
`docs/POST_STAGE_C_DIAGNOSTIC_PILOT_PROTOCOL_FREEZE_2026-07-25.md` (the R1
protocol freeze), which remains immutable.

## 1. Why R2 exists

The R1 diagnostic pilot was executed once under authorization
`post-stage-c-diagnostic-pilot-2026-07-25-one-use` (authorization commit
`079dd092030b51b1f25279ac7a20ffe54d0b50c5`, claim canonical SHA-256
`89e5d93c4a47db89f8da7eea58966574cb3dcf6dab41da4d1c6310b2a18fe9c3`, attempt
ID `e6b024440ff145709a1ba69365143cb2`). It is classified:

```text
DIAGNOSTIC PILOT VOID —
NO SCIENTIFIC INTERPRETATION;
NO RETRY UNDER THE CONSUMED AUTHORIZATION
```

R1 examined two candidates and then raised
`ValueError: qualification requires exactly one selected event` inside the
coordinator's candidate loop:

- **Candidate 0** — unqualified; FullKV answer unverifiable; R-KV hit the
  2,048-token cap; four eligible scored events existed; no intervention ran.
- **Candidate 1** — FullKV correct; fixed-trace R-KV correct; meaningful
  compression; two compaction events; zero eligible event plans;
  `selected_event_count = 0`; correctly unqualified; no intervention ran.

The first-eight scan stopped before candidate 2.

**R1 was void strictly before intervention. No swap gain, no answer margin,
no behavioural readout, and no no-op result was ever produced.** Nothing in
this document, and nothing in R2, may be described as a retry of R1, as a
reinterpretation of an R1 result, or as evidence that R1 produced one.

R1 is **not** classified as mechanically unqualified. It never completed the
qualification scan it was authorized to perform, because of an
implementation defect, not because of the evidence.

## 2. What R2 corrects

### 2.1 The zero-event qualification defect

`mechanically_qualifies` raised whenever `selected_event_count != 1`. That
treated a legitimate mechanically unqualified row —

```json
{
  "eligible_event_exists": false,
  "selected_event_has_two_candidates": false,
  "selected_event_count": 0
}
```

— as malformed input rather than as an ordinary non-qualifier, and the
coordinator called the helper in its candidate loop without treating
zero-event rows as ordinary non-qualifiers.

R2 freezes these semantics:

- **Zero selected events is a valid non-qualifying outcome.** It returns
  `False` and never raises.
- One selected event may still be non-qualifying if its pool contains fewer
  than two candidates.
- More than one selected event is structurally invalid and raises.
- Wrong field types remain structural errors and raise.
- Contradictory field combinations remain structural errors and raise.
- Malformed evidence is never silently converted to `False`.
- The coordinator continues to the next candidate from any legitimate
  non-qualifying row, and completes the full first-eight scan.
- The worker's `qualified` field and the coordinator's independent
  recomputation must agree.

### 2.2 The bounded candidate × restore-width grid

R1 tested every frozen candidate at width one but only the rank-zero
candidate at width two, so candidate × restore-width interaction was
untestable. **Because R1 produced zero intervention outcomes, adding the
missing cells is a pre-outcome correction, not post-hoc threshold movement.**

R2 evaluates the complete bounded factorial grid for the single frozen event
of each selected example:

| | width 1 | width 2 |
|---|---|---|
| candidate rank 0 | ✔ | ✔ |
| candidate rank 1 | ✔ | ✔ |
| candidate rank 2 | ✔ | ✔ |
| candidate rank 3 | ✔ | ✔ |

plus exactly one exact no-op.

```text
maximum pairs per selected example = 4 candidates x 2 widths + 1 no-op = 9
maximum total pairs                = 3 selected examples x 9           = 27
```

Every cell shares: the same event, the same layer, the same intervention
time, the same donor, the same 48-token readout, and the same baseline. No
extra free-running generation, no cross-layer restore, no multi-token
restore. Width two means both KV heads `{0, 1}` — never query heads. The
candidate pool is frozen before any intervention runs.

Each persisted pair record carries `candidate_pool_rank`,
`candidate_absolute_position`, `restore_width`, `kv_head_indices`,
`is_rank_zero_candidate`, `diagnostic_only`, and
`deployable_performance = false`.

### 2.3 Naming

The four-token pool is a **bounded score-prioritized candidate pool**, also
referred to as a **bounded local candidate maximum**. It is never an "oracle
candidate", a "global upper bound", or a "causal oracle".

## 3. What is unchanged

| Item | Frozen value |
|---|---|
| Gain threshold | `0.01` nats — unchanged |
| Model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Model/tokenizer revision | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| Dataset | `HuggingFaceH4/MATH-500`, revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` |
| R-KV revision | `45eaa7d69d20b7388321f077020a610d9afb65bd` |
| Candidate manifest canonical SHA-256 | `b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42` |
| First-eight qualification order | unchanged |
| Maximum qualification candidates | 8 |
| Maximum selected examples | 3 |
| Events per selected example | 1 |
| Maximum selected events | 3 |
| Maximum candidate pool | 4 |
| Restore widths | `[1, 2]` |
| Scored horizon | 48 tokens |
| Runtime limit | 5,400 seconds |
| VRAM limit | 22 GiB |
| Hardware | one RTX 3090, no CPU offload |
| Maximum invocations | 1 |
| Automatic retries | 0 |

Changed relative to R1, and only this: the per-example pair maximum (6 → 9)
and total pair maximum (18 → 27), which follow arithmetically from filling
the missing width-two cells.

## 4. Classification categories

Exactly one category is assigned, in this precedence order. Precedence is
evaluated over the pooled cells of all selected examples, and it is what makes
the "while X does not" qualifiers in B, C and D self-enforcing: by the time B is
reached, no rank-zero width-one cell anywhere exceeded 0.01, so the
same-example qualifier in B is already satisfied; by the time C and D are
reached, no width-one cell anywhere exceeded 0.01, so the same-candidate
qualifier in C is already satisfied. The qualifiers are stated below for
readability and are never evaluated independently of this ordering.

**H. Void.** Execution, evidence, no-op, binding or reconstruction fails. No
scientific interpretation.

**G. Mechanically unqualified.** Fewer than three examples qualify after the
complete first-eight scan. No scientific interpretation.

**A. Current candidate works.** At least one rank-zero width-one gain
exceeds 0.01. *The restoration mechanism exists at 1.5B under the current
deployable candidate; the previous 8B null may be model-, row-, event- or
implementation-regime dependent.*

**B. Candidate-selection rescue.** A non-rank-zero width-one candidate
exceeds 0.01 while the rank-zero width-one candidate on the same example
does not. *Candidate selection is implicated.*

**C. Restore-width rescue.** For the same candidate, width two exceeds 0.01
while width one does not. *Single-head restoration is too narrow.*

**D. Candidate × width interaction.** Only a non-rank-zero width-two
intervention exceeds 0.01, while all width-one candidates remain at or below
0.01 and rank-zero width two remains at or below 0.01. *Candidate choice and
KV-head width interact.*

**E. Readout-only movement.** No NLL gain exceeds 0.01, but a predeclared
answer-margin sign change, fixed-trace extracted-answer change, or
fixed-trace correctness change occurs. *The mean-NLL readout is insensitive
to a behavioural response.*

**F. Flat bounded diagnostic.** No gain exceeds 0.01 and no behavioural
readout changes.

> Single-token, single-layer KV restoration did not move under the frozen
> candidate pool, event selection, intervention time, donor construction,
> 1.5B operating point and fixed readouts.

**A flat R2 result does not kill all single-token KV restoration, and F is
never generalized to the 8B operating point.**

## 5. Reported quantities

The full per-example gain matrix (candidate rank × restore width) is
reported, with at minimum:

- rank-zero width-one gain;
- rank-zero width-two gain;
- best non-rank-zero width-one gain;
- best non-rank-zero width-two gain;
- best gain over the full bounded grid;
- width increment per candidate, `gain_width_2 - gain_width_1`;
- every gain above 0.01;
- fixed-trace behavioural changes;
- the exact no-op.

No favourable post-hoc subset may be selected.

## 6. Scale boundary

A 1.5B result does not settle the 8B operating point. It neither confirms
nor refutes the immutable 8B R2 null.

## 7. Governance

- No B2B is authorized by this document.
- No method is designed, claimed, or implemented by this document.
- A method direction may be proposed only after the R2 result identifies a
  specific, supported failure mechanism — never before.
- R2 execution requires its own separate one-use authorization document,
  exact-SHA CI, and an independent implementation audit.
- Once the R2 claim is created, the R2 authorization is permanently
  consumed; no retry is permitted for any reason.
