# Post-Stage-C Diagnostic Pilot R2 — Result (2026-07-25)

```text
DIAGNOSTIC PILOT R2 MECHANICALLY UNQUALIFIED (G) —
FEWER THAN THREE EXAMPLES QUALIFIED AFTER THE COMPLETE FIRST-EIGHT SCAN;
NO SCIENTIFIC INTERPRETATION;
NO RETRY UNDER THE CONSUMED AUTHORIZATION

ZERO EXAMPLES QUALIFIED. ZERO INTERVENTIONS RAN.
NO GAIN, MARGIN, OR BEHAVIOURAL OUTCOME EXISTS.
NO B2B. NO METHOD CLAIM. THIS SAYS NOTHING ABOUT 8B.
```

## 1. Execution identity

| Item | Value |
|---|---|
| Authorization ID | `post-stage-c-diagnostic-pilot-r2-2026-07-25-one-use` |
| Authorization commit | `40fae9465877ae4f55c50c9b2fdecc9015443129` |
| Audited implementation SHA | `d4ad573c67615daaa9fa62e51de39abf874b2e48` |
| Attempt ID | `e8b8fa846b5640a3836192038faf2bca` |
| Claim canonical SHA-256 | `5029f809c023056e4e566c2b5938287b61309fe19a1a3a4c59afb879fe6a9ec6` |
| R2 protocol SHA-256 | `8d5fd74e9a94ab5ba067ed7c6ffd9c23590593b1131813f53317693e37e2f3ee` |
| R2 runtime canonical SHA-256 | `e2ea738c89a22a76f3cb71f5b4e01136a519183f916779908c1647a7d8a7c200` |
| Candidate manifest canonical SHA-256 | `b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42` |
| Model / tokenizer revision | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| R-KV revision | `45eaa7d69d20b7388321f077020a610d9afb65bd` |
| `final.json` SHA-256 | `0f8b3f5a7a54de2f4e491f6f7531be599fd5113c4429175074ed5e5ccafe0f2a` |
| Exit code | 0 |
| Started / ended (UTC) | 06:10:06.573 → 06:56:10.535 |
| Runtime | 2,757.9 s (limit 5,400 s) |
| Peak tracked CUDA | 3,829,399,552 B = 3.57 GiB (limit 22 GiB) |
| Invocations | 1 — the authorization is permanently consumed |

## 2. Accepted classification

**G — MECHANICALLY UNQUALIFIED.**

Zero of eight candidates qualified. The protocol requires three. **No
scientific interpretation of any kind is permitted from this run.** Nothing
here supports or refutes any claim about single-token KV restoration, about
candidate selection, about restore width, or about the 8B operating point.

The run itself was mechanically clean: exit code 0, complete first-eight scan,
runtime and VRAM far inside limits, all evidence reconstructable, R1 evidence
untouched. It is unqualified because the *rows* did not meet the frozen
conditions, not because the execution failed.

## 3. Complete first-eight qualification scan

All eight candidates were attempted in frozen manifest order. No candidate was
skipped and the scan did not stop early.

| Ord | Row | Cap hit | Compaction events | Eligible plans | Scored events | Selected events | FullKV / R-KV answer | Failed conditions |
|---|---|---|---|---|---|---|---|---|
| 0 | `test/precalculus/1313.json` | yes | 9 | 7 | 4 | 1 | unverifiable / incorrect | fullkv_execution_valid, rkv_replay_mechanically_valid, correctness_status_matched |
| 1 | `test/number_theory/631.json` | no | 2 | 0 | 0 | 0 | correct / correct | rkv_replay_mechanically_valid, eligible_event_exists, selected_event_has_two_candidates |
| 2 | `test/geometry/538.json` | yes | 10 | 8 | 8 | 1 | correct / correct | rkv_replay_mechanically_valid |
| 3 | `test/algebra/853.json` | yes | 10 | 7 | 7 | 1 | incorrect / incorrect | fullkv_execution_valid, rkv_replay_mechanically_valid |
| 4 | `test/prealgebra/1356.json` | no | 0 | 0 | 0 | 0 | correct / correct | rkv_replay_mechanically_valid, meaningful_compression, eligible_event_exists, selected_event_has_two_candidates |
| 5 | `test/number_theory/838.json` | yes | 9 | 5 | 3 | 1 | incorrect / incorrect | fullkv_execution_valid, rkv_replay_mechanically_valid |
| 6 | `test/prealgebra/1388.json` | no | 0 | 0 | 0 | 0 | correct / correct | rkv_replay_mechanically_valid, meaningful_compression, eligible_event_exists, selected_event_has_two_candidates |
| 7 | `test/geometry/711.json` | yes | 9 | 4 | 4 | 1 | incorrect / incorrect | fullkv_execution_valid, rkv_replay_mechanically_valid |

## 4. Which qualification issue blocks inference

**`rkv_replay_mechanically_valid` failed on all eight candidates.** It is the
single universal blocker, and it decomposes into exactly two disjoint causes.

### 4.1 R-KV natural generation hit the 2,048-token cap — 5 of 8

Ordinals 0, 2, 3, 5, 7. Every one of these produced *ample* compaction
structure — 9 to 10 compaction events, 4 to 8 eligible event plans, 3 to 8
scored events, a valid selected event with a frozen candidate pool. Ordinal 2
in particular was correct under both FullKV and fixed-trace R-KV with 10 events
and 8 scored events, and failed on the cap alone.

The cap is a hard mechanical validity condition: a truncated natural trace
means the replayed suffix is not the model's own completed reasoning, so a
48-token readout after the intervention point is not comparable to a baseline
drawn from a completed trace.

### 4.2 No eligible event plan — 3 of 8

Ordinals 1, 4, 6.

- Ordinals 4 and 6 produced **zero compaction events**: generation finished
  well under the 1,024-token budget, so R-KV never compacted and there was
  nothing to intervene on. `meaningful_compression` correctly also failed.
- Ordinal 1 produced 2 compaction events but **zero eligible event plans** —
  no event survived the deterministic layer/head/candidate-donor selection with
  a donor present in every KV head.

**Ordinal 1 is the row that killed R1.** In this run it was classified as an
ordinary non-qualifier and the scan continued. Ordinals 4 and 6 have the same
zero-event shape, so the repaired path was exercised three times against real
weights.

### 4.3 A secondary blocker, not on the critical path

`fullkv_execution_valid` failed on 4 candidates (0, 3, 5, 7). Every one of
those also failed the cap condition, so repairing FullKV validity alone would
not have qualified a single additional row.

### 4.4 The binding constraint

The two causes are in tension under the frozen operating point:

- Rows long enough to generate rich compaction structure **overrun the 2,048
  cap**.
- Rows short enough to finish under the cap **generate too few compaction
  events** — often zero — to yield an eligible event.

Of eight frozen candidates, **not one landed in the window** where R-KV
compacts meaningfully *and* the natural trace completes. That window, not the
restoration mechanism, is what blocks inference.

## 5. Intervention evidence

None. Zero interventions ran, so every outcome slot is empty by construction —
not flat, not null: **absent**.

| Quantity | Value |
|---|---|
| Selected examples | 0 |
| Selected events | 0 |
| Candidate pools evaluated | 0 |
| Completed width-one pairs | 0 |
| Completed width-two pairs | 0 |
| Completed no-op pairs | 0 |
| Expected pair count | 0 |
| Rank-zero width-one gains | none |
| Rank-zero width-two gains | none |
| Non-rank-zero width-one gains | none |
| Non-rank-zero width-two gains | none |
| Largest gain over the bounded grid | none |
| Gains above 0.01 | 0 |
| Answer-margin sign changes | 0 |
| Fixed-trace extracted-answer changes | 0 |
| Fixed-trace correctness changes | 0 |
| No-op maximum absolute difference | not applicable (no no-op ran) |

The per-example candidate-rank × restore-width gain matrix is **empty**. There
is no favourable subset to select and none was sought.

## 6. Independent reconstruction

The stored summary was not trusted. Every reported quantity was recomputed from
the worker primitives, and the stored `scientific_summary.json` was read only
at the end for comparison. All checks agree:

- one authorization, one claim, one attempt, no retry, no failure sidecar;
- `invocation_ordinal = 1`, `automatic_retries = 0`, `retry_allowed = false`;
- protocol, runtime, manifest, model, tokenizer and R-KV bindings all exact;
- contiguous ordinals 0–7, stored qualification identical to worker primitives;
- each worker's `qualified` field agrees with independent recomputation under
  the strict contract;
- independent classification recomputes to **G**;
- runtime and VRAM inside limits; `verify_attempt` passes;
- both R1 trees byte-for-byte unchanged.

## 7. What this run does and does not establish

**Establishes:**

- The zero-event qualification defect is repaired against real weights. Three
  rows carrying the exact shape that voided R1 were handled as ordinary
  non-qualifiers and the scan completed all eight candidates.
- The complete first-eight scan is reachable and produces a valid
  mechanically-unqualified *result* rather than a crash artifact.
- Under the frozen 1.5B operating point and this frozen 8-row candidate
  manifest, the qualification window is empty.

**Does not establish anything about:**

- whether single-token, single-layer KV restoration moves the readout;
- candidate selection, restore width, or their interaction;
- the mean-NLL readout's sensitivity;
- the 8B operating point or the immutable 8B R2 null.

The bounded candidate × restore-width grid was implemented, tested and audited,
but **never executed**. It remains untested against real weights.

## 8. Governance

- The R2 authorization is permanently consumed. No retry is permitted for any
  reason.
- R1 remains void before intervention, immutable, and independently verifiable.
- No B2B. No Stage C. No method is designed or implemented from this result —
  see `docs/FAITHKV_METHOD_HANDOFF_FROM_DIAGNOSTIC_R2_2026-07-25.md`, which
  records that method design remains blocked.
