# FaithKV Method Handoff from Diagnostic R2 (2026-07-25)

```text
METHOD DESIGN REMAINS BLOCKED

R2 CLASSIFICATION: G — MECHANICALLY UNQUALIFIED
ZERO EXAMPLES QUALIFIED; ZERO INTERVENTIONS RAN
NO FAILURE MECHANISM WAS IDENTIFIED
NO METHOD PROTOTYPE IS IMPLEMENTED
NO B2B
```

## 1. Why no method is designed

The R2 protocol permits a method direction **only** when the result identifies
a unique candidate-, width-, or interaction-based failure mechanism. R2 was
classified **G — mechanically unqualified**: zero of eight candidates qualified
and zero interventions ran.

There is therefore no gain matrix, no width increment, no answer-margin
evidence, and no behavioural readout to reason from. Every mechanism-based
branch of the handoff rules is unreachable:

| Observed result | Handoff rule | Reachable? |
|---|---|---|
| A — current rank-zero width-one works | investigate the 8B/1.5B discrepancy | no — no gain exists |
| B — candidate-selection rescue | offline causal labels + deployable proxy | no |
| C — restore-width rescue | grouped KV protection | no |
| D — candidate × width interaction | joint token-head protection rule | no |
| E — readout-only movement | repair the faithfulness objective first | no |
| F — flat bounded diagnostic | close the single-token line; test coherent state units | no |
| **G — mechanically unqualified** | **state the blocking issue; implement nothing** | **this run** |

Designing a method now would mean inventing a mechanism the evidence does not
contain. This document therefore records the blocking issue and stops.

## 2. Exactly which qualification issue blocks inference

`rkv_replay_mechanically_valid` failed on **all eight** candidates. It is the
single universal blocker and decomposes into two disjoint causes.

**(a) R-KV natural generation hit the 2,048-token cap — ordinals 0, 2, 3, 5, 7.**
These rows produced abundant compaction structure (9–10 compaction events, 4–8
eligible event plans, 3–8 scored events, a valid selected event with a frozen
candidate pool). They failed only because the natural trace was truncated. A
truncated trace means the replayed suffix is not the model's own completed
reasoning, so the 48-token post-intervention readout is not comparable to its
baseline.

**(b) No eligible event plan — ordinals 1, 4, 6.**
Ordinals 4 and 6 produced **zero compaction events** — generation finished well
under the 1,024-token budget, so R-KV never compacted. Ordinal 1 produced two
compaction events but no event survived the deterministic layer/head/
candidate-donor selection with a donor present in every KV head.

**(c) The binding constraint.** These two causes are in direct tension at the
frozen operating point:

- rows long enough to compact meaningfully **overrun the 2,048-token cap**;
- rows short enough to complete under the cap **produce too few compaction
  events**, often zero.

None of the eight frozen candidates landed in the window where R-KV compacts
meaningfully *and* the natural trace completes. **That empty window — not the
restoration mechanism — is what blocks inference.**

A secondary blocker, `fullkv_execution_valid`, failed on ordinals 0, 3, 5 and
7; each of those also failed the cap condition, so repairing FullKV validity
alone would qualify no additional row.

## 3. What is now known that was not known before R2

1. The zero-event qualification defect is repaired against real weights, not
   only in tests. Three rows carrying the exact shape that voided R1 (ordinals
   1, 4, 6) were classified as ordinary non-qualifiers, and the scan completed
   all eight candidates.
2. A complete first-eight scan yielding zero qualified examples now produces a
   valid mechanically-unqualified **result**, not a crash artifact.
3. The frozen 8-row candidate manifest contains **no** row that satisfies the
   frozen qualification conditions at the 1.5B operating point. This is a
   property of the manifest-and-operating-point pair, established
   outcome-blind, before any intervention.

## 4. What remains untested

The bounded candidate × restore-width grid is implemented, unit-tested, and
independently audited, but **has never run against real weights**. Its nine
cells per example are the intended instrument for distinguishing categories A
through F, and that instrument remains uncalibrated.

## 5. The next hypothesis is about qualification, not about method

The next experiment must test whether a qualified example is *obtainable at
all* at this operating point, before any restoration question can be asked
again. That is a qualification-yield question, and it is answered by
observation of already-frozen, outcome-blind quantities — never by relaxing a
scientific gate to manufacture a qualifier.

Candidate directions, all requiring their own separate pre-registration,
protocol freeze, independent audit and one-use authorization:

1. **Measure the qualification window before spending another execution.** For
   a larger set of frozen MATH-500 rows, record natural generated length under
   FullKV and R-KV, compaction-event count, and eligible-plan count. This is
   outcome-blind and answers directly whether a
   compacts-meaningfully-and-completes window exists at 1.5B.
2. **Re-examine the two frozen limits jointly.** The 1,024-token compression
   budget and the 2,048-token generation cap together define the window. Any
   change to either is a scientific-setting change requiring its own dated
   authorization, and must be pre-registered as a qualification-yield change,
   never justified by an observed intervention outcome.
3. **Reconsider the candidate-manifest ordering.** The manifest is frozen and
   content-hash-ordered, which is correct for outcome-blindness. Whether eight
   candidates is enough depends on the window measured in (1).

**No option above may be selected on the basis of an intervention outcome,
because none exists.**

## 6. Explicit prohibitions carried forward

- No method prototype is implemented by this handoff — not behind a feature
  flag, not as a scorer, not as a KV-protection rule.
- No B2B authorization.
- No Stage C.
- No threshold, model, tokenizer, dataset, or R-KV revision change.
- No retry of the consumed R2 authorization, for any reason.
- Nothing here bears on the 8B operating point or the immutable 8B R2 null.
- R2 must not be described as showing that restoration does not work. It shows
  that no example qualified to test it.
