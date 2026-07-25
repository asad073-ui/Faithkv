# FaithKV Method Handoff — 8B Structured Restoration Geometry (2026-07-25)

## Status: method design BLOCKED

This phase's mandate was narrow: determine whether the original 8B
Stage-C R2 null (`docs/B2A_R3_STAGE_C_R2_RESULT_ACCEPTANCE_2026-07-25.md`)
was caused by candidate choice, KV-head coverage, layer coverage,
token-span coverage, or an insensitive readout — and implement a method
only if a mechanism is observed AND replicates on at least one additional
example. Neither condition was met.

## What was done

1. Verified both required preserved archives byte-for-byte against their
   recorded SHA-256 hashes.
2. Independently reconstructed the exact 8B Stage-C R2 operating point
   from primitive evidence (claim, attempt directory, pair records) —
   never copied from prose — and recomputed the original null's every
   statistic (mean, median, min, max gain, sign distribution, subwindow
   gains, Spearman correlation) from the immutable `pair_records.json`,
   matching the accepted result-acceptance document exactly.
3. Froze a structured restoration geometry protocol binding an anchor
   event (compaction event 5, layer 16, KV head 1), a bounded
   score-prioritized candidate pool (2 members), and 8 of a possible 10
   structured arms (candidate rank-0/rank-1, all-KV-heads, valid-layer-set,
   head×layer, token-span, span×head, no-op).
4. Implemented the missing structured-intervention harness (generalized
   multi-slot KV restore primitive, branch construction, pre-outcome
   layer-set/span-availability freezing, one-use authorization/claim
   mechanism, CLI wiring), with 123 CPU tests, all passing, reusing this
   repository's existing proven primitives (swap.py, pass1/pass2,
   real_model_adapter, branch_eval) rather than duplicating them.
5. Passed a bounded independent implementation audit (18/18 gates,
   after one repair round for a persistence-ordering defect).
6. Ran the primary GPU pilot. The first three authorized attempts failed
   before producing any scientific evidence — two rounds of call-signature
   bugs against the real model/config objects, then two rounds of design
   flaws this session had not anticipated (candidates evicted BY the
   anchor event need pre-event resolution; KV heads and layers evict
   independently, so availability cannot be assumed uniform). Each was
   diagnosed from real GPU behavior, fixed, covered by new CPU tests
   modeling the actual failure scenario, and re-authorized under a fresh
   ID (R1 through R4) since a consumed authorization is never retried.
7. The fourth attempt succeeded end-to-end. Every reported swap gain was
   independently reconstructed from the persisted primitive per-token NLL
   arrays, matching exactly; the classification was independently
   re-derived by calling `classify_geometry_pilot` directly against the
   persisted data outside the executing process.

## Primary result

**Classification H — structured restoration flat.**

- Candidate rank-0 (the original single-token/single-layer/single-head
  restoration): gain −0.00177 nats.
- Candidate rank-1: +0.00103 nats.
- All KV heads at the selected layer (5 of 8 resolvable): −0.00320 nats.
- Selected KV head across 11 valid layers: +0.00024 nats.
- All KV heads across those 11 valid layers (51 of up to 88 resolvable
  pairs): +0.00165 nats.
- No-op: exactly 0.0, byte-identical NLL arrays.
- Token span: structurally unavailable (a span neighbor was independently
  evicted with no valid post-event snapshot; no substitute donor invented).

Every branch remained far below the frozen `0.01`-nat threshold. Head
coverage and layer coverage — individually and combined — do not rescue
the original null on this example.

## Why method design is blocked

Per the frozen protocol and this task's own rules: classification `H` is
not one of `{B, C, D, E, F}` (candidate/head/layer/interaction/span rescue),
so replication is not authorized on this classification, the novelty-kill
review does not run (there is no mechanism to check for novelty), and no
method may be designed (method design requires a replicated mechanism from
one of the rescue classifications). This is not a failure of execution —
the mechanically-invalid classification (`I`) was explicitly not selected;
every check in the primary execution audit passed. It is a null result:
single-token and local-span KV restoration, at this row's this
compaction event, under head and layer coverage up to what proved
resolvable, did not move the model's behavior.

## What would be required to reopen this line

- A different frozen row/event from the original R2 candidate manifest,
  selected outcome-blind, exercising the identical bounded arm set.
- Or a genuinely different mechanism class not covered by this protocol
  (e.g., attention-pattern-level intervention rather than KV-content
  restoration).

Neither is authorized by the current task; both would require their own
separate, dated authorization following this same disciplined sequence
(preserve evidence → reconstruct operating point → freeze protocol →
implement → audit → authorize → execute → reconstruct → classify).

## Required baselines/benchmarks if a mechanism is ever found here

Not applicable — no mechanism to benchmark. If a future row/event
replicates a rescue classification, the required baselines remain
unchanged from this task's own specification: FullKV, R-KV, KIVI, ShotKV,
and the closest reasoning-specific token/head method, under a
matched-memory protocol with accuracy-neutral and faithfulness tests.

## B2B status

**BLOCKED**, unconditionally and independently of this phase's result —
this phase never had B2B authorization, and nothing here changes that.
