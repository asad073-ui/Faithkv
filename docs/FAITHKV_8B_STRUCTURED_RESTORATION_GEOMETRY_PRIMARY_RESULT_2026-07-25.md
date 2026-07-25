# FaithKV 8B Structured Restoration Geometry — Primary Result (2026-07-25)

Persistence and interpretation only. This document authorizes no further
execution of any kind. The primary pilot's claim
(`geometry-pilot-primary-r4-2026-07-25`) is scientifically consumed; it
is never rerun.

## 1. Execution result

| Item | Value |
|---|---|
| Authorization ID | `geometry-pilot-primary-r4-2026-07-25` (fourth authorization; R1-R3 permanently consumed by three preventable/design-level bugs, all fixed before this attempt — see the authorization document's own history section) |
| Attempt ID | `08d11fba6fbb4250a4472f50c0d4c124` |
| Attempt directory | `results/decisions/geometry_pilot_attempt_20260725T175735086366Z_08d11fba6fbb4250a4472f50c0d4c124` |
| Implementation commit | `3c27064fcecfbfc6b47e330b5c2be6f900f541dd` |
| Observed execution commit | `a119ba3e04f089159bfef22c2852fcb2d1ed11a6` |
| Claim canonical SHA-256 | `0a2a1b0010d1121e5fe168b0e55ffb6c37175096e7ca4f0e594a3237ed7859b5` (independently recomputed and matched) |
| Exit code | 0 |
| Natural answer | `36`, status `correct` (matches the frozen gold answer and the original R2 evidence) |
| Natural generated token count | 2784 (see note below; original R2 evidence: 1936) |
| Wall time | 302.46 s |
| Peak CUDA allocated | 16.61 GB / Peak reserved 16.73 GB (well under the 22 GiB authorized limit) |
| GPU-hours used | ≈0.084 (302 s), far under the 1.00-hour primary ceiling and the 4.00-hour combined ceiling |

**Generated-length note.** The natural run produced 2784 tokens here versus
1936 in the original R2 evidence for the identical model, config, seed,
and row. `apply_framework_seed`'s own documented contract records
`bitwise_determinism_guaranteed=False` for `flash_attention_2` — its
kernels are not guaranteed bitwise-deterministic across separate runs even
under an identical seed, so floating-point drift over a long autoregressive
chain can eventually flip an argmax and change total length. This does not
compromise the evidence measured here: Pass-2's own construction-time
validity checks (exact token identity against Pass-1's frozen trace at
every replayed position, the compaction event firing at exactly the
expected absolute position 1664 at layer 16, cross-pass survivor-identity
parity at KV head 1) all passed without exception — only possible if
generation was reproduced exactly through at least absolute position 1714
(bridge 1665 + 48 scored tokens), comfortably past this pilot's entire
read window. Any divergence, if real, happens strictly later and affects
nothing this pilot measures.

## 2. Branch results (independently reconstructed from primitive arrays)

| Arm | Mutation count | Layers | KV heads | Swap gain (nats) | Exceeds 0.01? |
|---|---|---|---|---|---|
| C1 (rank-0 candidate) | 1 | [16] | [1] | −0.00176856 | No |
| C2 (rank-1 candidate) | 1 | [16] | [1] | +0.00102890 | No |
| H (all KV heads, selected layer) | 5 of 8 possible (heads 0,1,4,5,6 resolved; 2,3,7 independently unavailable) | [16] | [0,1,4,5,6] | −0.00319949 | No |
| L (selected KV head, valid layers) | 11 | 11 valid layers (of 32 candidates) | [1] | +0.00024126 | No |
| HL (all KV heads, valid layers) | 51 of up to 88 possible | 11 valid layers | 0–7 | +0.00164803 | No |
| S / SH (token span) | — | — | — | unavailable — left neighbor (1158) independently evicted at (layer 16, head 1) with no valid post-event snapshot; no substitute donor invented | n/a |
| no-op | 1 | [16] | [1] | 0.0 exactly | n/a |

Every gain above independently recomputes to the exact stored value from
`mean(baseline_per_token_nll) - mean(swapped_per_token_nll)`. The no-op's
baseline and swapped 48-token NLL arrays are byte-identical; 0 key slots
and 0 value slots changed.

Sub-window (8-token, 16-token) gains were also reconstructed for every
arm and are recorded in `primary-result-reconstruction.json` — per protocol
§8 these are descriptive only and never function as a gate; the
classification below uses exclusively each arm's full 48-token mean-NLL
gain against the frozen `0.01`-nat threshold.

Arms H and HL realized fewer mutations than their theoretical maximum
(8 heads, 11×8=88 pairs respectively) because R-KV evicts independently
per `(layer, kv_head)` — some heads/layers had already evicted the
candidate at an earlier, untracked event and so cannot resolve it via
either the pre-event or post-event captured state. This is recorded
transparently, not treated as a failure.

## 3. Classification

```text
8B GEOMETRY PILOT VALID (H) —
EVERY AVAILABLE CAUSAL BRANCH REMAINED AT OR BELOW 0.01 NATS, NO
BEHAVIOURAL READOUT CHANGED, AND THE NO-OP WAS EXACT;
SINGLE-TOKEN AND LOCAL-SPAN RESTORATION ARE UNSUPPORTED UNDER THE
FROZEN PRIMARY-ROW CONDITIONS; THIS DOES NOT GENERALIZE BEYOND THEM;
NO B2B AUTHORIZATION
```

Independently reverified by calling
`kvcot.discovery.geometry_pilot_contract.classify_geometry_pilot` directly
against the persisted primitive readouts, outside of the executing
process — reproduces `STRUCTURED_RESTORATION_FLAT` / letter `H` exactly.

No fixed-trace behavioural-margin capture exists in this pilot's design
(`behavioural_change` is a hardcoded `False`, documented in
`geometry_pilot_execute.py` since this pilot performs no additional
free-running generation) — classification G is therefore structurally
unreachable by this implementation, not ruled out by evidence. This is a
readout-design limitation, recorded honestly rather than silently
defaulted into H.

## 4. What this result does and does not say

Restoring the original evicted content of the rank-zero candidate — at a
single KV head, across all available KV heads at the selected layer,
across all layers where it is independently resolvable, and across their
combination — did not measurably reduce the model's negative
log-likelihood over the 48-token scored horizon on this one MATH-500 row,
at this compaction event. Head coverage and layer coverage, individually
and combined, do not rescue the original Stage-C R2 null on this example.

It does **not** say restoration never matters on any row, event, or
model; this is one bounded example. It does not say the reasoning chain is
unfaithful. It does not establish anything about internal cognition.

## 5. Replication gate (task §20)

Classification `H` is not in `{B, C, D, E, F}` — the only classifications
that authorize replication. Per the frozen protocol: **"Classification H
closes the current structured restoration line."** No replication
example is selected, no second GPU authorization is created, and no
novelty-kill review is performed (§22 of the task applies only after a
mechanism replicates — there is no mechanism here to review).

## 6. Method status

**Method design remains blocked.** No mechanism was observed (classification
H, not A–F), so no method may be proposed under this task's own rules
(§23: method design requires classifications B–F to have replicated).
