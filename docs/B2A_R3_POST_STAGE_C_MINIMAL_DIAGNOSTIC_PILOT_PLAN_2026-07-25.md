# B2A-R3 Post-Stage-C Minimal Diagnostic Pilot Plan (dated 2026-07-25)

**This document is a design, not an authorization.** It authorizes no GPU
time, no CUDA, no model load, no execution of any kind. Running any part of
it requires a separate, dated, independently audited execution
authorization. It is explicitly **not** a B2B run.

Predecessor: `docs/B2A_R3_STAGE_C_R2_RESULT_ACCEPTANCE_2026-07-25.md`
(classification: frozen mechanism gate FAILED; B2B BLOCKED).

---

## 1. What the immutable R2 data already tells us

All figures below are recomputed from the 12 preserved real pairs. No new
measurement was taken.

| Observation | Value | Why it matters |
|---|---|---|
| Peak per-token \|ΔNLL\| | per-layer mean of per-pair peaks 5.7e-2 to 6.4e-2 nats (layers 6, 16); single-pair maximum 7.28e-2 | The intervention **does** perturb the model substantially — the largest single-token effect is ~52% of the 0.1394-nat mean baseline NLL |
| Window-mean gain | −4.7e-4 | Those large local effects **cancel** when averaged over 48 tokens |
| Sign split | 5 positive / 7 negative | Direction is essentially random w.r.t. "restoring the evicted token helps" |
| Tokens with \|Δ\| > 1e-9 | 17–38 of 48 across the 12 pairs | The perturbation propagates broadly, it does not decay quietly |
| Narrowing the window | 0/12 above 0.01 at **every** 8-token slice; first-token-only max is **negative** (−2.4e-7) | A shorter scoring window does **not** rescue the result |
| `score_margin_e_minus_r` | −1.1e-5 to −1.5e-4 | The selection score barely separates evicted from retained tokens |
| abs(Spearman rho) | 0.2777 | Deployable signals do not predict gain |
| Layer/head coverage | 3 of 256 (layer, kv-head) combinations = **1.2%** | Extremely narrow sampling |
| Layer 23 peak \|Δ\| | 1.5e-2 vs 5.7–6.4e-2 at layers 6/16 | Deep-layer restores propagate ~4× less |
| No-op control | bit-exact | Swap machinery is sound — rules out state corruption |

---

## 2. Hypothesis ranking

### H1 — Candidate-selection failure (STRONGEST)
The selected token was not causally load-bearing; the score tracks salience,
not counterfactual necessity.

| Field | Content |
|---|---|
| Evidence for | `score_margin_e_minus_r` is 1e-5–1.5e-4 — the scorer barely distinguishes the evicted from the retained token. abs(rho) = 0.2777 means the deployable signal does not predict gain. 7 of 12 gains negative. |
| Evidence against | Large peak per-token effects (6.4e-2) show the swapped slot *is* used by the model, so the token is not inert. |
| Testable prediction | Choosing candidates by a direct counterfactual criterion (highest measured future-attention mass, or oracle leave-one-out) yields materially larger positive gains than the current score. |
| Cheapest discriminating test | **Pilot arm A** below — oracle-selected vs score-selected candidates at identical events. |
| Expected GPU cost | ≤ 0.5 GPU-hours (1.5B model, 1 example) |
| Kill condition | If oracle-selected candidates also produce max gain < 0.01 nats, candidate selection is **not** the bottleneck; stop pursuing better scorers. |

### H2 — Intervention dilution / redundancy (STRONG)
Restoring one KV slot at one (layer, head) is too weak; information is
redundant across neighbouring heads, layers, and tokens.

| Field | Content |
|---|---|
| Evidence for | 1.2% layer/head coverage. Peak local effect is large but window-mean is ~0, exactly what redundancy predicts. Layer 23 shows 4× smaller propagation. |
| Evidence against | If redundancy were total, per-token \|Δ\| would be near zero, not 6.4e-2. |
| Testable prediction | Restoring the same token across N heads (or all heads of a layer) produces gain scaling super-linearly in N; single-head restores stay at floor. |
| Cheapest discriminating test | **Pilot arm B** — restore-width sweep N ∈ {1, 4, all-heads-in-layer}. |
| Expected GPU cost | ≤ 0.7 GPU-hours |
| Kill condition | If all-head restore still yields < 0.01 nats, the mechanism is not dilution-limited and the whole single-token-restore framing is dead. |

### H3 — Insensitive outcome metric (MODERATE — partially falsified)
Mean NLL over a fixed 48-token window does not capture answer-relevant
effects.

| Field | Content |
|---|---|
| Evidence for | Large per-token effects cancel in the mean; NLL is not the quantity the claim boundary cares about (answer behaviour is). |
| Evidence against | **Already partially falsified by R2 data**: no 8-token sub-window, and not even the first token alone, produces any pair above 0.01. Window length is not the problem. |
| Testable prediction | A *behavioural* metric (final-answer flip rate, or answer-token margin) shows effects where mean NLL shows none. |
| Cheapest discriminating test | **Pilot arm C** — recompute a behavioural readout on the same branches. Largely CPU-side if branches are re-scored. |
| Expected GPU cost | ≤ 0.3 GPU-hours |
| Kill condition | If answer-token margin is also unmoved, the metric is not the bottleneck. |

### H4 — Wrong intervention time (MODERATE)
The branch is evaluated too early or too late relative to the token's
influence.

| Field | Content |
|---|---|
| Evidence for | Scoring starts only 2 tokens after the event; mean \|Δ\| is non-monotone (dips at tokens 8–15, rises again at 32–39), so influence is not concentrated where scoring is densest. |
| Evidence against | Non-monotonicity is equally consistent with trajectory divergence rather than mistimed measurement. |
| Testable prediction | Shifting the scored horizon to the re-amplification region raises gains. |
| Cheapest discriminating test | Free — re-score existing R2 branches at shifted offsets (CPU only, no new GPU time). |
| Expected GPU cost | **0** |
| Kill condition | Already largely killed: no 8-token slice contains a pair above 0.01. |

### H5 — Donor contamination (MODERATE)
Donor replacement introduces its own effect, masking restoration benefit.

| Field | Content |
|---|---|
| Evidence for | Sign inconsistency (5+/7−) is what a competing donor effect looks like. Donor recency varies from −1281 to +976 with no relation to gain sign. |
| Evidence against | The no-op control (donor replaced by itself) is bit-exact, so the swap mechanism adds no artefact of its own. |
| Testable prediction | Matched-donor designs (donor held fixed across candidates) reduce gain variance. |
| Cheapest discriminating test | Fold into arm A as a fixed-donor stratum. |
| Expected GPU cost | shared with arm A |
| Kill condition | If variance is unchanged with a fixed donor, donor choice is not contaminating. |

### H6 — Selection-range problem (WEAK-MODERATE)
The 3 chosen events do not represent the strongest observable failure cases.

Evidence for: only 3 of 13 eligible events were used. Evidence against: all
three, across layers 6/16/23, agree on a null result. Testable by widening
event coverage; cheapest test is a by-product of arm B.

### H7 — State incompatibility (WEAK — effectively falsified)
Restored KV inconsistent with post-compression cache state.

The no-op control is bit-for-bit exact and every parity condition passed.
There is no evidence for this and it should not absorb further GPU budget.

---

## 3. The minimal diagnostic pilot

**Purpose:** discriminate H1 (token-selection) from H2 (layer/head dilution)
from H3 (metric insensitivity), at the lowest possible cost, before any
method work.

### Constraints (all satisfied)
- **≤ 3 examples** — uses 3.
- **1.5B model** (`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`) for initial
  discrimination, per the plan's preference and CLAUDE.md §1's primary model.
- **≤ 4 GPU-hours** total across all arms — budgeted at ≤ 1.5.
- **One factor changed at a time.**
- Reuses the existing B1B/B2A harness, swap primitive, and no-op machinery.
- Includes FullKV and current R-KV controls, plus an explicit no-op.
- Success and kill criteria predeclared **here, before any run**.

### Arms

| Arm | Factor changed | Everything else | GPU budget |
|---|---|---|---|
| **Control** | none — current R-KV config, current scorer, single-head restore | baseline | 0.3 h |
| **A — selection** | candidate chosen by oracle future-attention mass instead of the deployable score | restore width fixed at 1 head | 0.5 h |
| **B — width** | restore width N ∈ {1, 4, all heads in layer} | candidate fixed to the current scorer's pick | 0.7 h |
| **C — metric** | readout = answer-token margin + final-answer flip, alongside mean NLL | no intervention change; re-scores arms above | ~0 (shares forward passes) |
| **No-op** | donor replaced by itself | must be bit-exact | included |

Arms A and B are deliberately orthogonal: A holds width fixed and varies
selection; B holds selection fixed and varies width. C is a pure readout
added to both.

### Predeclared success criterion

> The pilot is **informative** iff at least one arm produces
> `max swap_gain > 0.01` nats on at least one example, using the unchanged
> frozen `0.01` threshold.

Note this is a criterion for *learning something*, not for declaring the
mechanism real. It reuses the existing frozen threshold precisely so that no
new number is invented after seeing data.

### Predeclared kill criteria

1. If **both** arm A (oracle selection) and arm B (all-head restore) stay
   below 0.01 nats, the single-token-KV-restore mechanism is dead at this
   operating point. Stop; do not design a method around it; do not run B2B.
2. If the no-op control is not bit-exact in any arm, the run is void — fix
   the harness, do not interpret the numbers.
3. If arm C's behavioural readout is also flat while arms A and B are flat,
   metric insensitivity is excluded as an explanation.

### What the outcomes mean

| Result | Conclusion |
|---|---|
| A moves, B flat | Selection was the bottleneck → invest in candidate scoring |
| B moves, A flat | Dilution was the bottleneck → single-slot restore is the wrong intervention unit |
| Both move | Both contribute; measure interaction before any method design |
| **Neither moves** | **Mechanism dead at this operating point — the most likely outcome given R2** |
| Only C moves | NLL was the wrong readout; revisit the metric before anything else |

### Model-scale caveat (stated explicitly, not buried)

R2 ran on `DeepSeek-R1-Distill-Llama-8B`. Arms A–C above are specified on
the 1.5B primary-pipeline model because it is the cheapest way to
*discriminate among mechanisms*. That is a deliberate deviation from R2's
operating point, and it has a hard consequence: **a 1.5B result can neither
confirm nor refute the 8B R2 null.** It can only tell us which mechanism to
interrogate. Any claim about the R2 operating point itself requires an 8B
arm, costed separately. The "no change to model" line below means no model
substitution *within* the frozen primary pipeline or *within* an arm — it
does not license reading a 1.5B outcome as an 8B conclusion.

### Explicitly out of scope
No B2B. No 12-example run. No method implementation. No faithfulness-aware
eviction. No threshold change. No new eviction criterion. No change to
model, dataset, revision, seed, budget, R-KV pin, or any frozen scientific
setting. No merge.

---

## 4. Honest expectation

Given that the largest of 12 observed gains is 9.46× below threshold, the
mean is negative, and narrowing the scoring window does not help, the most
probable pilot outcome is **kill criterion 1** — neither arm moves. That is
a legitimate and useful result: it would close the single-token-KV-restore
line cheaply, on ~1.5 GPU-hours, instead of on a full B2B pilot.

This plan is designed to be *cheap to be wrong on*, not to rescue the
hypothesis.
