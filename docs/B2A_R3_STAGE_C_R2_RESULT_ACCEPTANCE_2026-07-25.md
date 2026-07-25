# B2A-R3 Stage-C R2 Result Acceptance (dated 2026-07-25)

Persistence and interpretation only. This document authorizes no execution
of any kind: no GPU, no CUDA, no model or tokenizer load, no Stage C, no
B2B, no new authorization, no rerun of R1 or R2.

The R2 execution result is immutable. Nothing in this document rewrites,
regenerates, repairs, or reinterprets a single byte of the preserved claim
or attempt directory.

---

## 1. Execution result

The single authorized R2 invocation completed.

| Item | Value |
|---|---|
| Authorization ID | `stage-c-2026-07-24-r2` |
| Command | `kvcot run-b2a-r3-stage-c --authorization-document docs/B2A_R3_STAGE_C_EXECUTION_AUTHORIZATION_2026-07-24.md --execute` |
| Exit code | 0 |
| Invocation count | 1 (limit 1) |
| GPU | one NVIDIA GeForce RTX 3090, `GPU-66d7af95-59c1-5f33-5fbe-75a5dd2e6d4a`, driver 580.142, CUDA 12.4 |
| Device preflight | passed, `policy_satisfied: true`, `visible_gpu_count: 1`, no blockers |
| FullKV worker | launched, return code 0, no timeout, 146.73 s |
| R-KV worker | launched, return code 0, no timeout, 623.45 s |
| Actual wall time | 784.17 s (0.2178 GPU-hours) |
| Runtime projection | 2.4170 GPU-hours (limit 4.00) |
| Peak allocated VRAM | 15.883 GiB (limit 22) |
| Peak reserved VRAM | 16.064 GiB |
| Max `nvidia-smi` VRAM | 16,775 MiB = 16.382 GiB |
| CPU offload | none; every parameter on CUDA |
| Final-reference verification | `passed = True`, `reasons = ()` |
| Attempt artifacts | 30 files, all present, no `stage_c_execution_failure.json` |
| R1 evidence | byte-identical to its pre-execution hashes; never reused |
| Retry | prohibited and not performed |

The provenance-collector repair that R1 died on is confirmed working:
`required_ancestry` holds exactly the four Stage-C ancestors, the legacy B1
SHA `3c853cff…` is correctly absent from it, and
`starting_ancestor_verified` was genuinely computed by a real
`git merge-base --is-ancestor` query rather than a blind dict index.

---

## 2. Scientific observations

All values independently recomputed from primitive stored quantities
(`swap_gain = mean(baseline_per_token_nll) - mean(swapped_per_token_nll)`),
not copied from the summary. Discrepancy count against `pair_records.json`,
`scientific_summary.json`, `completion.json`, and `final.json`: **zero**.

| Quantity | Value |
|---|---|
| Observed compaction events | 15 |
| Eligible compaction events | 13 |
| Selected compaction events | 3 |
| Real pairs (expected / completed / failed) | 12 / 12 / 0 |
| No-op control pairs | 1 |
| Attrition | 0 |
| Observed retention ratio | 0.38236307268343095 |
| Minimum swap gain | −0.001960865853040672 |
| Maximum swap gain | **+0.0010573618851692501** |
| Mean swap gain | **−0.00046982425755099977** |
| Median swap gain | −0.0006066344272130193 |
| Count `swap_gain > 0` | 5 of 12 |
| Count `swap_gain > 0.001` | 2 of 12 |
| Count `swap_gain > 0.005` | 0 of 12 |
| Count `swap_gain > 0.01` | **0 of 12** |
| `abs(spearman_score_margin_vs_swap_gain)` | 0.2776840001936151 |
| No-op control | NLL arrays element-wise exactly equal; `swap_gain = 0.0` |

Context: mean baseline NLL over the scored window is 0.1394 nats, so the
`0.01`-nat threshold is 7.18% of baseline. The largest observed **positive**
gain is 0.76% of baseline — **9.46× below** the threshold. (The largest
gain in either direction, 0.00196, is 1.41% of baseline and is *negative*.)
Seven of twelve gains are negative.

---

## 3. Gate authority

| Question | Answer |
|---|---|
| Authoritative criterion | `docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §17 — "Scientific mechanism gate" |
| Criterion text | (A) `gain_above_0_01_count >= 4` of `real_pair_count == 12`, strict `swap_gain > 0.01` nats; (B) `abs(spearman_score_margin_vs_swap_gain) < 0.30`; (C) no-op control exactly zero |
| Authority commit | `93b6ba869eb5e555684704a6d1f2250f16884768` (2026-07-22), repaired `81e11cb…`, closed `382de26…` (2026-07-23) |
| Threshold origin | `0.01` nats and `0.30` inherited unchanged from B0.5 `b0_5_r2_1_repaired_gate_10` (`ac3e7d5…`, 2026-07-19) |
| Active at R2 | **Yes** — protocol last modified 2026-07-23, before the 2026-07-24 R2 execution |
| Superseded | **No** — no document in the repository supersedes §17 |
| Implemented behavior | `_MEANINGFUL_GAIN_THRESHOLD = 0.01` computes `gain_above_0_01_count` in `scientific_summary.py`; the value is persisted but **never consumed as a decision anywhere in `src/`** |
| Corrected behavior | **None. No code change is warranted.** |

§17 states its own non-wiring explicitly:

> It is, in the current codebase, descriptive rather than gating:
> `scientific_summary.py`'s fields are computed and persisted but are not
> wired into either frozen gate tuple in §16 as a pass/fail condition.

> The mechanical gate (§16) answers whether the harness result is valid.
> This scientific gate answers whether the causal-mismatch signal survives at
> all. **A mechanical pass does not imply scientific success.**

### The two wired gates

| Gate | Frozen count | Observed count | Mechanism-success conditions |
|---|---|---|---|
| Legacy (`b2a_contract.MANDATORY_GATE_CONDITIONS`) | 29 | **29** | **0** |
| Final (`final_contract.FINAL_MANDATORY_GATE_CONDITIONS`) | 30 | **30** | **0** |

Category split — legacy: 10 execution-integrity, 16 evidence-integrity, 3
resource. Final: 12 execution-integrity, 14 evidence-integrity, 4 resource.

### §17 condition-by-condition result

| Condition | Required | Observed | Result | Wired? |
|---|---|---|---|---|
| A — pairs above 0.01 nats | ≥ 4 of 12 | **0 of 12** | **FAIL** | no |
| B — abs(Spearman rho) | < 0.30 | 0.2777 | PASS | no |
| C — no-op control exact | exact | exact | PASS | yes (`no_op_exact_parity`, `no_op_numerical_parity`) |
| **§17 conjunction** | — | — | **FAIL** | — |

### Corrections to prior reporting

Recorded here rather than by editing immutable evidence:

1. The legacy gate has **29** conditions. The prior execution audit said
   "all 30 legacy conditions" while listing 29, and its combined total of
   "60" should be **59**. Verbal reporting of "31" was also wrong — the
   `gate_result` object carries 31 keys because it additionally holds the
   non-condition keys `passed` and `failed_conditions`.
2. Prior reporting described the outcome as "scientific gate PASS". That
   wording is **wrong**. `overall_gate_passed` covers §16's *mechanical*
   gates only. The word "scientific" does not appear in the executable's
   vocabulary; the mislabel was downstream human prose, not an artifact
   defect. The correct phrasing is: **mechanical gates PASS; scientific
   mechanism gate FAIL.**
3. §17 is not entirely unwired — condition C *is* enforced by the wired
   no-op parity conditions. The accurate statement is that neither wired
   gate contains a mechanism-**success** condition.

---

## 4. Accepted scientific classification

```text
STAGE-C EXECUTION VALID —
FROZEN MECHANISM GATE FAILED;
B2B BLOCKED
```

This is **not** a gate-implementation defect. The implementation is faithful
to the frozen protocol, which documented the non-wiring in the same
paragraph that froze the thresholds. Wiring condition A into a gate now,
after observing a failing result, would be a post-hoc gate change and is
prohibited. No threshold, comparator, required sample count, or scientific
primitive is changed by this acceptance.

A mechanically flawless execution produced a null causal result. Both facts
are recorded, and neither is allowed to stand in for the other.

---

## 5. B2B decision

```text
B2B BLOCKED
```

Three independent grounds, any one sufficient:

1. §17's mechanism criterion fails (0 of 12 versus a required 4).
2. Protocol §18's kill gate: gains at floor → do not run B2B.
3. The R2 authorization document itself states "Stage C only: no B2B", and
   CLAUDE.md §1c records "B2B authorization: **None.**"

Eligibility is not authorization, and here there is neither.

---

## 6. Evidence binding

| Artifact | Hash |
|---|---|
| R2 claim canonical SHA-256 | `b63b2ec9f392834dd643910741951029846aade9205d5717b922669d49b2fdde` |
| R2 claim timestamp | `2026-07-24T23:37:17.440992+00:00` |
| R2 attempt ID | `73cb2b597b83449eb0d990fcab2741b6` |
| R2 execution commit | `0673bbeba25a9b7a6d6e0a78b8a391f36d43f216` |
| Evidence snapshot manifest SHA-256 | `9017cbef043e444708c59b15d509b9c1603421fb84f08863d79e72570346345d` |
| Gate-authority timeline SHA-256 | `1fca4429912a17d8bae8c8331b3eaa8331dce8e3e5439ee013f25d103d65d09b` |
| Recomputed metrics SHA-256 | `5f728c188ad9f2db84291b9f95406f96e1035f3402cde0bde1b77e8d74e0d618` |
| Gate crosswalk SHA-256 | `a9e33031915afb3be5c9aadf109c5401b0882a607d471cb5ffaf56f85965646b` |
| R2 execution independent audit SHA-256 | `8666a88cd98663b7ee611a034ee22d0ce0580ad23aba77cf0c0c2a93c3a129c8` |
| Gate-consistency independent audit SHA-256 | `64aecb1d50c08b7080ac425ce9a4b61465f7727aed222aa907a9c6cadaf71a1b` |
| Repair authorization SHA | none — no repair was warranted |
| Repair SHA | none |

Raw claims and attempt artifacts are deliberately **not** committed to Git.
They are preserved outside the repository and in the destruction-safe
backups.

---

## 7. What this result does and does not say

It says: at this operating point, on this one MATH-500 row, restoring a
single evicted KV entry at a single (layer, head) did not measurably reduce
the model's negative log-likelihood over the scored horizon.

It does **not** say the reasoning chain is fake, decorative, or unfaithful,
and it does not establish anything about internal cognition. It is one
bounded counterfactual measurement on visible generated tokens, exactly as
CLAUDE.md §1's claim boundary requires. It is also a single example — the
correct description is a **null calibration result**, not a refutation.
