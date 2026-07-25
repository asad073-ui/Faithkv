# FaithKV 8B Structured Restoration Geometry — Primary GPU Authorization R3 (2026-07-25)

Authorizes exactly one execution attempt of the 8B structured restoration
geometry pilot. Two prior authorization IDs are now permanently consumed
by failed attempts, both preserved read-only and never reused:

- `geometry-pilot-primary-2026-07-25` (R1): claim created, model loaded,
  then crashed with an `AttributeError` before any branch outcome existed
  -- call-signature bugs (`git_commit()` argument count;
  `num_key_value_heads`/`max_new_tokens` read from the wrong config
  object; `reset_patched_state`/`ModelProvenance`/`build_math500_answer_fn`/
  `build_real_branch_step_fn_restore_once` call-shape mismatches), fixed in
  commits `714b653` and `5f29992`. Evidence preserved at
  `/workspace/faithkv-8b-geometry/r1-failed-attempt-preserved/`.
- `geometry-pilot-primary-r2-2026-07-25` (R2): claim created, model loaded,
  natural generation and Pass-2 capture both succeeded, then crashed
  building Arm C1's mutation ("candidate 1159 not resolvable at layer 16,
  head 1") -- a genuine design flaw, not a typo: the rank-zero candidate is
  evicted BY the anchor event at the anchor (layer, head) itself, so it is
  only resolvable via the PRE-event captured state, never the POST-event
  snapshot. Fixed in commit `fb1aee5`, which also fixed a related Arm S/SH
  span-donor design flaw. Evidence preserved at
  `/workspace/faithkv-8b-geometry/r2-failed-attempt-preserved/`.

Commit `fb1aee5` (2129 passed, 4 pre-existing/unrelated skips, 0 failed;
exact-SHA CI green) is the implementation this authorization binds. No
scientific threshold, event, candidate, layer, or arm definition has
changed across any of these repair rounds.

## Binding

| Field | Value |
|---|---|
| Authorization ID | `geometry-pilot-primary-r3-2026-07-25` |
| Repository | `asad073-ui/Faithkv` |
| Branch | `research/8b-structured-restoration-geometry` |
| Implementation (audited + twice-repaired) commit SHA | `fb1aee59f49d70ef5cf16a840ed5c513d80e16b4` |
| Protocol document | `docs/FAITHKV_8B_STRUCTURED_RESTORATION_GEOMETRY_PROTOCOL_2026-07-25.md` |
| Protocol document SHA-256 | `b2f80bef26c07916e88e06a7e1dab6efc451b7c94458cdb51041730e92298df5` |
| Operating-point JSON | `/workspace/faithkv-8b-geometry/authoritative-8b-operating-point.json` |
| Operating-point SHA-256 | `89b2a866adee2780087457d90d569744f5e2d3a1eb8912b515af83fd4ae93ed0` |
| Candidate-pool manifest canonical SHA-256 | `efc9a113f518a57193b1cdd0607fee2d45e2c879c862b8e8190b2be9fead1349` |
| Implementation-audit report | `/workspace/faithkv-audits/faithkv-8b-structured-restoration-geometry-implementation-audit.md` |
| Implementation-audit report SHA-256 | `d254662688e40976eef6e32969554c267be8d59efcd63f0ec30892bdbbd28ad5` |
| Superseded (permanently consumed, failed) authorizations | `geometry-pilot-primary-2026-07-25`, `geometry-pilot-primary-r2-2026-07-25` |
| Model repository / revision | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` / `6a6f4aa4197940add57724a7707d069478df56b1` |
| Tokenizer repository / revision | same |
| R-KV upstream revision | `45eaa7d69d20b7388321f077020a610d9afb65bd` |
| Selected row | `test/number_theory/631.json` (`example_index=287`) |
| Exact command | `kvcot run-8b-geometry-pilot --authorization-document docs/FAITHKV_8B_STRUCTURED_RESTORATION_GEOMETRY_PRIMARY_AUTHORIZATION_R3_2026-07-25.md --execute` |
| Maximum branches | 8 (of a 10-branch protocol ceiling) |
| Maximum invocations | 1 |
| Automatic retries | 0 |
| Hardware | exactly one RTX 3090, no CPU/disk/meta offload |
| VRAM limit | 22 GiB peak allocated |
| Primary runtime limit | 3600 seconds (1.00 GPU-hour) — conservative; the natural-generation phase alone (identical to original R2's 784s comparable workload) dominates wall time |
| Combined primary+replication runtime ceiling | 4.00 GPU-hours — R1 and R2's failed attempts together used well under 2 minutes of real GPU time (both crashed during/shortly after model load, before any inference-heavy work), so this ceiling remains essentially untouched |
| Claim path (repository-relative) | `results/decisions/geometry_pilot_authorization_claims/geometry-pilot-primary-r3-2026-07-25.json` |

The claim path uses a distinct authorization ID from both prior failed
attempts, so it cannot collide with either preserved claim.

## Prohibitions

- No B2B execution or authorization of any kind.
- No second attempt under this authorization ID once a claim exists at the
  bound claim path.
- No change to the `0.01`-nat mechanism threshold, the frozen anchor event
  (compaction event 5, layer 16, KV head 1), the frozen candidate pool, or
  the frozen arm set.
- No CPU offload, no multi-GPU, no unpinned model/tokenizer/R-KV revision.
- No free-running generation beyond the existing 48-token teacher-forced
  scored horizon.

## Non-consuming preparation requirement

Before this authorization's claim is created, the executing session must
prove, in the same process: CUDA remains uninitialized, no model weights
are loaded, the claim path is absent, and the attempt output root is
absent — via `kvcot run-8b-geometry-pilot --authorization-document
docs/FAITHKV_8B_STRUCTURED_RESTORATION_GEOMETRY_PRIMARY_AUTHORIZATION_R3_2026-07-25.md
--dry-run`.

Once FullKV/R-KV inference begins under this authorization, the attempt is
scientifically consumed — no automatic or unauthorized second attempt.
