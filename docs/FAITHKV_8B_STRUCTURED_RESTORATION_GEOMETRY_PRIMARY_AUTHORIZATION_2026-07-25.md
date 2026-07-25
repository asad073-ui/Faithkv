# FaithKV 8B Structured Restoration Geometry — Primary GPU Authorization (2026-07-25)

Authorizes exactly one execution attempt of the 8B structured restoration
geometry pilot implemented and independently audited on
`research/8b-structured-restoration-geometry`. Does not authorize B2B, does
not authorize a second attempt, does not change any threshold, model,
dataset, revision, or gate frozen by CLAUDE.md or by
`docs/FAITHKV_8B_STRUCTURED_RESTORATION_GEOMETRY_PROTOCOL_2026-07-25.md`.

## Binding

| Field | Value |
|---|---|
| Authorization ID | `geometry-pilot-primary-2026-07-25` |
| Repository | `asad073-ui/Faithkv` |
| Branch | `research/8b-structured-restoration-geometry` |
| Implementation (audited) commit SHA | `8ade1db36fbbb0bfe0ef3d2fafce1ba77f7e203a` |
| Protocol document | `docs/FAITHKV_8B_STRUCTURED_RESTORATION_GEOMETRY_PROTOCOL_2026-07-25.md` |
| Protocol document SHA-256 | `b2f80bef26c07916e88e06a7e1dab6efc451b7c94458cdb51041730e92298df5` |
| Operating-point JSON | `/workspace/faithkv-8b-geometry/authoritative-8b-operating-point.json` |
| Operating-point SHA-256 | `89b2a866adee2780087457d90d569744f5e2d3a1eb8912b515af83fd4ae93ed0` |
| Candidate-pool manifest canonical SHA-256 | `efc9a113f518a57193b1cdd0607fee2d45e2c879c862b8e8190b2be9fead1349` |
| Implementation-audit report | `/workspace/faithkv-audits/faithkv-8b-structured-restoration-geometry-implementation-audit.md` |
| Implementation-audit report SHA-256 | `d254662688e40976eef6e32969554c267be8d59efcd63f0ec30892bdbbd28ad5` |
| Model repository / revision | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` / `6a6f4aa4197940add57724a7707d069478df56b1` |
| Tokenizer repository / revision | same |
| R-KV upstream revision | `45eaa7d69d20b7388321f077020a610d9afb65bd` |
| Selected row | `test/number_theory/631.json` (`example_index=287`) |
| Exact command | `kvcot run-8b-geometry-pilot --authorization-document docs/FAITHKV_8B_STRUCTURED_RESTORATION_GEOMETRY_PRIMARY_AUTHORIZATION_2026-07-25.md --execute` |
| Maximum branches | 8 (of a 10-branch protocol ceiling) |
| Maximum invocations | 1 |
| Automatic retries | 0 |
| Hardware | exactly one RTX 3090, no CPU/disk/meta offload |
| VRAM limit | 22 GiB peak allocated |
| Primary runtime limit | 3600 seconds (1.00 GPU-hour) — conservative; the original R2 execution (a comparable natural-generation-plus-Pass-2-plus-12-pair workload) completed in 784 s wall time |
| Combined primary+replication runtime ceiling | 4.00 GPU-hours (task-level bound; this authorization alone never approaches it) |
| Claim path (repository-relative) | `results/decisions/geometry_pilot_authorization_claims/geometry-pilot-primary-2026-07-25.json` |

The claim path is intentionally an in-repository, untracked path (never
committed — this repository's existing B2A-R3/diagnostic-pilot claim and
attempt directories under `results/decisions/` follow the identical
convention and are likewise never git-added), disjoint from every existing
claims root.

## Prohibitions

- No B2B execution or authorization of any kind.
- No second attempt under this authorization ID once a claim exists at the
  bound claim path — the claim's exclusive creation is itself the
  consumption event.
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
docs/FAITHKV_8B_STRUCTURED_RESTORATION_GEOMETRY_PRIMARY_AUTHORIZATION_2026-07-25.md
--dry-run`.

Once FullKV/R-KV inference begins under this authorization, the attempt is
scientifically consumed — no automatic or unauthorized second attempt.

## Amendment (pre-consumption CLI fix, same day)

The non-consuming dry-run/preflight check caught a `TypeError` in
`cmd_run_8b_geometry_pilot`'s `--execute` branch
(`git_commit(".")` — that function takes zero arguments; every other call
site in `src/kvcot/cli.py` already calls it bare) before `run_execute` was
ever entered — no claim was created, no CUDA was initialized, the
authorization remained unconsumed. Commit `8ade1db` fixes the one-line
call site, re-passes the full non-GPU CPU suite and exact-SHA CI, and does
not touch any scientific threshold, event, candidate, layer, or arm
definition. The "Implementation (audited) commit SHA" above is updated to
this fix commit; nothing else in this authorization's binding changes.
