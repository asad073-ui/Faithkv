# FaithKV 8B Structured Restoration Geometry — Primary GPU Authorization R2 (2026-07-25)

Authorizes exactly one execution attempt of the 8B structured restoration
geometry pilot, superseding nothing about the frozen protocol or the
scientific configuration. This is a fresh authorization ID because
`geometry-pilot-primary-2026-07-25` (§ see
`docs/FAITHKV_8B_STRUCTURED_RESTORATION_GEOMETRY_PRIMARY_AUTHORIZATION_2026-07-25.md`)
is now **permanently consumed**: that attempt's claim was created, the real
model was loaded (FullKV/R-KV inference began within the meaning of this
project's existing Stage-C contract), and it then failed with an
`AttributeError` before any branch outcome existed, entirely because of
call-signature bugs in `geometry_pilot_workers.capture_geometry_anchor`
(`num_key_value_heads`/`num_hidden_layers` sourced from the wrong object,
a missing `cache_factory` argument, an uninitialized per-layer
`ModelProvenance`, a missing `tokenizer` argument to
`build_math500_answer_fn`, and a missing `model` argument to
`build_real_branch_step_fn_restore_once`). That failed attempt's evidence
(claim, empty attempt directory, execution log) is preserved read-only at
`/workspace/faithkv-8b-geometry/r1-failed-attempt-preserved/` and is never
reused, retried, or deleted — exactly this project's existing precedent for
a consumed-and-failed Stage-C R1 attempt.

Commit `714b653` fixes those five bugs (CPU-tested: 2122 passed, 4
pre-existing/unrelated skips, 0 failed; exact-SHA CI green). Commit
`5f29992` fixes a sixth, distinct bug found by re-reading the execute path
before spending GPU time again: `geometry_pilot_execute._to_jsonable`
delegated to `dataclasses.asdict()`, which does not convert a
`KVMutationSpec`'s `torch.Tensor` fields (`replacement_key`/
`replacement_value`) into JSON-serializable values -- every existing test
used an empty `mutations=()` tuple, which hid this entirely; a real branch
readout with a real (non-empty) mutation would have raised
`TypeError: Object of type Tensor is not JSON serializable` inside
`run_execute`, AFTER a real GPU run had already completed every branch's
evaluation. Neither fix changes any scientific threshold, event,
candidate, layer, or arm definition.

## Binding

| Field | Value |
|---|---|
| Authorization ID | `geometry-pilot-primary-r2-2026-07-25` |
| Repository | `asad073-ui/Faithkv` |
| Branch | `research/8b-structured-restoration-geometry` |
| Implementation (audited + repaired) commit SHA | `5f29992e660fe8ae02da47028db456e287df58af` |
| Protocol document | `docs/FAITHKV_8B_STRUCTURED_RESTORATION_GEOMETRY_PROTOCOL_2026-07-25.md` |
| Protocol document SHA-256 | `b2f80bef26c07916e88e06a7e1dab6efc451b7c94458cdb51041730e92298df5` |
| Operating-point JSON | `/workspace/faithkv-8b-geometry/authoritative-8b-operating-point.json` |
| Operating-point SHA-256 | `89b2a866adee2780087457d90d569744f5e2d3a1eb8912b515af83fd4ae93ed0` |
| Candidate-pool manifest canonical SHA-256 | `efc9a113f518a57193b1cdd0607fee2d45e2c879c862b8e8190b2be9fead1349` |
| Implementation-audit report | `/workspace/faithkv-audits/faithkv-8b-structured-restoration-geometry-implementation-audit.md` |
| Implementation-audit report SHA-256 | `d254662688e40976eef6e32969554c267be8d59efcd63f0ec30892bdbbd28ad5` |
| Superseded (permanently consumed, failed) authorization | `geometry-pilot-primary-2026-07-25` |
| Superseded claim (preserved, read-only) | `/workspace/faithkv-8b-geometry/r1-failed-attempt-preserved/geometry-pilot-primary-2026-07-25.json` |
| Model repository / revision | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` / `6a6f4aa4197940add57724a7707d069478df56b1` |
| Tokenizer repository / revision | same |
| R-KV upstream revision | `45eaa7d69d20b7388321f077020a610d9afb65bd` |
| Selected row | `test/number_theory/631.json` (`example_index=287`) |
| Exact command | `kvcot run-8b-geometry-pilot --authorization-document docs/FAITHKV_8B_STRUCTURED_RESTORATION_GEOMETRY_PRIMARY_AUTHORIZATION_R2_2026-07-25.md --execute` |
| Maximum branches | 8 (of a 10-branch protocol ceiling) |
| Maximum invocations | 1 |
| Automatic retries | 0 |
| Hardware | exactly one RTX 3090, no CPU/disk/meta offload |
| VRAM limit | 22 GiB peak allocated |
| Primary runtime limit | 3600 seconds (1.00 GPU-hour) — conservative; original R2's comparable workload completed in 784 s |
| Combined primary+replication runtime ceiling | 4.00 GPU-hours (task-level bound; the R1 failed attempt used well under 1 minute of GPU time before crashing, so this ceiling is essentially untouched) |
| Claim path (repository-relative) | `results/decisions/geometry_pilot_authorization_claims/geometry-pilot-primary-r2-2026-07-25.json` |

The claim path uses a distinct authorization ID from the failed R1
attempt, so it cannot collide with, overwrite, or be confused with the
preserved R1 claim.

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
docs/FAITHKV_8B_STRUCTURED_RESTORATION_GEOMETRY_PRIMARY_AUTHORIZATION_R2_2026-07-25.md
--dry-run`.

Once FullKV/R-KV inference begins under this authorization, the attempt is
scientifically consumed — no automatic or unauthorized second attempt.
