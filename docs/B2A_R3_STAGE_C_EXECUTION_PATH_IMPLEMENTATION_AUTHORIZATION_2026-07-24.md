# B2A-R3 Stage-C Execution-Path Implementation Authorization

Date: 2026-07-24

This document authorizes a narrow CPU-only implementation phase for the
missing B2A-R3 Stage-C execution infrastructure. It does not authorize
Stage-C execution, CUDA initialization, model loading, tokenizer loading for
execution, R-KV import, worker launch, GPU rental, claim consumption, or
creation of a real Stage-C attempt directory.

## Bound Authorities

- Repository: `asad073-ui/Faithkv`
- Branch: `research/b2a-r3-runtime-qualified-calibration`
- Starting SHA: `b4b1676a08c55a0918c0f216a3e109ca88f9d905`
- Required CI run: `30110379400`, completed successfully at the starting SHA
- R-KV SHA: `45eaa7d69d20b7388321f077020a610d9afb65bd`
- Readiness audit: `/workspace/faithkv-audits/faithkv-b2a-r3-stage-c-readiness-audit.md`
- Readiness audit SHA-256: `39573b450be3c78222e8e878092baff85fd4d3df03625fe665f68dfb84619c49`
- Freezer persistence post-repair re-audit: `/workspace/faithkv-audits/faithkv-b2a-r3-freezer-persistence-post-repair-reaudit.md`

## Fixed Inputs

- Config path: `configs/discovery/llama8b_math500_b1024.yaml`
- Config byte SHA-256: `de8ac65a348c307c4f00089da07914666332935981bcaa7c98a150a9e7e778b3`
- Candidate manifest path: `configs/discovery/b2a_r3_candidate_manifest.json`
- Candidate manifest canonical SHA-256: `b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42`
- Qualification artifact path: `results/decisions/b2a_r3_qualification.json`
- Qualification artifact canonical SHA-256: `4349edc97a273819d4f5a3e75812af80437971f584071b66b25c858ffa02ff1d`
- Selected manifest path: `configs/discovery/b2a_one_example_manifest.json`
- Selected manifest hash: `dea628339f4b82678fa18bdc86c8dafa11c5ed87714a8b3a79b588884cab02e0`
- Selected manifest hash algorithm: `B2AOneExampleManifest.manifest_hash-v1`
- Selection provenance path: `results/decisions/b2a_r3_selection_provenance.json`
- Selection provenance canonical SHA-256: `2be6ef3097bf6362bfb029caee702cfd802ac95f00f8bb4a39ad83e6de593442`
- Accepted selected ordinal: `1`
- Accepted selected row: `test/number_theory/631.json`
- Model/tokenizer: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- Model/tokenizer revision: `6a6f4aa4197940add57724a7707d069478df56b1`

## Authorized Implementation Scope

The implementation may add:

- A fixed-path B2A-R3 Stage-C production orchestration module.
- An internal deterministic Stage-C authorization-claim payload constructor.
- Fixed-path Stage-C dry-run and execute planning.
- Semantic Stage-C authorization verification.
- Atomic single-use claim consumption through the existing authorization mechanism.
- Post-claim device preflight.
- Delegation to the existing B2A scientific calibration coordinator.
- Stage-C binding/provenance persistence before any GPU action.
- CLI wiring for a fixed-path Stage-C dry-run/execute entry point.
- CPU-only tests and documentation.

The implementation may make minimal non-scientific metadata changes needed
for Stage-C binding artifacts to be inventoried and verified by final
attempt manifests.

## Explicit Prohibitions

This authorization prohibits:

- Real Stage-C execution.
- Real Stage-C claim consumption.
- Creation of a real Stage-C attempt directory.
- CUDA initialization.
- Model or tokenizer loading for execution.
- R-KV import or worker launch.
- Running `b2a-calibrate --execute`.
- Modifying the selected manifest or selection provenance.
- Modifying candidate or qualification evidence.
- Modifying thresholds, seeds, model, tokenizer, dataset, budget, bridge,
  scoring-window, intervention, no-op, attrition, worker math, R-KV pin, or
  scientific semantics.
- Merge, rebase, reset, amend, or force-push.

The only permitted execution in this phase is CPU-only dry-run or mocked
execution in temporary repositories. The real Stage-C command must remain
blocked until this implementation is independently audited and a separate,
single-file Stage-C execution authorization document is committed.

## Authorized Outcome

The authorized end state of this implementation phase is:

```text
STAGE-C EXECUTION PATH IMPLEMENTED;
CPU TESTS GREEN;
NO REAL AUTHORIZATION CONSUMED;
NO GPU EXECUTION PERFORMED;
INDEPENDENT RE-AUDIT REQUIRED
```
