# Post-Stage-C Diagnostic Pilot R2 — One-Use Execution Authorization (2026-07-25)

```text
POST-STAGE-C DIAGNOSTIC PILOT R2 EXECUTION AUTHORIZED — ONE USE ONLY

AUTHORIZATION ID: post-stage-c-diagnostic-pilot-r2-2026-07-25-one-use
MAXIMUM INVOCATIONS: 1
AUTOMATIC RETRIES: 0

NOT STAGE C
NOT B2B
NOT A RETRY UNDER THE CONSUMED R1 AUTHORIZATION

ONCE THE R2 CLAIM IS CREATED THIS AUTHORIZATION IS PERMANENTLY CONSUMED.
NO RETRY IS PERMITTED FOR ANY REASON.
```

This authorizes exactly one execution of the repaired, tested, audited
post-Stage-C diagnostic pilot R2. It authorizes nothing else. It does not
reinterpret, retry, or supersede the consumed R1 attempt, which remains
**void before intervention** with no scientific interpretation.

## 1. Preconditions, all satisfied before this document was written

| Precondition | Evidence |
|---|---|
| Implementation complete | `d4ad573c67615daaa9fa62e51de39abf874b2e48` |
| Focused diagnostic suites pass | 5 suites, 0 failed |
| Full CPU suite passes | 2034 passed, 14 deselected, 0 skipped, 0 failed |
| Exact-SHA CI succeeds | GitHub Actions run `30146383564`, conclusion `success` |
| Independent implementation audit passes | `AUDIT VERDICT: PASS`, no blocking findings |
| R2 runtime preparation verifies | canonical `e2ea738c…c200`, CUDA never initialized |
| R2 claim path absent | verified absent |
| R2 output root absent | verified absent |

The audit round before this one returned FAIL on a cross-generation path
collision defect; it was repaired, re-run through exact-SHA CI, and re-audited
to PASS. Both rounds were fresh, independent, read-only auditors.

## 2. Exact authorized command

Run exactly once, from the repository root, on the authorization commit:

```
kvcot run-post-stage-c-diagnostic-pilot --authorization-document docs/POST_STAGE_C_DIAGNOSTIC_PILOT_R2_EXECUTION_AUTHORIZATION_2026-07-25.md --execute
```

## 3. Bound identities and limits

| Binding | Value |
|---|---|
| Repository | `asad073-ui/Faithkv` |
| Branch | `research/post-stage-c-diagnostic-pilot` |
| Authorized implementation SHA | `d4ad573c67615daaa9fa62e51de39abf874b2e48` |
| R2 protocol document SHA-256 | `8d5fd74e9a94ab5ba067ed7c6ffd9c23590593b1131813f53317693e37e2f3ee` |
| R2 runtime canonical SHA-256 | `e2ea738c89a22a76f3cb71f5b4e01136a519183f916779908c1647a7d8a7c200` |
| Candidate-manifest canonical SHA-256 | `b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42` |
| Model revision | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| Tokenizer revision | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| R-KV revision | `45eaa7d69d20b7388321f077020a610d9afb65bd` |
| Implementation-audit SHA-256 | `34e3ebdc13b12ece5c93c6b890fe22d40b30309a9ad5735684db402ee80d25cf` |
| Maximum qualification candidates | 8 |
| Maximum selected examples | 3 |
| Maximum selected events | 3 |
| Maximum candidate pool | 4 |
| Restore widths | `[1, 2]` |
| Maximum pairs per selected example | 9 |
| Maximum total pairs | 27 |
| Maximum invocations | 1 |
| Automatic retries | 0 |
| Hardware | one RTX 3090 |
| CPU offload | prohibited |
| VRAM maximum | 22 GiB (`23622320128` bytes) |
| Runtime maximum | 5,400 seconds |
| R2 runtime path | `/workspace/faithkv-post-stage-c-diagnostic-runtime-r2/runtime_config.json` |
| R2 output root | `/workspace/faithkv-post-stage-c-diagnostic-execution-r2` |
| R2 claim path | `/workspace/faithkv-post-stage-c-diagnostic-execution-r2/authorization-claim.json` |
| Gain threshold | 0.01 nats — unchanged |

## 4. What is prohibited

- Any second invocation, for any reason, including SSH disconnection, timeout,
  CUDA failure, worker failure, coordinator failure, evidence failure, test
  failure, or agent interruption.
- Re-executing the consumed R1 authorization. Only the R2 generation is
  executable; the implementation refuses an R1 authorization before it reads
  any runtime input.
- Writing anywhere inside the R1 output root or the R1 runtime root.
- Stage C, B2B, any method implementation, any threshold change, any model,
  tokenizer, dataset, or R-KV revision change, CPU offload, more than one GPU,
  or an unpinned model snapshot.
- Any scientific interpretation if the result is mechanically unqualified or
  void.

## 5. Scale boundary

A 1.5B result does not settle the 8B operating point. A flat result does not
kill all single-token KV restoration; it is reported only as the narrow
statement frozen in the R2 protocol.

## 6. Authorization payload

<!-- DIAGNOSTIC_PILOT_AUTHORIZATION_JSON_BEGIN -->
{
  "authorization_id": "post-stage-c-diagnostic-pilot-r2-2026-07-25-one-use",
  "authorized_branch": "research/post-stage-c-diagnostic-pilot",
  "authorized_implementation_sha": "d4ad573c67615daaa9fa62e51de39abf874b2e48",
  "automatic_retries": 0,
  "candidate_manifest_canonical_sha256": "b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42",
  "canonical_sha256": "dbae064324b3a65d2c9bfc5ea87347e46ae6a3dbf7562d5e1224cfecaacf63e0",
  "claim_path": "/workspace/faithkv-post-stage-c-diagnostic-execution-r2/authorization-claim.json",
  "cpu_offload": false,
  "events_per_example": 1,
  "exact_command": "kvcot run-post-stage-c-diagnostic-pilot --authorization-document docs/POST_STAGE_C_DIAGNOSTIC_PILOT_R2_EXECUTION_AUTHORIZATION_2026-07-25.md --execute",
  "implementation_audit_path": "/workspace/faithkv-audits/faithkv-post-stage-c-diagnostic-pilot-r2-implementation-audit.md",
  "implementation_audit_sha256": "34e3ebdc13b12ece5c93c6b890fe22d40b30309a9ad5735684db402ee80d25cf",
  "implementation_audit_verdict": "PASS",
  "kv_restore_widths": [
    1,
    2
  ],
  "maximum_candidate_pool_size": 4,
  "maximum_invocations": 1,
  "maximum_pairs_per_selected_example": 9,
  "maximum_qualification_candidates": 8,
  "maximum_selected_events": 3,
  "maximum_selected_examples": 3,
  "maximum_total_pairs": 27,
  "model_revision": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
  "output_root": "/workspace/faithkv-post-stage-c-diagnostic-execution-r2",
  "protocol_document_sha256": "8d5fd74e9a94ab5ba067ed7c6ffd9c23590593b1131813f53317693e37e2f3ee",
  "repository": "asad073-ui/Faithkv",
  "rkv_revision": "45eaa7d69d20b7388321f077020a610d9afb65bd",
  "runtime_config_canonical_sha256": "e2ea738c89a22a76f3cb71f5b4e01136a519183f916779908c1647a7d8a7c200",
  "runtime_config_path": "/workspace/faithkv-post-stage-c-diagnostic-runtime-r2/runtime_config.json",
  "runtime_limit_seconds": 5400,
  "single_rtx3090": true,
  "tokenizer_revision": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
  "vram_limit_bytes": 23622320128
}<!-- DIAGNOSTIC_PILOT_AUTHORIZATION_JSON_END -->
