# B2A-R3 Stage-C Execution Authorization (dated 2026-07-24)

This document authorizes exactly one B2A-R3 Stage-C execution claim at the
clean execution commit that contains this document.

Authority is limited to the committed fixed-path, provenance-gated B2A-R3
Stage-C entry point. This document does not authorize Stage-B reruns,
freezer reruns, B2B execution, model/config/dataset/manifest changes,
scientific semantic changes, R-KV pin changes, CPU offload, repeated
invocation, automatic retry, or any committed claim JSON.

The independently audited implementation commit is:

```text
817f4300fd237e1efe9de979a091539fde192991
```

The authorization verifier must reject use unless the current clean `HEAD`
is the later execution-authorization commit that contains this exact
document, the audited implementation commit above is its ancestor, and the
diff from that implementation commit to the execution-authorization commit
contains only this file.

Exact authorized command:

```bash
kvcot run-b2a-r3-stage-c \
  --authorization-document \
  docs/B2A_R3_STAGE_C_EXECUTION_AUTHORIZATION_2026-07-24.md \
  --execute
```

Frozen Stage-C bindings:

- Maximum invocations: 1.
- Automatic retries: 0.
- GPU: exactly one RTX 3090.
- Maximum runtime: 4 GPU-hours.
- Maximum VRAM: 22 GiB.
- CPU offload: prohibited.
- Fixed selected row: `test/number_theory/631.json`.
- Partial evidence: must be preserved.
- Stage C only: no B2B.

<!-- BEGIN B2A-R3 AUTHORIZATION JSON -->
```json
{
  "authorization_document_schema_version": "faithkv-b2a-r3-stage-authorization-document-v2",
  "authorization_id": "stage-c-2026-07-24-r1",
  "authorization_stage": "b2a_r3_execution",
  "authorized_repository": "asad073-ui/Faithkv",
  "authorized_branch": "research/b2a-r3-runtime-qualified-calibration",
  "authorized_code_commit_sha": "817f4300fd237e1efe9de979a091539fde192991",
  "required_ancestor_shas": [],
  "required_rkv_sha": "45eaa7d69d20b7388321f077020a610d9afb65bd",
  "candidate_manifest_canonical_sha256": "b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42",
  "maximum_candidates": null,
  "phase_wall_time_limit_seconds": null,
  "qualification_artifact_canonical_sha256": "4349edc97a273819d4f5a3e75812af80437971f584071b66b25c858ffa02ff1d",
  "selected_manifest_sha256": "dea628339f4b82678fa18bdc86c8dafa11c5ed87714a8b3a79b588884cab02e0",
  "selected_manifest_hash_algorithm": "B2AOneExampleManifest.manifest_hash-v1",
  "created_at_utc": "2026-07-24T18:07:04+00:00"
}
```
<!-- END B2A-R3 AUTHORIZATION JSON -->
