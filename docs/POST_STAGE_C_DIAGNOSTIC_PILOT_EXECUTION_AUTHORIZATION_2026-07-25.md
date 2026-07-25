# Post-Stage-C Diagnostic Pilot Execution Authorization

Date: 2026-07-25

This document authorizes one invocation of the bounded post-Stage-C diagnostic
pilot described by the frozen protocol. It does not authorize Stage C, B2B,
threshold changes, CPU offload, multiple GPUs, or an automatic retry. Once the
claim is created, this authorization is permanently consumed regardless of the
execution outcome.

<!-- DIAGNOSTIC_PILOT_AUTHORIZATION_JSON_BEGIN -->
{
  "authorization_id": "post-stage-c-diagnostic-pilot-2026-07-25-one-use",
  "authorized_branch": "research/post-stage-c-diagnostic-pilot",
  "authorized_implementation_sha": "961db4045297d30a9681e0bdc38360295d95ac82",
  "automatic_retries": 0,
  "candidate_manifest_canonical_sha256": "b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42",
  "canonical_sha256": "780ba2db1d7d77ce7d7c940b43c0813add5df8585effbc4d4d1ca061a6445bf8",
  "claim_path": "/workspace/faithkv-post-stage-c-diagnostic-execution/authorization-claim.json",
  "cpu_offload": false,
  "events_per_example": 1,
  "exact_command": "kvcot run-post-stage-c-diagnostic-pilot --authorization-document docs/POST_STAGE_C_DIAGNOSTIC_PILOT_EXECUTION_AUTHORIZATION_2026-07-25.md --execute",
  "implementation_audit_path": "/workspace/faithkv-audits/faithkv-post-stage-c-diagnostic-pilot-implementation-audit.md",
  "implementation_audit_sha256": "fa67ba2fc3558cd9532f30adcec1e42de74110b2441c7ce5bc7b3d97e7a83c42",
  "implementation_audit_verdict": "PASS",
  "kv_restore_widths": [
    1,
    2
  ],
  "maximum_candidate_pool_size": 4,
  "maximum_interventions_per_selected_example": 6,
  "maximum_invocations": 1,
  "maximum_qualification_candidates": 8,
  "maximum_selected_events": 3,
  "maximum_selected_examples": 3,
  "model_revision": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
  "output_root": "/workspace/faithkv-post-stage-c-diagnostic-execution",
  "protocol_document_sha256": "f0607b75e344bba4b4ad824c4a0c49df209cad66ea16c04be519904f4398f1e6",
  "repository": "asad073-ui/Faithkv",
  "rkv_revision": "45eaa7d69d20b7388321f077020a610d9afb65bd",
  "runtime_config_canonical_sha256": "f6d0b02805de7e9c4df04290a222c68e587a183231ebc6d4a0548e1c64d134ac",
  "runtime_config_path": "/workspace/faithkv-post-stage-c-diagnostic-runtime/runtime_config.json",
  "runtime_limit_seconds": 5400,
  "single_rtx3090": true,
  "tokenizer_revision": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
  "vram_limit_bytes": 23622320128
}
<!-- DIAGNOSTIC_PILOT_AUTHORIZATION_JSON_END -->

The exact authorized command is the `exact_command` value above. It may be
invoked at most once and only after the non-consuming preflight succeeds.
