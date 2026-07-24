# B2A-R3 Stage-C Execution-Path Implementation

Date: 2026-07-24

This implements the missing fixed-path, provenance-gated B2A-R3 Stage-C
entry point authorized by
`docs/B2A_R3_STAGE_C_EXECUTION_PATH_IMPLEMENTATION_AUTHORIZATION_2026-07-24.md`.
It does not execute Stage C in this session and does not consume a real
Stage-C authorization.

## Readiness-Audit Failure Addressed

The Stage-C readiness audit failed because the repository had no production
Stage-C command, no production consumer for a `b2a_r3_execution`
authorization claim, and no fixed-path replacement for the older
caller-configurable `b2a-calibrate --execute` path.

## Reuse Map

- Frozen config loading: `load_discovery_config()` and `config_identity()`
  against `configs/discovery/llama8b_math500_b1024.yaml`.
- Selected manifest loading: `B2AOneExampleManifest.model_validate_json()`
  against `configs/discovery/b2a_one_example_manifest.json`.
- Candidate verification: `verify_candidate_manifest_structure()`.
- Qualification verification: `verify_qualification_artifact()`.
- Persisted Stage-B authorization binding:
  `verify_persisted_stage_b_authorization_binding()`.
- Selection-provenance verification: `verify_selection_provenance()`.
- Authorization document parsing:
  `parse_authorization_document()` / `AuthorizationDocumentR3`.
- Semantic authorization verification:
  `verify_authorization_preconditions()`.
- Single-use claim consumption: `claim_authorization()`.
- Attempt-directory naming:
  `b2a_r3_contract.attempt_directory_name()`.
- Immutable JSON writes: `atomic_write_json()`.
- CPU-safe provenance collection: `collect_execution_provenance()`.
- Device/RTX-3090 preflight: `verify_single_rtx3090()`.
- FullKV/R-KV subprocess execution and final artifact writing:
  `run_b2a_calibration()`.
- Pair artifacts and final reference-manifest verification:
  existing `b2a_execute.py` calls to `verify_pair_record_artifacts()`,
  `verify_attempt_artifacts()`, and `verify_final_reference_manifest()`.

No FullKV/R-KV worker math, swap evaluation, bridge logic, scoring window,
NLL calculation, attrition, runtime projection, VRAM gate, or pair
verification formula was reimplemented.

## Entry Point

New public APIs:

- `plan_b2a_r3_stage_c_execution(authorization_document_path=..., repository_root=".")`
- `run_b2a_r3_stage_c_execution(authorization_document_path=..., repository_root=".")`

The public APIs do not accept alternate config, candidate, qualification,
manifest, provenance, output, attempt directory, model/tokenizer identity,
dataset identity, R-KV path, budget, seed, event count, threshold, device
metadata, runner callback, force, resume, or retry arguments.

New CLI:

```text
kvcot run-b2a-r3-stage-c --authorization-document PATH --dry-run
kvcot run-b2a-r3-stage-c --authorization-document PATH --execute
```

Exactly one mode is required.

## Execution Ordering

The execute path is ordered as follows:

1. Load and verify fixed CPU artifacts.
2. Parse and verify the committed Stage-C authorization document.
3. Construct the Stage-C claim payload internally.
4. Run semantic authorization preconditions.
5. Reverify Git/worktree state immediately through `claim_authorization()`.
6. Consume the authorization claim atomically.
7. Confirm the deterministic claim exists.
8. Create the exact claimed B2A-R3 attempt directory.
9. Write invocation and `stage_c_binding.json`.
10. Write CPU-safe provenance while filtering only the consumed claim path.
11. Reverify post-claim provenance with active claim/attempt paths.
12. Perform RTX-3090/CUDA device preflight.
13. Invoke the existing scientific coordinator.
14. Verify the final reference manifest.
15. Return a typed result.

No CUDA, tokenizer, model, R-KV import, worker subprocess, or device
inspection occurs before successful claim consumption.

## Binding Artifact

`stage_c_binding.json` is written before any GPU action and includes the
authorization ID, document path/hash, claim hash/path, authorized code SHA,
execution authorization commit SHA, repository and branch, R-KV SHA, fixed
config/candidate/qualification/selection hashes, selected row,
model/tokenizer/dataset revisions, attempt ID/directory, command identity,
single-invocation policy, runtime ceiling, VRAM ceiling, and CPU-offload
prohibition. The attempt-reference manifest now recognizes it with semantic
role `stage_c_binding`.

## Failure And Single-Use Semantics

If the claim path already exists, Stage C refuses before device preflight,
worker launch, or attempt creation. After claim consumption, failures
preserve the claim and any partial attempt evidence; `retry_allowed` is
false.

## Validation Scope

Implemented tests cover fixed public API shape, unsafe authorization
document paths, fixed-path dry-run with no writes, claim-before-device and
claim-before-coordinator ordering, single-use refusal, selected-row
tampering refusal, and CLI mode/argument behavior.

Actual execute was not invoked. A real Stage-C authorization document was
not consumed. A real Stage-C claim was not created. A real Stage-C attempt
directory was not created. Stage C remains blocked pending independent
implementation audit and a separate one-use execution authorization.
