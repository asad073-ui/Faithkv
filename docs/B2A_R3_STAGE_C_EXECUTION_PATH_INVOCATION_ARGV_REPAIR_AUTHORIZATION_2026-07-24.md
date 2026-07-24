# B2A-R3 Stage-C Invocation Argv Repair Authorization

Date: 2026-07-24

This document authorizes one bounded CPU-only repair to the B2A-R3
Stage-C execution-path implementation at
`4d402fcb1004b6b3af7a2ae6bccad45e4039f847`.

## Audit Authority

- Independent implementation audit:
  `/workspace/faithkv-audits/faithkv-b2a-r3-stage-c-execution-path-independent-audit.md`
- Audit SHA-256:
  `70534ef20ea447104e2a1963ff9f6b10c193b42459f890e09852e7eb9d15837e`
- Verdict: FAIL
- Exact-SHA CI already green for audited implementation:
  run `30113990698`, SHA `4d402fcb1004b6b3af7a2ae6bccad45e4039f847`

## Blocking Finding

Normal Stage-C CLI execution writes the literal
`--authorization-document` flag and a filename containing `AUTHORIZATION`
into `invocation.json["argv"]`. The existing final attempt verifier treats
argv entries containing `authorization` as credential-like and rejects the
attempt. That could consume the one-use Stage-C authorization, run workers,
and then fail final attempt verification from the Stage-C command's own
required flag.

## Authorized Repair

Authorized changes are limited to:

- Sanitizing Stage-C `invocation.json["argv"]` so the required Stage-C CLI
  flag and document path cannot trigger the existing credential-marker
  rejection.
- Keeping the exact authorization-document path and command identity in
  dedicated Stage-C evidence fields (`authorization_document_path` and
  `stage_c_binding.json`) rather than unsanitized argv.
- Adding CPU tests proving a Stage-C CLI-shaped invocation writes verifier-
  compatible sanitized argv.
- Minimal documentation/status updates for this repair.

## Prohibitions

This repair does not authorize:

- Real Stage-C execution.
- Real Stage-C claim consumption.
- Creation of a real Stage-C attempt directory.
- CUDA initialization, model/tokenizer loading for execution, R-KV import,
  or worker launch.
- Changes to selected manifest, selection provenance, candidate evidence,
  qualification evidence, config values, R-KV pin, worker math, thresholds,
  swap/no-op/bridge/scoring-window/attrition semantics, or scientific gates.
- Merge, rebase, reset, amend, or force-push.

The repaired implementation must be committed separately, validated with
CPU tests and exact-SHA CI, and submitted to a second fresh independent
auditor before any one-use Stage-C execution authorization is created.
