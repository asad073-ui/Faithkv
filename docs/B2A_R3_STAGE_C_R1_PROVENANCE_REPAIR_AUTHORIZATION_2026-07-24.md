# B2A-R3 Stage-C R1 Provenance-Collector Repair Authorization

Date: 2026-07-24

This document authorizes one bounded CPU-only repair to
`src/kvcot/discovery/attempt_artifacts.py`'s `collect_execution_provenance`,
following the first real Stage-C execution attempt.

## Failed Attempt Identity

```text
failed execution SHA:
df6f006a7483b11efddeca271f897577b551fd1b

consumed authorization:
stage-c-2026-07-24-r1

consumed claim canonical hash:
56c6807e97aee9db8cde1c5350adbf54fd0e3a8f80f3e66bd6eecb138d39dfe8

failed attempt ID:
d88e5c48443947e0a7eaf5bc0464789c

forensic audit hash:
bd779a895a1039df06796872e20723f58db5915c631d49d62522371a5aa4ed75

failure:
KeyError on the backward-compatibility starting-ancestor field

CUDA initialized:
false

workers launched:
false

scientific gate evaluated:
false
```

Independent root-cause reconstruction:
`/workspace/faithkv-audits/faithkv-b2a-r3-stage-c-r1-provenance-defect-reconstruction.md`
(SHA-256: `f118e802782c62a998000ba9c3a5ebea4e8232a774c639303688194d16b38ecb`).

## Root Cause (summary)

`collect_execution_provenance` unconditionally indexes the literal legacy
SHA `3c853cff34e52d792cd0e5a96d1a5369f17f8047` into the `ancestry` dict it
just built from a caller-supplied `required_ancestor_shas` tuple. Stage-C
(`b2a_r3_stage_c.py:578`) legitimately supplies a custom tuple beginning
with `plan.authorized_code_commit_sha` that has no reason to contain that
B1-era SHA, so the dictionary lookup raised `KeyError` after claim
consumption but strictly before device preflight, CUDA initialization, or
any worker launch. `stage-c-2026-07-24-r1` is consumed and permanently
non-reusable as a direct consequence.

## Authorized Repair

Authorized changes are limited to:

- `src/kvcot/discovery/attempt_artifacts.py`: make the legacy
  `starting_ancestor_verified` field computed independently of whatever
  `required_ancestry` map was built for the caller's actual required
  ancestors, so a custom Stage-C tuple that omits the legacy SHA never
  triggers a `KeyError`, never fabricates `True`, never mutates
  `required_ancestry` to insert the legacy key, and still performs a real
  Git ancestry check.
- Auditing (not assuming) the `None` vs. `()` contract of
  `required_ancestor_shas` against existing callers and tests, and, if and
  only if warranted by that audit, replacing the truthiness fallback with
  an explicit `is None` branch with equivalent behavior for every existing
  caller.
- Tests for `attempt_artifacts.py` and Stage-C orchestration/provenance
  covering the exact R1 regression shape.
- Minimal documentation/governance updates recording this repair
  (`CLAUDE.md`, `CHANGELOG.md`, `PLAN.md`, and the implementation document
  this authorization requires next).

## Prohibitions

This repair does not authorize changes to:

- `src/kvcot/discovery/b2a_workers.py`, scientific coordinator logic, swap
  logic, no-op logic, bridge logic, scoring-window logic, pair
  calculations, runtime formulas, VRAM formulas.
- `configs/`, `results/`, `third_party/R-KV/`.
- The selected manifest, selection provenance, qualification artifact, or
  candidate manifest.
- The consumed R1 claim or the R1 attempt directory (both remain in the
  original evidence checkout, `/workspace/Faithkv`, untouched).

It also does not authorize:

- Real Stage-C execution, real claim consumption, or creation of a real
  Stage-C attempt directory.
- CUDA initialization, model/tokenizer loading for execution, R-KV import,
  FullKV/R-KV worker launch.
- Any new one-use Stage-C execution authorization (that is a separate,
  later artifact, ID `stage-c-2026-07-24-r2`, created only after this
  repair passes independent audit).
- Merge, rebase, amend, or force-push.

All repair work happens only in the clean clone at
`/workspace/Faithkv-stagec-repair`. The original evidence checkout at
`/workspace/Faithkv` is never modified, reset, cleaned, stashed, or pulled
by this authorization.

The repaired implementation must be committed separately from this
authorization, validated with CPU tests and exact-SHA CI, and submitted to
a fresh independent auditor — one that did not implement the repair or
author its tests — before any Stage-C execution authorization may be
created.
