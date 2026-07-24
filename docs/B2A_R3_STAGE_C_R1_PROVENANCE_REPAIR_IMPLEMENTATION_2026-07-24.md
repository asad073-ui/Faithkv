# B2A-R3 Stage-C R1 Provenance Repair Implementation - 2026-07-24

## Scope

This implementation consumes the repair authorization committed at
`6a43d5b7ed39b01c22fcbb71450306cfd059d785` and repairs only the Stage-C R1
provenance infrastructure failure. It does not authorize or execute another
Stage-C attempt.

The failed R1 execution authorization was:

- Authorization ID: `stage-c-2026-07-24-r1`
- Authorization commit: `df6f006a7483b11efddeca271f897577b551fd1b`
- Claim canonical SHA-256:
  `56c6807e97aee9db8cde1c5350adbf54fd0e3a8f80f3e66bd6eecb138d39dfe8`
- Attempt ID: `d88e5c48443947e0a7eaf5bc0464789c`
- Attempt directory:
  `results/decisions/b2a_r3_attempt_20260724T182645975076Z_d88e5c48443947e0a7eaf5bc0464789c`
- Forensic audit SHA-256:
  `bd779a895a1039df06796872e20723f58db5915c631d49d62522371a5aa4ed75`

R1 remains permanently consumed and non-retryable.

## Failure

R1 failed inside `src/kvcot/discovery/attempt_artifacts.py` with:

```text
KeyError: '3c853cff34e52d792cd0e5a96d1a5369f17f8047'
```

The failure happened during CPU-safe provenance collection, before device
preflight. CUDA initialization was not reached. Tokenizers, model weights,
FullKV, R-KV, and worker subprocesses were not launched.

The root cause was a provenance compatibility bug. Stage C supplies a
context-specific `required_ancestor_shas` tuple derived from the authorization
document. That tuple is not required to contain the historical B1 compatibility
SHA `3c853cff34e52d792cd0e5a96d1a5369f17f8047`. The old collector built
`required_ancestry` from the caller-specific tuple and then directly indexed
the historical SHA when filling `starting_ancestor_verified`.

## Repair

The repair introduces `LEGACY_STARTING_ANCESTOR_SHA` as a named constant and
preserves the compatibility fields:

- `starting_ancestor`
- `starting_ancestor_verified`

The collector no longer assumes the legacy SHA is present in the caller's
required ancestry set. If the legacy SHA is present, its computed ancestry
value is reused. If it is absent, the collector performs a separate real
`git merge-base --is-ancestor` check. A missing legacy commit now records
`starting_ancestor_verified = false` instead of raising or fabricating success.

The caller-specific `required_ancestry` remains exactly caller-specific. Stage C
does not silently inherit the historical B1 SHA.

The `required_ancestor_shas` contract is now explicit:

- `None` means use the historical `B1_REQUIRED_ANCESTOR_SHAS` defaults.
- `()` means explicitly require no context ancestors.

## Tests

Added focused provenance coverage in
`tests/unit/discovery/test_attempt_execution_provenance.py`:

- custom Stage-C-style ancestor tuple without the legacy SHA does not raise;
- the legacy compatibility field is computed independently;
- a custom tuple containing the legacy SHA reuses the existing ancestry result;
- a missing legacy commit reports `false`, not an exception;
- `None` preserves historical B1 defaults;
- an explicit empty tuple remains empty;
- provenance collection does not import `torch` as a side effect.

Added a Stage-C high-level regression in
`tests/unit/discovery/test_b2a_r3_stage_c.py` showing the real provenance
collector now reaches mocked device preflight and coordinator execution for the
same custom-ancestry shape that failed R1.

## Validation

Validation was run with CUDA hidden and Hugging Face offline mode enabled.

- `python -m compileall -q src tests`: passed
- Focused provenance tests: `7 passed`
- Stage-C focused tests: `10 passed`
- Affected focused tests:
  `tests/unit/discovery/test_attempt_artifacts.py`,
  `tests/unit/discovery/test_attempt_execution_provenance.py`, and
  `tests/unit/discovery/test_b2a_r3_stage_c.py`: `19 passed`
- Full CPU-safe suite: `1924 passed, 14 deselected, 0 failed, 0 skipped`
- Collection count: `1938 tests collected`

Frozen inputs were reverified:

- Config byte SHA-256:
  `de8ac65a348c307c4f00089da07914666332935981bcaa7c98a150a9e7e778b3`
- Candidate manifest canonical SHA-256:
  `b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42`
- Qualification artifact canonical SHA-256:
  `4349edc97a273819d4f5a3e75812af80437971f584071b66b25c858ffa02ff1d`
- Selected manifest SHA-256:
  `dea628339f4b82678fa18bdc86c8dafa11c5ed87714a8b3a79b588884cab02e0`
- Selection provenance canonical SHA-256:
  `2be6ef3097bf6362bfb029caee702cfd802ac95f00f8bb4a39ad83e6de593442`
- Selected row: `test/number_theory/631.json`
- R-KV SHA: `45eaa7d69d20b7388321f077020a610d9afb65bd`

The frozen YAML was rematerialized locally as CRLF in the repair checkout to
match `.gitattributes`. This was a checkout-materialization correction only and
is not part of the implementation commit.

## Non-Changes

No scientific behavior was changed. The repair does not modify B2A-R3
calculation logic, swap logic, no-op controls, bridge logic, scoring windows,
pair calculations, runtime projection, VRAM gates, model loading, tokenizer
loading, R-KV integration, configuration values, seeds, dataset selection, or
budgets.

R1 remains non-retryable. A new, distinct Stage-C authorization is required
before any future GPU execution.
