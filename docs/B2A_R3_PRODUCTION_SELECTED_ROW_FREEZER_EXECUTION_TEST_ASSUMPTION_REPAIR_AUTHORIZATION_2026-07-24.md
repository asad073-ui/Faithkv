# B2A-R3 Production Selected-Row Freezer Execution Test-Assumption Repair Authorization — 2026-07-24

Exact-SHA GitHub Actions CPU CI on the freezer execution-acceptance commit
(`6e808cc4a57bd7d0c653eef003499d48f883bff7`, "Persist accepted B2A-R3
selected-row freeze"), run
[30109659503](https://github.com/asad073-ui/Faithkv/actions/runs/30109659503),
failed with exactly 1 of 1904 non-GPU tests failing (1899 passed, 4 skipped
-- the 4 real cache-dependent tests, expected to skip on a CI runner
without the tokenizer snapshot cached -- 14 deselected):

```text
tests/unit/discovery/test_manifest.py::test_frozen_manifest_file_loads_and_validates
AssertionError: assert 287 == 365
```

## Root cause

This test's own docstring states it encodes
`example_index=365 (unique_id="test/number_theory/820.json") since the
B2A-R2 row freeze` -- a hard-coded golden value from the *prior* B2A-R2
freeze, asserted as if it were a permanent invariant of
`load_b2a_one_example_manifest()` rather than a snapshot of whatever row
happens to be currently frozen. The B2A-R3 production freezer execution
accepted in the immediately preceding commit intentionally and correctly
replaced that row with `test/number_theory/631.json`
(`example_index=287`), per the accepted Stage-B qualification evidence
(§1n/§1o) and the independently audited freezer repair (§1s) and execution
(§1t). This is a test-environment/test-assumption defect exposed by an
intended, authorized repository state change -- exactly the same shape of
defect §1o already repaired once for a different pair of tests -- not a
production code defect.

`src/kvcot/discovery/manifest.py`,
`configs/discovery/b2a_one_example_manifest.json`,
`results/decisions/b2a_r3_selection_provenance.json`, and every other file
under `src/`, `configs/`, `third_party/R-KV/`, and `results/` are
unaffected by this repair.

```text
B2A-R3 PRODUCTION FREEZER EXECUTION TEST-ASSUMPTION REPAIR AUTHORIZED — TEST FILE ONLY

AUTHORIZED:
NARROW REPAIR OF ONE TEST THAT HARD-CODED THE PRIOR (PRE-B2A-R3) FROZEN
ROW'S IDENTITY AS A PERMANENT INVARIANT, UPDATED TO THE NEWLY ACCEPTED
FROZEN ROW, PLUS STRENGTHENED STRUCTURAL/SCHEMA ASSERTIONS THAT DO NOT
DEPEND ON WHICH ROW IS CURRENTLY FROZEN

PROHIBITED:
PRODUCTION SOURCE CHANGES
SCIENTIFIC CONFIGURATION CHANGES
SELECTED-MANIFEST OR SELECTION-PROVENANCE CHANGES
CANDIDATE-MANIFEST / QUALIFICATION-ARTIFACT / CONSUMED-CLAIM CHANGES
MODEL INFERENCE
FULLKV/R-KV EXECUTION
STAGE C
```

## Planned repair

- `test_frozen_manifest_file_loads_and_validates`: update the hard-coded
  `example_index == 365` / `unique_id == "test/number_theory/820.json"`
  golden values to the newly accepted frozen row
  (`example_index == 287`, `unique_id == "test/number_theory/631.json"`),
  correct the docstring to cite the B2A-R3 freezer execution acceptance
  (`docs/B2A_R3_PRODUCTION_SELECTED_ROW_FREEZER_EXECUTION_ACCEPTANCE_2026-07-24.md`)
  instead of the superseded B2A-R2 freeze, and add a stronger,
  row-identity-independent structural assertion (e.g. `unique_id` matches
  the `dataset_split/dataset_config` namespaced-path pattern, `len(
  raw_content_hash) == 64`, `prompt_token_ids_sha256` present and 64 lowercase
  hex characters) so a future accepted re-freeze cannot trivially re-break
  this test the same way twice.
- No other test, fixture, or assertion changes. No file under `src/`,
  `configs/`, `third_party/R-KV/`, or `results/` changes.

## Status

```text
B2A-R3 PRODUCTION FREEZER EXECUTION TEST-ASSUMPTION REPAIR AUTHORIZED —
TEST FILE ONLY;

REPAIR IMPLEMENTATION IS THE NEXT REQUIRED ACTION;
STAGE C REMAINS BLOCKED
```
