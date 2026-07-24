# B2A-R3 Production Selected-Row Freezer Execution Acceptance — 2026-07-24

This document records the acceptance of the single authorized production
execution of the B2A-R3 selected-row freezer
(`kvcot freeze-b2a-r3-selected-row --execute`) into the repository. It is a
persistence record only: no Stage-C activity, no FullKV/R-KV execution, and
no further freezer execution is authorized, described, or implied by this
document.

## Repair and re-audit identity

```text
repair SHA (HEAD at execution time):  1ab9632d8fdfb525ad0adae5746a9d1b25ee0244
repair authorization SHA:             e5fbee6a1b8da9d7c36bf1caaf39128477083eb3
prior failed implementation SHA:      56d257236874d8947d5f127bf2074824cff62395
exact-SHA repair CI run:              30082773710 (conclusion: success, head SHA 1ab9632d8fdfb525ad0adae5746a9d1b25ee0244)
R-KV pin (unchanged):                 45eaa7d69d20b7388321f077020a610d9afb65bd
```

Independent freezer-repair re-audit (fresh, isolated, read-only auditor;
did not implement the repair):

```text
report:      /workspace/faithkv-audits/faithkv-b2a-r3-freezer-repair-independent-reaudit.md
report SHA-256: 178257e38ff812350518a2db7ab1e4fe886b08721c5401a8a51a5269c7d6c55c
verdict:     INDEPENDENT FREEZER REPAIR RE-AUDIT PASS
```

Required CPU suite reproduced by that audit: 1918 collected, 1904 passed,
14 deselected, 0 failed, 0 skipped. All 4 real cache-dependent tests and
all 3 real-evidence dry-run CLI tests passed for real (not skipped) on
this Vast.ai host.

## Freezer execution

Executed exactly once by the main controller (not the auditor), CPU-only,
offline (`CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`):

```text
command:            kvcot freeze-b2a-r3-selected-row --execute
exit code:           0
execution log:       /workspace/faithkv-audits/b2a-r3-freezer-execution.log
execution log SHA-256: d07cee75032a56f0683f97eaa65f08e6fdeee28f195d941279ef731d1f378de1
invocation count:    1 (single `.git` lock-held invocation; no retry)
publication_state_before: state_a_initial
publication_state_after:  state_b_complete ("complete" -- src/kvcot/discovery/b2a_r3_freeze.py PUBLICATION_STATE_B_COMPLETE)
already_frozen:      False
verification_passed: True
```

Dry-run performed immediately beforehand (`--dry-run`) reported exactly:
`would_freeze=True`, `selected_ordinal=1`,
`selected_unique_id=test/number_theory/631.json`,
`would_load_tokenizer_for_execution=False`,
`would_write_selected_manifest=False`,
`would_write_selection_provenance=False` -- consistent with a pure
metadata/tokenizer-hash freeze that never performed model inference.

## Selected row

```text
selected candidate ordinal:  1
selected unique ID:          test/number_theory/631.json
historical row (pre-freeze): test/number_theory/820.json
```

## Selected manifest

```text
path:                          configs/discovery/b2a_one_example_manifest.json
raw file byte SHA-256:         c22d57aef1a43e4ad0bad873b4c5e0b98247eb1191750ac15af5a057a0a580e7
manifest_hash (B2AOneExampleManifest.manifest_hash-v1): dea628339f4b82678fa18bdc86c8dafa11c5ed87714a8b3a79b588884cab02e0
```

The two hashes above are different quantities, not a discrepancy: the
freezer's own reported `selected_manifest_sha256` and the value recorded
in selection provenance are both the `manifest_hash-v1` canonical value
(`dea62833...`), never the raw pretty-printed file's byte hash
(`c22d57ae...`). Both are recorded here for independent auditability.

## Selection provenance

```text
path:                results/decisions/b2a_r3_selection_provenance.json
raw file byte SHA-256: 1f6ae8ec668ddfec8feba8840a03988ca9014b69964bb8dca14007cd809a9961
canonical_sha256:     2be6ef3097bf6362bfb029caee702cfd802ac95f00f8bb4a39ad83e6de593442
```

Provenance payload binds, and independently re-verifies against:

```text
qualification_artifact_canonical_sha256: 4349edc97a273819d4f5a3e75812af80437971f584071b66b25c858ffa02ff1d
candidate_manifest_canonical_sha256:     b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42
selected_manifest_sha256:                dea628339f4b82678fa18bdc86c8dafa11c5ed87714a8b3a79b588884cab02e0
selected_ordinal:                        1
selected_unique_id:                      test/number_theory/631.json
tokenizer_revision_used_for_prompt_hash: 6a6f4aa4197940add57724a7707d069478df56b1
```

## Frozen model / tokenizer identity (unchanged, reasserted here for binding)

```text
model:               deepseek-ai/DeepSeek-R1-Distill-Llama-8B
model revision:      6a6f4aa4197940add57724a7707d069478df56b1
tokenizer:           deepseek-ai/DeepSeek-R1-Distill-Llama-8B
tokenizer revision:  6a6f4aa4197940add57724a7707d069478df56b1
```

## Execution environment

```text
CPU-only:            CUDA_VISIBLE_DEVICES=""
Offline:             HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1
Model inference:     none (dry-run and execution logs both confirm no model
                     weights were opened; this is a metadata/tokenizer-hash
                     freeze operation only)
FullKV/R-KV execution: none
```

## Independent freezer-output audit

Fresh, isolated, read-only auditor (different from the repair re-auditor;
did not execute the freezer):

```text
report:       /workspace/faithkv-audits/faithkv-b2a-r3-freezer-output-independent-audit.md
report SHA-256: 2db7935e595df080bf0d89dea7e0843dda9f0885a03e5e9d5e78ad6ceabe56a4
verdict:      INDEPENDENT FREEZER OUTPUT AUDIT: PASS
```

That audit independently re-verified all three canonical hashes above, the
selected ordinal/row, the selected-manifest and selection-provenance
hashes, the complete provenance chain, that the only working-tree
differences are the two expected files, and that no CUDA/model
inference/dataset access occurred. It also ran
`kvcot verify-b2a-r3-selection` (pass) and
`tests/unit/discovery/test_b2a_r3_freeze.py` (73/73 passed), and flagged
one full-suite failure --
`tests/unit/discovery/test_manifest.py::test_frozen_manifest_file_loads_and_validates`
-- as a stale pre-freeze golden-value assumption (hard-codes
`example_index == 365` / `unique_id == "test/number_theory/820.json"` from
the prior B2A-R2 freeze), not a freezer defect. Per the same pattern as
§1o, this is intentionally **not** folded into this commit: it is left for
this commit's own exact-SHA CI to reproduce (or not), and addressed only by
a separate, narrow, bounded, dated repair authorization and implementation
afterward if confirmed.

## Explicit non-claims

* No R-KV or FullKV execution occurred during this freeze.
* No CUDA initialization or model-weight loading occurred.
* Stage C has not been authorized, planned in a binding way, or executed.
* No scientific threshold, budget, model, tokenizer, dataset, seed, or
  R-KV pin was changed.
* This document does not authorize a second freezer execution. The
  freezer's own `already_frozen` guard now reports `True` for any further
  `--dry-run`/`--execute` attempt against this repository state.
