# Post-Stage-C Diagnostic Pilot CPU Implementation Authorization (2026-07-25)

This document authorizes CPU-only implementation and testing of the protocol
in `docs/POST_STAGE_C_DIAGNOSTIC_PILOT_PROTOCOL_FREEZE_2026-07-25.md` from
starting SHA `ffc78bad874952e205032845b401df14516f2e69`.

It authorizes only:

- the corrected and frozen diagnostic protocol documents;
- new diagnostic-pilot contracts and schemas;
- a diagnostic-pilot manifest/qualification coordinator;
- separate FullKV and diagnostic R-KV worker entry points;
- the smallest diagnostic-only multi-KV-head restore primitive;
- prepare, verify, dry-run, and execution CLI entry points;
- CPU tests and synthetic fake workers/tensors;
- `PLAN.md`, `CHANGELOG.md`, and `CLAUDE.md` only where necessary to record
  this bounded authorization and status.

It explicitly prohibits:

- changing the frozen B2A-R3 `0.01`-nat threshold or any other frozen
  B2A-R3 scientific value;
- changing Stage-C orchestration, authorization semantics, R1/R2 evidence,
  either existing authorization claim, or either existing attempt directory;
- changing production R-KV behavior outside the new diagnostic path;
- GPU/CUDA initialization, model loading, pilot execution, Stage-C rerun,
  B2B, method implementation, multiple GPUs, CPU offload, or an unpinned
  model/tokenizer revision;
- committing generated claims, attempts, GPU evidence, or model weights.

Implementation must retain coordinator/process isolation, defer Torch and
Transformers imports on production worker paths, make dry-run non-consuming,
consume any later execution claim atomically before CUDA initialization, set
maximum invocations to one and automatic retries to zero, and preserve partial
evidence after consumption.

This is not an execution authorization. After CPU implementation, the exact
implementation SHA must pass CPU CI and a fresh independent read-only audit.
Only then may a separate one-use GPU execution authorization be created.
