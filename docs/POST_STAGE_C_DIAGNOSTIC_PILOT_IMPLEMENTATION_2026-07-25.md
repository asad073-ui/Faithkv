# Post-Stage-C Diagnostic Pilot Implementation (2026-07-25)

This implementation follows the CPU authorization in
`POST_STAGE_C_DIAGNOSTIC_PILOT_IMPLEMENTATION_AUTHORIZATION_2026-07-25.md`.
It is not a GPU execution authorization and changes no Stage-C module,
threshold, evidence, claim, attempt, or R-KV submodule content.

The implementation adds an isolated `diagnostic_pilot_*` namespace with:

- pure pinned protocol and result-classification contracts;
- CPU-only runtime preparation and canonical hashing;
- outcome-blind mechanical qualification and candidate/event ordering;
- a diagnostic-only same-layer KV restore primitive accepting widths one
  and two and rejecting width four;
- separate FullKV and diagnostic R-KV subprocess roles;
- an atomic create-if-absent one-use claim whose complete, synced bytes are
  exposed through a same-directory hard link;
- an exception-preservation boundary beginning immediately after that claim,
  with attempt-local completion/final records and a claim-directory fallback
  if attempt creation itself fails;
- claim-call failure preservation when create-if-absent has already consumed
  the path but a subsequent claim write or sync operation raises;
- non-consuming dry-run preflight;
- immutable attempt artifacts, partial-failure preservation, independent
  primitive-to-summary reconstruction, and final reference hashes;
- the three requested CLI command families.

The ordinary FullKV worker is reused unchanged. The diagnostic R-KV worker
uses the existing deterministic selectors, natural Pass 1, token-identical
score-only replay across all eligible events, a one-event Pass 2 snapshot,
strict local model loaders, placement checks, R-KV runtime verification,
restore-once branch stepper, and fixed 48-token NLL convention. Score replay
reduces and releases each event record immediately and retains no full-model
snapshot; only the already-frozen event receives a complete snapshot.
It never imports or calls Stage-C orchestration.

For each candidate row, FullKV completes in its own process before the
diagnostic R-KV process starts. R-KV constructs every event/candidate choice
from deployable pre-intervention scores. It evaluates no branch for a row
that fails mechanical qualification. The first three passing rows stop the
canonical scan; fewer than three yields the predeclared mechanically
unqualified result.

For each eligible event, layer and KV head are derived with the existing
seeded selectors; the event score is the greatest existing R-KV final score
among eligible evicted tokens at that frozen target. Event ties use event,
layer, then KV-head index. The existing seeded donor selector supplies one
absolute donor identity before intervention. An event is mechanically
excluded if that identity has no corresponding post-event slot in both KV
heads, allowing Arm B to change only head width rather than donor identity.

The control is Arm A's first candidate and is not recomputed. One baseline
branch is reused within an example; up to four width-one candidate branches,
one width-two branch, and one no-op branch are evaluated sequentially. The
no-op requires exact scored token IDs, exact float NLL arrays, zero gain,
identical final state hashes, and a primitive no-op mutation report.

Generation is capped at 2,048 new tokens so the frozen worst-case eight-row
scan remains bounded inside the later 1.5-hour authorization ceiling without
removing an arm or reducing the three-example target.

The resulting upper bound is 65,536 natural/replay decode selections (eight
rows times one FullKV and at most three R-KV passes times 2,048) plus 1,029
branch feed steps (three examples times seven baseline/intervention branches
times 49 feeds). These values are repeated in the canonical runtime artifact.

Arm C stores compact token-level log-probability primitives only at frozen
answer-span positions that occur inside the evaluated 48-token branch. It
labels fixed-trace extraction separately and never claims a free-running
answer flip.

All production Torch, Transformers, and R-KV imports are deferred to worker
bodies. Importing the CLI, coordinator, dry-run, contracts, authorization,
or worker module does not import Torch or Transformers and cannot initialize
CUDA.

The final preflight also hashes the checked-out protocol document, diagnostic
config, and frozen candidate manifest and compares those bytes with the
canonical runtime binding. Attempt reconstruction validates model/R-KV
identity, the 12-query-head/2-KV-head architecture, candidate ranks and
control identity, per-head mutation counts and donor slots, and primitive
answer-token log probabilities before accepting any derived projection or
summary.

Runtime verification reconstructs every candidate prompt from the exact
candidate-manifest row and pinned tokenizer and requires exact agreement with
the external prompt artifact, including qualification order. The execution
authorization output root must equal the runtime binding. Reconstruction also
re-derives FullKV/R-KV row identity, correctness, placement/device evidence,
compression, cap status, qualification, and completion limits from raw worker
records; embedded final summary/completion copies must equal their referenced
files.

No worker is launched with a nonpositive remaining budget, R-KV cap hits fail
mechanical qualification, and a result cannot complete beyond the 5,400-second
ceiling. Allocated and reserved CUDA peaks remain separate while their maximum
is the tracked ceiling value. Any exception after success-shaped immutable
artifacts appear adds a distinct failure marker without rewriting evidence.
