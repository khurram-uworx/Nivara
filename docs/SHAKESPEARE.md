# SHAKESPEARE — Batched MatMul + TinyShakespeare Transformer

## Purpose

This document breaks the NivaraChat-NEXT "Idea A" workstream (see `samples/NivaraChat/NEXT.md`) into concrete, assignable tasks for coding agents. The workstream has two linked goals:

1. **Improve Nivara's MatMul story** — add batched (rank-3) MatMul support and reduce allocation pressure in the existing rank-2 forward/backward path, following the discipline: *implement → measure → ensure correctness → improve → measure*.
2. **Land Idea A** — train a batched causal transformer on TinyShakespeare with Nivara, wrap it in a standard `Microsoft.Extensions.AI.IChatClient`, and prove it is DI-wireable in the .NET AI ecosystem.

Correctness is gated by the NivaraTorch parity technique (PyTorch reference fixtures) *before* the sample is built. This is a single end-to-end plan; the sample is the payoff, not an afterthought.

## How To Use

- Replace-agnostic: tasks are sized so one coding agent can own each end-to-end.
- Tasks 1–5 are core library + measurement + correctness; Task 6 is the sample; Task 7 is the finale (docs + roadmap retirement).
- The two **standing instructions** below are deliberately repeated inside every task so each task is a self-contained handout. Do not skip them.

## Standing Instructions (apply to EVERY task)

1. **Root-cause failing tests.** When an existing test fails after a code change, do not rush to a quick fix. Take a breath and assess the root cause first: is the failure caused by the change (a real regression), or has the test's expectation become invalid because the contract/behavior intentionally changed? Do the right engineering thing — fix the code if it is a bug, update the test only if the expectation is genuinely stale (and document why), and add coverage if it is missing. Never patch symptoms.
2. **File issues for the backlog.** While working, if you identify any interesting thing, gap, loop, or problem (surprising allocation pattern, missing op, kernel inefficiency, doc inconsistency, API smell), create a GitHub issue so it lands on the backlog. Use `gh issue create --repo khurram-uworx/Nivara`. Write the body to a temp file first and use `--body-file` to avoid PowerShell backtick/backslash escaping problems. Do not fix silently — capture it.

## Suggested Execution Order

1. Task 1: baseline measurements
2. Task 2: batched MatMul core (API-shape decision gate)
3. Task 3: allocation reduction in rank-2 MatMul
4. Task 4: correctness gate (PyTorch parity fixtures + tests)
5. Task 5: re-measure and data-driven improvement
6. Task 6: batched transformer chat sample (Idea A)
7. Task 7: finalize docs — update README, delete NEXT.md

## Coordination Notes

- Tasks 2 and 3 both touch `ReverseGradOperations.cs`, `GradKernels.cs`, and `TensorsHelper.cs`. Assign them to the same agent, or sequence them, to avoid merge conflicts.
- Task 4 depends on Tasks 2 and 3 (tests reference the new APIs).
- Task 5 depends on Tasks 1–3 (needs before/after numbers).
- Task 6 depends on Task 4 passing (the correctness gate) and Task 2 (the batched op it dogfoods).
- Task 7 depends on Task 6 (only retire the roadmap after the feature lands).
- **`gen_reference.py` RNG discipline:** new fixture cases must be appended at the *end* of `run()` — never inserted mid-stream — or every subsequent fixture changes. Commit regenerated fixtures and the C# manifest as one unit.
- Shared files that may conflict: `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`, `src/Nivara/Tensors/TensorsHelper.cs`, `tests/Nivara.PerformanceTests/Program.cs`, `Nivara.slnx`.

---

## Task 1: Baseline MatMul measurement scenarios

### Priority

High

### Goal

Extend `tests/Nivara.PerformanceTests` with MatMul-focused scenarios at transformer-relevant shapes so the before-numbers for time and allocations exist before any core change.

### Why this exists

There are no measurements for MatMul at the shapes Idea A will use (batch×seq × embed, attention scores). Without a baseline we cannot quantify improvement in Task 5.

### Decision required

None.

### Scope

- Add rank-2 `ReverseGradOperations.MatMul` scenarios at transformer shapes: `[2048, 64] @ [64, 64]` (B·L × D), `[32, 64] @ [64, 64]`, and attention-score shapes `[B·H, L, D/H]`.
- Cover both forward and forward+backward (the training step cost).
- Add a "flattened attention workaround" scenario replicating the NEXT.md approach, as the direct comparison point for the batched op.
- Record `ops/s`, `ns/op`, `B/op`, `gen0/op` for each scenario.

### Constraints

- Use the existing hand-rolled `Run` harness in `Program.cs` — do not add BenchmarkDotNet.
- Measure the default hot path (diagnostics disabled); optionally note the enabled-path cost.

### Suggested implementation path

- Mirror the existing `Linear forward [32x256]` / `TransformerBlock forward` scenario patterns.
- Use the transformer config from NEXT.md (B=16, L=128, D=64, H=4) so numbers transfer to Task 6.

### Acceptance criteria

- Harness prints every MatMul scenario with before-numbers.
- Results are saved (pasted into this doc or a notes file) for comparison in Task 5.

### Files likely involved

- `tests/Nivara.PerformanceTests/Program.cs`

### Standing instructions (apply to this task)

1. If an existing test fails after a code change, root-cause first: fix the code if it is a bug, update the test only if its expectation is genuinely invalid, add coverage if it is missing — never patch symptoms.
2. If you identify any interesting gap/problem while working, file a backlog issue with `gh issue create --repo khurram-uworx/Nivara` (use `--body-file` with a temp file).

---

## Task 2: Batched MatMul core support

### Priority

High

### Goal

Add batched (rank-3) MatMul to core: a shared-B overload `[B, M, K] @ [K, N] -> [B, M, N]` (Q/K/V projections, MLP — same weight per batch) and a batched-B overload `[B, M, K] @ [B, K, N] -> [B, M, N]` (attention scores), plus `ReverseGradOperations.BatchedMatMul` with a correct backward pass.

### Why this exists

NEXT.md's attention workaround flattens `[B*H, L, D/H]` and pays cross-batch waste (or loops per-head). A proper batched op is the real fix — it is the library gap Idea A exposes, and it makes the batched transformer clean.

### Decision required

- API shape: overloads vs. distinct method names; where the batch dimension lives; whether a batched `MatMulTransposedB` (inference, transpose-free) variant is included. Keep symmetry with the existing `MatMul` / `MatMulTransposedB` pair.

### Scope

- `TensorsHelper.BatchedMultiplyCore` with float/double `MemoryMarshal.Cast` dispatch and `ShouldParallelize` mirroring `MultiplyCore`.
- Shared-B overload hoists B's transpose out of the batch loop (transpose once, reuse for every batch element).
- `GradKernels.BatchedMatMul`.
- `ReverseGradOperations.BatchedMatMul` (+ batched `MatMulTransposedB` for inference) with per-batch grads for `a` and grads for shared `b` accumulated across the batch.
- Shape validation with clear exceptions.

### Constraints

- Reverse-mode graph nodes only created inside `GradientUtils.Grad()` (inference-default direction).
- Follow existing span / `TensorPrimitives` conventions; no `NivaraColumn.Data` access in kernels.
- Type constraint `IFloatingPointIeee754<T>`.

### Suggested implementation path

- Reuse the row-dot kernel; loop the batch dimension and share the transposed B buffer for the shared-B overload.
- Mirror the existing `MatMul` backward structure (`OpNode<T>`, `AccumulateGradient`).

### Acceptance criteria

- Batched forward numerically matches a per-batch loop of rank-2 `MatMul`.
- Backward grads for shared `b` equal the sum of the per-batch grads.
- Incorrect shapes throw clear exceptions.
- `dotnet build Nivara.slnx` passes.

### Files likely involved

- `src/Nivara/Tensors/TensorsHelper.cs`
- `src/Nivara/AutoDiff/Operations/GradKernels.cs`
- `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`
- `tests/Nivara.Tests/AutoDiff/GradOperationsTests.cs` (or a new `BatchedMatMulTests.cs`)

### Standing instructions (apply to this task)

1. If an existing test fails after a code change, root-cause first: fix the code if it is a bug, update the test only if its expectation is genuinely invalid, add coverage if it is missing — never patch symptoms.
2. If you identify any interesting gap/problem while working, file a backlog issue with `gh issue create --repo khurram-uworx/Nivara` (use `--body-file` with a temp file).

---

## Task 3: Allocation reduction in rank-2 MatMul

### Priority

Medium

### Goal

Reduce transient allocations in `ReverseGradOperations.MatMul` forward/backward by pooling the ~3 transient transpose arrays via `ArrayPool<T>.Shared`.

### Why this exists

Each call allocates ~3 transient `new T[]` transpose buffers (plus owned results). Transformers invoke MatMul hundreds of times per training step, so this shows up as gen0 pressure and `B/op`.

### Decision required

None — owned result arrays must stay owned by `NivaraColumn` (it wraps and owns the buffer, so those cannot be pooled).

### Scope

- Pool the forward `bTArr` and the backward `bTArr`/`aTArr` transpose buffers.
- Keep result/grad-result arrays owned by `NivaraColumn` (unavoidable).

### Constraints

- Return every rented array in a `finally` block (`clearArray: true`).
- Zero behavioral change; outputs must be identical.

### Suggested implementation path

- Mirror the `ArrayPool` pattern already used inside `TensorsHelper.MultiplyCore*`.
- Consider a small rent-transpose-return helper to keep the op readable.

### Acceptance criteria

- `B/op` for MatMul forward and forward+backward measurably reduced vs. the Task 1 baseline.
- All existing MatMul tests still pass.

### Files likely involved

- `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`
- `tests/Nivara.PerformanceTests/Program.cs` (re-run baseline scenarios to confirm the reduction)

### Standing instructions (apply to this task)

1. If an existing test fails after a code change, root-cause first: fix the code if it is a bug, update the test only if its expectation is genuinely invalid, add coverage if it is missing — never patch symptoms.
2. If you identify any interesting gap/problem while working, file a backlog issue with `gh issue create --repo khurram-uworx/Nivara` (use `--body-file` with a temp file).

---

## Task 4: Correctness gate — PyTorch parity fixtures and tests

### Priority

High

### Goal

Append batched MatMul cases to `samples/NivaraTorch/gen_reference.py` and add NivaraTorch-style NUnit tests in `tests/Nivara.Tests/NivaraTorch/` that compare forward *and backward* against PyTorch fixtures.

### Why this exists

The NivaraTorch parity technique is the project's correctness gate before any sample lands. Batched MatMul is new autograd surface; it must match PyTorch before it is dogfooded in Task 6.

### Decision required

None.

### Scope

- Append cases to `gen_reference.py` at the **end** of `run()`: batched shared-B forward, batched attention-scores forward, and backward (grads w.r.t. `a` and shared `b`).
- Regenerate fixtures and commit the full `samples/data/torch-comparison/` tree as one unit.
- New `tests/Nivara.Tests/NivaraTorch/BatchedMatMulTests.cs` comparing forward + backward to fixtures.
- Follow the RNG-order discipline (append only, never mid-stream).

### Constraints

- Fixtures + C# manifest are regenerated/committed together.
- Tests skip with a message when fixtures are missing (existing pattern).

### Suggested implementation path

- Mirror the existing `MatMul_MatchesPyTorch` / `MatMulTransposedB_MatchesPyTorch` test structure.
- For backward parity, compare gradients (not just forward outputs).

### Acceptance criteria

- All new parity tests pass.
- Existing NivaraTorch fixtures are unchanged (no mid-stream RNG perturbation).
- `dotnet test --filter "FullyQualifiedName~NivaraTorch"` passes.

### Files likely involved

- `samples/NivaraTorch/gen_reference.py`
- `samples/data/torch-comparison/*` (regenerated)
- `tests/Nivara.Tests/NivaraTorch/BatchedMatMulTests.cs`

### Standing instructions (apply to this task)

1. If an existing test fails after a code change, root-cause first: fix the code if it is a bug, update the test only if its expectation is genuinely invalid, add coverage if it is missing — never patch symptoms.
2. If you identify any interesting gap/problem while working, file a backlog issue with `gh issue create --repo khurram-uworx/Nivara` (use `--body-file` with a temp file).

---

## Task 5: Re-measure and data-driven improvement

### Priority

Medium

### Goal

Re-run the Task 1 harness, compare before/after, and drive any further kernel improvement strictly by data.

### Why this exists

This is the "we implement; we measure; we improve; we measure" loop. Only tune kernel details (cache blocking, parallel threshold, transpose caching) if the measurements show a real win; otherwise record the idea as a backlog issue.

### Decision required

Whether to make further kernel changes this cycle or capture them as backlog issues and stop.

### Scope

- Re-run the `tests/Nivara.PerformanceTests` MatMul scenarios.
- Compare batched vs. flattened-workaround vs. per-batch-loop of rank-2 (time and allocations).
- Record before/after numbers in this doc (results table).
- File issues for any identified inefficiency.

### Constraints

- No speculative optimization — every change must be justified by a measurement.
- Keep all regression tests passing.

### Suggested implementation path

- Add a small before/after results table to this document.
- If an optimization is justified, implement it and re-measure in the same task.

### Acceptance criteria

- Before/after numbers recorded in the doc.
- Every further optimization (if any) is backed by a measurement and an issue/notes.
- Build and relevant tests pass.

### Files likely involved

- `tests/Nivara.PerformanceTests/Program.cs`
- `src/Nivara/Tensors/TensorsHelper.cs` (only if an optimization is justified)
- `docs/SHAKESPEARE.md` (results table)

### Standing instructions (apply to this task)

1. If an existing test fails after a code change, root-cause first: fix the code if it is a bug, update the test only if its expectation is genuinely invalid, add coverage if it is missing — never patch symptoms.
2. If you identify any interesting gap/problem while working, file a backlog issue with `gh issue create --repo khurram-uworx/Nivara` (use `--body-file` with a temp file).

---

## Task 6: Build the batched transformer chat sample (Idea A)

### Priority

High

### Goal

Land `samples/NivaraChatClient/` per NEXT.md: a batched causal transformer trained on TinyShakespeare, wrapped in `IChatClient`, DI-wireable, using the new batched MatMul for attention.

### Why this exists

Dogfoods the new op end-to-end and completes Idea A. MicroGpt proved Nivara can *train* a transformer; NivaraChatClient proves it can *serve* one in an ecosystem-compatible way.

### Decision required

- Follow NEXT.md's proposed separate project `samples/NivaraChatClient/` (default).
- Before writing new core-adjacent code, verify what is already implemented (per AGENTS.md: `LayerNorm`, `Embedding` as `Module<T>` with batched `Forward`, softmax `dim`) — do not reimplement what exists.

### Scope

- `BatchedTransformer<T>` (`Module<T>`), `MultiHeadAttention` using the new `BatchedMatMul` (shared-B for projections, batched-B for scores), `PositionEncoding` (sinusoidal), word-level `TextTokenizer`, TinyShakespeare downloader.
- `NivaraChatClient : IChatClient` (`GetResponseAsync`, streaming, `GetService`, `Dispose`).
- Training via `TrainingLoop<T>` + `DataLoader<T>`, AdamW, `CrossEntropyLoss`, `ModelSerializer` save/load.
- CLI (`--train` / `--load` / `--prompt` / `--interactive`) and a DI wiring showcase.
- Register the project in `Nivara.slnx`.

### Constraints

- CPU-only, float32.
- No KV cache, top-p, or beam search (explicitly out of scope per NEXT.md).
- Expected training time: tens of minutes (~600 steps, ~1.4M params).

### Suggested implementation path

- Follow the NEXT.md architecture diagram; use the batched op for attention instead of the flatten workaround.
- Reuse `TextTokenizer` from `samples/Nivara.Samples` if the shape fits; otherwise a local copy per NEXT.md.

### Acceptance criteria

- Model trains on TinyShakespeare and saves/loads via `ModelSerializer`.
- `IChatClient` returns generated text; streaming yields token-by-token updates.
- DI wiring compiles (`AddChatClient<NivaraChatClient>()`).
- `dotnet build Nivara.slnx` passes; the sample runs end-to-end.

### Files likely involved

- `samples/NivaraChatClient/` (new: `Program.cs`, `BatchedTransformer.cs`, `MultiHeadAttention.cs`, `PositionEncoding.cs`, `NivaraChatClient.cs`, `TextTokenizer.cs`, `TinyShakespeareDownloader.cs`, `.csproj`)
- `Nivara.slnx`

### Standing instructions (apply to this task)

1. If an existing test fails after a code change, root-cause first: fix the code if it is a bug, update the test only if its expectation is genuinely invalid, add coverage if it is missing — never patch symptoms.
2. If you identify any interesting gap/problem while working, file a backlog issue with `gh issue create --repo khurram-uworx/Nivara` (use `--body-file` with a temp file).

---

## Task 7: Finalize docs — update README, delete NEXT.md

### Priority

High

### Goal

Update `samples/NivaraChat/README.md` to cover Idea A / `NivaraChatClient`, and delete `samples/NivaraChat/NEXT.md` — but **only after** confirming every item it asks for has been taken care of and nothing is left to salvage.

### Why this exists

NEXT.md was the roadmap; once Idea A lands, its purpose is served. Its durable content (architecture, CLI spec, gaps, exercises-vs-MicroGpt, not-doing list) must be preserved in the README before deletion. The explicit requirement: ensure everything asked for is taken care of and nothing is left to salvage.

### Decision required

None.

### Scope

- Salvage durable NEXT.md content into README: Idea A description, batched transformer architecture, CLI interface, gap decisions, the "What This Exercises vs. MicroGpt" comparison, and the not-doing list.
- Add the `NivaraChatClient` mode(s) to the README quick start + CLI table (or a clear pointer to `samples/NivaraChatClient/`).
- Cross-check every NEXT.md claim against the actual repo state. For anything done differently, record the delta. For anything still missing that is still valuable, **file an issue** (standing instruction #2) rather than silently dropping it.
- Delete `samples/NivaraChat/NEXT.md`.
- Update `AGENTS.md` references to NEXT.md if any exist.

### Constraints

- Deletion is the **last** step, only after the salvage review passes.
- Do not delete while real, un-addressed items remain — surface them first.

### Suggested implementation path

- Read `README.md` and `NEXT.md` side by side; transfer, then delete.
- Grep the repo for `NEXT.md` references to catch dangling links.

### Acceptance criteria

- README documents Idea A and the new sample without losing durable NEXT.md content.
- No dangling references to `NEXT.md` remain anywhere.
- `NEXT.md` is deleted.
- Backlog issues filed for anything valuable but un-addressed.

### Files likely involved

- `samples/NivaraChat/README.md`
- `samples/NivaraChat/NEXT.md` (deleted)
- `AGENTS.md` (only if it references NEXT.md)

### Standing instructions (apply to this task)

1. If an existing test fails after a code change, root-cause first: fix the code if it is a bug, update the test only if its expectation is genuinely invalid, add coverage if it is missing — never patch symptoms.
2. If you identify any interesting gap/problem while working (including NEXT.md content that turns out to be un-addressed), file a backlog issue with `gh issue create --repo khurram-uworx/Nivara` (use `--body-file` with a temp file).

---

## Additional Tasks

None — the seven tasks above cover the full workstream. If a task balloons in scope during execution, split it (keep one task per clear outcome) and record the reason in the doc.

## Suggested Agent Handout Batches

### Batch A: decision-critical

- Task 2 (batched MatMul API shape)
- Task 6 (sample scope decisions)

### Batch B: implementation

- Task 1
- Task 2
- Task 3
- Task 4
- Task 5

### Batch C: tests and docs

- Task 4 (parity tests — can move here if B batches it later)
- Task 6
- Task 7

## Final Checklist

- every task has a clear owner-sized scope — yes
- every task has acceptance criteria — yes
- decision-gate tasks are clearly marked — yes (Tasks 2 and 6)
- likely files are listed to reduce agent search time — yes
- execution order reflects real dependencies — yes
- standing instructions (root-cause analysis; file GitHub issues) are present in **every** task — yes
- Task 7 (README update + NEXT.md deletion) is last and gated on the salvage review — yes
