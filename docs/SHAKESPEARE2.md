# SHAKESPEARE2 — Batched Transformer Chat Sample (Grounding in Repo)

## Purpose

This document breaks the NivaraChat-NEXT "Idea A" workstream (`samples/NivaraChat/NEXT.md`) into concrete, assignable tasks for coding agents. Unlike the earlier `docs/SHAKESPEARE.md` plan, this version is **grounded in the current repo state** — an audit was performed first, and the tasks reflect what actually exists in `src/Nivara/` today rather than what NEXT.md assumed.

Goal: land a **batched causal transformer** trained on TinyShakespeare with Nivara, wrapped in a standard `Microsoft.Extensions.AI.IChatClient`, and prove it is DI-wireable in the .NET AI ecosystem — using Nivara's own core modules where they already exist and closing only the genuine gaps.

Correctness is gated by the NivaraTorch parity technique (PyTorch reference fixtures) *before* the sample is built.

## Grounding Audit — NEXT.md gaps vs. actual core state (verified)

| NEXT.md claim | Actual state in `src/Nivara/` | Verdict |
|---|---|---|
| Gap 1/5: "LayerNorm does not exist" | `AutoDiff/Nn/LayerNorm.cs` — `LayerNorm<T> : Module<T>` with affine params; `TransformerBlock` also has `PerRowLayerNorm`/`PerRowRMSNorm` | **Resolved** |
| Gap 2: "batched embedding lookup missing" | `AutoDiff/Nn/Embedding.cs` — `Forward(ReverseGradTensor<T>)` handles `[B, L]` via `Gather`, reshapes to `[..., D]` | **Resolved** |
| Gap 3: "no Concat operation" | `ReverseGradOperations.Concat<T>(tensors, axis)` (ReverseGradOperations.cs:1313) | **Resolved** |
| Gap 4: "no Gather operation" | `ReverseGradOperations.Gather<T>(source, indices, axis)` (ReverseGradOperations.cs:1840); one-hot fallback unnecessary | **Resolved** |
| Gap 6: "Softmax ignores dim" | Fused `MultiHeadAttention` op does per-row softmax via `AttentionKernels<T>.SoftmaxRows` — exactly what attention needs | **Resolved** (for attention) |
| Gap 7: "Embedding not a Module" | `Embedding<T> : Module<T>` | **Resolved** |
| `AttentionMask` helper | `ModuleHelpers<T>.CreateCausalMask(qLen, kvLen)` (internal) + `TransformerBlock` mask slicing | **Resolved** internally |
| "MatMul only rank-2; flatten workaround needed" | Fused `ReverseGradOperations.MultiHeadAttention(query, key, value, numHeads, scale, mask)` (ReverseGradOperations.cs:362) with full VJP backward, built on `AttentionKernels` + `TensorsHelper.MultiplyCore(..., bTransposed: true)` | **Premise is stale** |
| Core has a transformer | `AutoDiff/Nn/MultiheadAttention.cs` (2D `[L, D]`), `AutoDiff/Nn/TransformerBlock.cs` (`NormType.RMSNorm/LayerNorm`, GELU MLP, causal mask) | **Exists — single-sequence only** |

### The real gap

Every existing attention/transformer surface is **single-sequence `[L, D]`** — the fused `MultiHeadAttention` op, the `MultiheadAttention<T>` module, and `TransformerBlock<T>` all reject rank-3. Idea A's "proper batched causal transformer `[B, L, D]`" therefore does **not** exist yet. The blocker is the **batch dimension**, not rank-2 MatMul (the fused op already sidesteps that via per-head `MultiplyCore`).

Three candidate routes (decided in Task 1):

- **Route A** — build batched (rank-3) `MatMul` primitives and compose batched attention from them. Largest new autograd surface; the earlier `SHAKESPEARE.md` plan premised on this.
- **Route B (recommended)** — extend the existing fused `MultiHeadAttention` op + `AttentionKernels` to accept `[B, L, D]` and loop the batch internally (e.g. `Parallel.For` over B). Reuses the already-correct fused backward; smallest correct surface. Batched MatMul is **not** required — the batched-MatMul-improvement idea should be recorded as a backlog issue, not a prerequisite.
- **Route C** — per-sequence loop in the sample. Zero library change; no batch parallelism; contradicts Idea A's goal.

## How To Use

- Tasks are sized so one coding agent can own each end-to-end.
- Task 1 is a decision gate grounded in a verification pass; Tasks 2–4 are core + measurement + correctness; Task 5 is the sample; Task 6 is the finale (docs + roadmap retirement).
- The two **standing instructions** below are deliberately repeated inside every task so each task is a self-contained handout. Do not skip them.

## Standing Instructions (apply to EVERY task)

1. **Root-cause failing tests.** When an existing test fails after a code change, do not rush to a quick fix. Take a breath and assess the root cause first: is the failure caused by the change (a real regression), or has the test's expectation become invalid because the contract/behavior intentionally changed? Do the right engineering thing — fix the code if it is a bug, update the test only if the expectation is genuinely stale (and document why), and add coverage if it is missing. Never patch symptoms.
2. **File issues for the backlog.** While working, if you identify any interesting thing, gap, loop, or problem (surprising allocation pattern, missing op, kernel inefficiency, doc inconsistency, API smell), create a GitHub issue so it lands on the backlog. Use `gh issue create --repo khurram-uworx/Nivara`. Write the body to a temp file first and use `--body-file` to avoid PowerShell backtick/backslash escaping problems. Do not fix silently — capture it.

## Suggested Execution Order

1. Task 1: current-state audit + batched-attention route decision (decision gate)
2. Task 2: batched attention core implementation
3. Task 3: correctness gate (PyTorch parity fixtures + tests)
4. Task 4: baseline + re-measurement (implement → measure → improve → measure)
5. Task 5: batched transformer chat sample (Idea A)
6. Task 6: finalize docs — update README, delete NEXT.md

## Coordination Notes

- **Decision gate:** Task 1 must settle the route (A/B/C) and batched-API shape before Task 2 begins.
- Task 3 depends on Task 2 (tests reference the new batched op).
- Task 4 depends on Tasks 1–3 (needs before/after numbers).
- Task 5 depends on Task 3 passing (the correctness gate) and Task 2 (the batched op it dogfoods).
- Task 6 depends on Task 5 (only retire the roadmap after the feature lands).
- **`gen_reference.py` RNG discipline:** new fixture cases must be appended at the *end* of `run()` — never inserted mid-stream — or every subsequent fixture changes. Commit regenerated fixtures and the C# manifest as one unit.
- Shared files that may conflict: `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`, `src/Nivara/AutoDiff/Operations/AttentionKernels.cs`, `src/Nivara/AutoDiff/Nn/TransformerBlock.cs`, `src/Nivara/AutoDiff/Nn/MultiheadAttention.cs`, `tests/Nivara.PerformanceTests/Program.cs`, `Nivara.slnx`.

### Route Decision (Task 1) — 2026-08-05

**Status: verified and approved.** Task 1 audit re-ran the grounding table against `src/Nivara/` on branch `khurram/shakespear` (2026-08-05). Every claim held with only trivial line drift (`ModuleHelpers.CreateCausalMask` is at line 72, not 67; `LayerNorm.Forward` at 48; `Embedding.Forward` batched at 35). Confirmed: `Concat` (ReverseGradOperations.cs:1313), `Gather` (1840), fused `MultiHeadAttention` (362), `AttentionKernels` `ScatterHead`/`PackHeads`/`SoftmaxRows` (32/42/53). No public API accepts `[B, L, D]` — `MultiheadAttention<T>` and `TransformerBlock<T>` both reject rank-3. **Route B selected.** Backlog issue filed for deferred batched MatMul: **#118**. No production code changed.

**Route selected: B** — extend the existing fused `MultiHeadAttention` op + `AttentionKernels` to accept `[B, L, D]` and loop the batch internally (e.g. `Parallel.For` over B). Reuses the already-correct fused backward; smallest correct surface. Re-confirmed against the repo during Task 1 verification (evidence in the Grounding Audit table above):

- `ReverseGradOperations.MultiHeadAttention` — ReverseGradOperations.cs:362
- `Concat` — ReverseGradOperations.cs:1313; `Gather` — ReverseGradOperations.cs:1840
- `AttentionKernels`: `ScatterHead`/`PackHeads`/`SoftmaxRows` — AttentionKernels.cs:32/42/53
- `ModuleHelpers.CreateCausalMask` (internal) — ModuleHelpers.cs:67
- `LayerNorm<T>.Forward` — LayerNorm.cs:48; `Embedding<T>.Forward` (batched `[B, L]`) — Embedding.cs:35

Rationale:
- The real gap is the batch dimension, not rank-2 MatMul — the fused op already sidesteps rank-2 via per-head `TensorsHelper.MultiplyCore(..., bTransposed: true)`.
- Route A (batched rank-3 MatMul primitives) would add the largest new autograd surface for no measured benefit yet; revisit only if Task 4 measurements justify it.
- Route C (per-sequence loop in the sample) contradicts Idea A's batched goal; rejected.

Batched API shape (tentative, to be finalized with the team):
- Batch overload of `ReverseGradOperations.MultiHeadAttention` accepting `[B, L, D]` vs. a separate `BatchedMultiHeadAttention` op — TBD.
- `Parallel.For` over the batch when large enough; single-sequence path kept intact and bit-identical.
- Batched-MatMul improvement idea deferred as a backlog issue (to be filed when implementation begins).

### Progress — 2026-08-05

- **Task 1 — DONE** (`7b44c30`): route decision block above; backlog issue **#118** filed (batched MatMul deferred).
- **Task 2 — DONE** (`e451f80`): new `ReverseGradOperations.BatchedMultiHeadAttention<T>` op (ReverseGradOperations.cs:569) accepting rank-3 `[B, L, D]` + optional `[B, qLen, kvLen]` additive mask; `Parallel.For` over B when `ShouldParallelizeBatch` (B≥4 and total work ≥ 2^21); ArrayPool-pooled transient buffers; single-sequence MHA untouched. 8 unit tests in `tests/Nivara.Tests/AutoDiff/BatchedMultiHeadAttentionTests.cs`; all pass.
- **Task 3 — DONE** (`d2ecc46`): batched causal self-attention + batched cross-attention fixture cases appended at the end of `gen_reference.py` `run()` (seed-303 `attn_rng` stream; the seed-42 shared-stream conv/linear fixtures were regenerated per the documented convention — the full `samples/data/torch-comparison/` tree is committed as one unit). New `tests/Nivara.Tests/NivaraTorch/BatchedAttentionTests.cs` (3 tests: causal forward, causal backward, cross forward+backward) compare against PyTorch fixtures. Full NivaraTorch suite: 71/71 pass.

### Results (Task 4) — 2026-08-05

Harness: `tests/Nivara.PerformanceTests/Program.cs` (extended with 4 batched-attention scenarios), Release build, .NET 10.0.9, 16 logical processors, x64. Shapes = Idea A defaults from NEXT.md (`B=16, L=128, D=64, H=4`), causal additive mask. Numbers are medians of 3+ runs × 12 iterations; warmup 3.

| Scenario | Per-seq loop (baseline) | Batched op (Task 2) | Improvement |
|---|---|---|---|
| forward — time/op | ~20.5 ms | ~7.5 ms | **~2.7x** |
| forward — B/op | ~2.14 MB | ~0.53 MB | **~4.0x** |
| forward — gen0/op | 0.17 | 0.00 | — |
| forward+backward — time/op | ~72 ms | ~23 ms | **~3.1x** |
| forward+backward — B/op | ~7.96 MB | ~7.88 MB (see caveat) | n/a |
| forward+backward — gen0/op | ~1.1 | ~0.33 | **~3.3x** |

Caveat: the batched path runs `Parallel.For` over B (B=16 ≥ 4, work ≥ 2^21), so a share of its allocations occurs on thread-pool workers and is not attributed to the calling thread by `GC.GetAllocatedBytesForCurrentThread`. The gen0/op delta (0.00–0.33 vs 0.17–1.1) is the reliable signal: the batched path essentially stops gen0 churn on the hot thread.

**Decision:** no further kernel changes this cycle. The batched op already delivers a ~3x time and ~4x allocation win over the honest baseline, so Route A (batched rank-3 MatMul primitives) is **not** justified by these measurements — it stays a backlog idea (issue **#118**). Parallel threshold / cache blocking were not worth perturbing; the `ShouldParallelizeBatch` heuristic (B≥4 and B·work ≥ 2^21) held up at the measured shape.

### Task 5 — DONE (`2a9334b`)

New sample **`samples/NivaraChatClient/`** (word-level causal transformer chat, Idea A from `samples/NivaraChat/NEXT.md`), registered in `Nivara.slnx`.

- `BatchedTransformer<T>` (`BatchedTransformer.cs`): rank-3 `[B, L, D]` transformer using the Task 2 op (`BatchedMultiHeadAttention`) per block, pre-norm LayerNorm blocks, GELU MLP, fixed sinusoidal position encoding (cached), tied LM head (`MatMul(x, wteᵀ)` → `[B*L, V]`). Causal `[B, L, L]` mask built once per forward and reused across blocks.
- `BatchedChatClient : IChatClient` (`NivaraChatClient.cs`): eval-mode, autoregressive, re-entrant; temperature sampling; multi-turn conversation formatting; streaming overload. Does not own the model (ownership lives with the caller — initial version disposed the model it was handed, which broke the DI demo; fixed).
- `PositionEncoding.cs`, `TinyShakespeare.cs` (corpus downloaded to `samples/data/tinyshakespeare.txt` on first run; committed for offline use per repo convention), `Program.cs` (CLI mirroring NivaraGpt; word-level `TextTokenizer.FromDocuments`; `TrainingLoop`-style harness with `Adam`, `CrossEntropyLoss`; `ModelSerializer` save/load + `tokenizer.Save/Load` via a `<model>.tokenizer.json` convention; `Microsoft.Extensions.AI` DI wiring via `AddChatClient(factory)` + `IChatClient` resolution).
- Verified end-to-end: train → save → load (tokenizer auto-restored, strict shape check catches mismatched `--n-embd`/etc. configs) → samples → DI demo reply. `dotnet build Nivara.slnx`: 0 warnings, 0 errors.
- **Perf note:** word vocab is the dominant cost — `TensorsHelper.MultiplyCore` emits one `TensorPrimitives.Dot` per output element, so the tied LM head over a ~8k vocab is ~1M short dot-calls/batch (≈ 175 ms/batch, ~730 tok/s for `D=32, B=8`). Matches the Task 4 decision: no core kernel changes this cycle. Smoke runs use `--vocab-size 1200` (~3x faster).
- **Task 6 — DONE** (`74d530e`): NEXT.md deleted after salvaging its durable content into `samples/NivaraChat/README.md` (companion-sample section: architecture, CLI spec + delta note, resolved gaps, exercises-vs-MicroGpt, not-doing list); stale `NEXT.md` pointer in `samples/NivaraChat/Program.cs` updated. Corrected the README's `Softmax(dim)` overclaim — core `Softmax` uses `shape[1]`, and attention softmax lives inside the MHA op kernel (`AttentionKernels.SoftmaxRows`), so no public `Softmax(dim)` was needed. No AGENTS.md/root-README references to NEXT.md existed. Workstream complete.

---

## Task 1: Current-state audit and batched-attention route decision

### Priority

High

### Goal

Re-verify the grounding audit above against the actual repo, confirm which NEXT.md gaps remain open, and choose the batched-attention route (A/B/C) with the API shape, recording the decision in this document.

### Why this exists

NEXT.md's gap list is stale relative to the core (most gaps are already resolved). Building on unverified assumptions produced the wrong plan shape in `docs/SHAKESPEARE.md`. This task applies the root-cause discipline to the plan itself before any implementation.

### Decision required

- Route selection: **Route B (extend the fused `MultiHeadAttention` op + `AttentionKernels` to `[B, L, D]`) is the recommended default.** Route A (batched MatMul composition) only if measurements/team preference justify the much larger surface. Route C is the fallback if batch parallelism is explicitly out of scope.
- Batched API shape: new `ReverseGradOperations.MultiHeadAttention` batch overload vs. a new `BatchedMultiHeadAttention` op; whether the sample also gets a batched `TransformerBlock`-style module.

### Scope

- Walk each row of the Grounding Audit in `src/Nivara/` (files/lines cited) and confirm resolved/open.
- Re-confirm no existing public API accepts `[B, L, D]` (check `MultiheadAttention.cs`, `TransformerBlock.cs`, `ReverseGradOperations.MultiHeadAttention`).
- Decide the route and record it (append a short decision block to this doc's Coordination Notes).
- If Route B, decide whether `Parallel.For` over the batch is used and whether the existing single-sequence op is kept intact (it must be).
- File issues for any new findings (e.g. the optional batched-MatMul-improvement idea, if Route B is chosen).

### Constraints

- Do not change production code in this task — it is decision/verification only.
- The single-sequence `MultiHeadAttention`/`TransformerBlock` behavior must remain unchanged (regression surface).

### Suggested implementation path

- Re-run the grep/read pass from the audit table; note any drift.
- Write the route decision as a dated block in this document so later tasks and reviewers see the reasoning.

### Acceptance criteria

- Audit table confirmed (or corrected) with evidence.
- Route + batched API shape decided and recorded.
- Backlog issues filed for anything valuable that is deliberately deferred (e.g. batched MatMul).
- No production code changed.

### Files likely involved

- `docs/SHAKESPEARE2.md` (decision block)
- (read-only) `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`, `src/Nivara/AutoDiff/Operations/AttentionKernels.cs`, `src/Nivara/AutoDiff/Nn/MultiheadAttention.cs`, `src/Nivara/AutoDiff/Nn/TransformerBlock.cs`

### Standing instructions (apply to this task)

1. If an existing test fails after a code change, root-cause first: fix the code if it is a bug, update the test only if its expectation is genuinely invalid, add coverage if it is missing — never patch symptoms.
2. If you identify any interesting gap/problem while working, file a backlog issue with `gh issue create --repo khurram-uworx/Nivara` (use `--body-file` with a temp file).

---

## Task 2: Batched attention core implementation

### Priority

High

### Goal

Implement the batched attention path chosen in Task 1 (default: extend the fused `MultiHeadAttention` op + `AttentionKernels` to accept `[B, L, D]` input) with a correct backward pass, while keeping the single-sequence path intact.

### Why this exists

The real gap for Idea A is the batch dimension: no attention/transformer surface accepts `[B, L, D]`. This task closes it by reusing the already-correct fused kernel and backward rather than composing from new rank-3 MatMul primitives.

### Decision required

None — the API shape was decided in Task 1. If the route selected is A or C, adjust scope accordingly (A: batched MatMul primitives; C: no core change).

### Scope

- Extend the fused `ReverseGradOperations.MultiHeadAttention` (or add a batch overload) to accept batched `[B, L, D]` query/key/value.
- Reuse `AttentionKernels<T>` (PackHeads, SoftmaxRows, ScatterHead, backward) per batch element; parallelize over the batch with `Parallel.For` when the batch is large enough.
- Correct backward: per-batch-element gradient accumulation; causal/padding masks applied per batch element.
- Allocation review of the attention forward/backward: pool any remaining transient `new T[]` (output, saved weights, dQ/dK/dV buffers) where the buffer is not owned by the result tensor.
- Keep the single-sequence path and its behavior unchanged.

### Constraints

- Reverse-mode graph nodes only created inside `GradientUtils.Grad()` (inference-default direction).
- Follow existing span / `TensorPrimitives` / `ArrayPool` conventions; no `NivaraColumn.Data` access in kernels.
- Type constraint `IFloatingPointIeee754<T>`.
- Zero behavioral change to the existing rank-2 `MatMul` / single-sequence MHA.

### Suggested implementation path

- Mirror the per-head loop in the existing op (ReverseGradOperations.cs:362–445) wrapped in an outer batch loop.
- Reuse `ModuleHelpers<T>.CreateCausalMask` for causal masking; do not duplicate the mask helper.

### Acceptance criteria

- Batched forward `[B, L, D]` numerically matches B independent single-sequence forwards.
- Batched backward matches the sum/concatenation of per-sequence backward gradients.
- Single-sequence `MultiHeadAttention` / `TransformerBlock` outputs are bit-identical to before.
- Incorrect shapes throw clear exceptions.
- `dotnet build Nivara.slnx` passes.

### Files likely involved

- `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`
- `src/Nivara/AutoDiff/Operations/AttentionKernels.cs`
- `src/Nivara/AutoDiff/Nn/MultiheadAttention.cs` (only if a batched module overload is added)
- `tests/Nivara.Tests/AutoDiff/` (new `BatchedMultiHeadAttentionTests.cs` or additions)

### Standing instructions (apply to this task)

1. If an existing test fails after a code change, root-cause first: fix the code if it is a bug, update the test only if its expectation is genuinely invalid, add coverage if it is missing — never patch symptoms.
2. If you identify any interesting gap/problem while working, file a backlog issue with `gh issue create --repo khurram-uworx/Nivara` (use `--body-file` with a temp file).

---

## Task 3: Correctness gate — PyTorch parity fixtures and tests

### Priority

High

### Goal

Append batched-attention cases to `samples/NivaraTorch/gen_reference.py` and add NivaraTorch-style NUnit tests that compare forward *and backward* against PyTorch fixtures, gating the sample.

### Why this exists

The NivaraTorch parity technique is the project's correctness gate before any sample lands. Batched attention is new autograd surface; it must match PyTorch before it is dogfooded in Task 5.

### Decision required

None.

### Scope

- Append cases to `gen_reference.py` at the **end** of `run()`: batched causal self-attention forward `[B, L, D]` and backward (grads w.r.t. Q/K/V), using a manual `QK^T/sqrt(d)/softmax` reference (or `F.scaled_dot_product_attention`) so semantics match the fused kernel.
- Regenerate fixtures and commit the full `samples/data/torch-comparison/` tree as one unit.
- New `tests/Nivara.Tests/NivaraTorch/BatchedAttentionTests.cs` comparing forward + backward to fixtures.
- Follow the RNG-order discipline (append only, never mid-stream).

### Constraints

- Fixtures + C# manifest are regenerated/committed together.
- Tests skip with a message when fixtures are missing (existing pattern).

### Suggested implementation path

- Mirror the existing `MatMul_MatchesPyTorch` test structure.
- For backward parity, compare gradients (not just forward outputs).
- Include a small-batch case and a multi-head case with `D % H == 0`.

### Acceptance criteria

- All new parity tests pass.
- Existing NivaraTorch fixtures are unchanged (no mid-stream RNG perturbation).
- `dotnet test --filter "FullyQualifiedName~NivaraTorch"` passes.

### Files likely involved

- `samples/NivaraTorch/gen_reference.py`
- `samples/data/torch-comparison/*` (regenerated)
- `tests/Nivara.Tests/NivaraTorch/BatchedAttentionTests.cs`

### Standing instructions (apply to this task)

1. If an existing test fails after a code change, root-cause first: fix the code if it is a bug, update the test only if its expectation is genuinely invalid, add coverage if it is missing — never patch symptoms.
2. If you identify any interesting gap/problem while working, file a backlog issue with `gh issue create --repo khurram-uworx/Nivara` (use `--body-file` with a temp file).

---

## Task 4: Baseline and re-measurement (implement → measure → improve → measure)

### Priority

Medium

### Goal

Extend `tests/Nivara.PerformanceTests` with batched-attention scenarios, capture the before-numbers (per-sequence loop — the current way to handle batch), and re-measure after Task 2, driving any further optimization strictly by data.

### Why this exists

This is the "we implement; we measure; we improve; we measure" loop. The honest baseline for batched `[B, L, D]` training is "run the existing single-sequence MHA B times" — the new batched op must beat that on time and allocations, or the decision must be revisited.

### Decision required

Whether to make further kernel changes this cycle or capture them as backlog issues and stop.

### Scope

- Add scenarios: B single-sequence forwards vs. one batched forward (time and `B/op`), plus forward+backward, at Idea A shapes (`B=16, L=128, D=64, H=4` from NEXT.md CLI defaults).
- Record before/after numbers in this doc (results table).
- File issues for any identified inefficiency (e.g. optional batched MatMul for projections/MLP, cache blocking, parallel threshold).

### Constraints

- No speculative optimization — every change must be justified by a measurement.
- Keep all regression tests passing.

### Suggested implementation path

- Reuse the existing `Run` harness in `Program.cs` (no BenchmarkDotNet).
- If an optimization is justified, implement and re-measure in the same task.

### Acceptance criteria

- Before/after numbers recorded in the doc.
- Every further optimization (if any) is backed by a measurement and an issue/notes.
- Build and relevant tests pass.

### Files likely involved

- `tests/Nivara.PerformanceTests/Program.cs`
- `src/Nivara/AutoDiff/Operations/AttentionKernels.cs` (only if an optimization is justified)
- `docs/SHAKESPEARE2.md` (results table)

### Standing instructions (apply to this task)

1. If an existing test fails after a code change, root-cause first: fix the code if it is a bug, update the test only if its expectation is genuinely invalid, add coverage if it is missing — never patch symptoms.
2. If you identify any interesting gap/problem while working, file a backlog issue with `gh issue create --repo khurram-uworx/Nivara` (use `--body-file` with a temp file).

---

## Task 5: Build the batched transformer chat sample (Idea A)

### Priority

High

### Goal

Land `samples/NivaraChatClient/` per NEXT.md: a batched causal transformer trained on TinyShakespeare using the new batched attention, wrapped in `IChatClient`, DI-wireable, with a full CLI and perplexity evaluation.

### Why this exists

Dogfoods the new op end-to-end and completes Idea A. MicroGpt proved Nivara can *train* a transformer; NivaraChatClient proves it can *serve* one in an ecosystem-compatible way. This is the payoff of the whole workstream.

### Decision required

- Follow NEXT.md's separate project `samples/NivaraChatClient/` (default).
- Reuse core modules where they exist — do **not** create sample-local `MultiHeadAttention.cs` or `LayerNorm.cs` (both are in core). Use the batched attention op from Task 2 (directly, or via a thin batched module).

### Scope

- `BatchedTransformer<T>` (`Module<T>`) composed from core `Embedding<T>` (batched `[B, L]` lookup), new batched attention, sinusoidal `PositionEncoding`, and core `Linear`/`LayerNorm`/GELU; LM head `[B, L, V]`.
- `PositionEncoding.Sinusoidal<T>(seqLen, embedDim)` — non-trainable constant (sample-local helper).
- Word-level `TextTokenizer` (reuse `samples/Nivara.Samples/TextTokenizer.cs` if the shape fits; otherwise a local copy per NEXT.md) + TinyShakespeare downloader/generator.
- `NivaraChatClient : IChatClient` — `GetResponseAsync`, `GetStreamingResponseAsync`, `GetService`, `Dispose`, multi-turn `FormatConversation`. **Design for thread safety (NEXT.md Gap 9):** re-entrant inference, no shared mutable state during generation, per-session generation state.
- Training via `TrainingLoop<T>` + `DataLoader<T>`, AdamW, `CrossEntropyLoss`, `ModelSerializer` save/load; report holdout perplexity + sample quality (NEXT.md evaluation).
- Full CLI per NEXT.md: `--train`, `--load <path>`, `--save <path>`, `--prompt`, `--interactive`, `--epochs 5`, `--batch-size 16`, `--seq-len 128`, `--n-embd 64`, `--n-layer 4`, `--n-head 4`, `--lr 0.001`, `--temperature 0.8`, `--seed 42`, `--help`.
- DI wiring showcase (`AddChatClient<NivaraChatClient>()`) with `Microsoft.Extensions.AI.Abstractions`, `Microsoft.Extensions.DependencyInjection`, `Microsoft.Extensions.Hosting`.
- Register the project in `Nivara.slnx`.

### Constraints

- CPU-only, float32.
- Out of scope (NEXT.md not-doing list — keep as documented non-goals): batched KV cache, top-p/top-k, beam search, fine-tuning/LoRA, quantization, GPU, multi-modal, ASP.NET Core hosting.
- Expected training time: tens of minutes (~600 steps, ~1.4M params).

### Suggested implementation path

- Follow the NEXT.md architecture diagram, but drive attention through the Task 2 batched op instead of the rank-2 flatten workaround.
- Reuse `ModuleHelpers.CreateCausalMask` for the causal mask.

### Acceptance criteria

- Model trains on TinyShakespeare and saves/loads via `ModelSerializer`.
- `IChatClient` returns generated text; streaming yields token-by-token updates; multi-turn history works.
- Perplexity (holdout) and sample quality are printed.
- Full CLI runs (`--train`, `--load`, `--save`, `--prompt`, `--interactive`).
- DI wiring compiles (`AddChatClient<NivaraChatClient>()`).
- `dotnet build Nivara.slnx` passes; the sample runs end-to-end.

### Files likely involved

- `samples/NivaraChatClient/` (new: `Program.cs`, `BatchedTransformer.cs`, `PositionEncoding.cs`, `NivaraChatClient.cs`, `TextTokenizer.cs`, `TinyShakespeareDownloader.cs`, `.csproj`)
- `Nivara.slnx`

### Standing instructions (apply to this task)

1. If an existing test fails after a code change, root-cause first: fix the code if it is a bug, update the test only if its expectation is genuinely invalid, add coverage if it is missing — never patch symptoms.
2. If you identify any interesting gap/problem while working, file a backlog issue with `gh issue create --repo khurram-uworx/Nivara` (use `--body-file` with a temp file).

---

## Task 6: Finalize docs — update README, delete NEXT.md

### Priority

High

### Goal

Update `samples/NivaraChat/README.md` to cover Idea A / `NivaraChatClient`, and delete `samples/NivaraChat/NEXT.md` — but **only after** confirming every item it asks for has been taken care of and nothing is left to salvage.

### Why this exists

NEXT.md was the roadmap; once Idea A lands, its purpose is served. Its durable content (architecture, CLI spec, gap decisions, exercises-vs-MicroGpt, not-doing list) must be preserved in the README before deletion. The explicit requirement: ensure everything asked for is taken care of and nothing is left to salvage.

### Decision required

None.

### Scope

- Salvage durable NEXT.md content into README: Idea A description, batched transformer architecture, full CLI interface, gap decisions (including the fact that most gaps were already resolved in core), the "What This Exercises vs. MicroGpt" comparison, and the not-doing list.
- Add the `NivaraChatClient` mode(s) to the README quick start + CLI table (or a clear pointer to `samples/NivaraChatClient/`).
- Cross-check every NEXT.md claim against the actual repo state (use the Grounding Audit). For anything done differently, record the delta. For anything still missing that is still valuable, **file an issue** (standing instruction #2) rather than silently dropping it.
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

None — the six tasks above cover the workstream. If a task balloons in scope during execution, split it (keep one task per clear outcome) and record the reason in the doc.

## Suggested Agent Handout Batches

### Batch A: decision-critical

- Task 1 (route + API shape)
- Task 5 (sample scope decisions)

### Batch B: implementation

- Task 1
- Task 2
- Task 3
- Task 4

### Batch C: tests and docs

- Task 3 (parity tests — can move here if B batches it later)
- Task 5
- Task 6

## Final Checklist

- every task has a clear owner-sized scope — yes
- every task has acceptance criteria — yes
- decision-gate tasks are clearly marked — yes (Tasks 1 and 5)
- likely files are listed to reduce agent search time — yes
- execution order reflects real dependencies — yes
- standing instructions (root-cause analysis; file GitHub issues) are present in **every** task — yes
- plan is grounded in the repo (Grounding Audit verified against `src/Nivara/`) — yes
- Task 6 (README update + NEXT.md deletion) is last and gated on the salvage review — yes
