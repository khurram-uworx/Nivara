# AutoDiff Refactor — Execution Plan (from `docs/NEXT-REFACTORING.md`)

## Purpose

This document reviews `docs/NEXT-REFACTORING.md`, checks its technical
assumptions against .NET 10 APIs (Microsoft Learn), and breaks the remaining
work into assignable tasks per `docs/TASKS-TEMPLATE.md`.

**Decisions already made (this plan's inputs):**

- Backwards compatibility is **not** a concern. Breaking changes are acceptable
  when they deliver net goodness. No obsolete-forwarding shims are required.
- Every user-visible breaking change is documented in `CHANGELOG.md` (no
  compat-shim obligation; behavior changes are logged, not preserved).
- The inference-default product direction is preserved: **predict by default,
  train explicitly** (`GradientUtils.Grad()`). No `NoGrad` primary API.
- ADR-001 (AutoDiff is a non-nullable domain) is settled and enforced at
  runtime. Null cleaning happens before entry.
- **Architecture (this refactor's one-way door): Option A — span boundary**,
  locked on 2026-08-03 and recorded in `docs/adr/002-autodiff-span-boundary.md`.
  `GradTensor<T>` keeps `NivaraColumn<T>` as the AutoDiff I/O boundary; the new
  `GradKernels` layer is span-in/span-out over generic `TensorPrimitives`; ops
  wrap results once; `OpNode`/`ComputationGraph` delegate types are unchanged;
  raw `Tensor<T>` backing is explicitly declined. Agents must not re-open this
  fork.

Branch: `khurram/refactoring` (created from `main`).

---

## Part A — How much of NEXT-REFACTORING.md is still valid

Reviewed against the current tree (line counts measured on `khurram/refactoring`
at `main`).

### A1. Completed — no work needed

| Plan item | Evidence |
|---|---|
| §1 Column storage consolidation into single `ColumnStorage<T>` | `src/Nivara/Storage/ColumnStorage.cs` (sole-owner `T[]` + optional `bool[]` mask + lazy `AsTensor()`). `MemoryStorage<T>`/`TensorStorage<T>` deleted. |
| §1 Factory simplification | `ColumnStorageFactory.cs` is 108 lines, direct `ColumnStorage<T>` construction, no 11-way type switches. `IsVectorizable<T>()` remains for kernel selection (fine). |
| §1 `NivaraColumn` dual-path collapse | Single span path; `NivaraColumn.AsSpan()`, `TryGetSpan`, `AsTensorView()` (#107, public). |
| §1 ADR-001 boundary (runtime throw) | `ReverseGradTensor`/`ForwardGradTensor` constructors throw `AutoGradException(Adr001Message)` on `HasNulls`; `ComputationGraph.AddNode` has a `Debug.Assert` Grad-scope guard. |
| Sequence steps 1–5 | Landed (see commit history for storage-consolidation PRs). |
| Zero-copy `AsTensorView()` | Public on `NivaraColumn<T>` and `NivaraSeries<T>` (issue #107). |
| Ergonomic additions | `GradientUtils.Grad()`, `StateDict()`/`LoadStateDict()`, state-dict JSON helpers — all landed. |

### A2. Partially done — residual work remains

| Plan item | Current state |
|---|---|
| §4 GradOperations / ForwardGradOperations rewrite | Ops are **span-ified** (use `TensorPrimitives` over spans internally) but every result is still wrapped in `NivaraColumn<T>.Create/CreateFromOwnedArray` (`ReverseGradOperations.cs:2360` `ResultTensor`). `OpNode<T>.BackwardFunction` is still `Action<NivaraColumn<T>>`; `ComputationGraph.Backward` still takes `NivaraColumn<T>?`. Files are **larger** than the plan's baseline: `ReverseGradOperations.cs` 2373 lines (plan said 1221→~600), `ForwardGradOperations.cs` 783 lines (plan said 990→~500) because Conv1d/Conv2d, BatchNorm, LayerNorm, RMSNorm, attention, VAE, transformer ops were added after the plan. |
| §5 Optimizers | SGD/Adam/AdamW read `tensor.Grad` (still `NivaraColumn<T>?`) via `TryGetSpan` and use `TensorPrimitives`, but still build results with `NivaraColumn<T>.Create(...)`. No `MergeNullMasks` remains (good). |
| §5 Boundary | `GradTensor<T>` still holds `NivaraColumn<T> Data`; `GradTensor<T>` still exposes `IsNull(int)` (delegates to column) and `HasNulls`-free domain is enforced only at construction. |

### A3. Outstanding — not started

- §2 `GradTensor<T>` backing onto `Tensor<T>` (see A4 — recommend NOT doing the
  raw rewrite; do the boundary-gate revision instead).
- §3 `GradKernels.cs` — does not exist. Activation/gradient/MatMul/Transpose
  kernels still live as `NivaraColumn<T>` extensions in
  `src/Nivara/Tensors/NivaraTensorExtensions.cs` (1887 lines).
- §5 Serialization: `ModelSerializer` still serializes a **null mask per
  parameter** (`ModelSerializer.cs:106-114, 171-176`). AutoDiff tensors are
  non-nullable by ADR-001, so this is dead weight.
- §5 `NivaraTensorExtensions` cleanup — **not** stripped. Still contains:
  activations/gradients/MatMul/Transpose/GELU (should move to AutoDiff kernels),
  the obsolete Series methods `AddTensor`, `MultiplyTensor`, `SumTensor`,
  `DotProduct`, `Norm`, `TransformTensor` (call sites: `NivaraSeriesIsValidTests.cs`
  only), and `MatrixMultiply` (0 call sites anywhere).
- §5 `NivaraSeries` cleanup — `Sum()`, `Min()`, `Max()` tensor math still
  present (`NivaraSeries.cs:854, 896, 917`).
- §6 Frame deprecations — `Dot`, `CosineSimilarity`, `ColumnNorms`, `RowNorms`
  still in core `NivaraFrame.cs` (lines 476, 530, 585, 632), marked
  `[Obsolete(..., false)]`, **not** moved to Extensions. `TENSORS.md` already
  directs them to a `Nivara.Extensions.Tensors` namespace.
- §7 initializers / training / datasets / extension factories — only column-based;
  minor cleanup.
- §8/§9 test gate and sequence — the remaining sequence steps (6–20) are
  effectively unstarted.

### A4. Superseded or needing revision

1. **Raw `Tensor<T>` backing is no longer the recommended target.** The plan's
   own "Observations" section (added later) supersedes §2: keep `NivaraColumn<T>`
   as the accepted boundary, validate `HasNulls == false` + span capability on
   entry (done), implement internal kernels over spans (largely done), and only
   replace AutoDiff storage with raw `Tensor<T>` if profiling proves the column
   wrapper is the actual problem. The bigger cleanup target is the
   extension-method boundary, not storage.
2. **`GradKernels` code sketch is outdated.** The plan's example branches
   `typeof(T) == typeof(float)` with `MemoryMarshal.Cast<T, float>`. .NET 10
   generic `TensorPrimitives` overloads make that unnecessary — call
   `TensorPrimitives.Sigmoid<T>(input, output)` directly under
   `where T : IExponentialFunctions<T>` (see Part B). The codebase already does
   this; a `GradKernels` file becomes thin generic wrappers, not a type-switch
   dispatcher.
3. **`Tensor.CreateUninitialized<T>([a.Length])` does not exist.** The .NET 10
   API is `Tensor.CreateFromShapeUninitialized<T>(ReadOnlySpan<nint>, bool pinned = false)`.
   In practice ops allocate `T[]` then wrap, so this is optional.
4. **Line-count deltas and file manifest are stale.** New ops added since the
   plan (Conv1d/2d, BatchNorm, LayerNorm, RMSNorm, MultiheadAttention,
   TransformerBlock, VAE, attention kernels, Im2Col) mean the "~50% line
   reduction" claim applies only to the element-wise + gradient core, not the
   whole operations files. `System.Numerics.Tensors` is now `10.0.10` (plan
   said `10.0.9`).
5. **"Only ever called from AutoDiff" is mostly true, not exactly.** Call-site
   analysis: activations/gradients/MatMul/Transpose/GELU are called from AutoDiff
   (`Activation.cs`, `ReverseGradOperations`, `ForwardGradOperations`, NN modules)
   **plus a handful of non-AutoDiff callers**: tests `NullHandlingPropertyTests`,
   `InferenceFastPathTests`, `ForwardParityTests`, `ForwardGradOperationsTests`;
   samples `Program.cs:130` (`LogSoftmax`), `TimeSeriesModel.cs` (`LeakyRelu`).
   These must be adapted or redirected when the extensions are stripped.
6. **Public `GradTensor<T>.Data` is used broadly.** ~374 `.Data` references in
   src/samples/tests, mostly `t.Data.Length` (parameter counting in sample models
   `MobileNetV2`, `ResNet18`, `DistilBertSst`, `Program.cs`, `StateDictLoader`,
   `MLNetInterop`, tests). Removing the property (plan §2) is a large mechanical
   churn; the smaller revision keeps it but stops using it in the hot path.

### A5. Validity verdict

| Section | Verdict |
|---|---|
| §1 Column storage | ✅ Done |
| §2 GradTensor onto Tensor<T> | ⚠️ Re-scope to boundary-gate revision (keep column) |
| §3 GradKernels | ✅ Valid, rework sketch to generic TensorPrimitives |
| §4 Ops rewrite | ⚠️ Half done (span-ified); revise estimates for added ops |
| §5 Downstream (optimizers, training, serialization, initializers, extensions, series) | ✅ Valid, mostly unstarted |
| §6 Frame deprecations | ✅ Valid, simplify (no shims) |
| §7 File manifest | ⚠️ Update for new files / new ops |
| §8 Risk & test strategy | ✅ Valid (esp. inference-default preservation) |
| §9 Sequence | ⚠️ Steps 1–5 done; 6–20 stand |

---

## Part B — .NET 10 API alignment (Microsoft Learn)

The plan's technical direction **aligns** with the .NET 10 platform approach.
Verified against Microsoft Learn:

### B1. Confirmed

- **`TensorPrimitives` methods are span-based and generic over generic-math
  interfaces** — not a hardcoded SIMD type list:
  - `TensorPrimitives.Tanh<T>(ReadOnlySpan<T>, Span<T>) where T : IHyperbolicFunctions<T>`
  - `TensorPrimitives.Exp10<T>(...) where T : IExponentialFunctions<T>`
  - `Exp`/`Sigmoid` use `IExponentialFunctions<T>`; `Log` uses
    `ILogarithmicFunctions<T>`; `Add`/`Subtract`/`Multiply` use operator +
    identity constraints (`IAdditionOperators<T,T,T>` + `IAdditiveIdentity<T,T>`).
  - `Half`, `BFloat16`, `Single`, `Double`, `NFloat` all implement these
    interfaces in .NET 10, so the current `IFloatingPointIeee754<T>` constraint
    on AutoDiff ops is the right generalization (consistent with AGENTS.md's
    Half/BFloat16 runtime validation).
- **`Tensor<T>.Create<T>(T[] array, ReadOnlySpan<nint> lengths)`** — wraps an
  array as a 1D/ND tensor (zero-copy), matching the plan's `FromArray`/`FromMatrix`
  "after" sketches.
- **`Tensor<T>.TryGetSpan(ReadOnlySpan<nint> startIndexes, int length, out Span<T>)`**
  and the `TensorSpan<T>`/`ReadOnlyTensorSpan<T>` family — span acquisition for
  `TensorPrimitives` calls; confirms the plan's risk note about
  `TryGetSpan`/`FlattenTo` fallback.
- **Spans are the currency.** `.Span` before calling `TensorPrimitives`, never
  `IReadOnlyMemory<T>` — this is the community/.NET 10 pattern and matches the
  plan's Observations.

### B2. Corrections to the plan's sketches

| Plan says | .NET 10 reality |
|---|---|
| `MemoryMarshal.Cast<T, float>` branch per concrete type in GradKernels | Unnecessary. Generic `TensorPrimitives.Sigmoid<T>(input, output)` works for any `T : IExponentialFunctions<T>`. Use the generic call. |
| `Tensor.CreateUninitialized<T>([a.Length])` | API is `Tensor.CreateFromShapeUninitialized<T>(ReadOnlySpan<nint>, bool pinned = false)`. Optional — ops can keep allocating `T[]` + `Tensor.Create(arr, [len])`. |
| `where T : unmanaged, INumber<T>` on GradKernels | Works, but `where T : IFloatingPointIeee754<T>` is the codebase-wide constraint and admits Half/BFloat16. Prefer it for consistency. |

### B3. Alignment conclusion

There is **no .NET community autograd framework built on
`System.Numerics.Tensors`** (TorchSharp is the de-facto .NET autodiff library but
uses its own tensor types). So "community alignment" means aligning with the BCL
tensor pattern the plan already describes: span-based kernels, generic-math
constraints, `TensorPrimitives` for vectorized ops, and a thin owned abstraction
(`GradTensor<T>`) over BCL types rather than a competing tensor hierarchy. The
recommended "smaller revision" is therefore the community-aligned and
lower-risk path; a full raw-`Tensor<T>` rewrite buys little because the ops
already consume spans and the column wrapper is a boundary, not the hot-path
cost.

---

## Part C — Recommended architecture (target state)

```
┌────────────────────────────────────────────────────┐
│ AutoDiff Layer                                     │
│  GradTensor / ReverseGradTensor / ForwardGradTensor│
│    Data = NivaraColumn<T> (non-null, ADR-001)      │  ← boundary
│    internal span access (AsSpan/AsTensor)          │
│  GradKernels (new) — span-in/span-out, generic     │
│    TensorPrimitives, no null handling              │
│  GradOperations / ForwardGradOperations → kernels  │
│  NN Modules  Optimizers  Training  Serialization   │
│  No null masks anywhere in the domain               │
├────────────────────────────────────────────────────┤
│ Columnar / DataFrame Layer                          │
│  NivaraColumn<T>  NivaraFrame  NivaraSeries        │
│  Null semantics  Schema  I/O  Joins  GroupBy       │
│  Reductions: Sum/Mean/Min/Max (null-aware)         │
│  Backed by ColumnStorage<T> (T[] + bool[]? mask)   │
├────────────────────────────────────────────────────┤
│ Storage Layer                                       │
│  ColumnStorage<T>  ColumnStorageFactory            │
│  KernelSelector (per-op vectorize/scalar decision) │
└────────────────────────────────────────────────────┘
```

Changes vs `NEXT-REFACTORING.md`:

- `GradTensor<T>` **keeps** `NivaraColumn<T>` backing (rejected: raw
  `Tensor<T>`). The plan's §2 table is re-scoped: no `Data` removal; instead add
  internal span access and keep `ToColumn()`/`ToSeries()`/`AsTensor()`.
- New `GradKernels` file uses generic `TensorPrimitives` calls, no type-switch
  dispatch, `where T : IFloatingPointIeee754<T>`.
- Ops produce raw `T[]` results and wrap once at the result-tensor boundary
  (already the pattern via `ResultTensor`); the win is deleting the
  `a.Data.Sigmoid()` / `typedGradOutput.Softmax(...)` column-extension calls in
  favor of direct kernels.
- Obsolete/duplicate API is **deleted**, not shimmed.

---

## Working Principles for Agents (apply to every task)

Standing instructions for any coding agent (or sub-agent) picking up a task in
this plan. Re-read this section before starting work.

### W1. Tensors, Vectors, SIMD / numerics intrinsics first

Nivara's identity is tensors and vectors: `System.Numerics.Tensors`,
`TensorPrimitives`, `Vector<T>`, and numeric intrinsics. Whenever a task touches
a numeric hot path, actively look for an opportunity to:

- use a **generic, span-based `TensorPrimitives`** overload instead of a scalar
  loop;
- use **`Vector<T>` / `Vector128<T>`/`Vector256<T>` intrinsics** where
  `TensorPrimitives` has no matching primitive;
- remove per-element null/branch checks so the vectorized path stays reachable
  (see W4).

Do not introduce custom vectorization abstractions or a custom tensor hierarchy —
.NET owns that now. Prefer the platform primitive; delete a scalar fallback only
when it is provably dead.

### W2. Learn official APIs from Microsoft Learn (MCP)

`System.Numerics.Tensors` and generic math are young and move quickly. Before
writing or changing tensor/kernel code, use the **microsoft-learn MCP** tools
(`microsoft_docs_search`, `microsoft_docs_fetch`, `microsoft_code_sample_search`)
to confirm:

- the exact signature and **generic-math constraint** of the primitive (e.g.
  `TensorPrimitives.Sigmoid<T>` requires `IExponentialFunctions<T>`);
- the correct overload for the version pinned in this repo
  (`System.Numerics.Tensors` **10.0.10**), e.g. `Tensor.Create<T>(array, lengths)`
  vs `CreateFromShapeUninitialized<T>` vs `TryGetSpan(...)`;
- whether `Half`/`BFloat16` satisfy the constraint.

Ground the code in official docs; do not rely on memory of API shapes.

### W3. Understand symbols before editing (code-memory MCP)

Before modifying a symbol, use the **code-memory MCP** tools to learn its
definition, callers, dependents, and test coverage:

- `code-memory_semantic_search` / `code-memory_sql_query` to find symbols;
- `code-memory_get_edit_context` / `code-memory_trace_dependency` /
  `code-memory_impact_analysis` to see who depends on what and which tests cover
  it;
- `code-memory_get_architecture_overview` / `code-memory_get_component_clusters`
  for layering.

When delegating to sub-agents, explicitly instruct them to use both MCPs
(microsoft-learn + code-memory) where they make sense, and to report what they
learned — not just what they changed.

### W4. Honor the ADRs; prune dead null branches, keep boundary checks

Read `docs/adr/` before implementing (currently
`docs/adr/001-autodiff-nonnullable-domain.md`, status: accepted). **AutoDiff is
a non-nullable domain.** While implementing:

- **Prune null handling that can no longer execute** in the AutoDiff hot paths
  whenever you touch them: `HasNulls` checks, `WithoutNulls()` calls,
  `TryGetNullMask` dual paths, mask OR-propagation, `IsNull` loops in backward
  kernels, and null-conditioned `Create`/`CreateFromSpans` branches. Each removal
  is the "subtract → lean" payoff of ADR-001 and directly enables W1 (single
  vectorized path).
- **Implement missing boundary checks.** Every entry point that accepts a
  nullable column — `ReverseGradTensor`/`ForwardGradTensor` constructors and the
  `FromColumn`/`FromSeries`/`FromArray`/`FromMatrix` factories, including the
  `ForwardGradTensor` tangent — must throw `AutoGradException` (message contains
  "ADR-001") when the input `HasNulls`. Do not assume a factory is guarded;
  verify each one.
- Null semantics stay in the columnar/storage layer (`NivaraColumn`,
  `ColumnStorage<T>`, `IColumn.IsNull`); null cleaning happens before entry
  (`DropNulls`/`FillNull`).

### W5. Test failures are diagnostics, not noise

When a test fails:

1. Identify the **root cause** first: is it (a) an intended behavior change from
   this refactor, (b) a design change that legitimately invalidates an old
   expectation, or (c) a real defect in the change?
2. Do not rush a quick fix or blanket `Skip`. Approach it as an engineering
   decision.
3. If the production change is correct and the old test expectation is stale
   (e.g. deleted methods, changed serialization format, simplified semantics),
   **update the test** to assert the new contract and record the behavior change
   in `CHANGELOG.md`.
4. If the change introduced a bug, fix the code; if the test exposes a latent
   codebase bug, fix the codebase.
5. Keep the full suite green at each validation gate (Task 13). During iteration
   run only the targeted test project; ask before running the long full
   `dotnet test` (per AGENTS.md).

### W6. Capture "later" work as GitHub issues

While implementing, when you encounter work that is out of scope but worth doing
later (design follow-ups, API gaps, perf ideas, doc debt), record it as a GitHub
issue immediately using `gh`:

- `gh issue create --repo khurram-uworx/Nivara --title "<short title>" --body-file <file>`
  (write the body to a temp file first — avoids the PowerShell backtick/backslash
  escaping gotcha in AGENTS.md).
- Reference the file/line and the reason it is deferred (scope, risk, or
  dependency).
- Do not silently leave bare TODO comments for deferred work; the issue is the
  record. PRs that resolve an issue reference `#<number>` in the description.

---

## Part D — Tasks

### Task 1: Record the architecture decision (decided — non-blocking)

**Priority:** High

**Goal:** Record the already-made architecture decision as ADR-002 and update
the affected docs. Do not re-litigate the decision.

**Why this exists:** NEXT-REFACTORING §2 conflicts with its own Observations
section. The decision was made explicitly (maintainer review, 2026-08-03);
agents must not reopen it.

**Decision (locked):** Option A — span boundary (ADR-002). `GradTensor<T>`
keeps `NivaraColumn<T>` as the AutoDiff I/O boundary; new `GradKernels` span
layer; ops wrap results once; no `OpNode`/`ComputationGraph` delegate changes;
raw `Tensor<T>` backing declined.

**Scope:**
- Write `docs/adr/002-autodiff-span-boundary.md` recording the accepted option,
  rejected options (B: raw `Tensor<T>` backing, C: internal `T[]`+shape) with
  rationale, and consequences (keep public `Data` + `ToColumn()`; add internal
  `AsSpan()`/`AsTensor()` in Task 7; remove `IsNull(int)`).
- Reconcile NEXT-REFACTORING §2/§3 narrative with the decision (mark the
  raw-`Tensor<T>` rewrite sections as superseded by ADR-002).
- Confirm `GradTensor<T>.Data` stays public; internal span access lands in
  Task 7.

**Acceptance criteria:**
- ADR-002 exists and matches the locked decision.
- NEXT-REFACTORING.md marks the superseded sections.
- Task 2 onward proceeds without re-opening the decision.

**Files likely involved:**
- `docs/adr/002-autodiff-span-boundary.md` (new)
- `docs/NEXT-REFACTORING.md`
- `docs/PLAN.md`

---

### Task 2: Create GradKernels (span-based kernel layer)

**Priority:** High

**Goal:** A new `src/Nivara/AutoDiff/Operations/GradKernels.cs` static class of
pure span kernels: `Sigmoid`, `SigmoidGradient`, `Tanh`, `TanhGradient`, `Relu`,
`ReluGradient`, `LeakyRelu`, `LeakyReluGradient`, `Exp`, `Log`, `LogGradient`,
`Softmax`, `SoftmaxGradient`, `LogSoftmax`, `LogSoftmaxGradient`, `Abs`,
`AbsGradient`, `Clamp`, `ClipGradient`, `Negate`, `Divide`, `MatMul`,
`Transpose`, plus the newer `Gelu`/`GeluGradient`/`GeluExact`/`GeluExactGradient`.

**Why this exists:** These live today as `NivaraColumn<T>` extensions in
`NivaraTensorExtensions.cs` (1887 lines) carrying null-mask and mixed-storage
branches that AutoDiff never needs.

**Scope:**
- Signatures `(ReadOnlySpan<T> input, Span<T> output)` (and grad variants),
  `where T : struct, IFloatingPointIeee754<T>`.
- Use generic `TensorPrimitives` directly — no `typeof(T)` dispatch,
  no `MemoryMarshal.Cast` (Part B2).
- Softmax/LogSoftmax/MatMul/Transpose take explicit dimension parameters.
- No null checks, no mask propagation, no allocations on the hot path.

**Acceptance criteria:**
- Kernels pass float/double (and Half/BFloat16 where sensible) known-value unit
  tests.
- Outputs match existing `NivaraTensorExtensions` numerics within tolerance.
- No `NivaraColumn` references in `GradKernels.cs`.

**Files likely involved:**
- `src/Nivara/AutoDiff/Operations/GradKernels.cs` (new)
- `tests/Nivara.Tests/AutoDiff/GradKernelsTests.cs` (new)

---

### Task 3: Migrate ReverseGradOperations to GradKernels + span results

**Priority:** High

**Goal:** Replace `a.Data.Sigmoid()`/`Softmax`/`MatMul`/`Transpose`/gradient
extension calls in `ReverseGradOperations.cs` (2373 lines) with `GradKernels`
calls on spans; keep the single wrap via `ResultTensor`.

**Why this exists:** Decouples the reverse-mode engine from the columnar
null-mask machinery and is the core "subtract" of the refactor.

**Scope:**
- Element-wise, activation, softmax, matmul/transpose, broadcast, slice/concat,
  norm ops — swap to span kernels.
- `AddBias`/`BroadcastAdd`/grad accumulation paths stay structurally the same.
- Keep `GradientUtils.ShouldTrackGrad` gating and `GradientUtils.Grad()`
  inference-default semantics exactly as-is.
- Keep `AsSpan<T>(this ReverseGradTensor<T>)` helper.

**Constraints:**
- OpNode closures may keep capturing input tensors; only the math changes.
- Do not change public method signatures.

**Acceptance criteria:**
- Existing AutoDiff tests (GradOperations, BackwardPass, Nn, Loss,
  InferenceGraphTests, ForwardParity) pass unchanged or with construction-only
  edits.
- `ReverseGradOperations.cs` no longer calls any `NivaraTensorExtensions`
  activation/gradient/MatMul/Transpose method.

**Files likely involved:**
- `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`
- `src/Nivara/AutoDiff/Operations/AttentionKernels.cs` (if it uses column ext)

---

### Task 4: Migrate ForwardGradOperations to GradKernels + span results

**Priority:** High

**Goal:** Same migration for `ForwardGradOperations.cs` (783 lines).

**Why this exists:** Keeps JVP parity tests green while removing the same column
coupling.

**Scope:**
- Swap `a.Data.Sigmoid()`/`Tanh()`/gradients/MatMul/Transpose calls to kernels.
- Preserve tangent semantics and `RequiresTangent`.

**Acceptance criteria:**
- `ForwardGradOperationsTests.cs` and `ForwardParityTests.cs` pass.
- No `NivaraTensorExtensions` activation/gradient/MatMul/Transpose calls remain.

**Files likely involved:**
- `src/Nivara/AutoDiff/Operations/ForwardGradOperations.cs`

---

### Task 5: Optimizers — stop wrapping results in columns

**Priority:** Medium

**Goal:** SGD/Adam/AdamW keep `tensor.Grad` as `NivaraColumn<T>?` (boundary) but
build updated parameters from spans/arrays with a single wrap, removing
per-update `NivaraColumn<T>.Create(...)` where it adds no value.

**Why this exists:** They already read spans via `TryGetSpan`; the wrapping is
redundant allocation on the training hot loop.

**Scope:**
- `ApplySgdUpdate`, `Adam<T>`, `AdamW<T>` result construction.
- Keep `ArrayPool` state buffers (already in place).

**Acceptance criteria:**
- Optimizer behavior tests pass (SGD/Adam/AdamW momentum and bias correction
  unchanged).
- Per-step allocation profile is flat or reduced (spot-check with a small perf
  test if one exists).

**Files likely involved:**
- `src/Nivara/AutoDiff/Optimizer/SGD.cs`
- `src/Nivara/AutoDiff/Optimizer/Adam.cs`
- `src/Nivara/AutoDiff/Optimizer/AdamW.cs`

---

### Task 6: Simplify ModelSerializer (drop per-parameter null masks)

**Priority:** Medium

**Goal:** Serialize parameters as shape + flat data (base64), no null mask.
Breaking format change is acceptable.

**Why this exists:** AutoDiff is non-nullable by ADR-001; null-mask
serialization per parameter (`ModelSerializer.cs:106-114, 171-176`) is dead
weight and bloats saved models.

**Scope:**
- Remove null-mask serialization from `StateDictToJson`/`JsonToStateDict` and
  `Save`/`Load`.
- Preserve the public workflow: `Module<T>.StateDict()`,
  `LoadStateDict(state, strict:)`, `ModelSerializer.Save/Load`,
  `StateDictToJson`/`JsonToStateDict<T>`.
- Bump/version the JSON payload so old files are rejected loudly, not misread.
- Log the format change as a breaking entry in `CHANGELOG.md`.

**Acceptance criteria:**
- `SerializationTests.cs` round-trips (save→load, JSON→state dict) pass with the
  new format; strict missing-key and shape-mismatch validation preserved.
- Saved file contains no mask arrays.

**Files likely involved:**
- `src/Nivara/AutoDiff/Serialization/ModelSerializer.cs`
- `src/Nivara/AutoDiff/Serialization/Checkpoint.cs`

---

### Task 7: Harden boundary gates on the three GradTensor types

**Priority:** Medium

**Goal:** `GradTensor<T>`, `ReverseGradTensor<T>`, `ForwardGradTensor<T>`
factories (`FromColumn`, `FromArray`, `FromMatrix`, `FromSeries`) validate the
ADR-001 contract once and expose `internal` span access for ops/modules.

**Why this exists:** Centralizes the "dense, non-null, numeric, span-capable"
contract instead of scattering `HasNulls` checks.

**Scope:**
- Keep the existing `HasNulls` runtime throws; consolidate the check path.
- Add `internal ReadOnlySpan<T> AsSpan()` / `Tensor<T> AsTensor()` on
  `GradTensor<T>` (reuse `Data.AsTensorView()`); update the internal
  `AsSpan<T>(this ReverseGradTensor<T>)` helper to delegate to it.
- Remove `IsNull(int)` from `GradTensor<T>` public surface (breaking OK) — it is
  meaningless in a non-null domain.

**Acceptance criteria:**
- `TypeSafetyTests.cs` still validates nullable→throw at factories.
- `NullHandlingTests.cs` boundary coverage kept.
- No compilation regressions in modules/optimizers after the `IsNull` removal.

**Files likely involved:**
- `src/Nivara/AutoDiff/GradTensor.cs`
- `src/Nivara/AutoDiff/ReverseGradTensor.cs`
- `src/Nivara/AutoDiff/ForwardGradTensor.cs`

---

### Task 8: Strip NivaraTensorExtensions + delete dead Series/Frame methods

**Priority:** High

**Goal:** Reduce `NivaraTensorExtensions.cs` (1887 lines) to null-aware column
reductions only: `Sum`, `Mean`, `Min`, `Max`. Delete the obsolete Series
extensions (`AddTensor`, `MultiplyTensor`, `SumTensor`, `DotProduct`, `Norm`,
`TransformTensor`) and `MatrixMultiply`.

**Why this exists:** These are AutoDiff-only or dead code. The plan's call-site
claim is confirmed except for a handful of tests/samples that must be redirected
(see scope).

**Scope:**
- Keep: `Sum`, `Mean`, `Min`, `Max` (null-aware, two-path
  TensorPrimitives/scalar-mask).
- Delete: activations, gradients, MatMul, Transpose, GELU family (now in
  `GradKernels`), obsolete Series methods, `MatrixMultiply`.
- Redirect non-AutoDiff callers to the new kernels or column reductions:
  `tests/.../NullHandlingPropertyTests.cs`, `InferenceFastPathTests.cs`,
  `ForwardParityTests.cs`, `ForwardGradOperationsTests.cs`;
  `samples/Nivara.SampleApp/Program.cs:130` (`LogSoftmax`),
  `samples/NivaraTimeSeries/TimeSeriesModel.cs` (`LeakyRelu`).
- Remove the obsolete-method tests in `NivaraSeriesIsValidTests.cs`.

**Acceptance criteria:**
- `NivaraTensorExtensions.cs` contains only `Sum`/`Mean`/`Min`/`Max`.
- Build is green across src, samples, tests.
- Grep confirms zero remaining call sites for the deleted methods.

**Status: COMPLETE** — `NivaraTensorExtensions.cs` stripped to the four
reductions; activations/gradients/MatMul/Transpose/GELU family/obsolete Series
methods/`MatrixMultiply` deleted. `Subtract`/`Divide` promoted from extensions
to first-class `NivaraColumn<T>` members (with `NumericTensorKernels`
`TensorPrimitives` overloads) to keep `ExpressionEvaluator` and
`ReverseGradOperations.Mean` working. Test callers redirected: `PerfTests`
Gelu/Relu throughput → `GradKernels`; `PerformanceTests` ColumnSigmoid →
`TensorPrimitives.Sigmoid`; obsolete-method tests removed from
`NivaraSeriesIsValidTests`; Gelu null-propagation test + unused
`GeluTanhApprox` removed from `NullHandlingPropertyTests`;
`NivaraTensorExtensionsGeluTests.cs` deleted. `GradKernelsTests` rewritten to
known-value kernel tests (local Abramowitz–Stegun `Erf` helper — `Math.Erf` is
not available in this runtime). Plan's `Program.cs:130`/`TimeSeriesModel.cs`
redirects confirmed stale — no callers found there. Full AutoDiff suite 746/746
passing; Serialization/SeriesIsValid/NullHandling suites 33/33 passing; solution
+ all 11 samples build green. Gate C (Tasks 8–10) partially satisfied (Task 8
done; Tasks 9–10 pending).

**Files likely involved:**
- `src/Nivara/Tensors/NivaraTensorExtensions.cs`
- `src/Nivara/Tensors/TensorsHelper.cs` (check for duplication)
- `src/Nivara/Tensors/NullableTensor.cs` (check for duplication)
- the test/sample call sites above

---

### Task 9: NivaraSeries cleanup

**Priority:** Low

**Goal:** Remove `Sum()`, `Min()`, `Max()` tensor math from `NivaraSeries<T>`
(`NivaraSeries.cs:854, 896, 917`), leaving a labeled column wrapper.

**Why this exists:** Duplicates `NivaraColumn` reductions; the plan marks
NivaraSeries as a labeled-column-wrapper only.

**Scope:**
- Redirect AutoDiff/tests callers to `column.Sum()`.
- Keep `ToColumn()`, LINQ helpers, label indexer, `AsTensorView()`.

**Acceptance criteria:**
- No `Sum/Min/Max` members on `NivaraSeries<T>`.
- Aggregate/series tests updated and green.

**Status: COMPLETE** — removed public `Sum()`/`Min()`/`Max()` from
`NivaraSeries<T>` plus the now-dead private helpers (`sumVectorized`,
`minVectorized`, `maxVectorized`, `minTensorPrimitive`, `maxTensorPrimitive`);
`Average()` (and its `averageVectorized`/`sumTensorPrimitive`/`divideByCount`
helpers) retained per scope. Callers redirected to the column reductions:
`ForwardGradOperations.Sum/Mean/KlDivergence` and
`ReverseGradOperations.Sum/KlDivergence` now call `a.Data.Sum()`/`a.Tangent.Sum()`/
`klElements.Sum()` directly (no `NivaraSeries` wrapper); `ForwardParityTests`
uses `rx.Grad!.Sum()`/`expected.Sum()`. `NivaraSeriesAggregateTests` rewritten:
Sum/Min/Max tests now target `NivaraColumn<T>` (error messages updated to the
column wording; all-null Sum returns 0 instead of throwing; string/object
non-numeric Min/Max/Sum cases removed — column reductions are
`INumber<T>`-constrained); Average tests unchanged. `samples/.../AggregateExample.cs`
uses `series.Values.Sum()/Min()/Max()` and drops the string comparison demo.
Grep confirms zero remaining `NivaraSeries` Sum/Min/Max call sites. Core, solution,
and all 11 samples build with 0 warnings.

**Files likely involved:**
- `src/Nivara/NivaraSeries.cs`
- `tests/Nivara.Tests/NivaraSeriesAggregateTests.cs`

---

### Task 10: Remove Frame tensor methods from core

**Priority:** Medium

**Goal:** Delete `Dot`, `CosineSimilarity`, `ColumnNorms`, `RowNorms` from
`NivaraFrame.cs`. The plan's original relocation to
`Nivara.Extensions.Tensors.FrameTensorOperations` (per `TENSORS.md`) was **not
pursued** — the methods have no production callers, so they were deleted
outright per the "remove if not being used" directive. No obsolete shims
(breaking OK).

**Why this exists:** TENSORS.md directs column math to `TensorPrimitives` on
spans; frame columns are not tensor axes.

**Scope:**
- Remove from core (the four `[Obsolete]` methods + private
  `dotSpans`/`cosineSimilaritySpans`/`normSpan` helpers).
- Delete the now-dead `TensorsHelper.RowNorms` SIMD kernel (only consumer was
  `frame.RowNorms`).
- Remove/rewrite test callers.

**Acceptance criteria:**
- `NivaraFrame` has no tensor-axis methods.
- `TensorsHelper.RowNorms` gone; no CS0618 pragmas remain in tests.

**Status: COMPLETE** — deleted `Dot`/`CosineSimilarity`/`ColumnNorms`/`RowNorms`
from `NivaraFrame.cs` (and the `dotSpans`/`cosineSimilaritySpans`/`normSpan`
helpers; `using System.Numerics;` no longer needed). Removed the orphaned
`TensorsHelper.RowNorms` kernel (only caller was `frame.RowNorms`) and its 3
`TensorsHelperTests`. `TensorInteropTests`: 14 frame-method tests deleted; the 2
`RankingWorkflow_*` tests rewritten to compute per-column scores with
`TensorPrimitives.CosineSimilarity` on column spans (the recommended
replacement) before running `ArgSortDescending`/`TopKDescending`; CS0618 pragma
removed. `DiagnosticsTests`: 5 `FrameDot`/`FrameCosineSimilarity`/
`FrameColumnNorms`/`FrameRowNorms` diagnostics tests deleted along with the
`CreateDiagnosticsFrameWithNulls`/`AssertFrameBatchOperation` helpers; CS0618
pragma removed. `CopyToRowMajor`/`ToTensors` retained (public, still used).
Grep confirms zero remaining frame-method call sites; all remaining
`.Dot(`/`.CosineSimilarity(` hits are `TensorPrimitives` kernels.

**Files likely involved:**
- `src/Nivara/NivaraFrame.cs`
- `src/Nivara/Tensors/TensorsHelper.cs`
- `tests/Nivara.Tests/Tensors/TensorInteropTests.cs`
- `tests/Nivara.Tests/Tensors/TensorsHelperTests.cs`
- `tests/Nivara.Tests/Diagnostics/DiagnosticsTests.cs`

---

### Task 11: Initializers, datasets, and extension factories

**Priority:** Low

**Goal:** Remove extra column hops in initializers, `TensorDataset<T>`,
`DataLoader`, and `NivaraAutoGradExtensions` now that span access is internal.

**Why this exists:** Small mechanical cleanup to finish the boundary work.

**Scope:**
- Initializers: build `T[]` then single `CreateFromOwnedArray`/column wrap.
- `TensorDataset<T>`: span slices from column spans.
- `NivaraAutoGradExtensions`: keep `ToReverseGradTensor`, `ToFrame`,
  `ToGradientFrame`, `BatchBackward`; adapt to any surface changes.

**Acceptance criteria:**
- Training tests and sample apps compile and behave unchanged.

**Status: COMPLETE** — all 13 initializer implementations (6 static `Init<T>`
plus `KaimingUniformInitializer`, `KaimingNormalInitializer`,
`XavierUniformInitializer`, `XavierNormalInitializer`, `UniformInitializer`,
`NormalInitializer`, `PyTorchDefaultInitializer`) now wrap freshly allocated
weight arrays with `NivaraColumn<T>.CreateFromOwnedArray` (zero-copy) instead of
copying through `NivaraColumn<T>.Create(data)`. `TensorDataset<T>.BuildTensor`
slices column spans via `TryGetSpan` (columns looped outer, batch inner into the
row-major buffer) and throws `AutoGradException` (ADR-001) up front when any
source column `HasNulls`, removing the dead null-mask/`CreateFromSpans` path
that previously always threw at the `ReverseGradTensor` constructor.
`NivaraAutoGradExtensions` required no adaptation — it already routes through
`FromColumn`/`FromSeries` against the current surface. Verified: solution + 11
samples build 0 warnings; `NnTests`/`TrainingTests`/`DataParallelTests` 246/246;
full suite 2116/2116.

**Files likely involved:**
- `src/Nivara/AutoDiff/Nn/Initializers/*.cs`
- `src/Nivara/AutoDiff/Training/TensorDataset.cs`
- `src/Nivara/AutoDiff/Extensions/NivaraAutoGradExtensions.cs`

---

### Task 12: Tests + samples adaptation

**Priority:** High

**Goal:** Keep the full suite green at each validation point and adapt all
remaining call sites to the new kernels/surfaces, preserving explicit
`GradientUtils.Grad()` scopes for backward/training tests.

**Why this exists:** The refactor touches construction patterns across
AutoDiff tests and samples.

**Scope:**
- Adapt: `GradOperationsTests`, `ForwardGradOperationsTests`, `BackwardPassTests`,
  `NnTests`, `LossTests`, `ForwardParityTests`, `TypeSafetyTests`,
  `NivaraIntegrationTests`, `SerializationTests`, `GradientUtilsTests`.
- Delete obsolete-method tests in `NivaraSeriesIsValidTests.cs`.
- Update samples: `ForwardParityExample`, `AutoDiffExample`,
  `CrossFrameworkFraudNet`, `Program.cs`, `TimeSeriesModel.cs`, `MobileNetV2`,
  `ResNet18`, `DistilBertSst`, `StateDictLoader`, `MLNetInterop`.
- Preserve inference-default tests (no graph built outside `Grad()`).

**Acceptance criteria:**
- Full `dotnet test Nivara.slnx` passes (see Task 13 gate; ask before running
  the long test command per AGENTS.md).
- Inference-graph regression tests (`InferenceGraphTests.cs`) stay green.

**Files likely involved:**
- `tests/Nivara.Tests/AutoDiff/*`
- `samples/*` (as enumerated)

---

### Task 13: Final validation + docs update

**Priority:** High

**Goal:** Run the full suite, update docs, and mark the plan complete.

**Why this exists:** Test gate per the plan's §8 and keep AGENTS.md/AUTODIFF.md
accurate.

**Scope:**
- Full `dotnet test` green.
- Update `docs/NEXT-REFACTORING.md` status (mark sections done/re-scoped),
  `docs/AUTODIFF.md` (kernel location, serialization format),
  `docs/TENSORS.md` (frame methods moved), `AGENTS.md` known-issues.
- Record every user-visible breaking change (removed `Data`-adjacent surface,
  deleted methods, new serialization format) in `CHANGELOG.md` as it lands.
- Optionally delete this `docs/PLAN.md` after execution (or archive it).

**Acceptance criteria:**
- Suite green; docs reflect the target architecture; no stale references to
  deleted methods.

**Files likely involved:**
- `docs/NEXT-REFACTORING.md`, `docs/AUTODIFF.md`, `docs/TENSORS.md`,
  `AGENTS.md`

---

## Suggested Execution Order

1. Task 1 (record locked decision as ADR-002 — docs only, non-blocking; safe to
   run in parallel with Task 2)
2. Task 2 → Task 3 → Task 4 (kernel layer, then ops migration — sequential,
   share `GradKernels.cs` and the operations files)
3. Task 7 (boundary gates) — can start after Task 2
4. Task 5, Task 6 (optimizers, serializer) — independent of ops migration
5. Task 8, Task 9, Task 10, Task 11 — cleanup (parallel-safe once Tasks 2–4 land)
6. Task 12 (tests + samples) — continuous
7. Task 13 (final gate + docs)

## Validation Gates

- **Gate A** (after Task 2): `GradKernelsTests` pass; build green.
- **Gate B** (after Tasks 3–4): AutoDiff test set passes.
- **Gate C** (after Tasks 8–10): no references to deleted methods anywhere;
  src + samples + tests compile.
- **Gate D** (Task 13): full `dotnet test` (ask user before running).

## Coordination Notes

- **Merge-conflict risk files:** `NivaraTensorExtensions.cs`, `ReverseGradOperations.cs`,
  `ForwardGradOperations.cs`, `GradTensor.cs`. Run Tasks 2–4 on the same branch
  head sequentially; parallelize only Tasks 5–11 after the kernel layer lands.
- **Do not start until Task 1:** none — the architecture decision (Option A,
  span boundary) is locked and recorded in ADR-002. Tasks 2–11 proceed under it.
- **Decision gates:** none blocking. Tasks 5–6's breaking changes are pre-approved
  (backward compat not a concern).
- **Shared touchpoints:** samples `Program.cs`, `TimeSeriesModel.cs` and tests
  `NullHandlingPropertyTests.cs`, `InferenceFastPathTests.cs` are edited by both
  Task 8 and Task 12 — sequence Task 8 before Task 12.

## Suggested Agent Handout Batches

### Batch A: docs (non-blocking, parallel-safe)
- Task 1 (record ADR-002; reconcile NEXT-REFACTORING §2/§3)

### Batch B: implementation (sequential, shared files)
- Task 2, Task 3, Task 4, Task 7

### Batch C: implementation (parallel-safe)
- Task 5, Task 6, Task 8, Task 9, Task 10, Task 11

### Batch D: tests and docs
- Task 12, Task 13

## Final Checklist

- every task has an owner-sized scope ✅
- every task has acceptance criteria ✅
- decision-gate tasks clearly marked (none blocking; ADR-002 records the locked
  architecture) ✅
- likely files listed ✅
- execution order reflects real dependencies ✅
- full-suite validation gated behind user confirmation (per AGENTS.md) ✅
- agents honor Working Principles W1–W6 (SIMD-first, Microsoft-Learn MCP,
  code-memory MCP, ADR-001 null pruning + boundary checks, root-cause test
  triage, `gh issue` for deferred work) ✅
- every user-visible breaking change logged in `CHANGELOG.md` ✅
