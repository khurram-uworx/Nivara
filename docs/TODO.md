# TODO: Fused evaluator span-kernel execution (issue #166)

**Branch:** `khurram/166` (created off `main`). **Tracker:** relates to #155.

## Problem

`FusedExpressionEvaluator` (`src/Nivara/Expressions/FusedExpressionEvaluator.cs`) compiles the
`ColumnExpression` AST to a cached `System.Linq.Expressions` delegate. `Span<T>` is prohibited in
expression trees, so the delegate runs over `T[]` arrays:

- `SnapshotLeaves` (`:288-322`) copies every leaf into a fresh `T[]` per evaluation, even null-free
  columns (`TryGetSpan` → `CopyTo` into a new array).
- `ExecuteCompiled` invokes via `Delegate.DynamicInvoke` (`:213-221`) — reflection invocation boxing
  args every evaluation.
- The generic node-tree fallback (`FusedKernel.cs:40-73`) reads `column[index]` per-element (not
  span-based) with an O(leaves) reference scan per element, and `ComputeMask` runs as a separate
  O(n) pass.

## Goal

1. Eliminate leaf snapshot copies on the compiled path for null-free contiguous leaves.
2. Eliminate `DynamicInvoke` per-evaluation reflection.
3. Add a span-in/span-out generic kernel with fused mask computation (ADR-001 semantics).
4. Adaptive routing (decided with human): span kernel primary for **null-bearing uniform**
   generic-math plans; compiled zero-copy path primary for **null-free uniform** plans (preserves the
   JIT-vectorized ~11.6x fused win) and all heterogeneous plans.
5. Keep public `Evaluate`/`EvaluateBoolean` signatures and the three guardrail counters unchanged.
   All internals; zero public API break.

## Grounding

- `bool` is NOT in `TypeCompatibilityValidator.GetNumericTypes()` and does not satisfy generic math,
  so all bool-result plans (comparisons, And/Or/Not) stay on the compiled path. Only uniform
  `INumber<T>` arithmetic plans can route to the span kernel.
- `NivaraColumn<T>.AsSpan()` (`NivaraColumn.cs:1454`) is a zero-copy raw view (defaults at null
  slots); `Storage.NullMask` is `ReadOnlySpan<bool>`. Span kernel reads all leaves zero-copy.
- `ColumnStorage<T>` already has `internal ReadOnlyMemory<T> Data`; recover array+offset via
  `MemoryMarshal.TryGetArray` for zero-copy compiled leaves.

## Proposed changes

### Step 1 — Typed invocation (FusedExpressionEvaluator.cs)

`BuildCompiledDelegate` also emits a cached `Func<object[], int, Array>` wrapper (expression-compiled
once per signature) that casts each leaf `object[]` entry to its `T[]` and invokes the typed
`Action<T1[],...,TN[],R[]>` directly. `ExecuteCompiled` becomes `action(leafArrays, length)`.
`compiledKernelCache` value type changes `Delegate` → `Func<object[], int, Array>`.

### Step 2 — Zero-copy leaf reads (Interfaces.cs + FusedExpressionEvaluator.cs)

Add `internal ReadOnlyMemory<T> Data { get; }` to `IColumnStorage<T>` (already implemented by the
sole impl `ColumnStorage<T>`). In `SnapshotLeaf<T>`, when the leaf is null-free and
`MemoryMarshal.TryGetArray(Storage.Data)` yields offset 0, return the backing `T[]` directly; copy
only for sliced (offset > 0) or null-bearing leaves.

### Step 3 — Span kernel (FusedKernel.cs)

Add span-in/span-out core:

```csharp
internal static void Execute<T>(ColumnExpression expr, ReadOnlySpan<T>[] inputs,
    ReadOnlySpan<bool>[] masks, Span<T> output, bool[]? outMask) where T : struct, INumber<T>;
```

- Single pass; `ReferenceEqualityComparer` leaf-index map (like `BuildCompiledDelegate`) instead of
  per-element O(leaves) scan.
- Fused OR-mask: masked positions write `default(T)` and set `outMask[i] = true`.
- Keep `Evaluate<T>(expr, leaves, bool[]? mask = null)`; `mask == null` → compute from leaves.
  `CoerceLiteral`/`ApplyArithmetic` unchanged.

### Step 4 — Routing + counters (FusedExpressionEvaluator.cs)

In `EvaluateCore`, span-check first:

```
plan.IsGenericMath && plan.ResultType != typeof(bool)
    && plan.Columns.All(l => l.Column.ElementType == plan.ResultType)
    && plan.HasNulls
```

Compiled path otherwise, with span fallback when `BuildCompiledDelegate` throws `NotSupportedException`.
Span evals increment `NodeTreePathEvaluationCount` (3 counters kept, names unchanged);
`RecordDiagnostics(..., "Span fused kernel")`. `ComputeMask` stays only on the compiled path.
Replace `GetNodeTreeRunner` MethodInfo-invoke with a cached `CreateDelegate` per result type
(non-generic delegate signature) to avoid per-eval reflection on the span path.

### Step 5 — Tests

- Update assertions at `FusedExpressionEvaluatorTests.cs:67` and `:331`:
  `CompiledPathEvaluationCount == 1` → `NodeTreePathEvaluationCount == 1` (null-bearing uniform
  plans now span-routed).
- `Notes == "Compiled fused kernel"` assertions at `ExpressionEvaluatorTypedFastPathTests.cs:357,392`
  stay green (those plans remain compiled).
- Test at `FusedExpressionEvaluatorTests.cs:385` stays green via retained `mask` parameter.
- Add span-kernel null-mask propagation tests (two-leaf OR propagation, direct `Execute<T>` span
  values, span-routing counter).
- Re-validate `tests/Nivara.PerformanceTests/Program.cs` `CreateFusedChainScenario` (null-free
  uniform → stays compiled, bench intact).

## Blast radius

- **Files:** `src/Nivara/Expressions/FusedKernel.cs` (span rewrite),
  `src/Nivara/Expressions/FusedExpressionEvaluator.cs` (invoke, snapshot, routing),
  `src/Nivara/Interfaces.cs` (+`Data` on internal `IColumnStorage<T>`).
  `ColumnExpression.cs`, `ExpressionTypeInferer.cs`, the three ops, and `ParallelExecutionStrategy`
  untouched (contiguity/null checks happen at eval time against live columns).
- **All internal to the `Nivara` assembly.** Public surfaces (`Where`, `Select`, `OrderBy`,
  `SortByExpression`) unchanged → zero public API break.
- **Downstream callers:** `FilterOperation.cs:62`, `SelectOperation.cs:101`,
  `SortByExpressionOperation.cs:151`, `ParallelExecutionStrategy.cs:143` — call only
  `Evaluate`/`EvaluateBoolean`; unchanged.
- **Tests:** `FusedExpressionEvaluatorTests.cs` (guardrail counters), 
  `ExpressionEvaluatorTypedFastPathTests.cs` (Notes == "Compiled fused kernel"), Filter/Select/Sort
  tests, fused-chain performance scenario.

## Verification

- `dotnet build Nivara.slnx` after each step (AGENTS.md).
- Ask before `dotnet test` or any long-running verification.
- Regression guards: `FusedExpressionEvaluatorTests`, `ExpressionEvaluatorTypedFastPathTests`,
  Filter/Select/Sort tests, fused-chain performance scenario.

## Planned commits

1. `docs: plan fused span-kernel execution (#166) in TODO.md`
2. `refactor: replace DynamicInvoke with cached typed invocation in fused evaluator`
3. `perf: zero-copy leaf reads for null-free contiguous columns in compiled path`
4. `feat: add span-in/span-out generic kernel with fused mask to FusedKernel`
5. `refactor: route null-bearing uniform plans through span kernel; add cached span runner`
6. `test: span-kernel null-mask propagation; update guardrail counters`
7. `docs: remove TODO.md — plan executed`

## GitHub issues log

- No issues created yet. As each task executes, if deferred work or a concern is found (known
  limitations, follow-ups, refactors) outside this plan, create a GitHub issue immediately via
  `gh issue create --repo khurram-uworx/Nivara` and record its number here — don't rely on memory.
