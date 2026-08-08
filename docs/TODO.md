# TODO: Phase 2 — kernel fusion + generic-math collapse (POLARS-ROADMAP)

## Problem

Phase 1 (typed fast path + typed promoted path, PR #153) made `ExpressionEvaluator`
**typed and vectorized per operator**, but an expression like
`(Salary * 1.1) + 1000 - Tax` still materializes an intermediate column per
operator. The roadmap (Phase 2) calls for fusing the whole tree into a single
span pass with no `object?` boxing and no intermediates. In parallel, the
column layer's `float`/`double` type-switches should collapse onto generic math
(`INumber<T>` / `IFloatingPointIeee754<T>`), mirroring AutoDiff — which also
brings `Half`/`decimal` through the generic path. BFloat16 is explicitly **not**
a target (net11 only; it rides in automatically then).

Current gaps on `main`:

- No fused evaluator: `FilterOperation`, `SelectOperation`,
  `SortByExpressionOperation`, and `ParallelExecutionStrategy` each build a
  fresh `ExpressionEvaluator` and evaluate every operator separately.
- Boxed fallbacks remain: `ExpressionEvaluator` (`ApplyBinaryOperation` /
  `ApplyComparisonOperation` / `AddValues` … `CompareLessThanOrEqual`) and the
  `dynamic` scalar loops in `NivaraColumn<T>` arithmetic (e.g. `NivaraColumn.cs:58`)
  fire for any unsupported/opaque type instead of throwing.
- `MultiColumnComparer` (`SortOperation.cs:293`) boxes values via
  `IColumn.GetValue` + `IComparable.CompareTo`; `OrderBy(r => r["Salary"] * 1.1)`
  keys are typed/vectorized at materialization but compared boxed.
- `TypeCompatibilityValidator.GetNumericTypes()` / kernel dispatch cover
  10 primitives only; `Half`, `nint`/`nuint`, `Int128` etc. miss the typed path.

## Decision (confirmed with user)

- **Hybrid fused evaluator, compiled-first:** translate the `ColumnExpression`
  AST to `System.Linq.Expressions` → `Compile()`, cached per (expression
  signature, element type), run once over the whole column.
  **Span caveat:** ref structs (`Span<T>`) are prohibited in expression trees
  (confirmed via MS Learn), so the compiled delegate runs over `T[]` arrays,
  not spans. Straight-line IL auto-vectorizes; this is the primary path.
  A generic sealed **node-tree** target is the fallback (JIT-monomorphized
  single loop) for shapes where expression-tree build/compile fails. The
  existing boxed `ExpressionEvaluator` is removed as a fallback for arithmetic.
- **Remove the `dynamic`/boxed fallback and throw** for unsupported type
  combinations (clear `SchemaValidationException`/`NotSupportedException` with
  column/type context). No capability regression: Guid/string/DateTime/…
  comparisons keep working through the typed comparison dispatch and the fused
  compiled path (`Comparer<T>.Default`-based).
- **Typed sort comparer, no SIMD sort:** `MultiColumnComparer` compares via
  `Comparer<T>.Default` on `NivaraColumn<T>` values; `Array.Sort` stays.
- **Null semantics** (ADR-001 / roadmap): result null mask = OR of leaf column
  masks; comparisons produce masked-false at nulls; `Not` propagates operand
  mask. Same as the current evaluator so fused output is bit-equivalent.
- **Roadmap documentation:** after the plan is executed, update the
  "Sequencing rationale" section of `docs/plan/POLARS-ROADMAP.md` to mark
  Phase 1 and Phase 2 delivered, following the Phase 3 documentation pattern
  (`✅ Delivered` rows + `Status (…)` lines + "Scope (remaining)").

## Changes

### Task 1 — `src/Nivara/Expressions/ExpressionTypeInferer.cs` (new)

Walks a `Validate`d `ColumnExpression` AST and computes one promoted element
type for the whole subtree via `NumericPromoter` (chain-wise promotion, so
`A * 1.1 + 1000` unifies to `double`). Coerces `LiteralExpression.Value` /
`ScalarExpression.Scalar` to the unified type via `T.CreateChecked`
(generic-math) or `Convert.ChangeType`. Returns a small plan record
(`Type ElementType`, `bool IsGenericMath`, leaf column bindings) or signals
"not fusable" for opaque `object` subtrees.

### Task 2 — Fused evaluator: `src/Nivara/Expressions/FusedExpressionEvaluator.cs` + `FusedKernel.cs` (new)

- Entry points mirror `ExpressionEvaluator`: `Evaluate(...)` and
  `EvaluateBoolean(...)`.
- **Compiled target:** AST → expression tree over `T[]`; `Compile()` once,
  cache in a `ConcurrentDictionary<(string Signature, Type ElementType), Delegate>`.
  Rent destination `T[]` + `bool[]` mask via `ArrayPool<T>.Shared`; run once;
  null mask OR'd from leaf masks only when any leaf has nulls. Emits
  `NivaraColumn<T>.CreateFromSpans` or `Create`.
- **Node-tree target:** sealed generic nodes `FusedLiteral<T>`, `FusedColumnRef<T>`,
  `FusedBinary<T>`, `FusedComparison<T>`, `FusedNot<T>`; single loop; one result
  + one mask.
- Guardrail counters `FusedPathEvaluationCount` / `CompiledPathEvaluationCount`;
  record route + element type + `IsVectorizable` in `OperationDiagnostics`.

### Task 3 — Wire fused evaluator into operations

- `FilterOperation.Execute` (`FilterOperation.cs:62`), `SelectOperation.Execute`
  (`SelectOperation.cs:102`), `SortByExpressionOperation.Execute` +
  `MaterializeKeys` (`SortByExpressionOperation.cs:152`, `:185`),
  `ParallelExecutionStrategy.executeExpressionSortParallelSync`
  (`ParallelExecutionStrategy.cs:142`): `new ExpressionEvaluator()` →
  `new FusedExpressionEvaluator()`.

### Task 4 — Generic-math collapse + remove boxed/dynamic fallbacks

- `TypeCompatibilityValidator.GetNumericTypes()` and the kernel dispatch lists:
  add `Half`, `decimal` (already listed), `nint`/`nuint`, `Int128`/`UInt128`
  (whichever BCL supports) via `INumber<T>` kernels. BFloat16 excluded.
- `ExpressionEvaluator`: delete `ApplyBinaryOperation`, `ApplyComparisonOperation`,
  `AddValues`/`SubtractValues`/`MultiplyValues`/`DivideValues`, and the boxed
  `Compare*` helpers; unsupported combos throw. Move `Guid` (and any comparable
  type currently relying on the boxed path) onto the typed comparison dispatch
  (`Comparer<T>.Default` supports it) so capability is preserved.
- `NivaraColumn<T>` `dynamic` scalar loops (`NivaraColumn.cs:58,84,110,136,162,188`):
  route `Half`/`decimal`/`nint`/… through `NumericTensorKernels<T>` type-switch
  branches; the residual `dynamic` loop becomes a clear `NotSupportedException`
  for non-numeric `T`.
- Keep `CreateConstantColumn`'s generic `NivaraColumn<object>` creation (column
  creation, not a perf path). Public `NivaraFrame.Where(Func<dynamic, bool>)`
  and `NivaraSeries<T>` dynamic aggregate fallbacks are out of scope (breaking /
  lower priority) — file as follow-up issues (Task 8).

### Task 5 — Typed `MultiColumnComparer` (`SortOperation.cs:293`)

Dispatch on `IColumn is NivaraColumn<T> typed` → `Comparer<T>.Default.Compare(
typed[x], typed[y])` (nulls via `typed.IsNull`, existing `NullOrdering`
semantics preserved); keep the `IComparable` path only for non-`NivaraColumn<T>`
object columns. No SIMD sort.

### Task 6 — Tests

- New `tests/Nivara.Tests/Query/FusedExpressionEvaluatorTests.cs`:
  bit-equivalence + null-mask equivalence vs the existing evaluator (chained
  arithmetic, mixed int/double, comparisons with nulls, `And`/`Or`/`Not`);
  guardrails (`FusedPathEvaluationCount`/`CompiledPathEvaluationCount`);
  `Half`/`decimal` through the fused/generic path; unsupported arithmetic throws.
- Update `ExpressionEvaluatorTests.cs` Guid tests → typed comparison
  (`TypedPathEvaluationCount == 1`, `BoxedPathEvaluationCount == 0`) and any
  boxed-fallback assertions that now throw.

### Task 7 — Benchmark + diagnostics

- `tests/Nivara.PerformanceTests/Program.cs`: chained-arithmetic fused vs
  multi-pass (`Salary * 1.1 + 1000 - Tax`) at 100k/1M rows; assert fused wins at
  `Length >= vectorSize * 4` (`KernelSelector` heuristic).
- Surface fused route in `OperationDiagnostics` / `ExecutionEngine.LastDiagnostics`.

### Task 8 — File follow-up GitHub issues (`gh`)

Candidates (check existing issues first to avoid dupes):
- `NivaraFrame.Where(Func<dynamic, bool>)` — last public `dynamic` surface;
  typed conversion is a breaking change.
- `NivaraSeries<T>` dynamic aggregate fallback removal.
- `Over`/`Rank`/`DenseRank` window remainder (exists as follow-up already?).
- `Expression.Compile`-over-arrays (not spans) deviation note.

### Task 9 — Update "Sequencing rationale" in `docs/plan/POLARS-ROADMAP.md`

Mark Phase 1 and Phase 2 rows in the §2 table as `✅ Delivered` (Phase 1 → #153),
add `Status (…)` lines to the Phase 1 and Phase 2 sections following the Phase 3
pattern (`Status` line + `Scope (remaining)`), and refresh the §0
"Where we are today" paragraph to state the expression hot path is now
typed/fused with no `object?` boxing for numeric/vectorizable columns. Phase 3's
existing `✅ Delivered (#135)` documentation stays.

### Task 10 — Close-out (iterative-commit)

Review `docs/TODO.md`, confirm every item is done, remove it
(`git rm docs/TODO.md`), commit, then offer push + PR (human-confirmed).

## Verification

- `dotnet build Nivara.slnx` after each task.
- `dotnet test` — only after explicit human confirmation (AGENTS.md).
- Focused test runs for fused evaluator + sort + typed-promotion fixtures.

## Planned commits

1. `docs: plan Phase 2 kernel fusion + generic-math collapse in TODO.md`
2. `feat: add fused expression type inference (ExpressionTypeInferer)`
3. `feat: add fused expression evaluator with compiled + node-tree targets`
4. `feat: route filter/select/sort operations through the fused evaluator`
5. `refactor: collapse generic-math types and remove boxed/dynamic fallbacks`
6. `refactor: typed MultiColumnComparer without boxing`
7. `test: fused evaluator bit-equivalence, null masks, Half/decimal generic path`
8. `bench: fused vs multi-pass chained arithmetic`
9. `chore: file follow-up issues for dynamic API surfaces`
10. `docs: update roadmap sequencing rationale for delivered phases`
11. `docs: remove TODO.md — plan executed`

## Follow-ups (after this plan)

- `NivaraFrame.Where(Func<dynamic, bool>)` typed conversion (breaking).
- `NivaraSeries<T>` dynamic aggregate fallback removal.
- BFloat16 kernels at net11 (issue #137 already tracks).
