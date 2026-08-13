# Plan: NivaraColumn dispatch dedup (#204), operators (#210), nullable factory (#212)

Branch: `khurram/nivara-column` (base `main` @ `1e27618`)

## Problems

1. **#204 (performance/code-quality, medium)** — 12 duplicated `typeof(T)` runtime dispatch
   chains in `src/Nivara/NivaraColumn.cs` (lines 39–400: multiply/add/subtract/divide/equals/
   greaterThan/lessThan, scalar+span forms; 17-type arithmetic domain, 11-type comparison
   domain) plus `sumTensorPrimitive`/`divideByCount` in `src/Nivara/NivaraSeries.cs`
   (68–96 / 101–195). Every chain is identical except the concrete
   `NumericTensorKernels<X>` instantiation and the reinterpret casts. ~470 lines of
   mechanical duplication.

2. **#210 (enhancement, medium)** — `NivaraColumn<T>` only exposes `operator *` (3 forms) and
   `operator +` (column+column only). No `-` or `/` operators. Scalar/reverse arithmetic
   methods (`Subtract(T)`, `scalar - column`, `scalar / column`, `Add(T)`) are missing.

3. **#212 (code-quality, low)** — `NivaraColumn<T>.CreateFromNullable(Array)` (line 556)
   boxes every element (`values.GetValue(i)`). The issue claims a generic
   `CreateFromNullable(T?[] values)` overload exists at line 536 — **that claim is false**
   (line 536 is `CreateForReferenceType(ReadOnlySpan<T>)`), and per C# rules such an
   overload **cannot exist on the unconstrained class** `NivaraColumn<T>`: `T?` on an
   unconstrained type parameter erases to `T` for value-type substitutions (only
   `where T : struct` yields `Nullable<T>`). The boxing-free entry must be a static generic
   method with its own `U : struct` constraint.

## Approach

**#204** — new internal `src/Nivara/Helpers/NumericKernelDispatcher.cs`: a
`ConcurrentDictionary<(Type, Operation, Shape), Delegate>` cache. Constrained builder
methods (`where U : INumber<U>`) are closed once per `(type, op, shape)` via
`MakeGenericMethod`; they return typed delegates (never `Invoke` spans — avoids the
AGENTS.md reflection pitfall). NivaraColumn/NivaraSeries keep thin same-signature wrappers
so existing call sites (`applyElementwiseBinary(other, subtractTensorPrimitive)` method
groups) are unchanged. Behavior preserved exactly:
- Arithmetic domain = 17 types (float, double, int, long, short, ushort, uint, ulong,
  byte, sbyte, char, decimal, Half, nint, nuint, Int128, UInt128); anything else →
  `NotSupportedException` (bool passes `IsNumericType` validation then throws, unchanged).
- Comparison domain = 11 types (float, double, int, long, short, ushort, uint, ulong,
  byte, sbyte, char); others fall back to `EqualityComparer<T>`/`Comparer<T>` loops via
  Try-style entry points returning `false`, unchanged.
- `reinterpretReadOnly`/`reinterpretWritable`/`reinterpretScalar` (NivaraColumn) and
  `reinterpretBack` (NivaraSeries) become dead → removed; `SpanReinterpret.cs` is then
  unused anywhere → deleted (confirmed only referenced by these two files).

**#210** — add kernels to `NumericTensorKernels<T>`: `Subtract(span, scalar)`,
`SubtractFrom(T, span)` (scalar − column), `DivideBy(T, span)` (scalar / column),
`DivideByCount(T, int)` = `sum / T.CreateChecked(count)` (semantically identical to the
current per-type `Unsafe.As` casts incl. truncating integer division and char). Add public
`Subtract(T)`, `SubtractFrom(T)`, `DivideBy(T)`, `Add(T)` methods and operators so `*`,
`+`, `-`, `/` each accept column−column, column−scalar, scalar−column. Factor the three
near-identical scalar vectorized methods (`multiplyVectorized`/`divideVectorized`/new)
into one `applyScalarOp(name, fill)` helper that owns storage access + null-mask copy.

**#212** — new non-generic static `NivaraColumn` class with
`CreateFromNullable<T>(T?[] values) where T : struct` (boxing-free). Migrate the six
internal call sites (ColumnFactory, ArrowInterop ×2, ParquetReader, ParquetWriter,
JoinOperation ×2) to it. The `Array` overload stays **non-obsolete** because
`Directory.Build.props` sets `TreatWarningsAsErrors=true` and 34 test/sample files (153
call sites) bind `NivaraColumn<int>.CreateFromNullable(new int?[] {...})` to it; marking
it `[Obsolete]` would break the build. Follow-up issue logged instead (see log below).
JoinOperation's `GetMethod(name, [Nullable<T>[]])` reflection is replaced with
`MakeGenericMethod` on the new factory (removes fragile exact-type lookup), using the same
cached-kernel pattern already in the file.

## Blast radius

- `src/Nivara/NivaraColumn.cs` — dispatch methods 39–400 rewritten; `reinterpret*` deleted;
  scalar arithmetic methods + operators added; `CreateFromNullable(Array)` body delegates
  to the factory.
- `src/Nivara/NivaraSeries.cs` — `sumTensorPrimitive`/`divideByCount` collapse;
  `reinterpretBack` deleted; `System.Runtime.CompilerServices` using removed.
- `src/Nivara/Helpers/NumericTensorKernels.cs` — +4 kernels (all additive).
- `src/Nivara/Helpers/NumericKernelDispatcher.cs` — new (additive).
- `src/Nivara/Helpers/SpanReinterpret.cs` — deleted.
- `src/Nivara/NivaraColumn.cs` → new static `NivaraColumn` factory class (additive; the
  `NivaraColumnBuilder` static class already coexists with `NivaraColumn<T>`).
- Callers migrated: `Helpers/ColumnFactory.cs` (58), `Extensions/IO/ArrowInterop.cs`
  (682, 790), `Extensions/IO/ParquetReader.cs` (401), `Extensions/IO/ParquetWriter.cs`
  (678), `Operations/JoinOperation.cs` (736–739, 829–832).
- Downstream: all arithmetic/operator/comparison users (query, aggregation, windows,
  AutoDiff interop) — behavior identical; only the dispatch mechanism changes.
- Tests covering the changed surface: `NivaraColumnTests.cs` (operators ~236–345,
  545–570, 2290–2300; nullable ~1803+; extended domain), `NivaraSeriesAggregateTests.cs`
  (Sum/Average incl. char + extended domain), `JoinOperationTests.cs`,
  `Execution/ParallelExecutionStrategyTests.cs` (value-type coalesce/gather paths),
  `ColumnTransformationTests.cs`.

## Planned commits

1. `docs: plan #204/#210/#212 in TODO.md`
2. `refactor: add scalar/reverse arithmetic kernels to NumericTensorKernels`
3. `refactor: add NumericKernelDispatcher for centralized numeric dispatch`
4. `refactor: collapse NivaraColumn and NivaraSeries dispatch to NumericKernelDispatcher`
   (incl. deleting `SpanReinterpret.cs` + dead reinterpret helpers)
5. `feat: add scalar and reverse arithmetic methods and -, /, + operators to NivaraColumn`
6. `test: cover scalar/reverse arithmetic and new operators across numeric domain`
7. `refactor: add NivaraColumn.CreateFromNullable<T> static factory and migrate callers`
8. `test: cover static nullable factory CreateFromNullable<T>`
9. `docs: remove TODO.md — plan executed`

Build (`dotnet build Nivara.slnx`) after each commit. `dotnet test` runs are
human-confirmed before starting (AGENTS.md / iterative-work rule).

## Verification

- `dotnet build Nivara.slnx` after every commit (warnings-as-errors enabled).
- Targeted `dotnet test` on `NivaraColumnTests`, `NivaraSeriesAggregateTests`,
  `JoinOperationTests`, `ParallelExecutionStrategyTests`, then full suite — always with
  human confirmation first.
- Behavioral invariants to preserve: `bool` arithmetic → `NotSupportedException`; string
  `operator *` → `InvalidOperationException`; mismatched lengths → `ArgumentException`;
  comparison on decimal/Half/extended types uses comparer fallback; char/byte/short
  average truncation matches current results.

## GitHub issues log

- [ ] #NNN — mark `NivaraColumn<T>.CreateFromNullable(Array)` `[Obsolete]` after migrating
      test/sample callers (created while working on #212; blocked by
      `TreatWarningsAsErrors` — 153 call sites across 34 files bind to the Array overload).

Reminder: as each task executes, if you find deferred work or a concern, create a GitHub
issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number
here — don't rely on memory or wait until the end of the plan, as compaction during
execution can lose important items.
