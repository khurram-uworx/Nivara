# Plan: Issue #172 — NivaraSeries<T>.Average() for Half/nint/nuint/Int128/UInt128

## Problem

`NivaraSeries<T>.Average()` computes the sum via `sumTensorPrimitive`
(`src/Nivara/NivaraSeries.cs:69-97`, dispatches all 17 numeric types), then divides via
`divideByCount` (`NivaraSeries.cs:102-183`), which covers only 12 types and throws
`NotSupportedException` for **Half, nint, nuint, Int128, UInt128**. The sum path advertises
support but the average path does not — a runtime functional gap (docs/REVIEW.md high-priority
finding #1). bool is unsupported at the sum dispatch itself (consistent, out of scope).

Not fixed by #169: #169 addressed group-by `SumAggregation`/`MeanAggregation`; this is the
separate `NivaraSeries.Average()` path. Work lands on branch `khurram/169` per human instruction.

## Fix

Add the 5 missing cases to `divideByCount`, matching the existing 12-style:
- `Half`: `halfSum / (Half)count`
- `nint`: `nintSum / count`
- `nuint`: `nuintSum / (nuint)count`
- `Int128`: `int128Sum / count`
- `UInt128`: `uint128Sum / (UInt128)count`

Average semantics preserved: returns `T` (truncating division for integral types), same as the
existing int/long/ulong arms. This is the issue's suggested "add the 5 missing cases" option;
it keeps the documented same-type return contract (`NivaraSeries.Average()` returns `T`).

## Tests — `tests/Nivara.Tests/NivaraSeriesAggregateTests.cs` (Average region)

- `Average_HalfValues_ReturnsCorrectAverage`
- `Average_NIntValues_ReturnsCorrectAverage`
- `Average_NUIntValues_ReturnsCorrectAverage`
- `Average_Int128Values_ReturnsCorrectAverage`
- `Average_UInt128Values_ReturnsCorrectAverage`
- `Average_Int128WithNulls_ReturnsValidAverage` (null-filter path)

Use divisible values so same-type truncating division yields exact expected `T`.

## Docs

- `AGENTS.md`: update `divideByCount` note — now covers 17 types (all `GetNumericTypes()`
  minus `bool`); `NivaraSeries.Average` throws only for `bool`.
- `CHANGELOG.md`: entry under Fixed referencing #172.
- `docs/REVIEW.md`: high-priority finding #1 resolved (no edit unless the finding text exists).

## Blast radius

- **Files changed**: `src/Nivara/NivaraSeries.cs`,
  `tests/Nivara.Tests/NivaraSeriesAggregateTests.cs`, `AGENTS.md`, `CHANGELOG.md`.
- **Behavior**: `NivaraSeries<Half/nint/nuint/Int128/UInt128>.Average()` previously threw;
  now computes. No production caller depends on the throw.
- **Consistency**: `series.Average()` returns `T` while `series.Values.Mean()` returns `double`
  — existing, intentional contract; unchanged.
- **Tests**: `NivaraSeriesAggregateTests.cs` (existing Average/Min/Sum region).

## Verification

1. `dotnet build Nivara.slnx`.
2. `dotnet test --filter FullyQualifiedName~NivaraSeriesAggregateTests` (ask human first).
3. No full-suite run without confirmation.

## Planned commit list

1. `docs: plan issue #172 NivaraSeries.Average fix in TODO.md`
2. `Fix NivaraSeries.Average for Half/nint/nuint/Int128/UInt128`
3. `test: cover NivaraSeries.Average for new numeric types`
4. `docs: mark #172 resolved, update divideByCount note`
5. `docs: remove TODO.md — issue #172 plan executed`

## GitHub issues log

- [ ] #169 — Sum/Mean group-by aggregation full numeric domain (fixed on this branch, PR #182)
- [ ] #172 — `NivaraSeries<T>.Average()` throws for Half/nint/nuint/Int128/UInt128 (this plan)
