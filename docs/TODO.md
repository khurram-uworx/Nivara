# Plan: Remove `NivaraColumn<T>.CreateFromNullable(Array)` (#222)

## Problem

The generic-class Array overload `NivaraColumn<T>.CreateFromNullable(Array)` was already
reduced to a validation wrapper that delegates to the boxing-free static factory
`NivaraColumn.CreateFromNullable<T>(T?[])` (issue #212). It remains public API with 141
call sites across tests and samples, so every new caller can pick the slower, unvalidated
path. Remove the Array overload entirely and migrate all call sites to the static factory.
Acceptable to introduce a breaking change (compile-time).

## Proposed changes

### 1. Delete the overload (`src/Nivara/NivaraColumn.cs`)
- Remove `public static NivaraColumn<T> CreateFromNullable(Array values)` (~line 320-348).
- Remove `static readonly MethodInfo createFromNullableFactory` (~line 317-318), which only
  serves the deleted method.
- Remove `using System.Reflection;` (line 8) — verified it is the only reflection use in the
  file (`GetMethod`/`MethodInfo`/`BindingFlags`/`Invoke`/`MakeGenericMethod` appear nowhere else).

### 2. Fix dangling XML-doc cref (`src/Nivara/NivaraColumn.Factory.cs:5-10`)
- Class summary references `<see cref="NivaraColumn{T}.CreateFromNullable(Array)"/>`.
  `TreatWarningsAsErrors=true` (Directory.Build.props:4) turns the dangling cref into a
  CS1574 build error. Rewrite the summary without the deleted overload.

### 3. Migrate 141 call sites → `NivaraColumn.CreateFromNullable(...)`
- 34 files, all in `tests/Nivara.Tests/` plus `samples/Nivara.SampleApp/AggregateExample.cs`.
  Pattern: `NivaraColumn<T>.CreateFromNullable(<T?[] arg>)` → `NivaraColumn.CreateFromNullable(<arg>)`.
- Type inference covers every site (all args are typed `T?[]` literals/variables, including
  generic helper `CreateTestFrame<T>(...)` at `IO/ArrowInteropTests.cs:1122` and the type-switch
  branches at `:1149-1156`).
- Two `null!` sites need an explicit type argument:
  `NivaraColumn.CreateFromNullable<int>(null!)` (`NivaraColumnTests.cs:1999`,
  `NullHandlingErrorConditionTests.cs:211`).
- NOT migrated in step 3 (handled by the removal commit):
  - `NivaraColumnTests.cs:2004` and `NullHandlingErrorConditionTests.cs:227` — reference-type
    rejection tests; deleted when the runtime check becomes a compile-time constraint.
  - `NivaraColumnTests.cs:2068` (`StaticFactory_CreateFromNullable_MatchesArrayOverload`) —
    the parity comparison target is the deleted overload; reworked to assert directly.

### 4. Behavioral test updates (runtime checks → compile-time constraint)
- Delete `CreateFromNullable_WithReferenceType_ShouldThrowInvalidOperationException`
  (`NullHandlingErrorConditionTests.cs:218-232`).
- Delete the reference-type sub-assertion in `CreateFromNullable_ShouldThrowForInvalidInputs`
  (`NivaraColumnTests.cs:2001-2004`); keep the null-array case.
- Rework `StaticFactory_CreateFromNullable_MatchesArrayOverload` to assert factory output
  against `intValues` directly (values + mask), mirroring the existing double portion.

### 5. Docs
- `docs/AUTODIFF.md:1530` — migrate snippet to `NivaraColumn.CreateFromNullable(...)`.
- `CHANGELOG.md` — add `### Changed` breaking-change entry with migration path.
- `docs/REVIEW*.md` are historical records — left as-is.

## Blast radius

| Area | Files | Risk |
|---|---|---|
| Public API removal | `src/Nivara/NivaraColumn.cs` | Breaking; only `src` refs are the deleted method/field |
| XML doc cref | `src/Nivara/NivaraColumn.Factory.cs` | Build error if not fixed (CS1574) |
| Call-site migration | 34 test/sample files (141 sites) | Compile-driven; mechanical |
| Behavioral tests | `NivaraColumnTests.cs`, `NullHandlingErrorConditionTests.cs` | 2 deletions + 1 rework |
| Docs | `docs/AUTODIFF.md`, `CHANGELOG.md` | None |

Internal dispatch (`JoinOperation`, `ColumnFactory`, `FusedExpressionEvaluator`) already uses
the static factory and is unaffected. `JoinOperation`'s cached
`nameof(NivaraColumn.CreateFromNullable)` still resolves to the static generic method.

## Verification

- `dotnet build Nivara.slnx` before each commit (warnings-as-errors gate).
- `dotnet test` — ask the human before running (AGENTS.md).

## Commit list

1. `docs: plan khurram/222 — remove NivaraColumn<T>.CreateFromNullable(Array) in TODO.md`
2. `tests: migrate call sites to NivaraColumn.CreateFromNullable static factory`
3. `refactor: remove NivaraColumn<T>.CreateFromNullable(Array) overload (breaking)`
4. `docs: note CreateFromNullable(Array) removal in CHANGELOG and AUTODIFF.md`
5. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none yet — scan during execution for deferred work)
