# Plan: issue #1166 — `Equals` means three different things across NivaraColumn, NivaraFrame and Schema

Branch: `khurram/1166` off `main` (HEAD `591f99bc`), package version 1.5.0.

## Problem

`Equals` is overloaded with three different meanings:

- `NivaraColumn<T>.Equals(T)` and `Equals(NivaraColumn<T>)` return `NivaraColumn<bool>` — an element-wise comparison operator wearing an equality name.
- `NivaraFrame.Equals(object?)` (line 1308) is overridden to `ReferenceEquals`, so two independently built identical frames are "not equal" and `Distinct()`/`HashSet` do nothing useful.
- `Schema.Equals` is structural and already metadata-aware (#234, closed 2026-08-14, commit `f33f64d7`). `NivaraSeries<T>` has no equality at all.

There is no `IEquatable<NivaraColumn<T>>`, no `==`/`!=`, and the element-wise set is asymmetric (`Equals` exists, `NotEquals`/`>=`/`<=` do not). The issue body's line refs and Schema claim are stale (verified against HEAD).

## Blast radius

- `src/Nivara/NivaraColumn.cs` (public methods `Equals(T)`/`Equals(NivaraColumn<T>)` + comparison helpers) — rename + additions + structural members.
- `src/Nivara/Helpers/NumericTensorKernels.cs`, `NumericKernelDispatcher.cs` — 3 new comparison ops.
- `src/Nivara/Interfaces.cs` — two internal `IColumn` members.
- `src/Nivara/NivaraFrame.cs` — structural equality.
- `src/Nivara/Query/QueryFrame.cs` — remove misleading overrides.
- `src/Nivara/NivaraSeries.cs` — structural equality (in scope per decision).
- Call sites: `tests/Nivara.Tests/NivaraColumnTests.cs` (20), `ComplexScenarioIntegrationTests.cs:139`, `DiagnosticsTests.cs:142`, `MixedTypeIntegrationTests.cs:271`, `samples/Nivara.SampleApp/DiagnosticsExample.cs:66`, `tests/Nivara.PerformanceTests/ChunkedColumnProbe.cs:135-145`.
- `CHANGELOG.md`, `AGENTS.md`, `docs/REVIEW-2026-11-12.md` (one prose sentence).

No internal code keys dictionaries/HashSets by columns or frames (verified).

## Decisions (locked)

1. **Rename now**, CHANGELOG migration block (1.5.0 precedent for type renames). `Equals` → `EqualTo`/`NotEqualTo`; add `GreaterThanOrEqual`/`LessThanOrEqual`; structural `Equals`+`IEquatable` in the same 1.6.0 breaking entry.
2. **`==`/`!=` on `NivaraColumn<T>`** for column-vs-column only; scalar comparisons stay named (`EqualTo(5)`). Documented asymmetry; no silent wrong behaviour.
3. **`NivaraSeries<T>` in scope** — values + index labels, default positional index equal to an explicit `0..n-1` index.

## Planned commits

1. `feat: complete and rename the element-wise comparison surface on NivaraColumn` — kernels + dispatcher + `EqualTo`/`NotEqualTo`/`GreaterThanOrEqual`/`LessThanOrEqual` + call-site sweep + CHANGELOG migration + AGENTS gotcha.
2. `feat: structural equality for NivaraColumn<T>` — `IColumn` seam, `IEquatable`, `Equals`/`GetHashCode`, `==`/`!=`.
3. `feat: structural equality for NivaraFrame` — `IEquatable`, `Equals`/`GetHashCode`, `==`/`!=`.
4. `fix: drop the fake Equals/GetHashCode overrides on QueryFrame` — identity semantics.
5. `feat: structural equality for NivaraSeries<T>`.
6. `docs: correct the M3 sentence in REVIEW-2026-11-12.md` — Schema metadata claim.
7. `test: equality fixtures for columns, frames, series`.

## Semantics that must be documented

- Column: `EqualityComparer<T>.Default` (NaN equals NaN), null positions skipped, chunked-vs-flat equal, disposed throws.
- Frame: column order sensitive; names compared OrdinalIgnoreCase via `Schema`; `SortKeys` ignored; metadata included (default); disposed throws; O(n) per call.
- Series: default index == explicit positional index; hash agrees without allocation.
- `QueryFrame` keeps identity semantics (lazy/streaming state).

## Verification

- `dotnet build Nivara.slnx -c Release` (0 warnings).
- targeted filters for the touched fixtures.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- commit 6 only: `FullyQualifiedName~Nivara.Tests.Docs`.
- No full suite or benchmarks without asking.

## GitHub issues log

- [x] #1166 — Equals means three different things (this work).
- [ ] #234 — Schema metadata equality (already closed; referenced for context).
- [ ] #1163 — no deprecation channel (context for the breaking-rename decision).

No new deferred-work issues created yet.
