# TODO — #862 Additive `Chunk` / `ChunkedColumn` behind the existing facade

**Issue:** [#862](https://github.com/khurram-uworx/Nivara/issues/862)
**Plan:** [`docs/plan/862.md`](plan/862.md) (authoritative; this file tracks execution)
**Branch:** `khurram/862` (off `main`)

## Problem

The engine has exactly one storage shape: `ColumnStorage<T>` (one contiguous array plus an
optional byte-wise null mask). Chunk/block structure therefore cannot survive a `Slice`, a
`Concat`, or an Arrow round-trip — every path either copies or flattens. Issue #862 asks for an
internal chunk layer behind the existing `IColumnStorage<T>` seam so structure can be carried
without changing the public `NivaraColumn<T>` / `NivaraFrame` surface.

## Proposed changes (from the plan)

| Step | Commit | Files |
|---|---|---|
| 0 | `test(probe)` — `chunked-arrow` probe | `tests/Nivara.SimdProbe/ChunkedArrowProbe.cs`, `Program.cs`, `README.md`, csproj (Apache.Arrow ref) |
| 1 | `feat(storage)` — chunk layer | `src/Nivara/Storage/Chunk.cs`, `ChunkedColumn.cs`, `tests/Nivara.Tests/Storage/ChunkedColumnTests.cs` |
| 2 | `refactor(column)` — retire ten guards, `AsTensorView` throw | `src/Nivara/NivaraColumn.cs` |
| 3 | `feat(helpers)` — chunk-preserving concat | `src/Nivara/Helpers/ColumnFilterHelper.cs`, concat tests |
| 4 | `feat(io)` — opt-in Arrow chunk preservation | `ArrowConversionOptions.cs`, `ArrowInterop.cs`, `tests/Nivara.Tests/IO/ChunkedArrowRoundTripTests.cs` |
| 5 | `test(perf)` — structural guard | `tests/Nivara.Tests/...` (`[Category("Performance")]`) |
| 6 | `docs(arrow)` — roadmap | `docs/plan/ARROW-ROADMAP.md` |

Design invariants (plan §5): `Chunk<T>` is a readonly struct over one already-sliced
`ColumnStorage<T>`; `ChunkedColumn<T>` implements `IColumnStorage<T>` with **≥2 chunks** (one chunk
collapses to `ColumnStorage<T>`); cached prefix offsets give O(log n) indexing; `Data`/`AsSpan`/
`NullMask`/`NullMaskMemory` lazily materialise one contiguous copy; `TryGetSpan` is always `false`;
`AsWritableSpan` throws; `Dispose` never disposes chunks.

## Blast radius

- **`IColumnStorage<T>` implementors:** `ColumnStorage<T>` (unchanged) + new `ChunkedColumn<T>`.
  No interface member changes.
- **`NivaraColumn<T>`:** ten `Unsupported storage type…` guards removed; fast paths untouched.
  Affected operations: scalar/column arithmetic, scalar/column equality, `>`, `<`.
- **`AsTensorView()`:** new `NotSupportedException` only for a non-flat (chunked) backing.
- **`ColumnFilterHelper.ConcatenateColumns`:** chunk-preserving branch only when an input is
  chunked; flat path unchanged. Call sites: `StreamingWindowProcessor` (2), `ParallelExecutionHelper`,
  `ParallelExecutionStrategy` (2), `ConcatenationOperation` (delegates).
- **`AsWritableSpan` audit:** only `NivaraColumn.AsWritableSpan` (internal, no `src` callers) and
  `ColumnStorage` implement it; AutoDiff optimizer call sites are on `Tensor<T>`, not this seam.
  Throwing on a chunked storage is safe.
- **Arrow:** `ArrowConversionOptions` (new opt-in property, default `false`), `ArrowInterop`
  import/export. Default path unchanged.
- **Docs:** `docs/plan/ARROW-ROADMAP.md` is outside the scanned docs scope → no `DocsCoveragePins`
  bump expected. Re-check after the branch review.

## Verification

- `dotnet build Nivara.slnx` (targeted builds as work lands)
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Storage"`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~IO.Arrow"`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
- `dotnet run -c Release --project tests/Nivara.SimdProbe -- chunked-arrow` (Step 0 gate)

No full-suite run without explicit approval.

## GitHub issues log

- [ ] (none captured yet)

## Probe lifecycle

- **Step 0 probe origin:** new mode in the existing `tests/Nivara.SimdProbe` project (no temp
  harness) because the question — does `new Table(schema, columns)` preserve per-column chunk
  counts — is a one-off API gate that must run against the in-tree `Apache.Arrow` reference.
- **Reusability:** keep the mode as a documented probe of the Arrow table/column construction
  contract.
