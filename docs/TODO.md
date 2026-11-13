# Plan: Issue #1189 — dropped, partially-read lazy query source releases its handle only at finalization

## Problem

Issue #1168 deleted the internal `NivaraResourceManager` abandoned-resource tracker. The
removal left one gap: a dropped, partially-read lazy query source leaks its file handle
until the underlying stream is finalized. `docs/STREAMING.md` §"Resource management"
already documents the contract ("There is no safety net...").

Evidence:
- `src/Nivara/IO/JsonDataSource.cs` (`JsonLazySource`, `chunkReader` at :82) and
  `src/Nivara.Extensions/IO/ParquetDataSource.cs` (`ParquetLazySource.lazyReader` at :26)
  open readers lazily and hold them until `Dispose()`.
- Neither lazy source implements a finalizer or wraps its reader in a `SafeHandle`.
- `tests/Nivara.Tests/IO/ScanAsQueryFrameHandleTests.cs` (issue #498) probes the
  explicit-disposal path with `FileShare.None` but not the dropped-frame path.

## Decision (G1)

Option (1) — add a finalizer / `SafeHandle` to the lazy sources — cannot move the
release point:

- The handle is already owned by a `FileStream` → `SafeFileHandle` created at
  `src/Nivara/IO/JsonStreamReader.cs:86-87`, `src/Nivara.Extensions/IO/CsvDataSource.cs:587`,
  `src/Nivara.Extensions/IO/ParquetDataSource.cs:396` (opened with `FileMode.Open`,
  `FileShare.Read`, `leaveStreamOpen: false` at :399).
- A dropped frame's graph (frame → `QuerySourceHandle` → source → `Lazy<ParquetReader>` →
  reader → `FileStream` → `SafeFileHandle`) becomes unreachable together, so the
  `SafeFileHandle` critical finalizer closes the OS handle in the first finalization
  pass that collects it — with or without our own finalizer.
- Adding a finalizer would instead run managed `DisposeAsync(...).GetAwaiter().GetResult()`
  on the finalizer thread (deadlock/ordering hazard), force CA1063/CA1816 restructuring
  under `TreatWarningsAsErrors=true`, and put every scan on the finalization queue for
  zero observable benefit. Its acceptance test ("released within one GC cycle") is
  satisfied by the current tree, so it cannot fail before the change — a gate that
  proves nothing.

Option (2) chosen: document the contract precisely and verify the finalization claim
with a probe. No product code change.

## Proposed changes

### Step 0 — probe (investigation, no product change)
- `tests/Nivara.PerformanceTests/DroppedScanHandleProbe.cs` (new) — `static class`,
  `public static int Run()`, following `MetricsProbe` (`MetricsProbe.cs:11-17`).
- Wire `--dropped-scan` into the flag parser (`Program.cs:1068-1105`), dispatch
  alongside the other modes, usage line (`Program.cs:1123`), and document in the
  README's "On-demand helper modes" section.
- Per source (Parquet, Csv, Json): create temp file → negative control → partial read
  one chunk → `FileHandleProbe.AssertLocked` while the frame is referenced → drop via
  `[MethodImpl(NoInlining)]` helper → `GC.Collect`/`WaitForPendingFinalizers`/`Collect`
  ×3, reporting locked/unlocked after each round. Non-zero exit if still locked.

### Step 1 — permanent pin
- `tests/Nivara.Tests/IO/ScanAsQueryFrameHandleTests.cs`: one case per source family,
  `locked-while-referenced → drop → GC → unlocked`. Doc comment states it pins BCL
  `SafeFileHandle` finalization plus absence of Nivara-side retention (not a Nivara
  release path). No timing assertion → no `TimingGuards`, no `[Category("Performance")]`.

### Step 2 — repair #1168 text damage (own commit)
- `docs/STREAMING.md:266-273`: rejoin the split bullet and remove the duplicated
  "partial reads leak / Parquet holds its reader until disposal" sentence.
- `CHANGELOG.md:27-28`: move "All notable changes…" back under `# Changelog`.

### Step 3 — record the decision
- `ARCHITECTURE.md`: new "Lazy source ownership and release" subsection under
  `## File I/O Architecture` (`:640-650`); one-line pointer in the existing
  §Ownership and disposal list (`ARCHITECTURE.md:275-284`).
- `docs/STREAMING.md` §Resource management: sharpen the "no safety net" bullet — no
  safety net of ours; the OS handle is closed by the BCL when the stream is finalized;
  this is why Windows teardown fails intermittently. Prose only, no `X.cs::Symbol`
  citations, so `DocsCoveragePins` arithmetic is untouched.

### Step 4 — IntelliSense parity
- Add the one-liner disposal remark to the three `ScanAsQueryFrame` overloads
  (matching the existing `ScanQuery<T>` remark at `src/Nivara/IO/JsonExtensions.cs:74-75`);
  keep `docs/LINQ.md:74-94` consistent.

## Verification
- `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --dropped-scan`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.IO.ScanAsQueryFrameHandleTests"`
- `dotnet test ... -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Blast radius
- Probe + pin: test projects only; no production code.
- Docs: `docs/STREAMING.md`, `CHANGELOG.md`, `ARCHITECTURE.md` (no citations added).
- IntelliSense: three public `ScanAsQueryFrame` XML doc blocks.

## GitHub issues log
- (none yet; file a follow-up issue after the probe if it shows release needs >1
  collection or never happens, or to track "stop holding the handle between chunks"
  as a product option)
