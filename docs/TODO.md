# Plan — #498: handle-release gate for non-Incident `ScanAsQueryFrame` consumers

## Problem

`ParquetLazySource` / `CsvLazySource` / `JsonLazySource` open their backing file with
`FileShare.Read` and no `FileShare.Delete`, so any consumer that forgets to dispose its
`QueryFrame` blocks `File.Delete` / `Directory.Delete` on Windows. The misleading
`IOException: ... is being used by another process` is what made #496 read as a cross-process
race rather than a same-process handle leak.

#496 fixed the six `Analysis` entry points and added
`tests/Nivara.Tests/Incident/AnalysisResourceTests.cs`, which probes with `FileShare.None`
(fails while any other handle is open — deterministic, no GC involvement).

That gate is Incident-local. `FileShare.None` appears exactly twice in the whole test tree and
both are in that one file. Every other consumer is ungated.

### The consumer surface is six entry points, not the three the issue names

| Entry point | Location | Backing handle |
|---|---|---|
| `Csv.ScanAsQueryFrame` | `CsvExtensions.cs:63` | `CsvDataSource.cs:583` `FileShare.Read` |
| `Csv.ScanQuery<T>` | `CsvExtensions.cs:76` | same |
| `Json.ScanAsQueryFrame` | `JsonExtensions.cs:63` | `JsonStreamReader.cs:86-87` — **two** `FileShare.Read` streams |
| `Json.ScanQuery<T>` | `JsonExtensions.cs:76` | same |
| `NivaraParquetReader.ScanAsQueryFrame` | `NivaraParquetReader.cs:133` | `ParquetDataSource.cs:396` `FileShare.Read` |
| `NivaraParquetReader.ScanQuery<T>` | `NivaraParquetReader.cs:150` | same |

### Handle-release semantics differ per source — this decides the gate's shape

| Source | Released by `Collect()` alone? | Load-bearing on `Dispose()`? | Where |
|---|---|---|---|
| Parquet | **No** — `lazyReader` holds the stream | **Yes** | `ParquetDataSource.cs:499-509` |
| CSV | Yes — `ReadChunk` closes the chunk reader at EOF | Only for partial reads | `CsvDataSource.cs:405` |
| JSON | Yes — `ReadChunk` closes the chunk reader at EOF | Only for partial reads | `JsonDataSource.cs:370,376` |

**Consequence:** a gate shaped like the Incident one (open → dispose → probe) is **vacuous for
CSV and JSON**. A full-to-EOF read releases their handles regardless of disposal, so those
cases would pass even if disposal were removed entirely. That is exactly why the existing
`ParquetStreamingTests.cs:254` and `JsonStreamingTests.cs:262` assertions (via `File.Delete`)
are green today. To have teeth, CSV/JSON cases must exercise a **partial** read — break out of
`AsStream` mid-file — leaving `chunkCsvReader` / `chunkReader` non-null.

### Abandoned frames have no cleanup path at all

`QueryFrame.cs:33` tracks itself only `if (NivaraResourceManager.IsEnabled)`, and tracking is
**off by default** (`ResourceManagementPropertyTests.cs:20` calls `Enable()` explicitly). There
is no finalizer. An abandoned `QueryFrame` holds its handle until the inner `FileStream`'s
`SafeFileHandle` is finalized — non-deterministically, which is precisely the intermittency
#496 hit.

`tests/Nivara.Tests/Streamix/StreamixBridgeIntegrationTests.cs:273` is exactly that shape:

```csharp
using var expected = Csv.ScanAsQueryFrame(csvPath).Collect();
```

`using` binds to the `NivaraFrame`; the `QueryFrame` is an abandoned temporary. This is in a
file the issue names.

### CI is ubuntu-latest only, and `File.Delete` cannot fail there

On Unix, unlinking a file with an open handle always succeeds. So:

- The symptom #496 reported (`Directory.Delete` throwing) is Windows-only and never reproduced by CI.
- The existing `File.Delete`-based assertions are **structurally incapable of failing on CI**.
- Whether the `FileShare.None` probe has teeth on Linux depends on .NET mapping `FileShare.None`
  to `flock(LOCK_EX | LOCK_NB)` and honoring it across separate fds in one process.

A green gate might therefore be reporting nothing. See the negative control below.

### `NivaraQuery<T>` is not `IDisposable`

`NivaraQuery.cs:19` — `public sealed class NivaraQuery<T>`, no interface, no finalizer. So the
typed half of the surface has no `using`; the only release is `query.AsQueryFrame().Dispose()`.
The gate must assert release after **that specific call**, and say so in the failure text, so
nobody "fixes" a future red by adding a `using` to a non-disposable object.

## Decisions (confirmed with human before execution)

1. **Leave `FileShare.Read` alone.** `CsvLazySource.EnsureChunkPosition`
   (`CsvDataSource.cs:579-599`) and `JsonLazySource.EnsureChunkPosition`
   (`JsonDataSource.cs:487-493`) both **re-open the file by path** on a backward seek. Adding
   `FileShare.Delete` would let a concurrent delete succeed and then turn those re-opens into
   a mid-stream `FileNotFoundException` — trading a currently-correct read for cleanup
   convenience. It also only helps on Windows, and it would silence the very signal the gate
   exists to detect.
2. **Test-only scope.** The missing `IDisposable` on `NivaraQuery<T>` is a separate issue (public
   API addition with its own compatibility story).
3. **Negative control fails loudly on Linux** rather than `Assert.Ignore` — if the probe has no
   teeth there, the build says so instead of the gate reporting green for the wrong reason.

## Proposed changes

### 1. Shared helper — `tests/Nivara.Tests/IO/FileHandleProbe.cs`

Follows the `tests/Nivara.Tests/Execution/ExecutionTestHelpers.cs` precedent: a shared,
non-`[TestFixture]` type living in the area folder.

```csharp
static class FileHandleProbe
{
    // Opens with FileShare.None; that open throws while any other handle is open.
    static void AssertUnlocked(string path, string because);

    // Inverse: proves the probe would actually SEE a leaked handle, i.e. the gate has teeth
    // for this consumer on this platform. Without this, a vacuously-passing case is
    // indistinguishable from a correct one.
    static void AssertLocked(string path, string because);

    static void AssertReleasesAfter<T>(string path, string consumer, Func<T> open, Action<T> release);
}
```

The pre-probe is load-bearing: it makes a red verdict attributable (leaked by an earlier case
vs. leaked by this one) instead of an unlocalized red.

### 2. Delete the Incident-local copy

Repoint `AnalysisResourceTests.cs:90-111` at the helper (AGENTS.md rule 8: one authoritative
implementation). The issue explicitly asks for the gate to stop being Incident-local.

### 3. New gate fixture — `tests/Nivara.Tests/IO/ScanAsQueryFrameHandleTests.cs`

Six cases, one per entry point.

- **Parquet** (`ScanAsQueryFrame`, `ScanQuery<T>`): full `Collect()`. Disposal is load-bearing,
  so this is the case that catches a regression outright.
- **CSV and JSON** (`ScanAsQueryFrame`, `ScanQuery<T>` each): break out of `AsStream`
  **mid-file** so the chunk reader is provably still open when release happens. Each case
  asserts `AssertLocked` before releasing — proving the gate is testing something — then
  releases and asserts `AssertUnlocked`.
- **Typed cases** release via `query.AsQueryFrame().Dispose()` and say so in the failure text.

### 4. Negative control — the load-bearing new test

`Probe_DetectsDeliberatelyLeakedHandle`: open a handle with `FileShare.Read`, assert the probe
detects it, close, assert the probe passes. This is what turns "the gate passed" into "the
gate has teeth on this platform". It **fails loudly** rather than ignoring.

### 5. Strengthen the two weak existing tests

`ParquetStreamingTests.cs:254` and `JsonStreamingTests.cs:262` assert via `File.Delete`, which
cannot fail on Linux. Repoint them at the helper so their coverage stops being
platform-dependent.

## Verification steps

1. ~~**G1 — prove the negative control goes red by design on ubuntu-latest.**~~ **DONE.**
   Ran `FileHandleProbe`'s exact probe semantics in `mcr.microsoft.com/dotnet/sdk:11.0`
   (Ubuntu 26.04.1 LTS, .NET 11.0.0-rc.1) and on Windows 10.0.26300. Both, exit 0:

   | check | Linux | Windows |
   |---|---|---|
   | probe sees a `FileShare.Read` handle (negative control) | **PASS** | **PASS** |
   | probe sees two concurrent `FileShare.Read` handles | **PASS** | **PASS** |
   | probe sees a `FileShare.ReadWrite` handle | **PASS** | **PASS** |
   | probe clean after release (no sticky lock) | **PASS** | **PASS** |
   | `File.Delete` with a handle open | **succeeded ⇒ vacuous** | threw ⇒ carries weight |

   So the probe is **not** vacuous on the Linux CI runner: .NET maps share modes onto
   `flock(2)` and honours them across separate descriptors in one process. The gate has real
   teeth on CI. The same run confirms the plan's other claim — the existing `File.Delete`
   assertions in `ParquetStreamingTests.cs:254` / `JsonStreamingTests.cs:262` are structurally
   incapable of failing there, so repointing them (step 5) is a coverage gain, not a no-op.

   What the gate still cannot prove: the Windows `Directory.Delete` *symptom* #496 reported.
   Detection is proven; reproduction is not. Stated in the fixture doc comment.
2. `dotnet build Nivara.slnx` — no new warnings.
3. Targeted run of the new fixture + `AnalysisResourceTests` + the two repointed tests,
   `-c Release`.
4. Full suite `-c Release --filter "Category!=Performance"`, capturing the process exit status
   (not a pipeline filter's status).
5. Re-run the gate fixture to confirm the negative control is stable, not flaky.
6. G2 — two reviews before deleting this file.

## Planned commits

1. `docs: plan #498 in TODO.md`
2. `test: add shared FileHandleProbe helper with negative control`
3. `test: gate handle release across ScanAsQueryFrame/ScanQuery consumers`
4. `test: point AnalysisResourceTests at the shared FileHandleProbe`
5. `test: replace File.Delete assertions in streaming handle tests`
6. `docs: remove TODO.md — #498 plan executed`

If step 3 goes red against `StreamixBridgeIntegrationTests.cs:273`, that is a real defect, not a
bad assertion. Fix it as its **own** commit (`fix(test): dispose abandoned QueryFrame ...`)
rather than weakening the gate — one commit, one reason.

## Blast radius

- **Test-only.** No `src/` change, no public API change, no behavior change.
- `FileHandleProbe` is new, `internal`-by-`sealed`-in-test-assembly. No other fixture can
  collide with the name.
- `AnalysisResourceTests` keeps its six `[Test]` cases and its `IDisposable`-returning
  delegate shape; only the probe implementation moves.
- The two repointed streaming tests keep their existing assertions (data correctness, row
  counts) — only the handle-release mechanism changes from `File.Delete` to the probe.
- **Not covered:** the gate proves *detection* on Linux; it cannot reproduce the Windows
  `Directory.Delete` symptom #496 actually reported. A green CI run is a weaker claim than it
  looks, and the fixture doc comment must say so.

## GitHub issues log

- [ ] #498 — handle-release gate for non-Incident `ScanAsQueryFrame` consumers (this work)
- [x] #501 — `NivaraQuery<T>` is not `IDisposable`, so `ScanQuery<T>` consumers have no `using`
  and must call `query.AsQueryFrame().Dispose()`. Public API addition, split out of this gate.
- [x] #502 — `StreamixBridgeIntegrationTests.cs:273` abandons a `QueryFrame`
  (`Csv.ScanAsQueryFrame(...).Collect()` bound to the result, not the frame). Masked today
  because `CsvLazySource` self-releases at EOF and CI's delete-based assertions cannot fail.
  Deliberately **not** fixed here: #498 is the gate that catches this class; #496 set the same
  precedent (gate in `9813a9c`, consumer fix in `24d10e1`). Once the gate lands, confirm this
  line is genuinely red — if it passes unfixed, the gate's CSV/JSON cases are vacuous and that
  is the more important finding.

> As each task executes, if you find deferred work or a concern outside this plan, create a
> tracked issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its
> number above — do not rely on memory, as compaction during execution can lose items.