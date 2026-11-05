# Plan — #920: concurrency guard on RagIndexCache writes

## Problem

`RagIndexCache.Save` writes the shared Parquet cache and its manifest directly to their final
paths with no coordination. Two simultaneous `--rag`/`--rag-agent` launches can interleave writes
and leave a torn Parquet file. `IsValid`/`Load` can also observe a file while it is being written.

## Proposed changes

All changes are in `samples/NivaraChat/Helpers/RagIndexCache.cs`; public signatures stay unchanged.

1. Add a cross-process exclusive lock sidecar at `indexPath + ".lock"`.
   - `TryAcquireLock(lockPath, timeout, out FileStream?)` opens the sidecar with
     `FileMode.OpenOrCreate`, `FileAccess.ReadWrite`, `FileShare.None`, retrying for up to 2s on
     contention and failing fast on permanent errors. No unbounded wait.
2. `Save`:
   - Ensure the destination directory exists.
   - Try to acquire the lock; if it cannot be acquired within the timeout, skip the save rather
     than hang.
   - While holding the lock, write to GUID-suffixed temp files and then `File.Move(..., overwrite:
     true)` the parquet and manifest into place. Clean up temps in `finally`.
3. `IsValid`:
   - After the cheap checks, try the lock; if held, return `false` so callers recompute instead of
     reading a torn cache. Read the manifest while holding the lock.
4. `Load`:
   - Acquire the lock before reading (throw if unavailable) so a read cannot overlap a write.

## Verification

- Build `NivaraChat` and `Nivara.Tests`.
- Targeted tests: concurrent `Save` produces a loadable valid cache; lock-held `Save`/`IsValid`/
  `Load` complete without deadlock.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits

1. `docs: plan #920 concurrency guard in TODO.md`
2. `feat(NivaraChat): guard RagIndexCache writes with a file lock`
3. `test(NivaraChat): cover concurrent and locked RagIndexCache access`
4. `docs: remove TODO.md — plan executed`

## Blast radius

- `samples/NivaraChat/Helpers/RagIndexCache.cs` — lock, temp-file writes, reader coordination.
- `samples/NivaraChat/Helpers/RagSearch.cs` — unchanged call sites; `BuildAsync` already catches
  save failures.
- `tests/Nivara.Tests/Samples/` — new `RagIndexCacheTests` fixture.
- No public signatures or documentation citations change; `DocsCoveragePins` should not move.
  Re-check after the diff: if a new `docs/*.md` file or citation appears, extend the pins.

## GitHub issues log

- None yet.
