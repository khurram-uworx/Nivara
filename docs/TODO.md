# Plan: issue #1078 — chore: small hygiene follow-ups from the #862 chunk layer

## Problem

Follow-up hygiene items from #862 (PR #1073). Each small and independent; one PR carries all.

1. `AGENTS.md` implementation map does not mention the new chunk layer, so agents grep instead of finding it.
2. `src/Nivara/Storage/Chunk.cs` exposes `Chunk<T>.IsNull(int)`, specified in the #862 design, but nothing calls it — `ChunkedColumn<T>` reads `storage.NullMask` directly. Unused internal surface.
3. `ChunkedColumn<T>.Slice` allocates a `List<ColumnStorage<T>>` plus a new `ChunkedColumn<T>` (three arrays) per call. Element data is not copied (contract). Pooling deferred unless baseline shows slice churn.
4. `ChunkedColumn<T>.ChunkStorages` returns a span over the internal array; after `Dispose` it is cleared (pinned by `Dispose_DoesNotDisposeChunks`). Decide whether an internal consumer can see a cleared span at all, or whether it should throw `ObjectDisposedException` like every other member.

## Proposed changes

### Item 1 — AGENTS.md

Add one line to "Where to look (implementation map)" after the Storage bullet. No behavior change.

### Item 2 — route cross-boundary null lookup through `Chunk<T>.IsNull`

Keep the existing `IsNull` member (specified in #862 design) and make it used.

- `src/Nivara/Storage/ChunkedColumn.cs`: add `internal bool IsNull(int index)` that resolves the owning chunk via `FindChunk` and delegates to `Chunk<T>.IsNull(index - starts[chunkIndex])`, with disposed/bounds guards.
- `src/Nivara/NivaraColumn.cs`: in `IsNull(int index)`, when `storage is ChunkedColumn<T>`, delegate to `chunked.IsNull(index)` before the `storage.NullMask` path. This avoids whole-mask materialisation for a single probe.
- `tests/Nivara.Tests/Storage/ChunkedColumnTests.cs`: add `IsNull_AcrossChunkBoundaries_ReturnsNullStatus` covering flat + masked chunks.

### Item 3 — `Slice` pooling: deferred

No code change. Rationale: the cost baseline shows chunked arithmetic pays by design through one materialised contiguous copy per operand; slice already does not copy element data. Pooling would complicate ownership/lifetime of per-slice storage for unmeasured benefit. Revisit if the chunk-wise compute baseline (tracked via the `--chunked-baseline` probe) shows slice allocation dominating hot paths. A tracked follow-up issue will be created when deferring.

### Item 4 — `ChunkStorages` after `Dispose`: fail fast

Make `ChunkStorages` throw `ObjectDisposedException` like every other member (fail-fast over silently visible cleared span). Keep `Array.Clear` on both arrays in `Dispose` for GC/weak-reference safety. Update `Dispose_DoesNotDisposeChunks` to assert the throw instead of the cleared span; keep the assertions that the underlying chunks remain usable.

## Verification steps

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Targeted: `ChunkedColumnTests`, `ChunkedColumnOperationTests`, and `NivaraColumn` IsNull coverage (`dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~ChunkedColumn"`).
- Confirm `Chunk<T>.IsNull` now has a caller (`grep` / code-memory).
- Docs pin sanity: no docs/*.md change, so DocsCoveragePins unchanged.

## Planned commits

1. `docs: plan #1078 in TODO.md`
2. `docs: add chunk layer to AGENTS.md implementation map` (item 1)
3. `feat(storage): route chunked null probes through Chunk<T>.IsNull` (item 2)
4. `feat(storage): make ChunkStorages fail fast after dispose` (item 4)
5. (optional) `docs/tests: cross-boundary IsNull and disposed-state assertions` (fold into commits 3/4)
6. `docs: remove TODO.md - plan executed`

## GitHub issues log

- [x] #1093 — chunked-vs-flat `Slice` pooling follow-up (backlog revisit after baseline measurements) (created while deferring item 3)

## Blast radius

- `src/Nivara/Storage/Chunk.cs::Chunk<T>.IsNull` — now has a caller.
- `src/Nivara/Storage/ChunkedColumn.cs` — adds `IsNull`; `ChunkStorages` getter gains a guard; `Dispose` unchanged.
- `src/Nivara/NivaraColumn.cs::IsNull` — one added pattern-match branch, no semantic change; direct NullMask path retained for flat storage.
- `tests/Nivara.Tests/Storage/ChunkedColumnTests.cs` — new IsNull cross-boundary test; disposed-state assertion changed.
- `AGENTS.md` — documentation only.

Downstream callers of `ChunkStorages` (`ArrowInterop`, `ColumnFilterHelper`, `ChunkedColumn.Create`/`FromChunkStorages`) all run pre-dispose on live instances, so the new guard does not affect them.
