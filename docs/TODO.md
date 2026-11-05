# Plan: unit tests for RagIndexCache (#923)

## Problem
`RagIndexCache` has no coverage beyond the #920 concurrency guard. Regressions in cache
validity, Parquet round-trip of the dimension frame, or manifest (de)serialization would only
surface at runtime.

## Proposed changes

### 1. `samples/NivaraChat/Helpers/RagIndexCache.cs`
- `IsValid`: add a chunk-count check — load the persisted Parquet dimension frame and compare
  `RowCount` to `manifest.ChunkCount` (return `false` on load failure). This closes the
  "chunk-count mismatch → rebuild" invalidation trigger.

### 2. `tests/Nivara.Tests/Samples/RagIndexCacheTests.cs`
- Round-trip (CPU + Silk): metadata + embeddings exact.
- Null embeddings survive as all-null rows and stay excluded from Top-K.
- Invalidation: mtime bump, backend mismatch, bad manifest, chunk-count mismatch, missing files.
- Keep existing concurrency/lock tests.

## Verification
- `dotnet build Nivara.slnx -c Release`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~RagIndexCacheTests"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits
1. docs: plan #923 in TODO.md
2. fix: validate chunk count in RagIndexCache.IsValid
3. test: round-trip + null + invalidation coverage for RagIndexCache
4. docs: remove TODO.md — plan executed

## Blast radius
- `RagIndexCache.IsValid` callers: `RagSearch.RagIndex.BuildAsync` (cache validity gate). Adding a
  Parquet read inside the existing lock; lock-held path returns before the read.
- Tests: `tests/Nivara.Tests/Samples/RagIndexCacheTests.cs`.

## GitHub issues log
- (none yet)
