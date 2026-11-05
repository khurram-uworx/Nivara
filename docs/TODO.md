# Plan: #927 RagIndexCache.IsValid chunk-count and dimension enforcement

Branch: khurram/927 (off khurram/926)

## Problem

`RagIndexCache.IsValid` does not enforce the chunk count or dimension recorded in the
manifest, and its existing chunk-count comparison reads the entire Parquet file. A cache
whose row count or `E*` column count drifted from the manifest would still be treated
as valid.

## Proposed changes

- `samples/NivaraChat/Helpers/RagIndexCache.cs`
  - Replace the full-load chunk-count block with a footer-only Parquet metadata read:
    - `Parquet.ParquetReader.CreateAsync(indexPath, ...)` + `reader.Metadata.NumRows`.
    - Compare against `m.ChunkCount`.
    - Count `DataFields` named `E` + integer and compare against `m.Dimension`.
  - Add `using Parquet;`.
  - Keep cheap: no column reads.

## Blast radius

- Files: only `RagIndexCache.cs`.
- Callers: `RagSearch.cs` (IsValid gating) and `RagIndexCacheTests`.
- Tests: `tests/Nivara.Tests/Samples/RagIndexCacheTests.cs`.
- No change to public Nivara API; sample-only.

## Verification

- Build `Nivara.Tests` Release.
- Targeted: `dotnet test ... --filter FullyQualifiedName~RagIndexCacheTests`.
- Whitespace: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits

1. `docs: plan #927 in TODO.md`
2. `fix(NivaraChat): enforce chunk count and dimension in RagIndexCache.IsValid`
3. `docs: remove TODO.md — plan executed` (after review)

## GitHub issues log

(none created during this plan)
