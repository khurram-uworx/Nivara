# Plan — issue #926: feat(NivaraChat): --rag-agent cache parity

## Problem

`--rag-agent` historically re-chunked and re-embedded the whole corpus on every launch and applied the ACL
filter before embedding (pre-cache), so it could not reuse a `--rag` cache and could not share one cache
with `--rag`. The shared pipeline (`RagIndex.BuildAsync`) already resolves the cache, validates it, loads
chunks/embeddings, applies ACL at query time, and saves on miss, but the cache hit path still rebuilds
the convention frame from embeddings via `BuildFrame(allChunks, embeddings)`. That keeps a second in-memory
copy and is the remaining bit of parity work.

## Proposed changes

- `samples/NivaraChat/Helpers/RagSearch.cs::RagIndex.BuildAsync()`: on cache hit, reuse columns from the
  loaded parquet frame directly instead of `RagIndexCache.ToEmbeddings` + `BuildFrame`:
  - `allChunks = RagIndexCache.ToChunks(loaded, dim)`.
  - columns `ChunkId`, `Source`, `Title`, `SectionPath`, `AclGroup`, `Content` taken straight from
    `loaded` via `GetColumn<string>` (guarded by `HasColumn`).
  - `Embedding` produced once by `loaded.ToEmbeddingFrame("Embedding", dimension: dim)` and its column
    reused.
  - `frame = NivaraFrame.Create(cols.ToArray())`.
- No change to the miss path, `RagIndexCache.Save`, ACL handling, `ChunkId` typing, `HasVisibleChunks`,
  or `RagAgentMode`'s TextSearchProvider flow.
- `--rag` and `--rag-agent` both already route through `RagIndex.BuildAsync`, so both gain the benefit
  and share one cache file.

## Verification

- `dotnet build Nivara.slnx`
- `dotnet build samples/NivaraChat/NivaraChat.csproj`
- targeted: `RagSearchTests`, `RagIndexCacheTests`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits

1. `docs: plan #926 in TODO.md`
2. `feat(NivaraChat): reuse cached frame columns on RAG cache hit`
3. (G2) `docs: remove TODO.md — plan executed`

## Blast radius

- Only `RagIndex.BuildAsync` cache-hit path. Downstream: `RagMode`, `RagAgentMode`, `RetrieveAsync`,
  `RagSearch.Rank`, `PrintRoundTrip`. No public API change.

## GitHub issues log

(none yet)