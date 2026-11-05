# Plan: issue #925 — configurable `--chunk-size` for NivaraChat RAG

## Problem
Chunk size is hardcoded to ~500 chars in `RagChunker`. Different corpora/docs benefit
from different sizes, and tuning is impossible without editing code.

## Proposed changes
- Add `--chunk-size <int>` (default 500) to `Program.cs` and `ModeContext`.
- Thread through to `RagChunker.ChunkMarkdown` (add `maxChunkSize` param).
- Document in README.

## Files / blast radius
- `samples/NivaraChat/Modes/ModeContext.cs` — new record field `ChunkSize = 500`.
- `samples/NivaraChat/Helpers/DocumentChunk.cs` — `RagChunker.ChunkMarkdown(..., int maxChunkSize = 500)`; replace hardcoded 500 in `Math.Min`.
- `samples/NivaraChat/Helpers/RagSearch.cs` — pass `ctx.ChunkSize`; cache path/validity includes chunk size.
- `samples/NivaraChat/Program.cs` — parse `--chunk-size`; pass to `ModeContext`; usage line.
- `samples/NivaraChat/Helpers/RagIndexCache.cs` — hash/manifest/validity include chunk size.
- `samples/NivaraChat/README.md` — options table row + pipeline note.

## Verification
- Build `samples/NivaraChat`.
- Run `--rag --chunk-size 200` and `--chunk-size 1000`; confirm chunk counts differ.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits
1. docs: plan #925 in TODO.md
2. feat: add ChunkSize to ModeContext and Program.cs
3. feat: thread maxChunkSize through RagChunker
4. feat: pass ctx.ChunkSize in RagSearch
5. feat: cache validity includes chunk size
6. docs: document --chunk-size in README
7. docs: remove TODO.md

## GitHub issues log
- (none yet)
