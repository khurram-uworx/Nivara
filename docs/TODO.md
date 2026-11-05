# Plan: issue #940 — NivaraChat --rag-multiview GeneratedEmbeddings round-trip

## Problem

`--rag-multiview` does not demonstrate a round-trip through
`GeneratedEmbeddings<Embedding<float>>` for any view, unlike `--rag --roundtrip`.
`MultiViewIndexCache` persists/loads the dimension-frame form (`ContentE*` etc.)
but the sample never converts a view back through `ToEmbeddingFrame` /
`ToGeneratedEmbeddings` and checks parity.

## Proposed changes (sample-local, no `src/` change)

1. `samples/NivaraChat/Helpers/MultiViewIndex.cs`
   - Add `using Microsoft.Extensions.AI;` and `using Nivara.AI;`.
   - Add `public bool UseFrame => true;` (mirrors `RagIndex`).
   - Add `public void PrintRoundTrip()` converting the **Content** view:
     - `metadata.ToEmbeddingMatrix("VectorContent", "ContentE", Dimension)`
     - `matrix.ToEmbeddingFrame("VectorContent", "ContentE")`
     - `metadata.ToGeneratedEmbeddings("VectorContent", …)` then `.ToNivaraFrame("VectorContent")`
     - Print frame/matrix/rebuilt counts and mismatch counts (parity).
2. `samples/NivaraChat/Modes/RagMultiViewMode.cs`
   - Call `index.PrintRoundTrip()` when `ctx.ShowRoundTrip`.
3. `samples/NivaraChat/Helpers/MultiViewIndexCache.cs`
   - Add `using Microsoft.Extensions.AI;`.
   - Add cache helpers mirroring `RagIndexCache.ToEmbeddings`:
     - `ToEmbeddings(loaded, dimension)` and overloads returning `float[]?[]` (Content view).
     - `ToGeneratedEmbeddings(loaded, dimension, …)` returning `GeneratedEmbeddings<Embedding<float>>`.
4. `samples/NivaraChat/Program.cs`
   - Extend `--roundtrip` help text to mention `--rag-multiview`.
5. Docs (optional): update `docs/EMBEDDINGS.md` / `README.md` citations if a gated snippet is added.

## Blast radius

- Sample helpers only; no core changes.
- `MultiViewIndex` gains a public method and property; `MultiViewIndexCache` gains public static helpers.
- Existing tests in `tests/Nivara.Tests/Samples/MultiViewIndexCacheTests.cs` and `MultiViewScoringTests.cs` remain green.

## Verification steps

- Build sample: `dotnet build samples/NivaraChat/NivaraChat.csproj`
- Run targeted sample tests if human approves.

## Planned commits

1. `docs: plan #940 in TODO.md`
2. `feat(NivaraChat): add Content-view round-trip to MultiViewIndex`
3. `feat(NivaraChat): add cache helpers for Content-view embeddings`
4. `docs: note multi-view round-trip in help text`
5. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none yet)