# Plan — #918: `--rag-agent` parity with `--rag`

## Problem

After #912/#917 migrated `--rag` to the #867 embedding convention, `--rag-agent` was left behind:

- `RagAgentMode.Run` hard-requires `--ollama` and early-returns without it.
- `--roundtrip` is parsed and carried on `ModeContext.ShowRoundTrip` but never read by `--rag-agent`.
- `SearchAsync` computes `TopKDescending` scores and then discards them (`TextSearchResult` has no
  `Score`), so agent mode hides what was retrieved.
- The chunk/embed/cache/frame block in `RagAgentMode.cs` (~120 lines) is a near-verbatim copy of
  `RagMode.cs`, which is how the parity gap opened.
- Both modes resolve a score label back with `workChunks.FirstOrDefault(c => c.ChunkId == label)`.
  `ChunkId` is `"{file name}-{idx}"`, and the corpus has two `README.md` files
  (`docs/blog/README.md`, `samples/NivaraChat/README.md`), so the reported chunk can differ from the
  scored row.

Acceptance criteria from the issue:

- `--rag-agent` runs without `--ollama`, listing top-K chunks.
- `--roundtrip` works for `--rag-agent`.
- Retrieved chunks printed in both modes.
- Agent workflow path unchanged when `--ollama` is present.

## Proposed changes

### 1. New shared core — `samples/NivaraChat/Helpers/RagSearch.cs`

- `RetrievedChunk(string ChunkId, string Source, string Title, string Content, double Score)`.
- `RagSearch.Rank(NivaraFrame frame, IReadOnlyList<RagChunk> chunks, float[]? query,
  string? aclGroup, int topK)`: applies the ACL mask, scores with `RowCosineSimilarity`, takes
  `TopKDescending`, and maps **positional labels** back to the row-aligned visible chunk list. This
  removes the `ChunkId`-collision mismatch without changing the id scheme or cache format.
- `RagIndex`: owns `MiniLMEmbedder.Handle`, chunks, frame/collection, `UseFrame`, `Dimension`.
  - `BuildAsync(ModeContext)` — the chunk/embed/cache block currently duplicated in both modes.
  - `RetrieveAsync(query)` → `(results, elapsedMs)`; frame path calls `RagSearch.Rank`, inmemory path
    calls `collection.SearchAsync` and maps `Score ?? 0`.
  - `PrintRetrieved` / `PrintRoundTrip` — the exact report text `--rag` already prints.

### 2. `RagMode`

Replace the duplicated indexing block and inline retrieval with `RagIndex`. Output must stay
identical to the pre-change baseline (timings excluded).

### 3. `RagAgentMode`

- Remove the `--ollama` early return; build the index unconditionally.
- `--roundtrip` prints once after index build (the provider may search more than once per turn).
- With `--ollama`: keep `OllamaApiClient` → `TextSearchProvider` → `ChatClientAgent` → workflow.
  The provider delegate calls `RetrieveAsync`, prints the report, then maps to `TextSearchResult`.
- Without `--ollama`: single-shot or REPL that calls `RetrieveAsync` and prints; no agent, no LLM.

### 4. CLI help and README

- `Program.cs`: `--rag-agent` Ollama optional; `--roundtrip` covers both RAG modes; `--rag` LLM optional.
- `samples/NivaraChat/README.md`: examples, CLI table, `--rag` section correction, rewritten
  `--rag-agent` section for the two paths. Preserve the existing `docs/EMBEDDINGS.md` links so
  `ExpectedDocumentReferenceCount = 240` does not move.

### 5. Tests — `tests/Nivara.Tests/Samples/RagSearchTests.cs`

Hand-built `NivaraFrame` (no MiniLM model files): top-K ordering, duplicate-`ChunkId` row alignment,
ACL filtering, empty ACL set, null query, null embedding row, top-K > visible count.

## Grounding (G1)

- Microsoft Learn: Microsoft.Extensions.AI `TextSearchProvider` / `TextSearchResult` surface and the
  Agent Framework context-provider pattern (confirm `TextSearchResult` has no score and the delegate
  shape is `Func<string, CancellationToken, Task<IEnumerable<TextSearchResult>>>`).
- code-memory SQL: `RagMode`, `RagAgentMode`, `RagIndexCache`, `RagChunker`, `RowCosineSimilarity`,
  `TopKDescending`, `ToEmbeddingMatrix`/`ToEmbeddingFrame`/`ToGeneratedEmbeddings`, and callers of
  `RagAgentMode.Run` (only `Program.cs`).
- Confirmed: `TopKDescending` stringifies positional labels; `ToEmbedding*` handles fully-null rows
  (null → null embedding, no throw); `samples/data/minilm` is gitignored, so CI cannot exercise the
  post-model-load path.

## Blast radius

- `samples/NivaraChat/Modes/RagMode.cs`, `RagAgentMode.cs`, `Program.cs`,
  `samples/NivaraChat/Helpers/RagSearch.cs` (new), `samples/NivaraChat/README.md`.
- `tests/Nivara.Tests/Samples/RagSearchTests.cs` (new). `Nivara.Tests` already references
  `NivaraChat.csproj`, so no project changes.
- No `src/Nivara*` changes. No changes to `RagChunker`, `RagIndexCache`, `ModeContext`, or the cache
  format/manifest.
- Docs gates: README is not in the snippet/citation scope but is in `DocReferenceTests` scope for
  `docs/…md` links; keep that count unchanged.

## Residual concerns (documented, out of scope)

- `--store inmemory` still uses `ChunkId` as the store key, so duplicate `README.md` ids overwrite
  each other; fully fixing this needs path-qualified chunk ids (display + cache keys change).
- `--store inmemory` ignores `--silk` (its generator is CPU MiniLM) and `--acl-group` (no `AclGroup`
  on `DocumentChunk`).
- `--store inmemory` still computes embeddings it never uses.

## Verification

1. Baseline capture of `--rag` before the refactor; diff after (timings excluded).
2. `dotnet build Nivara.slnx`.
3. `dotnet run -c Release --project samples/NivaraChat -- --rag-agent --text "How does embedding search work?"`
   → retrieval report, no “requires --ollama”.
4. `--rag-agent --roundtrip`, `--acl-group adr --top-k 5`, `--store inmemory`.
5. `--rag-agent --ollama --text "…"` → report plus unchanged agent response.
6. `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~RagSearchTests"`.
7. `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
   if README links changed.
8. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
9. Review branch changes; revise `DocsCoveragePins` only if the arithmetic moved.

## Planned commits

1. `docs: plan #918 --rag-agent parity in TODO.md`
2. `feat(NivaraChat): extract shared RAG index and retrieval core`
3. `feat(NivaraChat): --rag-agent parity with optional Ollama and retrieval report`
4. `test(NivaraChat): cover RAG ranking, ACL and duplicate chunk ids`
5. `docs(NivaraChat): document optional Ollama and retrieved-chunk reporting`
6. `docs: remove TODO.md — plan executed`

## GitHub issues log

Created as discovered during execution:

- (none yet)
