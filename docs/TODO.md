# Plan — Multi-view retrieval, late fusion and disagreement analysis (#914)

## Problem

The offline RAG sample (#912) embeds each chunk once. A single cosine score cannot say
*which representation of the chunk matched* — a prose paragraph may match a query on its
section heading, a code block on its body, and the document as a whole on its title. The
issue asks for a sample-only demonstration that embeds four **views** per chunk (Content,
Title, Section, Code), scores each independently, fuses late, names the winning view
(BestSpace), and reports where the views disagree.

Everything is sample-local. No `src/` changes.

## Decisions (cleared with the human before execution)

| Question | Decision |
| --- | --- |
| Views vs models | Four views, one MiniLM model, one 384-d space |
| Title view | Document title (H1, else filename stem) — constant per file |
| Section view | `H1 > H2 > H3` breadcrumb |
| Code view | One chunk per fenced block; prose chunks carry a null Code view |
| Chunking | Fence-aware; ```` ``` ```` and `~~~` lines are never headings |
| Fusion weights | 0.5 Content / 0.2 Section / 0.2 Title / 0.1 Code, renormalized per row over non-null views; `--weights` override |
| Rank bands | Gap from the query's top fused score: Near <= 0.02, Distinct <= 0.08, Weak > 0.08 |
| Disagreement bands | Absolute cosine difference: Agree <= 0.05, Mild <= 0.15, Strong > 0.15 |
| Cache | In scope: separate cache type, own schema, `ChunkerVersion` in the hash |
| Output | Console table; `--report <path>` markdown under gitignored `samples/data/nivarachat/` |
| Docs | `samples/NivaraChat/README.md`, prose section in `docs/EMBEDDINGS.md`, CHANGELOG |
| Location | Sample-local; no promotion to `Nivara.AI` |
| Mode name | `--rag-multiview` with `--explore`, `--report`, `--weights` |

## Factual corrections to the issue (verified in code)

1. `NivaraFrame.WithColumn` calls `validateColumnTypeForAddition`, whose allow-list
   excludes `float[]`. The four vector columns therefore land in one `NivaraFrame.Create`;
   only derived scalar/string columns may use `WithColumn`.
2. The four views are four representations in one 384-d space, not four embedding spaces.
   Divergence is a difference between cosine scores.
3. `RagIndexCache.IsValid` counts columns literally named `E<int>` and shares
   `SchemaVersion = 1` with #912, so the multi-view cache must be a separate type.
4. The existing `RagChunker` treats any line starting with `#` as a heading, including
   inside fences, and overwrites Title/Section with the most recent heading. It must not be
   modified: cache validity hashes document content and chunk size, not chunker code.

## Proposed changes

### 1. Shared cache file operations (behavior-neutral refactor)

`Helpers/IndexCacheFile.cs` (new, internal): `TryAcquireLock`, `TempPath`, `TryDelete`,
manifest read/write, atomic temp-move. `RagIndexCache` delegates to it; its schema, hash,
`IsValid`, `Load`, `Save` and `ToChunks` semantics are unchanged. `RagIndexCacheTests` must
stay green.

### 2. `Helpers/MultiViewChunk.cs` (new)

```csharp
public sealed record MultiViewChunk(
    string ChunkId,      // "{relativeSourcePath}#{index}" — unique across duplicate file names
    string Source,       // file name
    string SourcePath,   // repo-relative path
    string Title,        // document title
    string Section,      // "H1 > H2 > H3"
    string AclGroup,     // first folder under docs/, else "public"
    string Content,      // never empty
    string? Code);       // fence body, or null for prose chunks

public static class MultiViewChunker
{
    public static List<MultiViewChunk> ChunkMarkdown(
        string content, string sourcePath, string repoRoot, int maxChunkSize = 500);
}
```

Behavior: fence-aware scan; H1 sets the document title; H2/H3 build the breadcrumb; H4+
does not extend the breadcrumb; prose splits at paragraph boundaries and then at
`maxChunkSize`; each fence body becomes its own chunk above a minimum length; empty input
yields no chunks.

### 3. `Helpers/MultiViewScoring.cs` (new)

```csharp
public enum ViewSpace { Content, Title, Section, Code }   // declaration order = tie order

public sealed record FusionWeights(float Content, float Title, float Section, float Code);
public sealed record ViewScores(float? Content, float? Title, float? Section, float? Code);

public sealed record MultiViewHit(
    int Row, MultiViewChunk Chunk, ViewScores Scores, float? Fused,
    ViewSpace? BestSpace, float? DivergenceContentTitle, float? DivergenceContentSection,
    float? DivergenceContentCode, float? MaxDivergence, float? CoherenceContentTitle,
    string? RankBand);

public static class MultiViewScoring
{
    public static NivaraFrame BuildFrame(
        IReadOnlyList<MultiViewChunk> chunks,
        IReadOnlyList<float[]?> contentVectors,
        IReadOnlyList<float[]?> titleVectors,
        IReadOnlyList<float[]?> sectionVectors,
        IReadOnlyList<float[]?> codeVectors);

    public static IReadOnlyList<MultiViewHit> Rank(
        NivaraFrame vectorFrame, NivaraFrame contentMatrix, NivaraFrame titleMatrix,
        NivaraFrame sectionMatrix, NivaraFrame codeMatrix, IReadOnlyList<MultiViewChunk> chunks,
        float[]? query, FusionWeights weights, int topK);
}
```

`BuildFrame` uses one `NivaraFrame.Create` with `VectorContent`, `VectorTitle`,
`VectorSection`, `VectorCode` plus metadata. `Rank` scores each matrix with
`RowCosineSimilarity(NivaraSeries<float>.Create(query))`, fuses with per-row renormalized
weights, derives BestSpace/divergence/coherence/band, then uses `TopKDescending` and maps
positional labels back to the row list (the `RagSearch.Rank` pattern, so duplicate ids
cannot misreport a row). A row with all four scores null fuses null and never enters Top-K.
ACL filtering and matrix construction happen once at index build, not per query.

`CoherenceContentTitle` is the cosine between the Content and Title vectors, computed
sample-locally from the two `float[]` columns (no `Abs` kernel exists; divergence is
computed in the same span loop).

### 4. `Helpers/MultiViewExplorer.cs` (new)

```csharp
public sealed record AgreementBucket(string Name, float UpperBound, int Count, IReadOnlyList<MultiViewHit> Examples);
public static class MultiViewExplorer
{
    public static IReadOnlyList<AgreementBucket> Bucketize(IReadOnlyList<MultiViewHit> hits);
    public static string Explain(MultiViewHit hit);
}
```

Buckets: Agree (max divergence <= 0.05), Mild (<= 0.15), Strong (> 0.15). A hit with fewer
than two non-null views has no divergence and lands in its own "InsufficientViews" bucket.

### 5. `Helpers/MultiViewIndexCache.cs` (new)

- One Parquet: `ContentE0..ContentE{d-1}`, `TitleE*`, `SectionE*`, `CodeE*` plus `ChunkId`,
  `Source`, `SourcePath`, `Title`, `Section`, `AclGroup`, `Content`, `Code`.
- Own `SchemaVersion = 1` and `ChunkerVersion` in both hash and manifest.
- Path `samples/data/nivarachat/rag_multiview_cache/rag_multiview_{hash}.parquet`; `--index`
  overrides; `--rebuild` / `--no-cache` behave as they do today.
- Validity: backend, docsDir, schema, chunkSize, chunkerVersion, docs max mtime, content
  hash, row count, and per-view dimension counted from prefixed footer fields.
- Load rebuilds each view via `ToEmbeddingFrame("VectorX", columnPrefix: "XE",
  dimension: d)` on a prefix-only frame, then merges the four with `NivaraFrame.Create`.

### 6. `Helpers/MultiViewIndex.cs` (new)

Build: resolve docs, chunk, deduplicate view texts (Title and Section repeat heavily),
embed the unique texts in batches, expand back to per-chunk vectors, build the vector frame
and four matrix frames, then apply the ACL mask to all five frames plus the chunk list once.
Missing model/corpus prints the reason and returns null, matching `RagIndex.BuildAsync`.

### 7. `Modes/RagMultiViewMode.cs` (new) + `Program.cs` + `ModeContext`

`--rag-multiview`: header, corpus stats (files, chunks, per-view non-null counts), then a
REPL or `--text` single shot. Prints the fused table (rank, fused, band, BestSpace,
C/T/S/Code, max divergence, section, chunk) and, with `--explore`, the agreement buckets.
`--report <path>` writes the same as markdown. New `ModeContext` fields:
`bool Explore = false`, `string? ReportPath = null`, `string? Weights = null`.

### 8. Tests (new)

- `MultiViewChunkerTests` — breadcrumbs, fences not headings, code chunks, prose split,
  ACL folder, no-H1 fallback, CRLF, H4+ handling, empty input.
- `MultiViewScoringTests` — four `float[]` columns in one frame; per-view scores against
  hand-computed cosines; four-column `FilterByMask`/`Slice` row alignment; null Code view
  renormalizes over three; all-null row is null and excluded; BestSpace tie order;
  divergence/coherence null propagation; band edges; `Explain` text.
- `MultiViewIndexCacheTests` — exact round-trip (including all-null Code rows), manifest
  invalidation per field, `--index` override, lock contention, four-view column count.

### 9. Documentation

- `samples/NivaraChat/README.md`: CLI rows and a mode section.
- `docs/EMBEDDINGS.md`: prose-only multi-view section (no fenced `csharp`), three
  symbol-form citations; pin arithmetic updated in the same commit from an actual run.
- `CHANGELOG.md`: one entry under Unreleased/Added.

## Blast radius

- New files only under `samples/NivaraChat/Helpers`, `samples/NivaraChat/Modes` and
  `tests/Nivara.Tests/Samples`; the only edits to existing code are `Program.cs`,
  `ModeContext.cs`, `RagIndexCache.cs`, `samples/NivaraChat/README.md`,
  `docs/EMBEDDINGS.md`, `CHANGELOG.md`, and `DocsCoveragePins.cs` (counts only).
- `RagIndexCache` is touched by the behavior-neutral extraction and by nothing else; #912's
  cache format and validity are unchanged.
- No `src/` changes. No public API changes.
- Tests that can catch a regression: `RagSearchTests`, `RagIndexCacheTests`,
  `MiniLMEmbedderBatchTests`, `DocsCoverageTests`, the Docs namespace suite, and the new
  `MultiView*` fixtures.

## Verification

- `dotnet build samples/NivaraChat/NivaraChat.csproj -c Release`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Samples.MultiView"` (ask first)
- `dotnet test ... --filter "FullyQualifiedName~Nivara.Tests.Samples.RagIndexCacheTests"` after the extraction (ask first)
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Docs namespace suite when `docs/` changes (ask first)
- CPU demo: `dotnet run -c Release --project samples/NivaraChat -- --rag-multiview --docs-dir docs/adr --top-k 5 --explore`
- Silk run and full suite: human-run.

## Planned commits

1. `refactor(samples): share index-cache file operations`
2. `feat(samples): add a fence-aware multi-view chunker`
3. `feat(samples): add multi-view scoring, late fusion and disagreement analysis`
4. `feat(samples): add the multi-view index cache`
5. `feat(NivaraChat): add the --rag-multiview mode` (+ CLI, README, CHANGELOG)
6. `docs: document the multi-view retrieval convention` (+ pin arithmetic)

## Risks

- 1536-column Parquet save/load is the least exercised path; cache tests use small
  dimensions plus one real 384-d shape.
- MiniLM truncates at 128 tokens on both backends; long content/code views lose their tail.
  Documented, not fixed.
- A full-corpus run embeds roughly 2000 unique view texts; `--docs-dir docs/adr` is the
  quick path.
- `--silk` may hard-fail today (#924); Silk parity is a human-run check.

## GitHub issues log

- [ ] none yet — create at discovery time during execution.
