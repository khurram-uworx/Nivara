# Plan: batch embedding during RAG ingestion (#921)

## Problem

NivaraChat ingests RAG documents by calling `MiniLMEmbedder.Handle.Embed(string)` once per
chunk. On a 287-chunk corpus this measured ~48 s (~167 ms/chunk). The CPU path already exposes
`NivaraEmbeddingGenerator.GenerateAsync(IEnumerable<string>)`, and the Silk runner already
supports a padded batch (`Forward(..., batch, seqLen)`), but neither is used.

Target: measurably lower ingestion time (target >5x on 1k chunks), while null/empty chunks still
produce null embeddings and the cache save/load path is unchanged.

## Proposed changes

### 1. `samples/Nivara.Samples/MiniLMEmbedder.cs`

- Add `Handle.EmbedBatch` as a `Func<IEnumerable<string>, float[][]>` alongside the existing
  `Embed` delegate. Keep a constructor overload so existing construction sites compile.
- Add a static helper `MiniLMEmbedder.EmbedBatch(Handle embedder, IEnumerable<string> texts)`
  that delegates to the handle.
- CPU (`CreateCpu`): split null/whitespace inputs (result `null`), call
  `gen.GenerateAsync(nonEmptyTexts)` once, and scatter the generated vectors back to their
  original indices.
- Silk (`CreateSilk`): encode every non-empty text to the fixed padded `maxLen = 128` form,
  flatten into `batch * 128` id/mask arrays, call `runner.Forward` in chunks capped at the
  runner's `MaxBatch = 8`, then extract and L2-normalize the CLS row per item.
- Keep the per-text `Embed` path and its result unchanged.

### 2. `samples/NivaraChat/Helpers/RagSearch.cs` (`RagIndex.BuildAsync`)

- Replace the per-chunk embedding loop with slices of `ctx.BatchSize`.
- Call `embedder.EmbedBatch(slice)`; on exception or length mismatch, fall back to the existing
  per-chunk `embedder.Embed` loop with the same try/catch-to-null behavior.
- Preserve progress reporting and leave `BuildFrame`, cache validity, save and load untouched.

### 3. `samples/NivaraChat/Modes/ModeContext.cs`

- Add `int BatchSize = 32` to the record so existing call sites keep compiling.

### 4. `samples/NivaraChat/Program.cs`

- Parse `--batch-size <int>` (default 32) and pass it to `ModeContext`; add the option to the
  usage text.

## Code sketches

```csharp
public sealed class Handle : IDisposable
{
    public Func<string, float[]> Embed { get; }
    public Func<IEnumerable<string>, float[][]> EmbedBatch { get; }
    public int Dimension { get; }
}

Func<IEnumerable<string>, float[][]> embedBatch = texts =>
{
    var input = texts?.ToList() ?? [];
    var result = new float[input.Count][];
    var toEmbed = new List<(int Index, string Text)>();
    for (int i = 0; i < input.Count; i++)
    {
        if (string.IsNullOrWhiteSpace(input[i])) { result[i] = null!; continue; }
        toEmbed.Add((i, input[i]));
    }
    if (toEmbed.Count > 0)
    {
        var generated = gen.GenerateAsync(toEmbed.Select(t => t.Text)).GetAwaiter().GetResult();
        for (int k = 0; k < toEmbed.Count; k++)
            result[toEmbed[k].Index] = generated[k].Vector.ToArray();
    }
    return result;
};
```

```csharp
for (int start = 0; start < allChunks.Count; start += ctx.BatchSize)
{
    int count = Math.Min(ctx.BatchSize, allChunks.Count - start);
    var texts = new string[count];
    for (int j = 0; j < count; j++) texts[j] = allChunks[start + j].Content;

    float[][]? batch = null;
    try { batch = embedder.EmbedBatch(texts); } catch { /* per-chunk fallback below */ }

    for (int j = 0; j < count; j++)
    {
        int i = start + j;
        if (batch is { } values && values[j] is { } vector) { embeddings[i] = vector; continue; }
        var content = allChunks[i].Content;
        if (string.IsNullOrWhiteSpace(content)) { embeddings[i] = null; continue; }
        try { embeddings[i] = embedder.Embed(content); } catch { embeddings[i] = null; }
    }
}
```

## Verification

- `dotnet build Nivara.slnx` (Release).
- Targeted tests for the touched area:
  - `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Samples"`
  - `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` if `docs/` changes
- Documentation snippet gate if any gated fenced block is added (none planned).
- Manual RAG run if the MiniLM model files are present (human-assisted); otherwise report that
  the model-dependent path was not exercised locally.

## Planned commits

1. `docs: plan #921 batch embedding during RAG ingestion in TODO.md`
2. `Add batched embedding to MiniLMEmbedder`
3. `Use batched embedding in RAG ingestion`
4. `Add configurable RAG embedding batch size`
5. `review: ...` / test-fix commits as needed
6. `docs: remove TODO.md — plan executed`

## Blast radius

- `samples/Nivara.Samples/MiniLMEmbedder.cs`: shared by NivaraChat RAG/retrieval and any sample
  using MiniLM. The new member is additive; `Embed` semantics unchanged.
- `samples/NivaraChat/Helpers/RagSearch.cs`: RAG ingestion only; retrieval, ACL filtering,
  frame construction and cache contract unchanged.
- `samples/NivaraChat/Modes/ModeContext.cs` and `Program.cs`: additive optional parameter and
  CLI flag; existing positional construction remains valid.
- Tests: existing `RagIndexCacheTests` cover cache behavior; no existing MiniLM tests (model
  files required). No core library (`src/`) change.

## GitHub issues log

- None yet.
