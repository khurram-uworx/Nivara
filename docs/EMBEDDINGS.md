# Embeddings in Nivara

Nivara does not own embedding math. It owns the **table**: typed columns, null
masks, labels, filtering, and the row-scoring kernels that already exist. This
document defines the convention that connects the two, and the glue in
`Nivara.AI` that crosses between Nivara and `Microsoft.Extensions.AI`.

The design decision behind this shape — a flat `[rows, dim]` convention rather
than a new column kind — is recorded in
[docs/plan/AISTACK-ROADMAP.md](plan/AISTACK-ROADMAP.md) (Phase 2) and follows the
positioning principle in [docs/TENSORS.md](TENSORS.md).

## The two forms

An embedding is a vector per row. Nivara represents that in exactly two shapes,
and each has one job:

| Form | Shape | Use it for |
| --- | --- | --- |
| **Vector column** | one `NivaraColumn<float[]>` | storage, interop with `Microsoft.Extensions.AI`, carrying vectors next to metadata |
| **Dimension frame** | `E0..E{n-1}`, all `float` | scoring with `RowCosineSimilarity` / `RowDot` |

The vector column is the durable form. The dimension frame is derived, and it is
what the row kernels consume because
`NivaraFrame.RowCosineSimilarity()` (`src/Nivara/NivaraFrame.cs::RowCosineSimilarity()`)
treats every column as one dimension of a row vector — so a frame's column count
*is* the embedding dimension.

Because of that, `ToEmbeddingMatrix` emits **only** the dimension columns.
Metadata and labels stay on the source frame and are passed to the scoring call
as `labels`. Convert once, score, discard the matrix; keeping both forms live
doubles the embedding bytes in memory.

## Generate, store, score, retrieve

```csharp
using Microsoft.Extensions.AI;
using Nivara.AI;

var generator = new NivaraEmbeddingGenerator<string>(
    text => [text.Length, 1f, 0f],
    embeddingDimension: 3);

NivaraFrame table = await generator.GenerateFrameAsync(
    ["first chunk", "second chunk", "third chunk"],
    embeddingColumnName: "Embedding");

NivaraFrame matrix = table.ToEmbeddingMatrix("Embedding");
NivaraSeries<float> scores = matrix.RowCosineSimilarity(NivaraSeries<float>.Create([1f, 0f, 0f]));

var top = scores.TopKDescending(2);
foreach (var (label, score) in top)
    Console.WriteLine($"{label}: {score:F4}");
```

`GenerateFrameAsync` is a convenience over `GenerateAsync` followed by
`ToNivaraFrame`. Use the two-step form when you also need to attach identity or
metadata columns at ingestion.

## Null semantics

Nivara has explicit null masks and no NaN sentinel, so a missing embedding is a
null row — never a zero vector, which would silently score 0 against everything.

| Situation | Result |
| --- | --- |
| null `float[]` row → dimension frame | every `E*` cell null in that row |
| null row scored | **null score** for that row only |
| null element in the query | **every** score null |
| `null` passed as a `float[]` query | every score null |
| fully null dimension row → vector column | null embedding |
| partially null dimension row → vector column | needs a `PartialEmbeddingRepair` |
| null row → `GeneratedEmbeddings` | needs a `NullEmbeddingReplacement` |

`TopKDescending` always places nulls last and excludes them from the result, so a
null embedding simply never reaches retrieval.

## When a null has no representation

Two conversions cannot express a null, because `Microsoft.Extensions.AI` has no
null vector and a `float[]` has no per-element nulls. Both require a policy
rather than a library guess; with nulls present and no policy they throw, naming
the row and the parameter.

A null row crossing into `Microsoft.Extensions.AI`:

```csharp
using Microsoft.Extensions.AI;
using Nivara.AI;

NivaraFrame table = NivaraFrame.Create(
    ("Embedding", (IColumn)NivaraColumn<float[]>.CreateForReferenceType(
        new float[][] { [1f, 2f], null!, [3f, 4f] })));

var generated = table.ToGeneratedEmbeddings(
    "Embedding",
    nullReplacement: (row, dimension) =>
    {
        Console.WriteLine($"row {row} had no embedding; substituting zeros of length {dimension}");
        return new float[dimension];
    });
```

A partially null row coming back out of a dimension frame:

```csharp
using Nivara.AI;

var matrix = NivaraFrame.Create(
    ("E0", (IColumn)NivaraColumn<float>.Create([1f])),
    ("E1", (IColumn)NivaraColumnFactory.CreateFromNullable(new float?[] { null })));

NivaraFrame rebuilt = matrix.ToEmbeddingFrame(
    "Embedding",
    repair: (row, values, nullDimensions) =>
    {
        var filled = (float[])values.Clone();
        foreach (var dimension in nullDimensions)
            filled[dimension] = 0f;

        return filled;
    });
```

Returning `null` from a repair collapses that row to a null embedding instead.
Row *selection* is a frame decision, not a policy one: filter with
`FilterByMask` **before** converting, so positional alignment is never lost.

## Labels, filtering, and retrieval

Pass a label column to keep row identity through scoring. `TopKDescending`
stringifies labels, so a positional result comes back as `"0"`, `"1"`, … — parse
it back to a row index, or use a natural key such as a chunk id.

```csharp
using Nivara.AI;

var embeddings = NivaraFrame.Create(
    ("Embedding", (IColumn)NivaraColumn<float[]>.CreateForReferenceType(
        new float[][] { [1f, 0f], [0f, 1f], [1f, 1f] })),
    ("ChunkId", (IColumn)NivaraColumn<object>.CreateForReferenceType(["c0", "c1", "c2"])),
    ("Group", (IColumn)NivaraColumn<string>.Create(new[] { "eng", "eng", "hr" })));

var visible = embeddings.FilterByMask(NivaraColumn<bool>.Create(new[] { true, true, false }));

NivaraSeries<float> scores = visible.RowCosineSimilarity(
    "Embedding",
    new[] { 1f, 0f },
    labels: visible.GetColumn<object>("ChunkId"));

var top = scores.TopKDescending(1);
Console.WriteLine($"{top[0].Label}: {top[0].Score:F4}");
```

## What this convention is not

- Not a column kind. There is no `NivaraColumn<Embedding<float>>`; the convention
  is a `float[]` column plus derived dimension columns.
- Not a similarity engine. Scoring is
  `NivaraFrame.RowCosineSimilarity()` (`src/Nivara/NivaraFrame.cs::RowCosineSimilarity()`)
  and `NivaraFrame.RowDot()` (`src/Nivara/NivaraFrame.cs::RowDot()`), which call
  `TensorPrimitives` over a materialised row-major buffer.
- Not a vector index. There is no ANN structure; retrieval is an exact scan
  followed by `TopKDescending`.
- Not a persistence format. Embeddings round-trip through the existing CSV,
  JSON, Arrow and Parquet adapters as a column, with their null mask.

The full surface lives in `src/Nivara.Extensions/AI/NivaraEmbeddingExtensions.cs`.
The MEAI generator it builds on is
`src/Nivara.Extensions/AI/NivaraEmbeddingGenerator.cs`.
