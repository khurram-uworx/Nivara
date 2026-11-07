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

An empty frame has no row to infer a dimension from, so `ToEmbeddingMatrix`
needs `dimension` passed explicitly there. The `RowCosineSimilarity` and
`RowDot` wrappers take the dimension from the query instead, so scoring an
empty (for example, fully filtered) frame needs nothing extra and yields an
empty series.

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
| a column with no non-null embedding, scored | **every** score null; the query supplies the dimension |
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

## Worked contrast: the primitive path and the convention path

NivaraChat carries both shapes deliberately, and reading them side by side is the
point of the two modes.

| | `--embed` (`EmbeddingMode.cs::Run()`) | `--rag` (`RagMode.cs::Run(ModeContext)`) |
| --- | --- | --- |
| Row identity | array index (`0`, `1`, …) | `ChunkId` label column, returned by `TopKDescending` |
| Storage | two parallel arrays (`string[]` documents + `GeneratedEmbeddings<Embedding<float>>`) | one `NivaraFrame`: metadata columns + `Embedding` |
| Scoring | per-row `TensorPrimitives.CosineSimilarity` loop | `RowCosineSimilarity("Embedding", query, labels: …)` |
| Ranking | `OrderByDescending(…).Take(4)` | `TopKDescending(ctx.TopK)` (nulls last, excluded) |
| Missing embedding | not representable — a null would throw while reading the vector | null row → all-null vector → null score, excluded |
| Row selection / ACL | none; would need parallel rebuilds | `FilterByMask` before scoring (`--acl-group`) |
| Round-trip | none | `--roundtrip`: `ToEmbeddingMatrix` → `ToEmbeddingFrame` → `ToGeneratedEmbeddings` |
| Persistence | re-embeds on every start | `RagIndexCache.cs::Save()` writes the dimension frame to Parquet; `RagIndexCache.cs::ToEmbeddings()` rebuilds the column |
| Backend | MiniLM CPU only | CPU or Silk (`--silk`), same retrieval |
| Top-K | fixed at 4 | `--top-k` |

The primitive loop is this, and it is the whole point of the mode:

<!-- gate
locals: float[] query = [1f, 0f, 0f]; Microsoft.Extensions.AI.GeneratedEmbeddings<Microsoft.Extensions.AI.Embedding<float>> vectors = new();
-->
```csharp
using Microsoft.Extensions.AI;

var scores = new float[vectors.Count];
for (var i = 0; i < scores.Length; i++)
    scores[i] = TensorPrimitives.CosineSimilarity(query, vectors[i].Vector.Span);

var ranked = scores
    .Select((score, index) => (score, index))
    .OrderByDescending(entry => entry.score)
    .Take(4)
    .ToList();
```

`--embed` is deliberately not migrated to the convention: keeping it as the
before-state is what makes the cost visible, and `--rag` is the same problem
solved the convention way. The two files are meant to be read side by side.
Issue #919 originally proposed migrating `--embed`; the decision was to keep it
as the contrast and let `--rag` carry the convention.

### What the convention does not remove

- **Label → row is still a lookup.** `TopKDescending` returns the stringified label;
  `RagMode` maps it back through its chunk list, because the frame has no
  lookup-by-key. Alignment is guaranteed for the columns the frame holds, not for
  caller-side bookkeeping.
- **The wrapper converts on every call.** `RowCosineSimilarity("Embedding", …)` builds
  `ToEmbeddingMatrix` each time. For many queries against one corpus, convert once
  and call the core `NivaraFrame.RowCosineSimilarity` kernel (as the first example
  in this document does).
- **No ANN.** Retrieval is an exact row scan followed by `TopKDescending`.

The behaviour above is pinned by
`tests/Nivara.Tests/AI/NivaraEmbeddingExtensionsTests.cs::EndToEnd_GenerateFilterScoreRetrieve()`.

## Multiple views of the same corpus

The convention above stores one vector per row. That is enough when the text that
gets embedded is the text that matters, but a chunk's *title*, its *section
heading*, and the *code* inside it each carry a different signal, and a single
vector averages them into one number. Multi-view retrieval (#914) keeps them
separate: every chunk is embedded several times in the same 384-d space, once
per representation, and the representations are scored independently before being
combined.

The four views used by the `--rag-multiview` sample are:

- **Content** — the chunk as written, prose and fenced blocks together.
- **Title** — the document title, constant per file.
- **Section** — the heading breadcrumb, `H1 > H2 > H3`. Headings at H4 and
  deeper bound a chunk but do not extend the breadcrumb.
- **Code** — only the fence bodies inside the chunk; missing when the chunk has
  no fenced blocks.

Chunks are **section-scoped**: a contiguous run of one section's prose and whole
fenced blocks, bounded by a size limit. A fence is never split, and a paragraph
longer than the limit is split at whitespace. That scoping is what keeps Content
and Code from being the same text — one chunk per fence would make their
divergence ~0 for every code chunk, so a hit whose winning view is `Code` would
say nothing.

Each view lands in the same convention frame as a `float[]` column, so the
existing kernel does the work:
`samples/NivaraChat/Helpers/MultiViewScoring.cs::MultiViewScoring.BuildFrame()`
assembles metadata plus the four vector columns plus a Content/Title coherence
column in one `NivaraFrame.Create(...)` call (a `float[]` column cannot be added
with `WithColumn`), and
`samples/NivaraChat/Helpers/MultiViewScoring.cs::MultiViewScoring.Rank()` scores
each view with `NivaraFrame.RowCosineSimilarity()`.

Fusion is **late** and **per row**: the four scores are combined with weights
renormalized over the views that actually have a score, so a chunk without code is
not penalized for its missing view. A missing view is dropped from fusion, from
`BestSpace`, and from divergence. The same rule treats a non-finite score as
missing — a zero-magnitude vector makes `TensorPrimitives.CosineSimilarity`
return `NaN`, which must never poison the fused score. Weights default to
0.5 Content / 0.2 Section / 0.2 Title / 0.1 Code and can be overridden through
`samples/NivaraChat/Helpers/MultiViewScoring.cs::FusionWeights.TryParse()`.

The winning view is named per hit (`BestSpace`, ties broken in declaration order:
Content, Title, Section, Code), and the widest disagreement between Content and
each other view is reported as an absolute cosine difference. Rank bands are
measured from the query's top fused score (Near ≤ 0.02, Distinct ≤ 0.08,
else Weak); agreement bands are the same kind of absolute difference (Agree ≤
0.05, Mild ≤ 0.15, else Strong). Chunks with fewer than two scored views land
in an `InsufficientViews` bucket instead of being counted as agreement — see
`samples/NivaraChat/Helpers/MultiViewExplorer.cs::MultiViewExplorer.Bucketize()`.

Two practical notes. Titles and sections repeat heavily, so view texts are
deduplicated before embedding — a full-corpus run embeds roughly 2,000 unique
texts rather than one per chunk per view, which is what
`samples/NivaraChat/Helpers/MultiViewIndex.cs::MultiViewIndex.BuildAsync()` does.
And MiniLM truncates at 128 tokens on both backends, so a long Content or Code
view loses its tail; the mode reports retrieval signal only and calls no LLM.

Early evaluation (#943, 100-query LLM-as-judge run): fused retrieval beat content-only
with nDCG@10 0.698 vs 0.596 (68 wins / 31 losses / 1 tie, sign-test p=0.0003,
Cohen's d=0.422). The top hits' winning views were Title 557 / Content 348 /
Section 69 / Code 26, and agreement buckets were 134 agree / 354 mild / 512 strong.
Dropping only Title, Section, or Code degraded fused only modestly (0.688-0.693),
while dropping Content fell to 0.601. So multi-view helps modestly on relevance and
clearly adds explainability; the design's fallback semantics (non-finite scores and
missing views excluded from fusion, BestSpace, and divergence) remain important.

The multi-view index has its own cache type with its own `SchemaVersion` and a
`ChunkerVersion` salt in both the file hash and the manifest, so it can never be
mistaken for the single-view cache described above:
`samples/NivaraChat/Helpers/MultiViewIndexCache.cs::MultiViewIndexCache.TrySave()`
writes metadata plus the four prefixed dimension blocks as one Parquet, and
`samples/NivaraChat/Helpers/MultiViewIndexCache.cs::MultiViewIndexCache.IsValid()`
rejects a cache whose footer does not hold one dimension block per view.

### Learning the fusion weights

Those four weights are a hand-picked default, not a fitted result, so `--rag-learn` (#915) trains
them and compares the result against the fixed baseline on queries neither model saw. The trainable
part reranks a candidate pool; candidate generation stays outside the gradient path, so this is a
reranking comparison rather than three different retrieval systems.

The fusion weights are parameterized as logits and pushed through a softmax, which keeps them
strictly positive and summing to one:

```
w     = softmax(theta)                  # theta in R^4
fused = ((S . M) @ w) / (M @ w)         # S = cosine scores [rows, 4], M = presence mask
```

That is the tensor form of the per-row renormalization the sample already does, and the denominator
cannot be zero: softmax is strictly positive and every pooled row has at least one scored view.
Initializing `theta = ln(0.5, 0.2, 0.2, 0.1)` makes `softmax(theta)` return the defaults exactly,
because those four numbers sum to one, so an untrained model is numerically the #914 baseline. That
parity is the point: it makes a measured delta attributable to learning rather than to a different
fusion formula, and both the model and the mode assert it
(`samples/NivaraChat/Helpers/RetrievalTrainer.cs::FusionModel`).

The query gate is a mixture of the four cosine scorers. Its default feature vector is each view's
mean and max cosine over the candidate pool, 8 features to 4 logits, 36 parameters; the gate weight
matrix starts at zero and the bias at those same logits, so it is invariant to its input until
training moves it. `--gate embed` swaps in a 384-to-4 linear layer over the raw query embedding for
comparison, and 36 parameters is the size expected to survive roughly a hundred training queries.

The loss is listwise softmax cross-entropy against a `2^grade` target, normalized per query, so
grade 2 outranks 1 outranks 0 and a query contributes one graph
(`samples/NivaraChat/Helpers/RetrievalTrainer.cs::RetrievalTrainer.Train()`). `--loss pairwise`
selects a logistic loss over ordered pairs instead. Both are gather-free, which is a constraint
rather than a preference: `ReverseGradOperations` has no row-gather op, so the listwise path uses
`LogSoftmax` and the pairwise path multiplies a constant ±1 selector matrix.

Candidate generation is deliberately weight-agnostic: the pool is the union of each view's own
top-K plus every labeled positive
(`samples/NivaraChat/Helpers/RetrievalTrainingData.cs::CandidatePool.Select()`). Drawing the pool
from the fixed baseline's top-K would let the baseline choose its own competitors.

Labels come from three sources, and the difference matters for what the numbers mean. The default is
the LLM-judge cache written by `--rag-eval`; the fallback is a committed fixture keyed by **document
path**, so it survives docs edits where a `path#chunk` id does not; and `--labels auto` uses
heading-derived pseudo-queries, which are self-fulfilling because Title and Section then score their
own positive near 1.0. Auto labels are therefore a leakage diagnostic, printed with a warning, never
a headline. The mode refuses to train when fewer than half the label rows resolve, since a stale
label file shrinks the relevant set and lowers every ranker without any of them being worse.

Checkpoint selection uses validation nDCG@K with patience and the test split is scored once at the
end; the split itself hashes the query string with FNV-1a because `string.GetHashCode()` is
randomized per process and would make two reports incomparable
(`samples/NivaraChat/Helpers/RetrievalTrainingData.cs::QuerySplit.Create()`). The paired comparison
is the same one `--rag-eval` already reports — per-query nDCG, win/loss/tie, a sign test and Cohen's
d — so neither a small consistent win nor a large noisy one is mistaken for the other
(`samples/NivaraChat/Helpers/RetrievalEvaluation.cs::RetrievalEvaluation.Compare()`).

Two behaviours are stated rather than implied. A learned model that loses on held-out queries is a
valid outcome, so the mode prints the comparison and does not claim a win. And a two-sided sign test
cannot reach p < 0.05 with fewer than six decisive queries, so the mode says so when the held-out
split is too small to support a headline. With `--silk` the MiniLM embeddings run on the OpenCL
backend; training and scoring stay on CPU, and GPU training is a documented follow-up.

## What this convention is not

- Not a column kind. There is no `NivaraColumn<Embedding<float>>`; the convention
  is a `float[]` column plus derived dimension columns.
- Not a similarity engine. Scoring is
  `NivaraFrame.RowCosineSimilarity()` (`src/Nivara/NivaraFrame.cs::RowCosineSimilarity()`)
  and `NivaraFrame.RowDot()` (`src/Nivara/NivaraFrame.cs::RowDot()`), which call
  `TensorPrimitives` over a materialised row-major buffer.
- Not a vector index. There is no ANN structure; retrieval is an exact scan
  followed by `TopKDescending`.
- Not a persistence format. A `float[]` column cannot be written by the Arrow or
  Parquet adapters (`UnsupportedTypeException`), and CSV/JSON are read sources.
  Persist the dimension frame instead: `ToEmbeddingMatrix` before writing,
  `ToEmbeddingFrame` after reading, which is what `RagIndexCache` does. A null
  row survives as all-null dimension cells and comes back as a null embedding.

The full surface lives in `src/Nivara.Extensions/AI/NivaraEmbeddingExtensions.cs`.
The MEAI generator it builds on is
`src/Nivara.Extensions/AI/NivaraEmbeddingGenerator.cs`.
