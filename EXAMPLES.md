# Nivara Examples

These examples show how Nivara works today for workflows we want actual ML and AI engineers to review.

The goal is not to present a finished ideal API. The goal is to keep practical examples visible so feedback can be grounded in real day-to-day usage: embeddings, ranking, tensor conversion, typed columns, null semantics, and the places where the current experience is still too manual.

## 1. Product Ranking With Embedding Similarity

This is the core ML example from the external review: rank products by cosine similarity to a user embedding.

This kind of workflow is common in recommendation systems, semantic search, retrieval ranking, personalization, and nearest-neighbor style prototypes.

```csharp
using Nivara;
using Nivara.Tensors;

var user = NivaraSeries<float>.Create(new[] { 0.8f, 0.1f, 0.6f, 0.3f });

var productNames = new[] { "ProductA", "ProductB", "ProductC" };

using var products = new NivaraFrame(new[]
{
    ("ProductA", (IColumn)NivaraColumn<float>.Create(new[] { 0.9f, 0.2f, 0.5f, 0.4f })),
    ("ProductB", (IColumn)NivaraColumn<float>.Create(new[] { 0.1f, 0.9f, 0.2f, 0.7f })),
    ("ProductC", (IColumn)NivaraColumn<float>.Create(new[] { 0.7f, 0.1f, 0.8f, 0.2f })),
});

using var scores = products.CosineSimilarity(user);
var ranking = scores.ArgSortDescending();

foreach (var position in ranking)
{
    Console.WriteLine($"{productNames[position]}: {scores[position]:0.000}");
}
```

Today this is much better than the original manual loop. `CosineSimilarity` now computes one score per product column, and `ArgSortDescending` returns ranking positions.

Current limitations to review:

- Product embeddings are represented as columns, not rows. That is natural for Nivara's columnar model, but many ML engineers expect samples as rows and features as columns.
- Null-containing vectors produce null-propagating results (mask-OR) in batch similarity operations — null inputs yield null outputs rather than throwing.
- The result ranking is positional. The result series has column names as its index, but the common ergonomic path is still to map positions back to names.

## 2. Direct Tensor Conversion For BCL Interop

Nivara can convert typed series and columns to `System.Numerics.Tensors.Tensor<T>` when users need to call BCL tensor APIs directly.

```csharp
using Nivara;
using System.Numerics.Tensors;

var embedding = NivaraSeries<float>.Create(new[] { 0.8f, 0.1f, 0.6f, 0.3f });

Tensor<float> tensor = embedding.ToTensor();

Console.WriteLine(tensor.Rank);       // 1
Console.WriteLine(tensor.Lengths[0]); // 4
```

For null-containing data, the policy is explicit:

```csharp
using Nivara;
using System.Numerics.Tensors;

var values = NivaraColumn<float>.CreateFromNullable(
    new float?[] { 1.0f, null, 3.0f });

using var series = new NivaraSeries<float>(values);

// Throws because the null policy is not implicit.
// Tensor<float> unsafeTensor = series.ToTensor();

// Replaces only null positions with the provided value.
Tensor<float> filledTensor = series.ToTensor(defaultValue: 0.0f);
```

Current limitations to review:

- `ToTensor()` returns a copy to preserve Nivara immutability.
- There is no nullable-preserving tensor result type yet.
- The conversion policy must stay consistent across `NivaraColumn`, `NivaraSeries`, `NivaraFrame`, and `TensorInterop`.

## 3. Batch Column Dot Products

For recommendation and retrieval workflows, dot product is often used directly when embeddings are already normalized.

```csharp
using Nivara;

var query = NivaraSeries<float>.Create(new[] { 0.8f, 0.1f, 0.6f, 0.3f });

using var candidates = new NivaraFrame(new[]
{
    ("A", (IColumn)NivaraColumn<float>.Create(new[] { 0.9f, 0.2f, 0.5f, 0.4f })),
    ("B", (IColumn)NivaraColumn<float>.Create(new[] { 0.1f, 0.9f, 0.2f, 0.7f })),
    ("C", (IColumn)NivaraColumn<float>.Create(new[] { 0.7f, 0.1f, 0.8f, 0.2f })),
});

using var scores = candidates.Dot(query);
var ranking = scores.ArgSortDescending();
```

Current limitations to review:

- Dot product operates across columns.
- Input length must match the frame row count.
- Nulls propagate via mask-OR (null input → null output) rather than throwing.

## 4. Existing One-Dimensional Tensor Helpers

Nivara still has useful 1D tensor-aware helpers on series.

```csharp
using Nivara;
using Nivara.Tensors;

var left = NivaraSeries<float>.Create(new[] { 1.0f, 2.0f, 3.0f });
var right = NivaraSeries<float>.Create(new[] { 4.0f, 5.0f, 6.0f });

var dot = left.DotProduct(right);
var norm = left.Norm();
```

Current limitations to review:

- These methods are good for individual vectors.
- Batch operations across a frame are newer and should be made consistent with these helpers.
- Null-throwing behavior varies across individual helpers; `Dot` and `CosineSimilarity` on `NivaraFrame` now propagate nulls, but legacy series-level helpers may still throw.

## 5. Frame-To-Tensor Shape Convention

When converting a frame to a 2D tensor, Nivara uses `[rows, columns]`.

```csharp
using Nivara;
using Nivara.Tensors;
using System.Numerics.Tensors;

using var frame = new NivaraFrame(new[]
{
    ("Feature1", (IColumn)NivaraColumn<float>.Create(new[] { 0.9f, 0.1f, 0.7f })),
    ("Feature2", (IColumn)NivaraColumn<float>.Create(new[] { 0.2f, 0.9f, 0.1f })),
    ("Feature3", (IColumn)NivaraColumn<float>.Create(new[] { 0.5f, 0.2f, 0.8f })),
    ("Feature4", (IColumn)NivaraColumn<float>.Create(new[] { 0.4f, 0.7f, 0.2f })),
});

Tensor<float> matrix = frame.ToTensor<float>();

Console.WriteLine(matrix.Lengths[0]); // rows: 3
Console.WriteLine(matrix.Lengths[1]); // columns: 4
```

This representation is closer to the usual ML convention: rows are examples, columns are features.

Current limitations to review:

- `products.CosineSimilarity(user)` currently treats each frame column as a candidate vector.
- `frame.ToTensor<T>()` treats rows as examples and columns as features.
- We need clearer named APIs for row-wise versus column-wise operations before adding axis syntax.

## 6. Semantic Search Over Document Embeddings

This example is another common ML workflow: score a query embedding against a table of document embeddings.

This is intentionally shown in the usual ML orientation: each row is a document, and each embedding dimension is a numeric feature column. Today, Nivara does not have row-wise batch cosine similarity, so this requires a manual row loop. (Note: `NivaraFrame.TopKDescending` and `RowNorms`/`ColumnNorms` are available for null-propagating batch similarity — see section 3 for column-wise batch dot product.)

```csharp
using Nivara;
using System.Numerics.Tensors;

var queryValues = new[] { 0.8f, 0.1f, 0.6f, 0.3f };
var query = NivaraSeries<float>.Create(queryValues);

using var documents = new NivaraFrame(new[]
{
    ("DocumentId", (IColumn)NivaraColumn<string>.CreateForReferenceType(new[] { "doc-101", "doc-102", "doc-103" })),
    ("e0", (IColumn)NivaraColumn<float>.Create(new[] { 0.9f, 0.1f, 0.7f })),
    ("e1", (IColumn)NivaraColumn<float>.Create(new[] { 0.2f, 0.9f, 0.1f })),
    ("e2", (IColumn)NivaraColumn<float>.Create(new[] { 0.5f, 0.2f, 0.8f })),
    ("e3", (IColumn)NivaraColumn<float>.Create(new[] { 0.4f, 0.7f, 0.2f })),
});

var documentIds = documents.GetColumn<string>("DocumentId");
var e0 = documents.GetColumn<float>("e0");
var e1 = documents.GetColumn<float>("e1");
var e2 = documents.GetColumn<float>("e2");
var e3 = documents.GetColumn<float>("e3");

var scores = new float[documents.RowCount];

for (int row = 0; row < documents.RowCount; row++)
{
    var documentVector = new[] { e0[row], e1[row], e2[row], e3[row] };
    scores[row] = TensorPrimitives.CosineSimilarity(documentVector, queryValues);
}

var ranking = scores
    .Select((score, position) => (score, position))
    .OrderByDescending(x => x.score)
    .Take(2)
    .ToArray();

foreach (var item in ranking)
{
    Console.WriteLine($"{documentIds[item.position]}: {item.score:0.000}");
}
```

Current limitations to review:

- Row-wise embedding search requires manual extraction of each feature column.
- Each row allocates a temporary `documentVector` in this simple version.
- Ranking is still manual LINQ over a raw score array.
- Metadata labels must be mapped back from positions.
- Null behavior is not represented in the ranking result.

## 7. What We Want Reviewers To React To

When showing these examples to ML and AI engineers, useful feedback is:

- Is the product-ranking workflow readable enough for real recommendation or retrieval work?
- Does column-wise similarity feel surprising?
- Should the primary embedding-table API treat rows as candidates instead?
- Is the row-wise document search example closer to how people would naturally store embeddings?
- Are explicit null policies acceptable, or too verbose?
- Which missing operation blocks realistic usage first: row-wise similarity, batch norms, broadcasting, matrix multiply, or fluent arithmetic?
- Are ranking APIs returning positions enough, or should there be a label-aware ranking result?
- Does direct `Tensor<T>` conversion solve enough interop, or do users expect richer tensor-shaped Nivara APIs?

## Current Roadmap Link

See `docs/TENSORS-GAPS.md` for the current tensor-improvement roadmap and the gaps we should address before expanding the API surface.
