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

using var products = new NivaraFrame(new[]
{
    ("ProductA", (IColumn)NivaraColumn<float>.Create(new[] { 0.9f, 0.2f, 0.5f, 0.4f })),
    ("ProductB", (IColumn)NivaraColumn<float>.Create(new[] { 0.1f, 0.9f, 0.2f, 0.7f })),
    ("ProductC", (IColumn)NivaraColumn<float>.Create(new[] { 0.7f, 0.1f, 0.8f, 0.2f })),
});

using var scores = products.CosineSimilarity(user);
var topProducts = scores.TopKDescending(3);

foreach (var item in topProducts)
    Console.WriteLine($"{item.Label}: {item.Score:0.000}");
```

Today this is much better than the original manual loop. `CosineSimilarity` now computes one score per product column, and `TopKDescending` returns label-aware ranking results.

Current limitations to review:

- Product embeddings are represented as columns, not rows. That is natural for Nivara's columnar model, but many ML engineers expect samples as rows and features as columns.
- Null-containing vectors produce null-propagating results (mask-OR) in batch similarity operations — null inputs yield null outputs rather than throwing.
- `TopKDescending` currently returns `(string? Label, T Score)` tuples and excludes null scores. A richer ranking result type may still be useful if callers need positions or explicit null state.

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
var topCandidates = scores.TopKDescending(3);
```

Current limitations to review:

- Dot product operates across columns.
- Input length must match the frame row count.
- Nulls propagate via mask-OR (null input → null output) rather than throwing.
- `TopKDescending` provides labeled top-k results, but it currently uses a full sort rather than a threshold-based heap path.

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
- Scalar reducers such as `DotProduct` and `Norm` still throw on null-containing series; `Dot`, `CosineSimilarity`, `ColumnNorms`, and `RowNorms` on `NivaraFrame` propagate nulls in their result masks.

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

This is intentionally shown in the usual ML orientation: each row is a document, and each embedding dimension is a numeric feature column. Today, Nivara does not have row-wise batch cosine similarity, so this still requires a manual row loop. The final ranking is simpler now because scores can be wrapped in a labeled `NivaraSeries<T>` and ranked with `TopKDescending`.

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

var scoreLabels = Enumerable.Range(0, documents.RowCount)
    .Select(row => documentIds[row])
    .ToArray();
using var scoreSeries = NivaraSeries<float>.Create(scores, scoreLabels);
var ranking = scoreSeries.TopKDescending(2);

foreach (var item in ranking)
    Console.WriteLine($"{item.Label}: {item.Score:0.000}");
```

Current limitations to review:

- Row-wise embedding search requires manual extraction of each feature column.
- Each row allocates a temporary `documentVector` in this simple version.
- `TopKDescending` removes the manual LINQ ranking and label lookup, but it does not expose positions or explicit null state.
- Null behavior in the row-wise loop is still manual because there is no row-wise cosine-similarity API yet.

## 7. Automatic Differentiation Over Nivara Columns

Nivara.Extensions includes an AutoDiff layer for reverse-mode gradients
over `float` and `double` columns. Users wrap columns (or entire DataFrames)
in `ReverseGradTensor<T>`, call static `GradOperations`, reduce to a scalar
loss, call `Backward()`, and optionally apply `SgdOptimizer.SgdUpdate`.

The full pipeline — from a NivaraFrame through gradient descent and back to
columns — is a single linear workflow:

```csharp
using Nivara;
using Nivara.Extensions.AutoDiff;
using Nivara.Extensions.AutoDiff.Operations;
using Nivara.Extensions.AutoDiff.Optimizer;

// 1. Training data as a NivaraFrame
var df = NivaraFrame.Create(
    ("x", NivaraColumn<float>.Create(new[] { 1.0f, 2.0f, 3.0f })),
    ("y", NivaraColumn<float>.Create(new[] { 2.0f, 4.0f, 6.0f }))
);

// 2. Convert data columns to gradient tensors (no grads needed for input)
var tensors = df.ToReverseGradTensors<float>(new[] { "x", "y" }, requiresGrad: false);
var x = tensors["x"];
var y = tensors["y"];

// 3. Trainable parameters with gradient tracking
var w = ReverseGradTensor<float>.FromArray(new[] { 0.5f, 0.5f, 0.5f }, requiresGrad: true);
var b = ReverseGradTensor<float>.FromArray(new[] { 0.1f, 0.1f, 0.1f }, requiresGrad: true);

// 4. Forward pass: prediction = w * x + b
var prediction = GradOperations.Add(GradOperations.Multiply(w, x), b);

// 5. MSE loss: mean((prediction - y)²)
var diff = GradOperations.Subtract(prediction, y);
var loss = GradOperations.Mean(GradOperations.Multiply(diff, diff));

// 6. Backward pass — computes gradients
loss.Backward();

// 7. SGD update — creates new tensors with updated values
var updatedW = SgdOptimizer.SgdUpdate(w, 0.01f);
var updatedB = SgdOptimizer.SgdUpdate(b, 0.01f);

// 8. Convert updated parameters back to Nivara columns
var wCol = updatedW.ToColumn();
var bCol = updatedB.ToColumn();

Console.WriteLine($"Loss: {loss[0]:F4}");
Console.WriteLine($"w.grad[0] = {w.Grad![0]:F4}, updated w[0] = {wCol[0]:F4}");

// Clean up
w.Dispose(); b.Dispose();
updatedW.Dispose(); updatedB.Dispose();
```

The longer sample app demonstrates reductions, activations, gradient clipping,
and graph inspection in
[`samples/Nivara.SampleApp/AutoDiffExample.cs`](samples/Nivara.SampleApp/AutoDiffExample.cs).

Current limitations to review:

- AutoDiff currently supports `float` and `double` only.
- `Backward()` can be called without an explicit gradient only on scalar
  tensors. Non-scalar outputs require a matching gradient argument.
- Operations are static methods today; there are no fluent methods or operator
  overloads for `ReverseGradTensor<T>` yet.
- `MatMul` and `Transpose` use flattened tensors plus explicit dimensions.
- The `SgdOptimizer.SgdUpdate` primitive is available for simple SGD weight updates.
- There is no layer, model, dataloader, or training-loop API yet.
- See [`docs/AUTODIFF-SUGGESTIONS.md`](docs/AUTODIFF-SUGGESTIONS.md) for the
  grounded follow-up list.

## 8. What We Want Reviewers To React To

When showing these examples to ML and AI engineers, useful feedback is:

- Is the product-ranking workflow readable enough for real recommendation or retrieval work?
- Does column-wise similarity feel surprising?
- Should the primary embedding-table API treat rows as candidates instead?
- Is the row-wise document search example closer to how people would naturally store embeddings?
- Are explicit null policies acceptable, or too verbose?
- Which missing operation blocks realistic usage first: row-wise similarity, batch norms, broadcasting, matrix multiply, or fluent arithmetic?
- Are ranking APIs returning positions enough, or should there be a label-aware ranking result?
- Does direct `Tensor<T>` conversion solve enough interop, or do users expect richer tensor-shaped Nivara APIs?
- Is the current AutoDiff surface useful as a low-level gradient layer, or do reviewers need optimizer and shape-aware matrix helpers first?

## Current Roadmap Link

See [`docs/TENSORS-GAPS.md`](docs/TENSORS-GAPS.md) and
[`docs/AUTODIFF-SUGGESTIONS.md`](docs/AUTODIFF-SUGGESTIONS.md) for the current
Tensor and AutoDiff improvement roadmaps.
