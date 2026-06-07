# Nivara Examples

## Positioning

Nivara is **not** a NumPy-for-.NET, tensor library, vector math library, embedding similarity engine, or AutoDiff framework.

It is:

> A typed, immutable, null-aware DataFrame/query layer for .NET
> with clean interop to BCL tensors, Microsoft.Extensions.AI, VectorData, Arrow, CSV, JSON, and Parquet.

.NET already has `System.Numerics.Tensors` for tensor operations (including SIMD-accelerated `TensorPrimitives`) and `Microsoft.Extensions.AI` for embedding abstractions. Nivara integrates with those instead of competing with them.

| Use .NET / Microsoft libraries for | Use Nivara for |
|---|---|
| `Tensor<T>`, `TensorPrimitives` | Typed columns, schemas |
| `ReadOnlySpan<float>`, `Vector<T>` | Null masks, null propagation |
| `IEmbeddingGenerator`, `VectorData` | Query planning, joins, grouping |
| Math kernels (Dot, Norm, CosineSimilarity) | Lazy file I/O (CSV, JSON, Parquet, Arrow) |
| AI abstractions | Labeling, row identity, schema validation |

## How to read these examples

Each real-world scenario shows:
1. **Python** — the baseline you'd find in tutorials or from ChatGPT (NumPy, pandas, PyTorch)
2. **.NET** — the .NET approach, using either BCL `System.Numerics.Tensors` directly (when BCL suffices) or Nivara (when tabular data management adds value)

When the BCL alone handles a scenario, Nivara backs off. We want .NET users to know they don't need Nivara for tensor math.

All .NET examples assume `using System.Numerics.Tensors;` for BCL tensor APIs and `using Nivara;` for Nivara APIs unless otherwise noted.

## Examples

### 1. Product Ranking With Embedding Similarity

Rank products by cosine similarity to a query embedding — a core recommendation, semantic search, and retrieval workflow.

**Python**

```python
import numpy as np

user = np.array([0.8, 0.1, 0.6, 0.3], dtype=np.float32)

products = {
    "ProductA": np.array([0.9, 0.2, 0.5, 0.4], dtype=np.float32),
    "ProductB": np.array([0.1, 0.9, 0.2, 0.7], dtype=np.float32),
    "ProductC": np.array([0.7, 0.1, 0.8, 0.2], dtype=np.float32),
}

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = {name: cosine(vec, user) for name, vec in products.items()}
top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

for label, score in top:
    print(f"{label}: {score:.3f}")
```

**.NET (BCL)**

```csharp
float[] user = [0.8f, 0.1f, 0.6f, 0.3f];

var products = new Dictionary<string, float[]>
{
    ["ProductA"] = [0.9f, 0.2f, 0.5f, 0.4f],
    ["ProductB"] = [0.1f, 0.9f, 0.2f, 0.7f],
    ["ProductC"] = [0.7f, 0.1f, 0.8f, 0.2f],
};

var topProducts = products
    .Select(p => new
    {
        Label = p.Key,
        Score = TensorPrimitives.CosineSimilarity(p.Value, user)
    })
    .OrderByDescending(x => x.Score)
    .Take(3);

foreach (var item in topProducts)
    Console.WriteLine($"{item.Label}: {item.Score:0.000}");
```

You don't need Nivara here because `TensorPrimitives.CosineSimilarity` already handles vector math directly with SIMD acceleration. This is pure numeric computation with no tabular requirements (schemas, nulls, joins, I/O).

### 2. Direct Tensor Conversion / BCL Interop

Convert embedding data to `Tensor<T>` for consumption by BCL APIs.

**Python**

```python
embedding = np.array([0.8, 0.1, 0.6, 0.3], dtype=np.float32)
print(embedding.ndim)      # 1
print(embedding.shape[0])  # 4

# Null handling
values = np.array([1.0, np.nan, 3.0], dtype=np.float32)
filled = np.nan_to_num(values, nan=0.0)
```

**.NET (BCL)**

```csharp
float[] embedding = [0.8f, 0.1f, 0.6f, 0.3f];
Tensor<float> tensor = Tensor.Create(embedding);

Console.WriteLine(tensor.Rank);       // 1
Console.WriteLine(tensor.Lengths[0]); // 4
```

For nullable data the BCL requires manual handling:

```csharp
float?[] values = [1.0f, null, 3.0f];
float[] filled = values.Select(v => v ?? 0.0f).ToArray();
Tensor<float> filledTensor = Tensor.Create(filled);
```

We're using Nivara here because `NivaraColumn.CreateFromNullable` preserves null semantics at the column level, and `ToTensor(defaultValue)` provides a single-call fill policy instead of manual `Select` — avoiding allocation of intermediate LINQ enumerables when the column is large.

**.NET (Nivara)** — when the data lives in a Nivara column with nulls, Nivara encapsulates the fill policy:

```csharp
var column = NivaraColumn<float>.CreateFromNullable(
    new float?[] { 1.0f, null, 3.0f });

Tensor<float> filledTensor = column.ToTensor(defaultValue: 0.0f);
```

### 3. Batch Column Dot Products

Score candidates by dot product (common when embeddings are pre-normalized).

**Python**

```python
query = np.array([0.8, 0.1, 0.6, 0.3], dtype=np.float32)

candidates = {
    "A": np.array([0.9, 0.2, 0.5, 0.4], dtype=np.float32),
    "B": np.array([0.1, 0.9, 0.2, 0.7], dtype=np.float32),
    "C": np.array([0.7, 0.1, 0.8, 0.2], dtype=np.float32),
}

scores = {label: np.dot(vec, query) for label, vec in candidates.items()}
top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

for label, score in top:
    print(f"{label}: {score:.3f}")
```

**.NET (BCL)**

```csharp
float[] query = [0.8f, 0.1f, 0.6f, 0.3f];

var candidates = new Dictionary<string, float[]>
{
    ["A"] = [0.9f, 0.2f, 0.5f, 0.4f],
    ["B"] = [0.1f, 0.9f, 0.2f, 0.7f],
    ["C"] = [0.7f, 0.1f, 0.8f, 0.2f],
};

var topCandidates = candidates
    .Select(c => new
    {
        Label = c.Key,
        Score = TensorPrimitives.Dot(c.Value, query)
    })
    .OrderByDescending(x => x.Score)
    .Take(3);
```

You don't need Nivara here because `TensorPrimitives.Dot` handles the vector math. No tabular features (schemas, null masks, I/O) are involved — just a dictionary and a LINQ projection.

### 4. One-Dimensional Tensor Helpers

Basic vector operations (dot product, norm).

**Python**

```python
left = np.array([1.0, 2.0, 3.0], dtype=np.float32)
right = np.array([4.0, 5.0, 6.0], dtype=np.float32)

dot = np.dot(left, right)
norm = np.linalg.norm(left)

print(dot)
print(norm)
```

**.NET (BCL)**

```csharp
float[] left = [1.0f, 2.0f, 3.0f];
float[] right = [4.0f, 5.0f, 6.0f];

float dot = TensorPrimitives.Dot(left, right);
float norm = TensorPrimitives.Norm(left);

Console.WriteLine(dot);
Console.WriteLine(norm);
```

You don't need Nivara here because `TensorPrimitives.Dot` and `TensorPrimitives.Norm` are the intended BCL APIs for basic vector math. There is no tabular data to manage — just two flat arrays.

### 5. Frame-To-Tensor Shape Convention

Convert a tabular frame of numeric features to a dense 2D tensor with shape `[rows, columns]`.

**Python**

```python
frame = pd.DataFrame({
    "Feature1": [0.9, 0.1, 0.7],
    "Feature2": [0.2, 0.9, 0.1],
    "Feature3": [0.5, 0.2, 0.8],
    "Feature4": [0.4, 0.7, 0.2],
}, dtype=np.float32)

matrix = frame.to_numpy()
print(matrix.shape[0])  # 3
print(matrix.shape[1])  # 4
```

**.NET (BCL)**

```csharp
float[] matrixData =
[
    0.9f, 0.2f, 0.5f, 0.4f,
    0.1f, 0.9f, 0.2f, 0.7f,
    0.7f, 0.1f, 0.8f, 0.2f,
];

Tensor<float> matrix = Tensor.Create(matrixData, lengths: [3, 4]);

Console.WriteLine(matrix.Lengths[0]); // 3
Console.WriteLine(matrix.Lengths[1]); // 4
```

We're using Nivara here because `frame.ToTensor<T>()` automatically discovers typed numeric columns, lays out data in row-major order, and handles null fill policies — instead of manually constructing a flat array and calculating strides.

**.NET (Nivara)** — when the features are typed columns in a NivaraFrame:

```csharp
var frame = NivaraFrame.Create(
    ("Feature1", NivaraColumn<float>.Create(new[] { 0.9f, 0.1f, 0.7f })),
    ("Feature2", NivaraColumn<float>.Create(new[] { 0.2f, 0.9f, 0.1f })),
    ("Feature3", NivaraColumn<float>.Create(new[] { 0.5f, 0.2f, 0.8f })),
    ("Feature4", NivaraColumn<float>.Create(new[] { 0.4f, 0.7f, 0.2f }))
);

Tensor<float> matrix = frame.ToTensor<float>();
// Shape: [rows=3, columns=4]
```

### 6. Semantic Search Over Document Embeddings

Score a query embedding against a table of documents, then rank by similarity.

**Python**

```python
query = np.array([0.8, 0.1, 0.6, 0.3], dtype=np.float32)

documents = pd.DataFrame({
    "DocumentId": ["doc-101", "doc-102", "doc-103"],
    "e0": [0.9, 0.1, 0.7],
    "e1": [0.2, 0.9, 0.1],
    "e2": [0.5, 0.2, 0.8],
    "e3": [0.4, 0.7, 0.2],
})

embedding_cols = ["e0", "e1", "e2", "e3"]
doc_vectors = documents[embedding_cols].to_numpy(dtype=np.float32)

query_norm = np.linalg.norm(query)
doc_norms = np.linalg.norm(doc_vectors, axis=1)
scores = doc_vectors @ query / (doc_norms * query_norm)

ranking = sorted(zip(documents["DocumentId"], scores), key=lambda x: x[1], reverse=True)[:2]

for label, score in ranking:
    print(f"{label}: {score:.3f}")
```

**.NET (BCL)**

```csharp
float[] query = [0.8f, 0.1f, 0.6f, 0.3f];

var documents = new[]
{
    new { DocumentId = "doc-101", Embedding = new float[] { 0.9f, 0.2f, 0.5f, 0.4f } },
    new { DocumentId = "doc-102", Embedding = new float[] { 0.1f, 0.9f, 0.2f, 0.7f } },
    new { DocumentId = "doc-103", Embedding = new float[] { 0.7f, 0.1f, 0.8f, 0.2f } },
};

var ranking = documents
    .Select(d => new
    {
        d.DocumentId,
        Score = TensorPrimitives.CosineSimilarity(d.Embedding, query)
    })
    .OrderByDescending(x => x.Score)
    .Take(2);

foreach (var item in ranking)
    Console.WriteLine($"{item.DocumentId}: {item.Score:0.000}");
```

**.NET (Nivara)** — when documents live in a NivaraFrame with metadata and you want labeled results:

```csharp
var query = new[] { 0.8f, 0.1f, 0.6f, 0.3f };

var documents = NivaraFrame.Create(
    ("DocumentId", NivaraColumn<string>.CreateForReferenceType(
        new[] { "doc-101", "doc-102", "doc-103" })),
    ("Embedding", NivaraColumn<float[]>.CreateForReferenceType(
        new[] {
            new[] { 0.9f, 0.2f, 0.5f, 0.4f },
            new[] { 0.1f, 0.9f, 0.2f, 0.7f },
            new[] { 0.7f, 0.1f, 0.8f, 0.2f },
        }))
);

var ids = documents.GetColumn<string>("DocumentId");
var embeddings = documents.GetColumn<float[]>("Embedding");

var scores = Enumerable.Range(0, documents.RowCount)
    .Select(i => TensorPrimitives.CosineSimilarity(embeddings[i], query))
    .ToArray();

using var results = NivaraSeries<float>.Create(scores, ids.ToArray());
var ranking = results.TopKDescending(2);

foreach (var item in ranking)
    Console.WriteLine($"{item.Label}: {item.Score:0.000}");
```

We're using Nivara here because the NivaraFrame provides typed schema, preserves document metadata alongside embeddings, handles null masks in the embedding column, and `TopKDescending` gives label-aware ranking with null exclusion — all of which would be manual work with anonymous objects and LINQ alone. The math stays with `TensorPrimitives.CosineSimilarity`.

### 7. Automatic Differentiation Over Columns (Experimental)

Train a simple linear model using gradient descent. Nivara.Extensions includes an experimental reverse-mode AutoDiff layer.

**Python**

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([2.0, 4.0, 6.0])

w = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
b = torch.tensor([0.1, 0.1, 0.1], requires_grad=True)

prediction = w * x + b
diff = prediction - y
loss = torch.mean(diff * diff)

loss.backward()

with torch.no_grad():
    updated_w = w - 0.01 * w.grad
    updated_b = b - 0.01 * b.grad

print(f"Loss: {loss.item():.4f}")
print(f"w.grad[0] = {w.grad[0].item():.4f}, updated w[0] = {updated_w[0].item():.4f}")
```

You don't need Nivara here if you only need forward inference — `TensorPrimitives` (Dot, Norm, SumOfSquares) can express the math. But there is no BCL autograd, so backpropagation requires manual gradient derivation and loops.

**.NET (BCL)** — no autograd framework; explicit gradient computation required:

```csharp
float[] x = [1.0f, 2.0f, 3.0f];
float[] y = [2.0f, 4.0f, 6.0f];
float[] w = [0.5f, 0.5f, 0.5f];
float[] b = [0.1f, 0.1f, 0.1f];
float lr = 0.01f;
int n = x.Length;

float[] pred = new float[n], diff = new float[n];
for (int i = 0; i < n; i++)
{
    pred[i] = w[i] * x[i] + b[i];
    diff[i] = pred[i] - y[i];
}

float loss = TensorPrimitives.SumOfSquares(diff) / n;

float[] gradW = new float[n], gradB = new float[n];
for (int i = 0; i < n; i++)
{
    gradW[i] = 2.0f * diff[i] * x[i] / n;
    gradB[i] = 2.0f * diff[i] / n;
}

float[] updatedW = new float[n], updatedB = new float[n];
for (int i = 0; i < n; i++)
{
    updatedW[i] = w[i] - lr * gradW[i];
    updatedB[i] = b[i] - lr * gradB[i];
}

Console.WriteLine($"Loss: {loss:F4}");
Console.WriteLine($"gradW[0] = {gradW[0]:F4}, updatedW[0] = {updatedW[0]:F4}");
```

We're using Nivara.Extensions here because it provides the only reverse-mode AutoDiff for Nivara column data — automatically building a computation graph, computing gradients via `Backward()`, and providing `SgdOptimizer.SgdUpdate` — instead of manual gradient derivation.

**.NET (Nivara.Extensions)** — reverse-mode AutoDiff with gradient tracking:

```csharp
using Nivara.Extensions.AutoDiff;
using Nivara.Extensions.AutoDiff.Operations;
using Nivara.Extensions.AutoDiff.Optimizer;

var df = NivaraFrame.Create(
    ("x", NivaraColumn<float>.Create(new[] { 1.0f, 2.0f, 3.0f })),
    ("y", NivaraColumn<float>.Create(new[] { 2.0f, 4.0f, 6.0f }))
);

var tensors = df.ToReverseGradTensors<float>(new[] { "x", "y" }, requiresGrad: false);
var x = tensors["x"];
var y = tensors["y"];

using var w = ReverseGradTensor<float>.FromArray(new[] { 0.5f, 0.5f, 0.5f }, requiresGrad: true);
using var b = ReverseGradTensor<float>.FromArray(new[] { 0.1f, 0.1f, 0.1f }, requiresGrad: true);

var prediction = GradOperations.Add(GradOperations.Multiply(w, x), b);
var diff = GradOperations.Subtract(prediction, y);
var loss = GradOperations.Mean(GradOperations.Multiply(diff, diff));

loss.Backward();

using var updatedW = SgdOptimizer.SgdUpdate(w, 0.01f);
using var updatedB = SgdOptimizer.SgdUpdate(b, 0.01f);

Console.WriteLine($"Loss: {loss[0]:F4}");
Console.WriteLine($"w.Grad[0] = {w.Grad![0]:F4}, updated w[0] = {updatedW.ToColumn()[0]:F4}");
```

See the longer sample in [`samples/Nivara.SampleApp/AutoDiffExample.cs`](samples/Nivara.SampleApp/AutoDiffExample.cs) and deferred work in [`docs/AUTODIFF-GAPS.md`](docs/AUTODIFF-GAPS.md).

> **Note:** This API is experimental. It supports `float` and `double` only, uses static `GradOperations` methods (no operator overloads yet), and has no layer/dataloader/training-loop abstractions.

## What We Want Reviewers To React To

- Is the Python → .NET mapping clear enough for engineers coming from a Python/NumPy/pandas background?
- Does backing off from tensor math and positioning Nivara as a tabular data layer feel like the right call?
- Are the boundary examples (6, 7) clear about where Nivara stops and BCL math begins?
- Which tabular features add the most value: typed columns, null semantics, schemas, labeled ranking, or file I/O?
- Is the experimental AutoDiff layer worth keeping as a gradient bridge for Nivara column data?
- What interop scenarios are missing (Microsoft.Extensions.AI, VectorData, Arrow zero-copy, etc.)?

## Related docs

- [`docs/TENSORS.md`](docs/TENSORS.md) — full positioning discussion
- [`docs/AUTODIFF-GAPS.md`](docs/AUTODIFF-GAPS.md) — AutoDiff roadmap and gaps
