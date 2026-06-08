# Nivara Examples

## Positioning

Nivara is **not** a NumPy-for-.NET, tensor library, vector math library, embedding similarity engine, or AutoDiff framework.

It is:

> A typed, immutable, null-aware DataFrame/query layer for .NET
> with clean interop to BCL tensors, Microsoft.Extensions.AI, VectorData, Arrow, CSV, JSON, and Parquet.

.NET already has `System.Numerics.Tensors` for tensor operations (including SIMD-accelerated `TensorPrimitives`) and `Microsoft.Extensions.AI` for embedding abstractions. Nivara integrates with those instead of competing with them.

The .NET Base Class Library (BCL) has been evolving rapidly for ML workloads — `TensorPrimitives`
(SIMD math) in .NET 7, `Tensor<T>` (n-dimensional arrays) in .NET 9,
`Microsoft.Extensions.AI` in .NET 10. Nivara fills what's still missing:
null-aware tabular data with schema and file I/O, so you can stay in .NET
end-to-end.

| Use .NET / Microsoft libraries for | Use Nivara for |
|---|---|
| `Tensor<T>`, `TensorPrimitives` | Typed columns, schemas |
| `ReadOnlySpan<float>`, `Vector<T>` | Null masks, null propagation |
| `IEmbeddingGenerator`, `VectorData` | Query planning, joins, grouping |
| Math kernels (Dot, Norm, CosineSimilarity) | Lazy file I/O (CSV, JSON, Parquet, Arrow) |
| AI abstractions | Labeling, row identity, schema validation |

## How to read these examples

Each scenario shows:
1. **.NET** — the .NET approach, using BCL `System.Numerics.Tensors` where it suffices or Nivara when tabular data adds value
2. **Python** — the equivalent in NumPy, pandas, or PyTorch, shown for comparison

This isn't a Python tutorial — it's a .NET developer's guide to what the
runtime and ecosystem already provide, and where Nivara fills the gap.

The examples are grouped into acts that progressively introduce where Nivara
adds value — starting with scenarios where BCL alone is sufficient and building
up to full tabular+tensor workflows with null propagation and schema.

When the BCL covers a scenario, Nivara backs off. We want .NET users to know
they can stay in the .NET ecosystem end-to-end.

All .NET examples assume `using System.Numerics.Tensors;` for BCL tensor APIs and `using Nivara;` for Nivara APIs unless otherwise noted.

## Examples

### Act 1: When pure tensor math is enough — BCL handles it

> The BCL's `TensorPrimitives` class provides SIMD-accelerated math (Dot, Norm,
> CosineSimilarity) for `float` and `double` arrays. When your data is already
> in flat arrays and you have no null-handling or schema needs, you don't need
> Nivara at all.

#### 1a. One-Dimensional Tensor Helpers

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

The BCL already covers this — `TensorPrimitives.Dot` and `TensorPrimitives.Norm` are the intended APIs for basic vector math. No tabular data to manage here.

#### 1b. Product Ranking With Embedding Similarity

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

The BCL is sufficient here — `TensorPrimitives.CosineSimilarity` handles the math directly. No schemas, nulls, or I/O involved.

#### 1c. Batch Column Dot Products

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

BCL's `TensorPrimitives.Dot` handles this. No tabular features needed — just a dictionary and a LINQ projection.

### Act 2: Nulls are where BCL gets awkward — enter NivaraColumn

> Once you introduce nullable values or missing data, BCL requires manual
> `HasValue` checks and element-wise loops. Nivara columns carry explicit
> null masks that propagate automatically through arithmetic operations.

#### 2a. Null propagation in element-wise addition

Two float columns with nulls at different positions — adding them element-wise,
where the result is null if either operand is null.

**Python**

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "A": [1.0, np.nan, 3.0, np.nan],
    "B": [np.nan, 2.0, 3.0, np.nan],
})

result = df["A"] + df["B"]
# [NaN, NaN, 6.0, NaN]
```

**.NET (BCL)**

```csharp
float?[] a = [1.0f, null, 3.0f, null];
float?[] b = [null, 2.0f, 3.0f, null];
float?[] result = new float?[4];

for (int i = 0; i < a.Length; i++)
    result[i] = a[i].HasValue && b[i].HasValue
        ? a[i].Value + b[i].Value
        : null;
// [null, null, 6.0, null]
```

**.NET (Nivara)** — null propagation is automatic:

```csharp
var colA = NivaraColumn<float>.CreateFromNullable(
    new float?[] { 1.0f, null, 3.0f, null });
var colB = NivaraColumn<float>.CreateFromNullable(
    new float?[] { null, 2.0f, 3.0f, null });

var result = colA.Add(colB);
// result.IsNull(0) → true, result.IsNull(1) → true
// result[2] → 6.0f, result.IsNull(3) → true
```

We're using Nivara here because `NivaraColumn.Add` (and all arithmetic operators) automatically apply mask-OR semantics — the result null mask is `a.IsNull OR b.IsNull`. Nivara expresses this as a single `Add()` call instead of an element-wise loop with manual `HasValue` checks.

#### 2b. Derived column with cross-column nulls

Compute `total = price × quantity` where `price` and `quantity` are nullable columns with nulls at different positions. The result should be null if either operand is null.

**Python**

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "price": [10.0, 20.0, np.nan, 40.0],
    "qty":   [1.0,  np.nan, 3.0,  2.0],
})

df["total"] = df["price"] * df["qty"]
# [10.0, NaN, NaN, 80.0]
```

**.NET (BCL)**

```csharp
float?[] price = [10.0f, 20.0f, null, 40.0f];
float?[] qty = [1.0f, null, 3.0f, 2.0f];
float?[] total = new float?[4];

for (int i = 0; i < price.Length; i++)
    total[i] = price[i].HasValue && qty[i].HasValue
        ? price[i].Value * qty[i].Value
        : null;
// [10.0, null, null, 80.0]
```

**.NET (Nivara)** — cross-column operations with automatic null propagation:

```csharp
var price = NivaraColumn<float>.CreateFromNullable(
    new float?[] { 10.0f, 20.0f, null, 40.0f });
var qty = NivaraColumn<float>.CreateFromNullable(
    new float?[] { 1.0f, null, 3.0f, 2.0f });

var total = price.Multiply(qty);
// [10.0, null, null, 80.0]
```

We're using Nivara here because columnar data means each typed column carries its own null mask independently, and cross-column operations propagate nulls automatically via mask-OR semantics. Nivara operations compose naturally — `price.Multiply(qty)` is one expression with automatic null propagation.

### Act 3: Tabular data with typed schema — NivaraFrame

> A NivaraFrame brings multiple named, typed columns together under a single
> schema. Columns are accessed by name, not position — and operations like
> filtering, computing derived columns, and projecting subsets compose
> naturally without manual index alignment across parallel arrays.

#### Product catalog with schema-aware operations

A product catalog has columns of different types — string SKUs, float prices,
int stock counts, string categories. You need to filter, derive a new column,
and select a subset for export.

**Python**

```python
import pandas as pd

products = pd.DataFrame({
    "SKU": ["A100", "A200", "A300", "A400", "A500"],
    "Price": [29.99, 49.99, 9.99, 199.99, 14.99],
    "Stock": [150, 0, 42, 8, 200],
    "Category": ["Electronics", "Books", "Electronics", "Home", "Books"],
    "Rating": [4.5, 3.8, 4.2, 4.9, 3.5],
})

# Filter: in-stock items
in_stock = products[products["Stock"] > 0]

# Compute: price after tax
in_stock["PriceWithTax"] = in_stock["Price"] * 1.08

# Project: select columns for export
result = in_stock[["SKU", "PriceWithTax", "Category", "Rating"]]
```

**.NET (BCL)**

```csharp
// Parallel arrays — one per column, manual index alignment
string[] sku = ["A100", "A200", "A300", "A400", "A500"];
float[] price = [29.99f, 49.99f, 9.99f, 199.99f, 14.99f];
int[] stock = [150, 0, 42, 8, 200];
string[] category = ["Electronics", "Books", "Electronics", "Home", "Books"];
float[] rating = [4.5f, 3.8f, 4.2f, 4.9f, 3.5f];

// Filter + compute + project — all manual, no schema
var result = new List<(string SKU, float PriceWithTax, string Category, float Rating)>();
for (int i = 0; i < stock.Length; i++)
{
    if (stock[i] > 0)
        result.Add((sku[i], price[i] * 1.08f, category[i], rating[i]));
}
```

**.NET (Nivara)** — typed schema with named columns:

```csharp
var products = NivaraFrame.Create(
    ("SKU", NivaraColumn<string>.CreateForReferenceType(
        ["A100", "A200", "A300", "A400", "A500"])),
    ("Price", NivaraColumn<float>.Create(
        [29.99f, 49.99f, 9.99f, 199.99f, 14.99f])),
    ("Stock", NivaraColumn<int>.Create(
        [150, 0, 42, 8, 200])),
    ("Category", NivaraColumn<string>.CreateForReferenceType(
        ["Electronics", "Books", "Electronics", "Home", "Books"])),
    ("Rating", NivaraColumn<float>.Create(
        [4.5f, 3.8f, 4.2f, 4.9f, 3.5f]))
);

// Schema is explicit — column names and types
Console.WriteLine($"Columns: {string.Join(", ", products.ColumnNames)}");
// SKU, Price, Stock, Category, Rating

// Filter: Stock > 0
var inStock = products.FilterByMask(
    products.GetColumn<int>("Stock").GreaterThan(0));

// Compute: PriceWithTax = Price * 1.08
var withTax = inStock.WithColumn(
    "PriceWithTax",
    inStock.GetColumn<float>("Price").Multiply(1.08f));

// Project: select output columns
var result = withTax.SelectColumns(
    "SKU", "PriceWithTax", "Category", "Rating");
```

We're using Nivara here because a NivaraFrame provides a unified typed schema
— each column is named and strongly typed, accessible by name (`"Price"`)
not array index (`arrays[1]`). Filtering, derived columns, and projection
compose without manual index alignment across parallel arrays. The schema is
self-documenting at the type level.

#### 3b. Column-wise operations vs row-wise LINQ

A common question is "why a Frame instead of `List<T>` with LINQ?" The answer
is column-wise vs row-wise semantics — Nivara operates on entire columns as
first-class values, not row-by-row anonymous projections.

**Python**

```python
import pandas as pd

customers = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "Diana"],
    "Spend": [1200.0, 300.0, 2500.0, 800.0],
    "Frequency": [15.0, 5.0, 30.0, 10.0],
})

customers["LoyaltyScore"] = customers["Spend"] * customers["Frequency"] / 100.0
top = customers.nlargest(2, "LoyaltyScore")[["Name", "LoyaltyScore"]]
```

**.NET (LINQ, row-wise)** — each row projected individually:

```csharp
var customers = new[]
{
    (Name: "Alice", Spend: 1200.0, Frequency: 15.0),
    (Name: "Bob", Spend: 300.0, Frequency: 5.0),
    (Name: "Charlie", Spend: 2500.0, Frequency: 30.0),
    (Name: "Diana", Spend: 800.0, Frequency: 10.0),
};

var top = customers
    .Select(c => new
    {
        c.Name,
        LoyaltyScore = c.Spend * c.Frequency / 100.0
    })
    .OrderByDescending(c => c.LoyaltyScore)
    .Take(2);
```

**.NET (Nivara, column-wise)** — columns are first-class values:

```csharp
var customers = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(
        ["Alice", "Bob", "Charlie", "Diana"])),
    ("Spend", NivaraColumn<float>.Create(
        [1200f, 300f, 2500f, 800f])),
    ("Frequency", NivaraColumn<float>.Create(
        [15f, 5f, 30f, 10f]))
);

// Column-wise: entire LoyaltyScore column computed in one pass
var loyalty = customers.GetColumn<float>("Spend")
    .Multiply(customers.GetColumn<float>("Frequency"))
    .Multiply(1f / 100f);

var top = customers
    .WithColumn("LoyaltyScore", loyalty)
    .OrderBy("LoyaltyScore", ascending: false)
    .Take(2)
    .SelectColumns("Name", "LoyaltyScore");
```

We're using Nivara here because columns are first-class values —
`spendCol.Multiply(freqCol)` operates on the entire column in a single
vectorized pass, not row-by-row. LINQ over `IEnumerable<T>` or `DbSet<T>`
projects every row individually — 100K rows means 100K anonymous object
allocations and 100K per-row arithmetic operations. A NivaraFrame also
supports dynamic schema, schema introspection, and tensor interop —
capabilities that LINQ-over-POCOs don't offer.

### Act 4: Full workflow — tabular data meets tensor math

> When you need both tabular data management (typed schema, null propagation,
> labeled rows) and tensor math (similarity scoring, ranking), Nivara and the
> BCL work together. The frame auto-discovers numeric columns and lays them
> out row-major as a 2D tensor; `TensorPrimitives` does the math.

#### Semantic Search Over Document Embeddings

Score a query embedding against a table of documents, then rank by similarity.
The embeddings are stored as individual float columns per dimension so the
frame can extract them as a dense 2D tensor in one call.

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

// Manual flat array — must track stride and row alignment
float[] flatEmbeddings =
[
    0.9f, 0.2f, 0.5f, 0.4f,
    0.1f, 0.9f, 0.2f, 0.7f,
    0.7f, 0.1f, 0.8f, 0.2f,
];
string[] docIds = ["doc-101", "doc-102", "doc-103"];
int dims = 4;

var scores = new float[docIds.Length];
for (int i = 0; i < docIds.Length; i++)
{
    var row = flatEmbeddings.AsSpan(i * dims, dims);
    scores[i] = TensorPrimitives.CosineSimilarity(row, query);
}

var ranking = docIds
    .Select((id, i) => (DocumentId: id, Score: scores[i]))
    .OrderByDescending(x => x.Score)
    .Take(2);
```

**.NET (Nivara)** — the frame handles schema and row-major layout:

```csharp
var query = new[] { 0.8f, 0.1f, 0.6f, 0.3f };

var documents = NivaraFrame.Create(
    ("DocumentId", NivaraColumn<string>.CreateForReferenceType(
        ["doc-101", "doc-102", "doc-103"])),
    ("e0", NivaraColumn<float>.Create([0.9f, 0.1f, 0.7f])),
    ("e1", NivaraColumn<float>.Create([0.2f, 0.9f, 0.1f])),
    ("e2", NivaraColumn<float>.Create([0.5f, 0.2f, 0.8f])),
    ("e3", NivaraColumn<float>.Create([0.4f, 0.7f, 0.2f]))
);

// Select embedding columns → single 2D tensor, row-major
var docVectors = documents
    .SelectColumns("e0", "e1", "e2", "e3")
    .ToTensor<float>();
// Shape: [rows=3, columns=4]

// TensorPrimitives does the math
var scores = new float[docVectors.Lengths[0]];
var dims = (int)docVectors.Lengths[1];
for (int i = 0; i < scores.Length; i++)
{
    var row = docVectors.AsSpan().Slice(i * dims, dims);
    scores[i] = TensorPrimitives.CosineSimilarity(row, query);
}

// Labels stay aligned via the frame
var ranking = documents.GetColumn<string>("DocumentId")
    .Select((id, i) => (Label: id, Score: scores[i]))
    .OrderByDescending(x => x.Score)
    .Take(2);
```

We're using Nivara here because the frame auto-discovers numeric columns,
selects them by name, and lays them out row-major as a 2D `Tensor<float>`
— replacing manual flat-array construction with stride calculations. The
schema keeps DocumentId aligned with embedding rows end-to-end. The math
stays with `TensorPrimitives.CosineSimilarity`.

### Act 5: Experimental — AutoDiff over columns

> For gradient-based optimization, Nivara.Extensions provides the only
> reverse-mode AutoDiff that works directly with Nivara column data — building
> a computation graph, computing gradients via `Backward()`, and applying
> updates via `SgdOptimizer.SgdUpdate`.

#### Automatic Differentiation Over Columns (Experimental)

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

The BCL covers forward inference — `TensorPrimitives` (Dot, Norm, SumOfSquares) can express the math. But there is no BCL autograd, so backpropagation requires manual gradient derivation and loops.

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
- Does the progressive act structure (BCL-only → null-aware columns → frames → full tabular+tensor workflow → AutoDiff) make it clear when and why to reach for Nivara?
- Does backing off from tensor math and positioning Nivara as a tabular data layer feel like the right call?
- Which tabular features add the most value: typed columns, null semantics, schemas, labeled ranking, or file I/O?
- Is the experimental AutoDiff layer worth keeping as a gradient bridge for Nivara column data?
- What interop scenarios are missing (Microsoft.Extensions.AI, VectorData, Arrow zero-copy, etc.)?

## Related docs

- [`docs/TENSORS.md`](docs/TENSORS.md) — full positioning discussion
- [`docs/AUTODIFF-GAPS.md`](docs/AUTODIFF-GAPS.md) — AutoDiff roadmap and gaps
