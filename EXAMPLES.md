# Nivara Examples

## Positioning

Nivara is **not** a NumPy-for-.NET, tensor library, vector math library, embedding similarity engine, or AutoDiff framework.

It is:

> A typed, immutable, null-aware DataFrame/query layer for .NET
> with clean interop to the Base Class Library's (BCL) tensors,
> Microsoft.Extensions.AI, VectorData, Arrow, CSV, JSON, and Parquet.

.NET already has `System.Numerics.Tensors` (`TensorPrimitives`, `Tensor<T>`) for SIMD math and n-dimensional arrays, and `Microsoft.Extensions.AI` for embedding abstractions. Nivara integrates with those instead of competing with them.

| Use .NET / Microsoft libraries for | Use Nivara for |
|---|---|
| `Tensor<T>`, `TensorPrimitives` | Typed columns, schemas |
| `ReadOnlySpan<float>`, `Vector<T>` | Null masks, null propagation |
| `IEmbeddingGenerator`, `VectorData` | Query planning, joins, grouping |
| Math kernels (Dot, Norm, CosineSimilarity) | Lazy file I/O (CSV, JSON, Parquet, Arrow) |
| AI abstractions | Labeling, row identity, schema validation |

Each scenario shows Python (NumPy/pandas/PyTorch) and Nivara code side-by-side. All Nivara examples assume `using Nivara;`.

---

### Act 1: When pure tensor math is enough — BCL handles it

> `TensorPrimitives` (part of .NET's Base Class Library) provides SIMD-accelerated operators — Dot, Norm, CosineSimilarity, and many more — for float/double. When your data is already in flat arrays with no null-handling or schema needs, you don't need Nivara at all.

#### Embedding Similarity Ranking

**Python**
```python
query = np.array([0.8, 0.1, 0.6, 0.3], dtype=np.float32)
products = {"A": np.array([0.9, 0.2, 0.5, 0.4], dtype=np.float32),
            "B": np.array([0.1, 0.9, 0.2, 0.7], dtype=np.float32),
            "C": np.array([0.7, 0.1, 0.8, 0.2], dtype=np.float32)}
scores = {n: np.dot(v, query) / (np.linalg.norm(v) * np.linalg.norm(query))
          for n, v in products.items()}
top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
```

**.NET (BCL)**
```csharp
float[] query = [0.8f, 0.1f, 0.6f, 0.3f];
var products = new Dictionary<string, float[]>
{
    ["A"] = [0.9f, 0.2f, 0.5f, 0.4f],
    ["B"] = [0.1f, 0.9f, 0.2f, 0.7f],
    ["C"] = [0.7f, 0.1f, 0.8f, 0.2f],
};
var top = products
    .Select(p => new { p.Key, Score = TensorPrimitives.CosineSimilarity(p.Value, query) })
    .OrderByDescending(x => x.Score).Take(3);
```

---

### Act 2: Nulls are where BCL gets awkward — enter NivaraColumn

> Nivara columns carry explicit null masks that propagate automatically through arithmetic operations — no manual `HasValue` checks.

#### 2a. Null propagation in element-wise addition

**Python**
```python
df = pd.DataFrame({"A": [1.0, np.nan, 3.0, np.nan], "B": [np.nan, 2.0, 3.0, np.nan]})
result = df["A"] + df["B"]
# [NaN, NaN, 6.0, NaN]
```

**Nivara** — automatic null mask propagation:
```csharp
var colA = NivaraColumn<float>.CreateFromNullable(new float?[] { 1.0f, null, 3.0f, null });
var colB = NivaraColumn<float>.CreateFromNullable(new float?[] { null, 2.0f, 3.0f, null });
var result = colA.Add(colB);
// result.IsNull(0) → true, result[2] → 6.0f, result.IsNull(3) → true
```

#### 2b. Derived column with cross-column nulls

**Python**
```python
df = pd.DataFrame({"price": [10.0, 20.0, np.nan, 40.0], "qty": [1.0, np.nan, 3.0, 2.0]})
df["total"] = df["price"] * df["qty"]
# [10.0, NaN, NaN, 80.0]
```

**Nivara** — mask-OR semantics:
```csharp
var price = NivaraColumn<float>.CreateFromNullable(new float?[] { 10.0f, 20.0f, null, 40.0f });
var qty = NivaraColumn<float>.CreateFromNullable(new float?[] { 1.0f, null, 3.0f, 2.0f });
var total = price.Multiply(qty);
// [10.0, null, null, 80.0]
```

---

### Act 3: Tabular data with typed schema — NivaraFrame

> A NivaraFrame brings multiple named, typed columns under a single schema. Filtering, derived columns, and projection compose without manual index alignment across parallel arrays.

#### 3a. Product catalog with schema-aware operations

**Python**
```python
products = pd.DataFrame({
    "SKU": ["A100", "A200", "A300", "A400", "A500"],
    "Price": [29.99, 49.99, 9.99, 199.99, 14.99],
    "Stock": [150, 0, 42, 8, 200],
    "Category": ["Electronics", "Books", "Electronics", "Home", "Books"],
    "Rating": [4.5, 3.8, 4.2, 4.9, 3.5],
})
in_stock = products[products["Stock"] > 0]
in_stock["PriceWithTax"] = in_stock["Price"] * 1.08
result = in_stock[["SKU", "PriceWithTax", "Category", "Rating"]]
```

**Nivara** — typed schema with named columns:
```csharp
var products = NivaraFrame.Create(
    ("SKU", NivaraColumn<string>.CreateForReferenceType(["A100", "A200", "A300", "A400", "A500"])),
    ("Price", NivaraColumn<float>.Create([29.99f, 49.99f, 9.99f, 199.99f, 14.99f])),
    ("Stock", NivaraColumn<int>.Create([150, 0, 42, 8, 200])),
    ("Category", NivaraColumn<string>.CreateForReferenceType(["Electronics", "Books", "Electronics", "Home", "Books"])),
    ("Rating", NivaraColumn<float>.Create([4.5f, 3.8f, 4.2f, 4.9f, 3.5f]))
);

var result = products
    .FilterByMask(products.GetColumn<int>("Stock").GreaterThan(0))
    .WithColumn("PriceWithTax", products.GetColumn<float>("Price").Multiply(1.08f))
    .SelectColumns("SKU", "PriceWithTax", "Category", "Rating");
```

#### 3b. Column-wise vs row-wise

Column-wise operates on entire columns in one vectorized pass, not row-by-row.

**Python**
```python
customers = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "Diana"],
    "Spend": [1200.0, 300.0, 2500.0, 800.0],
    "Frequency": [15.0, 5.0, 30.0, 10.0],
})
customers["LoyaltyScore"] = customers["Spend"] * customers["Frequency"] / 100.0
top = customers.nlargest(2, "LoyaltyScore")[["Name", "LoyaltyScore"]]
```

**Nivara** — column-wise, single vectorized pass:
```csharp
var customers = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(["Alice", "Bob", "Charlie", "Diana"])),
    ("Spend", NivaraColumn<float>.Create([1200f, 300f, 2500f, 800f])),
    ("Frequency", NivaraColumn<float>.Create([15f, 5f, 30f, 10f]))
);

var top = customers
    .WithColumn("LoyaltyScore",
        customers.GetColumn<float>("Spend")
            .Multiply(customers.GetColumn<float>("Frequency"))
            .Multiply(1f / 100f))
    .OrderBy("LoyaltyScore", ascending: false)
    .Take(2)
    .SelectColumns("Name", "LoyaltyScore");
```

---

### Act 4: Full workflow — tabular data meets tensor math

> The frame auto-discovers numeric columns and lays them out row-major as a 2D tensor; `TensorPrimitives` does the math.

#### Semantic Search Over Document Embeddings

**Python**
```python
query = np.array([0.8, 0.1, 0.6, 0.3], dtype=np.float32)
documents = pd.DataFrame({
    "DocumentId": ["doc-101", "doc-102", "doc-103"],
    "e0": [0.9, 0.1, 0.7], "e1": [0.2, 0.9, 0.1],
    "e2": [0.5, 0.2, 0.8], "e3": [0.4, 0.7, 0.2],
})
doc_vectors = documents[["e0", "e1", "e2", "e3"]].to_numpy(dtype=np.float32)
scores = doc_vectors @ query / (np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query))
ranking = sorted(zip(documents["DocumentId"], scores), key=lambda x: x[1], reverse=True)[:2]
```

**Nivara** — frame handles schema and row-major layout:
```csharp
var query = new[] { 0.8f, 0.1f, 0.6f, 0.3f };

var documents = NivaraFrame.Create(
    ("DocumentId", NivaraColumn<string>.CreateForReferenceType(["doc-101", "doc-102", "doc-103"])),
    ("e0", NivaraColumn<float>.Create([0.9f, 0.1f, 0.7f])),
    ("e1", NivaraColumn<float>.Create([0.2f, 0.9f, 0.1f])),
    ("e2", NivaraColumn<float>.Create([0.5f, 0.2f, 0.8f])),
    ("e3", NivaraColumn<float>.Create([0.4f, 0.7f, 0.2f]))
);

var docVectors = documents.SelectColumns("e0", "e1", "e2", "e3").ToTensor<float>();

var scores = new float[docVectors.Lengths[0]];
var dims = (int)docVectors.Lengths[1];
for (int i = 0; i < scores.Length; i++)
    scores[i] = TensorPrimitives.CosineSimilarity(docVectors.AsSpan().Slice(i * dims, dims), query);

var ranking = documents.GetColumn<string>("DocumentId")
    .Select((id, i) => (Label: id, Score: scores[i]))
    .OrderByDescending(x => x.Score).Take(2);
```

---

### Act 5: Querying with LINQ — The QueryFrame

> Column-wise query pipeline with deferred execution and plan inspection. Familiar `Where().OrderBy().Select()` syntax — but operates on columns, builds a plan you can inspect before touching data.

#### 5a. Employee Directory

Filter active Engineering employees, sort by salary, project name + salary.

**Python**
```python
employees = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "Diana"],
    "Department": ["Engineering", "Sales", "Engineering", "Engineering"],
    "Salary": [120000, 90000, 150000, 110000],
    "IsActive": [True, True, False, True],
})
result = employees[
    (employees["Department"] == "Engineering") & (employees["IsActive"])
].sort_values("Salary")[["Name", "Salary"]]
```

**Nivara** — lazy plan with ExplainPlan(), schema errors caught at plan time:
```csharp
var employees = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(["Alice", "Bob", "Charlie", "Diana"])),
    ("Department", NivaraColumn<string>.CreateForReferenceType(["Engineering", "Sales", "Engineering", "Engineering"])),
    ("Salary", NivaraColumn<int>.Create([120000, 90000, 150000, 110000])),
    ("IsActive", NivaraColumn<bool>.Create([true, true, false, true]))
);

var query = employees.AsQueryFrame()
    .Where(x => x["Department"] == "Engineering" && x["IsActive"])
    .OrderBy(x => x["Salary"])
    .Select(x => x["Name"], x => x["Salary"]);

Console.WriteLine(query.ExplainPlan());
// Schema errors surface here, not at runtime
var result = query.ToNivaraFrame();
```

Sample `ExplainPlan()` output:
```
Query Execution Plan:
├─ Source: MemoryQuerySource (Name, Department, Salary, IsActive)
├─ Operations:
│  ├─ 1. Filter → Schema unchanged
│  ├─ 2. Sort   → Schema unchanged
│  └─ 3. Select → Name (String), Salary (Int32)
└─ Result Schema: Name (String), Salary (Int32)
```

#### 5b. Plan-time validation catches typos

**Nivara**
```csharp
var query = employees.AsQueryFrame().Where(x => x["Deprtment"] == "Engineering");
Console.WriteLine(query.ExplainPlan());
// SchemaValidationException: Column 'Deprtment' not found. Available: Name, Department, Salary, IsActive
```

---

### Act 6: Scale — Query Engine and Execution Strategies

> Swap execution strategies (Lazy, Parallel, Streaming) without changing the query. Same pipeline, different throughput/memory trade-offs.

#### Risk scoring pipeline

Score customers against a risk embedding, filter out VIPs, rank by risk, select top accounts.

**Python**
```python
customers = pd.read_csv("customers.csv")
risk_pattern = np.array([0.8, 0.1, 0.6, 0.3], dtype=np.float32)
active = customers[customers["Segment"] != "vip"]
embeddings = active[["e0", "e1", "e2", "e3"]].to_numpy(dtype=np.float32)
scores = embeddings @ risk_pattern / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(risk_pattern))
active["RiskScore"] = scores
top = active.nlargest(100, "RiskScore")[["CustomerId", "RiskScore"]]
```

**Nivara** — build once, swap strategies:
```csharp
var customers = NivaraFrame.Create(
    ("CustomerId", NivaraColumn<string>.CreateForReferenceType(["C001", "C002", "C003"])),
    ("Segment", NivaraColumn<string>.CreateForReferenceType(["standard", "vip", "standard"])),
    ("e0", NivaraColumn<float>.Create([0.9f, 0.1f, 0.7f])),
    ("e1", NivaraColumn<float>.Create([0.2f, 0.9f, 0.1f])),
    ("e2", NivaraColumn<float>.Create([0.5f, 0.2f, 0.8f])),
    ("e3", NivaraColumn<float>.Create([0.4f, 0.7f, 0.2f]))
);

var embeddings = customers.SelectColumns("e0", "e1", "e2", "e3").ToTensor<float>();

float[] riskPattern = [0.8f, 0.1f, 0.6f, 0.3f];
var riskScores = new float[customers.RowCount];
for (int i = 0; i < customers.RowCount; i++)
    riskScores[i] = TensorPrimitives.CosineSimilarity(embeddings.AsSpan().Slice(i * 4, 4), riskPattern);

var withScores = customers.WithColumn("RiskScore", NivaraColumn<float>.Create(riskScores));

var query = withScores.AsQueryFrame()
    .Where(x => x["Segment"] != "vip")
    .OrderByDescending(x => x["RiskScore"])
    .Select(x => x["CustomerId"], x => x["RiskScore"]);

// Execute with different strategies
var plan = query.ToQueryPlan();
var engine = new ExecutionEngine();

var lazyResult = engine.Execute(plan);

var parallelCtx = new NivaraExecutionContext(ExecutionStrategy.Parallel)
{
    MaxDegreeOfParallelism = Environment.ProcessorCount,
    ExecutionDiagnostics = new ExecutionDiagnostics()
};
var parallelResult = engine.Execute(plan, parallelCtx);

Console.WriteLine(engine.LastDiagnostics?.GenerateReport());
// Strategy: Parallel | Elapsed: 142ms | Parallelism: 8 threads
```

---

### Act 7: AutoDiff over columns

> Gradient-based optimization with reverse-mode AutoDiff that works directly with Nivara column data. Build a computation graph, compute gradients via `Backward()`, apply updates via `SgdOptimizer.SgdUpdate`.

#### Linear model with gradient descent

**Python**
```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([2.0, 4.0, 6.0])
w = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
b = torch.tensor([0.1, 0.1, 0.1], requires_grad=True)
prediction = w * x + b
loss = torch.mean((prediction - y) ** 2)
loss.backward()
with torch.no_grad():
    updated_w = w - 0.01 * w.grad
    updated_b = b - 0.01 * b.grad
```

**Nivara** — reverse-mode AutoDiff with gradient tracking:
```csharp
using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Optimizer;

var df = NivaraFrame.Create(
    ("x", NivaraColumn<float>.Create([1.0f, 2.0f, 3.0f])),
    ("y", NivaraColumn<float>.Create([2.0f, 4.0f, 6.0f]))
);

var tensors = df.ToReverseGradTensors<float>(["x", "y"], requiresGrad: false);
var (x, y) = (tensors["x"], tensors["y"]);

using var w = ReverseGradTensor<float>.FromArray([0.5f, 0.5f, 0.5f], requiresGrad: true);
using var b = ReverseGradTensor<float>.FromArray([0.1f, 0.1f, 0.1f], requiresGrad: true);

var prediction = GradOperations.Add(GradOperations.Multiply(w, x), b);
var loss = GradOperations.Mean(
    GradOperations.Multiply(
        GradOperations.Subtract(prediction, y),
        GradOperations.Subtract(prediction, y)));

loss.Backward();

using var updatedW = SgdOptimizer.SgdUpdate(w, 0.01f);
using var updatedB = SgdOptimizer.SgdUpdate(b, 0.01f);
```

See the full sample in [`samples/Nivara.SampleApp/AutoDiffExample.cs`](samples/Nivara.SampleApp/AutoDiffExample.cs) and the current roadmap in [`docs/AUTODIFF-PLAN.md`](docs/AUTODIFF-PLAN.md).

---

## Related docs

- [`docs/TENSORS.md`](docs/TENSORS.md) — full positioning discussion
- [`docs/AUTODIFF-PLAN.md`](docs/AUTODIFF-PLAN.md) — AutoDiff roadmap
