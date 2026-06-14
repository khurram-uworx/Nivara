# Nivara Examples

## Positioning

Nivara is **not** a NumPy-for-.NET, tensor library, vector math library, embedding similarity engine, or general-purpose AutoDiff framework.

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

#### 2c. Null-preserving tensor export

**Nivara** — data plus explicit mask:
```csharp
using Nivara.Tensors;

var score = NivaraColumn<float>.CreateFromNullable(new float?[] { 0.9f, null, 0.4f });
NullableTensor<float> tensor = score.ToNullableTensor();

// Tensor data is available for platform APIs; the mask remains authoritative.
var isMissing = tensor.NullMask!.AsTensorSpan()[1]; // true
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

**Nivara** — frame handles labels, schema, and row-major layout:
```csharp
var query = new[] { 0.8f, 0.1f, 0.6f, 0.3f };

var documents = NivaraFrame.FromRows(
    [
        ("doc-101", new[] { 0.9f, 0.2f, 0.5f, 0.4f }),
        ("doc-102", new[] { 0.1f, 0.9f, 0.2f, 0.7f }),
        ("doc-103", new[] { 0.7f, 0.1f, 0.8f, 0.2f })
    ],
    columnNames: ["e0", "e1", "e2", "e3"],
    labelColumnName: "DocumentId"
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
var vectors = NivaraFrame.FromRows(
    [
        ("C001", new[] { 0.9f, 0.2f, 0.5f, 0.4f }),
        ("C002", new[] { 0.1f, 0.9f, 0.2f, 0.7f }),
        ("C003", new[] { 0.7f, 0.1f, 0.8f, 0.2f })
    ],
    columnNames: ["e0", "e1", "e2", "e3"],
    labelColumnName: "CustomerId");

var customers = vectors.WithColumn(
    "Segment",
    NivaraColumn<string>.CreateForReferenceType(["standard", "vip", "standard"]));

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

### Act 7a: AutoDiff — low-level gradient operations

> Reverse-mode AutoDiff that works directly with Nivara column data. Build a computation graph, compute gradients via `Backward()`, apply updates via `SGD<T>.SgdUpdate`.

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

**Nivara** — reverse-mode AutoDiff with gradient tracking (operator overloads for `+`, `-`, `*`):
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

var prediction = w * x + b;
var loss = GradOperations.Mean((prediction - y) * (prediction - y));

loss.Backward();

using var updatedW = SGD<float>.SgdUpdate(w, 0.01f);
using var updatedB = SGD<float>.SgdUpdate(b, 0.01f);
```

See the full sample in [`samples/Nivara.SampleApp/AutoDiffExample.cs`](samples/Nivara.SampleApp/AutoDiffExample.cs).

---

### Act 7b: AutoDiff — module-based model with TrainingLoop

> Build models using `Module<T>` / `Linear<T>`, train with `TrainingLoop<T>` — no manual graph construction or gradient management needed.

**Python**
```python
# Not shown — the pattern is the same as Act 7a, but using nn.Module
```

**Nivara** — declare model architecture as a class and train with `TrainingLoop`:
```csharp
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Training;

class LinearModel : Module<float>
{
    Linear<float> L1;

    public LinearModel()
    {
        L1 = new Linear<float>(3, 1);
        RegisterModules(L1);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> x)
        => L1.Forward(x);
}

var frame = NivaraFrame.Create(
    ("x0", NivaraColumn<float>.Create([1.0f, 2.0f, 3.0f, 4.0f])),
    ("x1", NivaraColumn<float>.Create([2.0f, 3.0f, 4.0f, 5.0f])),
    ("x2", NivaraColumn<float>.Create([3.0f, 4.0f, 5.0f, 6.0f])),
    ("y", NivaraColumn<float>.Create([6.0f, 9.0f, 12.0f, 15.0f]))
);

var loader = new DataLoader<float>(
    new TensorDataset<float>(frame, ["x0", "x1", "x2"], "y"),
    batchSize: 2, shuffle: false);

var model = new LinearModel();
var optimizer = new SGD<float>(lr: 0.01f);
optimizer.AddParameterGroup(model.GetParameters().Values, learningRate: 0.01f);

var loop = new TrainingLoop<float>(
    model, loader,
    (pred, target) => new MSELoss<float>().Forward(pred, target),
    optimizer,
    epochs: 5);

var result = loop.Run();
result.PrintSummary();
// Epoch   1 | Loss:     --- | Batches:  2 | Time: 0.02s
// Epoch   5 | Loss:     --- | Batches:  2 | Time: 0.02s
```

**What changed from Act 7a:**
- No raw tensor creation — `Linear<T>` registers weight/bias as parameters automatically
- No manual forward pass with `GradOperations.Add/Multiply/Subtract` — just `L1.Forward(x)`
- No manual `Backward()` + `SgdUpdate` — `TrainingLoop` handles forward/backward/step/zero-grad
- No `using` statements per tensor — `Module<T>` implements `IDisposable` and disposes child modules

---

### Act 8: Production-style training pipeline

> Load real data, train with data-parallel multi-core execution, save the model, and run inference.

#### Fraud detection classifier

**Nivara**
```csharp
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Training;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Optimizer;

// 1. Define model
class FraudNet : Module<float>
{
    Linear<float> L1, L2, L3;

    public FraudNet()
    {
        L1 = new Linear<float>(8, 64);
        L2 = new Linear<float>(64, 32);
        L3 = new Linear<float>(32, 1);
        RegisterModules(L1, L2, L3);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> x)
    {
        var h = GradOperations.Relu(L1.Forward(x));
        h = GradOperations.Relu(L2.Forward(h));
        return L3.Forward(h);
    }
}

// 2. Load data (from CSV or in-memory)
var frame = NivaraFrame.Create(
    ("amount", NivaraColumn<float>.Create([100.0f, 5000.0f, 50.0f, 20000.0f, 75.0f])),
    ("hour", NivaraColumn<float>.Create([14, 2, 10, 3, 18])),
    ("distance", NivaraColumn<float>.Create([5.0f, 300.0f, 2.0f, 500.0f, 1.0f])),
    ("prev_attempts", NivaraColumn<float>.Create([0, 3, 0, 5, 1])),
    ("country_change", NivaraColumn<float>.Create([0, 1, 0, 1, 0])),
    ("device_new", NivaraColumn<float>.Create([0, 1, 0, 1, 0])),
    ("amount_ratio", NivaraColumn<float>.Create([1.0f, 10.0f, 0.5f, 20.0f, 0.8f])),
    ("velocity", NivaraColumn<float>.Create([0.0f, 4.0f, 0.0f, 6.0f, 0.0f])),
    ("is_fraud", NivaraColumn<float>.Create([0.0f, 1.0f, 0.0f, 1.0f, 0.0f]))
);

var featureCols = new[] { "amount", "hour", "distance", "prev_attempts",
                          "country_change", "device_new", "amount_ratio", "velocity" };
var loader = new DataLoader<float>(
    new TensorDataset<float>(frame, featureCols, "is_fraud"),
    batchSize: 2, shuffle: true);

// 3. Train with data parallelism (uses all cores)
var model = new FraudNet();
var optimizer = new Adam<float>(beta1: 0.9, beta2: 0.999);

// Add all model parameters to the optimizer
optimizer.AddParameterGroup(model.GetParameters().Values, learningRate: 0.001f);

var trainer = new DataParallelTrainer<float>(
    model, loader,
    (pred, target) => new BCEWithLogitsLoss<float>().Forward(pred, target),
    optimizer,
    epochs: 10);

var result = trainer.Run();
result.PrintSummary();
// Epoch   1 | Loss:   0.693100 | Workers:  8 | Chunks:    3 | Grad Norm:   0.542100 | Time: 0.15s
// Epoch  10 | Loss:   0.082300 | Workers:  8 | Chunks:    3 | Grad Norm:   0.012100 | Time: 0.12s

// 4. Save trained model
ModelSerializer.Save(model, "fraud_model.json");

// 5. Inference on new data
var loaded = new FraudNet();
ModelSerializer.Load(loaded, "fraud_model.json");
loaded.Eval();

var newTx = ReverseGradTensor<float>.FromMatrix(
    new float[] { 250.0f, 22, 10.0f, 0, 0, 0, 1.2f, 0.0f },
    rows: 1, cols: 8, requiresGrad: false);

var score = loaded.Forward(newTx);
// score[0] ≈ logit; apply sigmoid for probability:
var prob = 1.0f / (1.0f + MathF.Exp(-score[0]));
Console.WriteLine($"Fraud probability: {prob:P2}");
```

**What the pipeline does:**
| Step | Component |
|------|-----------|
| Model declaration | `FraudNet : Module<float>` with 3 `Linear` layers |
| Data packaging | `TensorDataset` → `DataLoader` with shuffle |
| Multi-core training | `DataParallelTrainer` splits rows across cores via `Parallel.For` |
| Loss function | `BCEWithLogitsLoss` — numerically stable binary classification |
| Optimizer | `Adam` with bias-corrected adaptive learning rates |
| Model persistence | `ModelSerializer.Save` / `ModelSerializer.Load` (JSON + base64 binary) |
| Inference | `Eval()` mode, no gradient tracking |

---

## Related docs

- [`docs/AUTODIFF-GAPS.md`](docs/AUTODIFF-GAPS.md) — AutoDiff gaps
