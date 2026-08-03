# NIVARA and Tensors/AI primitives

We are **not** positioning Nivara as:

```text
NumPy for .NET
Tensor library
Vector math library
Embedding similarity engine
AutoDiff framework
```

We are positioning it as:

```text
A typed, immutable, null-aware DataFrame/query layer for .NET
with clean interop to BCL tensors, Microsoft.Extensions.AI, VectorData, Arrow, CSV, JSON, and Parquet.
```

.NET already has `System.Numerics.Tensors` for tensor operations, including SIMD-accelerated primitives, and `Microsoft.Extensions.AI` defines embedding abstractions like `IEmbeddingGenerator` / `Embedding`. Nivara should integrate with those instead of competing with them. ([NuGet][1])

The examples should be rewritten to say this explicitly:

```text
Use Nivara for tabular data:
- typed columns
- null masks
- schema validation
- query planning
- joins
- grouping
- file I/O
- labels / row identity

Use .NET / Microsoft libraries for numerical and AI primitives:
- Tensor<T>
- TensorPrimitives
- ReadOnlySpan<float>
- VectorData
- IEmbeddingGenerator
```

A good example shape would be:

```csharp
// Nivara owns the table.
using var products = NivaraFrame.Create(
    ("ProductId", NivaraColumn<string>.CreateForReferenceType(["A", "B", "C"])),
    ("Embedding", NivaraColumn<float[]>.CreateForReferenceType([
        [0.9f, 0.2f, 0.5f, 0.4f],
        [0.1f, 0.9f, 0.2f, 0.7f],
        [0.7f, 0.1f, 0.8f, 0.2f],
    ]))
);

// BCL owns the math.
float[] query = [0.8f, 0.1f, 0.6f, 0.3f];

var ids = products.GetColumn<string>("ProductId");
var embeddings = products.GetColumn<float[]>("Embedding");

var ranked = Enumerable.Range(0, products.RowCount)
    .Select(i => new
    {
        ProductId = ids[i],
        Score = TensorPrimitives.CosineSimilarity(embeddings[i], query)
    })
    .OrderByDescending(x => x.Score)
    .Take(3);
```

That is much healthier than the removed frame tensor-axis API:

```csharp
// deleted in the AutoDiff refactor (Task 10) — use TensorPrimitives on spans
products.CosineSimilarity(query)
```

because it avoids pretending that dataframe columns are tensor axes.

We should update the docs with a principle like this:

```text
Nivara does not try to replace System.Numerics.Tensors or Microsoft.Extensions.AI.
For tensor math, vector operations, embeddings, and model-facing APIs, prefer the .NET platform libraries.
Nivara focuses on typed, null-aware, immutable tabular data and provides interop points to platform primitives.
```

The deprecated tensor APIs on `NivaraFrame`/`NivaraSeries`
(`Dot`, `CosineSimilarity`, `ColumnNorms`, `RowNorms`, `DotProduct`, `Norm`)
have been **removed** (Task 10 of the AutoDiff refactor) rather than moved to a
separate namespace — they had no production callers, and the platform
`TensorPrimitives` kernels are the sanctioned replacement. Column math is done
on spans via `TryGetSpan`; row-major operations assemble spans with
`CopyToRowMajor`.

Core Nivara should stay boring and strong:

```text
DataFrame
Column
Schema
Null semantics
Query
Join
GroupBy
I/O
Interop
```

That is the maintainable path.

# NIVARA Possible Future Strategy

The question is no longer:

> "Can we build NumPy for C#?"

The question is:

> "What does Nivara own that .NET 10 does not?"

That's a much healthier framing.

---

## The danger

The danger is trying to compete with the platform.

For example:

```csharp
NivaraSeries<float>.DotProduct(...)
```

is increasingly becoming:

```csharp
TensorPrimitives.Dot(...)
```

with extra wrappers.

That's technical debt.

---

## Option A: Double Down on Columnar Analytics

This is the option we personally find strongest.

Become:

```text
Arrow-inspired analytical dataframe
for .NET
```

instead of:

```text
NumPy for .NET
```

Focus on:

```text
Columns
Schemas
Nulls
Joins
Grouping
Query Planning
Lazy Execution
Arrow
Parquet
CSV
JSON
```

Treat tensors as interop.

Example:

```csharp
Tensor<float> tensor =
    frame["Embedding"]
        .AsTensor();
```

and stop there.

### The zero-copy `AsTensor()` surface

The one tensor interop surface core Nivara exposes is a **lazy zero-copy
`Tensor<T>` view**: internal `ColumnStorage<T>.AsTensor()` wraps the storage's
sole-owner `T[]` with `Tensor.Create(data, [length])` — no copy, no flattened
cache (unmanaged `T` only; `Half`/`BFloat16` pass, reference types throw).
`NivaraColumn<T>.AsTensorView()` and `NivaraSeries<T>.AsTensorView()` are the
**public** guarded entries, and `GradTensor<T>.AsTensor()` is the public
AutoDiff accessor. This is the intended way to hand a dense, non-null,
contiguous column to `System.Numerics.Tensors` consumers. The view shares the
backing array, so callers must not mutate it. Reserve `Tensor.FlattenTo` for
non-contiguous/multi-dim tensors — a contiguous column already *is* its
`Tensor<T>`.

Columns otherwise interop with BCL tensors element-wise via
`TensorInteropExtensions` (`Series`/`Frame` ↔ `Tensor<T>`), never by pretending
columnar ops are tensor ops.

No custom tensor ecosystem.

No tensor operators.

No tensor hierarchy.

No tensor math duplication.

---

## Option B: Become "Polars for .NET"

The getting-started guide is accidentally drifting here already.

We see:

```csharp
Lazy queries
Optimization
Predicate pushdown
Projection pushdown
Execution engine
Streaming
```

Those are Polars ideas.

Not NumPy ideas.

The roadmap could become:

```text
Nivara = Polars-style engine
for .NET developers
```

That's a much larger vision than:

```text
Tensor wrappers
```

---

## Option C: AI Data Infrastructure

This is the only AI direction we'd consider.

Not:

```text
Tensor library
```

but:

```text
Structured AI datasets
Embedding columns
Vector search
RAG data preparation
Feature engineering
Evaluation datasets
```

Example:

```csharp
frame
    .GenerateEmbeddings(...)
    .StoreVectors(...)
    .ChunkDocuments(...)
```

This complements Microsoft's AI stack instead of competing with it.

---

## What we should probably remove

If we inherited the project tomorrow, we'd put these on the chopping block:

```text
Custom tensor abstractions
Tensor helper duplication
Tensor math duplication
Vector math wrappers
```

because .NET 10 is already doing that work.

---

## What we should invest in

We'd put almost all energy into:

```text
Columnar engine
Schema system
Arrow interoperability
Parquet
Lazy query execution
GroupBy/Join performance
Null semantics
Expression engine
```

because those are the pieces where Microsoft has not yet provided a first-class answer.

The biggest insight from the getting-started document is that Nivara already looks much more like a **columnar analytics engine with some tensor features attached** than a **tensor library with dataframe features attached**.

If that's true, then the tensor surface may actually be distracting maintainers from the part of the project that has the best chance of becoming strategically relevant.
