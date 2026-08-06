# NIVARA and Tensors/AI primitives

## Positioning

Nivara is **not** positioning itself as:

```text
NumPy for .NET
Tensor library
Vector math library
Embedding similarity engine
AutoDiff framework
```

Nivara is positioning itself as:

```text
A typed, immutable, null-aware DataFrame/query layer for .NET
with clean interop to BCL tensors, Microsoft.Extensions.AI, VectorData, Arrow, CSV, JSON, and Parquet.
```

.NET already owns the numerical and AI primitives:

- `System.Numerics.Tensors` for tensor operations — stable in .NET 10, with SIMD-accelerated
  `TensorPrimitives` kernels and a multi-dimensional `Tensor<T>` ([package][1], [what's new][2]).
- `Microsoft.Extensions.AI` for embedding abstractions like `IEmbeddingGenerator` / `Embedding`.

Nivara integrates with those instead of competing with them.

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

A good example shape:

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

That is the deliberate shape: Nivara owns the table, the BCL owns the math. We do not
pretend dataframe columns are tensor axes.

The principle:

```text
Nivara does not try to replace System.Numerics.Tensors or Microsoft.Extensions.AI.
For tensor math, vector operations, embeddings, and model-facing APIs, prefer the .NET platform libraries.
Nivara focuses on typed, null-aware, immutable tabular data and provides interop points to platform primitives.
```

Core Nivara stays boring and strong:

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

---

## What Nivara actually exposes

### The zero-copy `AsTensorView()` surface

The one tensor interop surface core Nivara exposes is a **lazy zero-copy `Tensor<T>` view**:

- `ColumnStorage<T>.AsTensor()` — internal; wraps the storage's sole-owner `T[]` with
  `Tensor.Create(data, [length])` (slices use `Tensor.Create(data, start, lengths, strides)`),
  no copy, no flattened cache. Unmanaged `T` only
  (`RuntimeHelpers.IsReferenceOrContainsReferences<T>()` guard); `Half` passes, reference types throw.
- `NivaraColumn<T>.AsTensorView()` and `NivaraSeries<T>.AsTensorView()` — **public** guarded entries
  (throw on null-containing columns or reference element types).
- `GradTensor<T>.AsTensor()` — public AutoDiff accessor.

This is the intended way to hand a dense, non-null, contiguous column to
`System.Numerics.Tensors` consumers. The view shares the backing array, so callers must not
mutate it. Reserve `Tensor.FlattenTo` for non-contiguous/multi-dim tensors — a contiguous column
already *is* its `Tensor<T>`.

Columns otherwise interop with BCL tensors element-wise via `TensorInteropExtensions`
(`Series`/`Frame` ↔ `Tensor<T>`), never by pretending columnar ops are tensor ops.

### What was removed, and why

The deprecated tensor APIs on `NivaraFrame`/`NivaraSeries`
(`Dot`, `CosineSimilarity`, `ColumnNorms`, `RowNorms`, `DotProduct`, `Norm`) were **removed**
(AutoDiff refactor, Task 10) rather than moved to a separate namespace — they had no production
callers, and the platform `TensorPrimitives` kernels are the sanctioned replacement.
`NivaraTensorExtensions` was stripped to null-aware column reductions (`Sum`, `Mean`, `Min`, `Max`).
Column math is done on spans via `TryGetSpan`; row-major operations assemble spans with
`CopyToRowMajor`.

No custom tensor ecosystem. No tensor operators. No tensor hierarchy.

---

## The platform owns the math (.NET 10 grounding)

Facts grounded in the official docs ([what's new in .NET 10][2]):

- `System.Numerics.Tensors` is **stable** in .NET 10 — no longer `[Experimental]`. The APIs still
  live in the `System.Numerics.Tensors` NuGet package, but they are finalized.
- `IReadOnlyTensor` gives nongeneric access to `Lengths`/`Strides`; slice operations are zero-copy.
- C# 14 extension operators provide arithmetic (`tensor + tensor`) when `T` implements the
  relevant generic-math interfaces (`INumber<T>`, `IAdditionOperators<...>`, etc.).
- `TensorPrimitives` exposes ~200 generic overloads for `INumber<T>` / `IRootFunctions<T>` /
  `ITrigonometricFunctions<T>` and friends — e.g. `CosineSimilarity<T>`, `Dot<T>`, `Norm<T>` —
  SIMD-accelerated on spans.

### BCL swap targets

`TensorsHelper` is the shared internal kernel store (MatMul, Transpose) with **BCL swap-target
annotations** (see [ADR-002][3] and [ADR-003][4]). `Tensor.Transpose<T>` and
`Tensor.MatrixMultiply<T>` are net-11-preview APIs; when they ship, they replace the handwritten
kernels. `GradKernels` is the AutoDiff facade over span-in/span-out kernels. These are internal —
no permanent public tensor API is added on top of a stopgap kernel while the BCL matmul story is
still in flux.

### AutoDiff keeps the span boundary

[ADR-002][3]: `GradTensor<T>.Data` remains `NivaraColumn<T>`; operations compute over spans and
wrap results once. [ADR-003][4]: batch is a first-class dimension handled **inside** fused
ops/modules via internal loops, not by adding general rank-N primitive ops.

### Type support note

AutoDiff is constrained to `IFloatingPointIeee754<T>` (float, double, Half). `BFloat16` is
net11-only (not testable at the net10.0 TFM); today's BFloat16 support is load-time BF16→F32
widening in `SafeTensorsLoader`. Kernel tests for BFloat16 are deferred to the net11 migration.

---

## Committed direction

The question "what does Nivara own that .NET 10 does not?" is now answered by committed roadmaps,
not open options:

- **Columnar analytics** (was "Option A") — the execution engine is built; the roadmap forward is
  `docs/POLARS-ROADMAP.md` (query/expression engine) and `docs/ARROW-ROADMAP.md` (columnar physics).
- **Polars-style engine** (was "Option B") — committed as `docs/POLARS-ROADMAP.md`.
- **AI data infrastructure** (was "Option C") — committed as `docs/AISTACK-ROADMAP.md`.

The "danger" of competing with the platform is handled by the boundary above: Nivara does not
re-wrap what `TensorPrimitives` already does, and the tensor surface is interop, not an engine.

---

## Future strategy / where to go next

### Columnar engine (`docs/POLARS-ROADMAP.md`)

- Unified typed expression engine (make-or-break)
- Kernel fusion + generic-math collapse
- Window functions
- Async-first streaming
- Source generators

### Columnar physics (`docs/ARROW-ROADMAP.md`)

- Chunked column model in core
- Layout separation + variable-binary strings
- Validity as an explicit bitmap
- Real zero-copy interop, both directions
- Dictionary encoding
- Interchange-native scanning

### Local .NET AI stack (`docs/AISTACK-ROADMAP.md`)

- ONNX export/import (train in Nivara, exchange with the ecosystem)
- Embedding columns as a first-class data type
- Dataset & data-loader layer
- Evaluation & metrics module
- Safe read-only tensor views — **already delivered** via public `AsTensorView()` (#107)

### Scoped tensor ambitions (GitHub issues)

Row-wise frame scoring (#138), row-slice `TensorPrimitives` kernels (#141), benchmark coverage
(#142), and a `Standardize` data-prep alias (#143) are tracked as standalone issues to pick up
opportunistically between releases — **before** the committed roadmaps. They are scoped as
**interop conveniences** — not a change to the column-first storage model, not BLAS-level matrix
multiplication in core.

### Explicit non-goals

```text
Custom tensor abstractions
Tensor helper/math duplication with a public API surface
Vector math wrappers that re-implement TensorPrimitives
BLAS-level matrix multiplication in core
```

.NET 10 is already doing that work.

---

## Related documents

- `docs/plan/POLARS-ROADMAP.md` / `docs/plan/POLARS-REVIEW.md` — columnar engine lens
- `docs/plan/ARROW-ROADMAP.md` / `docs/plan/ARROW-REVIEW.md` — columnar physics lens
- `docs/plan/AISTACK-ROADMAP.md` / `docs/plan/AISTACK-REVIEW.md` — local .NET AI stack lens
- Scoped tensor ambitions are tracked as standalone GitHub issues: [row-wise scoring #138](https://github.com/khurram-uworx/Nivara/issues/138), [row-slice kernels #141](https://github.com/khurram-uworx/Nivara/issues/141), [benchmarks #142](https://github.com/khurram-uworx/Nivara/issues/142), [Standardize #143](https://github.com/khurram-uworx/Nivara/issues/143)
- `docs/adr/001-autodiff-nonnullable-domain.md` — the null-boundary rule
- `docs/adr/002-autodiff-span-boundary.md` — the AutoDiff span boundary
- `docs/adr/003-batch-fused-ops-not-rank-n-primitives.md` — the batch-dimension rule
- `docs/AUTODIFF.md` — the AutoDiff subsystem reference

[1]: https://www.nuget.org/packages/System.Numerics.Tensors
[2]: https://learn.microsoft.com/en-us/dotnet/core/whats-new/dotnet-10/libraries#systemnumerics
[3]: docs/adr/002-autodiff-span-boundary.md
[4]: docs/adr/003-batch-fused-ops-not-rank-n-primitives.md
