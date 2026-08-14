# Nivara

A high-performance, columnar DataFrame library for .NET, focused on **type safety**, **explicit null semantics**, **query planning**, and clean interop with platform tensor and data APIs.

Nivara is designed for developers who want predictable behavior, strong typing, and performance-oriented data processing without relying on dynamic or NaN-based conventions.

---

## Why Nivara

- https://khurram-uworx.github.io/2026/01/12/LLMs-Equalizers.html

Most DataFrame-style libraries trade correctness and type safety for convenience. Nivara takes a different approach:

- **Strong typing end-to-end** — column types are explicit and enforced
- **Explicit null handling** — no NaN-based semantics or hidden behavior
- **Immutable data model** — operations return new data structures
- **Interop with .NET primitives** — use Nivara for tabular data and `System.Numerics.Tensors` for tensor math
- **Schema-aware query planning** — errors surface early, not at runtime

If you care about correctness, debuggability, and performance in .NET data processing, Nivara is built for you.

---

## Installation

Core library:

```bash
dotnet add package Nivara
```

Optional extensions and I/O integrations (install when you need file formats, Arrow interoperability, or ML integration):

```bash
dotnet add package Nivara.Extensions
```

---

## Quick Start

```csharp
using Nivara;
using Nivara.Linq;

// Create typed columns
NivaraColumn<int> ages = [25, 30, 35];
var names = NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie" });

// Combine into a DataFrame
var frame = NivaraFrame.Create(
    ("Name", names),
    ("Age", ages)
);

// Query with lazy evaluation — strongly typed lambdas over a POCO
public sealed class Person { public string Name { get; set; } public int Age { get; set; } }

var typed = frame.Query<Person>()
    .Where(p => p.Age > 30)
    .Select(p => new { p.Name })
    .ToObjects();   // IReadOnlyList<anonymous> — { Name = "Charlie" }

// Or materialize to a NivaraFrame
var adults = frame.Query<Person>()
    .Where(p => p.Age > 30)
    .Collect();     // NivaraFrame — 1 row (Charlie)
```

---

## Core Features

### Typed Columns and DataFrames
- Strongly typed, immutable columns with automatic storage selection
- Schema-aware frames with validation and type safety
- Explicit null handling using validity masks (no NaN semantics)

### Query Engine
- Typed object LINQ — `frame.Query<T>()` maps a POCO to the frame schema and compiles typed lambdas into query plans (predicates, projections, `OrderBy`/`ThenBy` with per-key `SortDirection`/`NullOrdering`, `Distinct`/`DistinctBy`, `SelectRows`, `Skip`/`Take`, `GroupBy` with `g.Key` + `Average`/`Sum`/`Count`/`Min`/`Max` aggregates), materializing to a `NivaraFrame` or `IReadOnlyList<TResult>`
- Lazy typed file-source queries — `Json.ScanQuery<T>()` (core) and `Csv.ScanQuery<T>()` (Extensions) defer I/O until execution; `ReadFrame`/`ScanFrame` cover eager/lazy frame loading
- Automatic query optimization (predicate pushdown, projection pushdown, operation fusion)
- Multiple execution strategies (lazy, eager, streaming, parallel) — all fully implemented with integrated performance diagnostics

### Tensor, AI, and AutoDiff Interop
- Convert columns, series, and frames to `Tensor<T>` for platform math APIs
- Preserve null masks through `NullableTensor<T>` when crossing tensor boundaries
- Ingest 2D tensors and labeled row vectors into schema-aware frames
- Keep tensor math in `System.Numerics.Tensors`, not custom DataFrame APIs
- Run lightweight reverse-mode AutoDiff when you need local training; inference is the default, manual training is explicit with `GradientUtils.Grad()`, and module state can be copied via `StateDict()` / `LoadStateDict()`
- Broader type support with `IFloatingPointIeee754<T>` constraint — `Half`/F16 and BFloat16 now pass runtime validation alongside `float` and `double`
- NLP and vision building blocks out of the box: `Embedding<T>`, `SparseEmbedding<T>`, `Conv1d<T>` (im2col-rewritten, PyTorch-compatible layout), `Conv2d<T>` (grouped conv, 1×1 fast path, PatchLocation lookup, InputGrad specializations), `ConvTranspose2d<T>`, `BatchNorm1d<T>` (now accepts 3D `[B,C,L]` input), `BatchNorm2d<T>`, `LayerNorm<T>` (SIMD via `TensorPrimitives.Dot`), `DepthwiseSeparableConv2d<T>`, `TransformerBlock<T>` (RMSNorm/LayerNorm + GELU), `MultiheadAttention<T>` (self/cross/causal), `ConvVAE<T>`, `VAE<T>` (optional conditioning), `MaxPool2d<T>`, `AdaptiveAvgPool2d<T>`, `GELU`, `TextTokenizer`, and `Sampler<T>` — all differentiable and composable with the existing module system (ready-to-use `TextClassifierModel<T>` / `TokenClassifierModel<T>` ship as sample code in `samples/Nivara.Samples/`)

### Performance
- Vectorized execution where semantics are simple and measurable
- SIMD-accelerated optimizer and normalization kernels (Adam, AdamW, PerRowRMSNorm backward, LayerNorm sum-of-squares via TensorPrimitives chains)
- ArrayPool-backed buffer management in hot paths (AccumulateGradient, Gather backward, Adam/AdamW state)
- Automatic storage backend selection for supported types
- Scalar fallbacks that preserve explicit null semantics

### Data Operations
- **Row Operations**: Filtering, slicing, sorting with null-aware semantics
- **Column Operations**: Transformations, projections, renaming, computed columns
- **Join Operations**: Inner, Left, Right, Full Outer joins with flexible key mapping
- **Aggregation**: GroupBy operations with vectorized aggregate functions
- **Concatenation**: Vertical and horizontal DataFrame combination

### Data Sources and I/O
- CSV and JSON lazy data sources with schema inference
- Parquet file I/O with compression support (via `Nivara.Extensions`)
- Apache Arrow interoperability (via `Nivara.Extensions`)

### Developer Experience
- Comprehensive error handling with structured exceptions
- Performance diagnostics and query plan inspection
- Fluent API with method chaining
- Early error detection through schema validation

---

## Getting Started

For detailed examples and tutorials, see [**GETTING-STARTED.md**](https://github.com/khurram-uworx/nivara/blob/main/GETTING-STARTED.md).

For comprehensive API documentation and advanced usage patterns, explore the [**samples/**](https://github.com/khurram-uworx/nivara/tree/main/samples) directory — including a character-level GPT trained on Nivara AutoDiff, a neural chess evaluator, a hybrid Nivara+LLM agent workflow, a variational autoencoder for synthetic pattern generation, a PyTorch parity benchmark suite showing <0.04% loss-curve divergence, a MiniLM inference pipeline, a DistilBERT fine-tuning pipeline for SST-2 (samples/NivaraFineTuning), a MobileNetV2/ResNet-18 inference pipeline (samples/NivaraInference), and a time-series anomaly detection sample (samples/NivaraTimeSeries).

---

## Current Capabilities

Nivara aims to bring **predictable, high-performance data processing** to the .NET ecosystem — without sacrificing correctness or clarity.

Nivara currently supports:

- **Core Data Structures**: Typed, immutable columns and frames with automatic storage selection
- **Null Handling**: Explicit null handling with fill and drop operations, comprehensive null mask tracking
- **Tensor Interop**: `Tensor<T>` and nullable tensor conversion helpers, plus matrix/labeled-row ingestion
- **Performance**: Vectorized arithmetic and comparisons where semantics are safe
- **Storage**: High-performance tensor-backed storage for numeric types, memory-based storage for reference types
- **Query Engine**: Schema-aware lazy query construction with automatic optimization, `OperationType` constants, diagnostics and plan inspection
- **Typed Object LINQ**: `frame.Query<T>()` with eager POCO→column mapping, typed predicates/projections, GroupBy aggregates, and row-factory materialization (`Collect`/`ToList` → `NivaraFrame`, `ToObjects`/`ToRows` → `IReadOnlyList<TResult>`); unsupported expressions fail fast with `UnsupportedQueryExpressionException`
- **Data Sources**: CSV and JSON lazy data sources with automatic schema inference
- **Row Operations**: Filtering with boolean masks, slicing with Take/Skip operations, and arbitrary row range selection
- **Sorting Operations**: Multi-column sorting with configurable direction, null ordering, and stable sort semantics
- **Column Transformations**: Type-safe element-wise transformations with null propagation and exception handling
- **Column Projections**: Flexible column selection, renaming, exclusion, and computed column generation
- **Join Operations**: Inner, Left, Right, and Full Outer joins with flexible key mapping, column disambiguation, and null-aware matching
- **Aggregate Functions**: Sum, Average, Min, Max with vectorized operations and null-aware computation
- **Grouping Operations**: Hash-based GroupBy with composite key support and efficient group management
- **Aggregation Framework**: Extensible aggregation system with built-in functions (Count, Sum, Min, Max, Mean) and vectorized execution
- **Parquet I/O**: Full read/write support with compression, streaming, and batch operations (via `Nivara.Extensions`)
- **Apache Arrow**: Bidirectional conversion (via `Nivara.Extensions`)
- **ML.NET Integration**: ML.NET conversion helpers for machine learning workflows (via `Nivara.Extensions`)
- **Performance Optimization**: Buffer pooling, memory management, query optimization engine, async I/O operations, and integrated execution diagnostics (plan inspection via `ExplainPlan()`, per-operation timings)
- **Automatic Differentiation**: Reverse-mode autodiff with inference by default, explicit manual training via `GradientUtils.Grad()`. Type constraint broadened to `IFloatingPointIeee754<T>` — `Half`/F16 and BFloat16 supported alongside `float`/`double`. Full training stack: module system (`Linear`, `Sequential`, `Embedding`, `SparseEmbedding`, `Conv1d` (im2col + Dot, PyTorch-compatible layout), `Conv2d` (grouped conv, 1×1 fast path, PatchLocation, InputGrad specializations), `ConvTranspose2d`, `BatchNorm1d`/`2d` (fused span-kernel, 3D input support), `LayerNorm` (SIMD `TensorPrimitives.Dot`), `DepthwiseSeparableConv2d`, `TransformerBlock` (RMSNorm/LayerNorm + GELU), `MultiheadAttention`, `ConvVAE`, `VAE` (optional conditioning), `MaxPool2d`, `AdaptiveAvgPool2d`), NLP utilities (`TextTokenizer`, `Sampler`), activations (`GELU`), operations (`MeanPool`, `TransposeAxes`, `SparseEmbeddingBag`, `Gather` with zero-copy forward, `Softmax`, `LogSoftmax`, `Dropout`), optimizers (`SGD`, `Adam`, `AdamW`) with SIMD-accelerated kernels, training loops, data-parallel training, model serialization, and 55 PyTorch-validated functional tests

---

## Documentation

- [**GETTING-STARTED**](https://github.com/khurram-uworx/nivara/blob/main/GETTING-STARTED.md) — tutorials, examples, and step-by-step guides
- [**ARCHITECTURE**](https://github.com/khurram-uworx/nivara/blob/main/ARCHITECTURE.md) — design and internal architecture
- [**AUTODIFF**](https://github.com/khurram-uworx/nivara/blob/main/docs/AUTODIFF.md) — automatic differentiation subsystem (operations, modules, optimizers, forward-mode AD, training)
- [**CONTRIBUTING**](https://github.com/khurram-uworx/nivara/blob/main/CONTRIBUTING.md) — how to contribute to the project
- [**GUIDELINES**](https://github.com/khurram-uworx/nivara/blob/main/GUIDELINES.md) — architectural rationale, lessons learned, and known gotchas
- [**CHANGELOG**](https://github.com/khurram-uworx/nivara/blob/main/CHANGELOG.md) — Notable changes and release history
- [**RELEASING**](https://github.com/khurram-uworx/nivara/blob/main/RELEASING.md) — how to cut a release and publish to NuGet

