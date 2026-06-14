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

// Query with lazy evaluation
// Query with lazy evaluation (LINQ-like)
var adults = frame.AsQueryFrame()
    .Where(x => x["Age"] > 30)
    .Select(x => x["Name"])
    .ToNivaraFrame();

Console.WriteLine(adults.RowCount); // 1 (Charlie)
```

---

## Core Features

### Typed Columns and DataFrames
- Strongly typed, immutable columns with automatic storage selection
- Schema-aware frames with validation and type safety
- Explicit null handling using validity masks (no NaN semantics)

### Query Engine
- Lazy query construction with true LINQ-like syntax (Where, Select, OrderBy)
- Automatic query optimization (predicate pushdown, projection pushdown, operation fusion)
- Multiple execution strategies (lazy, eager, streaming, parallel) — all fully implemented with integrated performance diagnostics

### Tensor, AI, and AutoDiff Interop
- Convert columns, series, and frames to `Tensor<T>` for platform math APIs
- Preserve null masks through `NullableTensor<T>` when crossing tensor boundaries
- Ingest 2D tensors and labeled row vectors into schema-aware frames
- Keep tensor math in `System.Numerics.Tensors`, not custom DataFrame APIs
- Run lightweight reverse-mode AutoDiff when you need local training; inference is the default, and manual training is explicit with `GradientUtils.Grad()`

### Performance
- Vectorized execution where semantics are simple and measurable
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

For comprehensive API documentation and advanced usage patterns, explore the samples in the `samples/` directory.

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
- **Apache Arrow**: Bidirectional conversion with zero-copy optimization support (via `Nivara.Extensions`)
- **ML.NET Integration**: ML.NET conversion helpers for machine learning workflows (via `Nivara.Extensions`)
- **Performance Optimization**: Buffer pooling, memory management, query optimization engine, async I/O operations, and integrated execution diagnostics via `ExecutionEngine.LastDiagnostics`
- **Automatic Differentiation**: Reverse-mode autodiff for `float` and `double` columns with inference by default, explicit manual training via `GradientUtils.Grad()`, plus a full training stack — module system (`Linear`, `Sequential`), optimizers (`SGD`, `Adam`, `AdamW`), training loops, data-parallel training, and model serialization (core)

---

## Documentation

- [**GETTING-STARTED**](https://github.com/khurram-uworx/nivara/blob/main/GETTING-STARTED.md) — tutorials, examples, and step-by-step guides
- [**ARCHITECTURE**](https://github.com/khurram-uworx/nivara/blob/main/ARCHITECTURE.md) — design and internal architecture
- [**CONTRIBUTING**](https://github.com/khurram-uworx/nivara/blob/main/CONTRIBUTING.md) — how to contribute to the project
- [**GUIDELINES**](https://github.com/khurram-uworx/nivara/blob/main/GUIDELINES.md) — architectural rationale, lessons learned, and known gotchas

