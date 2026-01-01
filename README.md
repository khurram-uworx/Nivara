# Nivara

A high-performance, columnar DataFrame library for .NET, focused on **type safety**, **explicit null semantics**, and **vectorized execution**.

Nivara is designed for developers who want predictable behavior, strong typing, and performance-oriented data processing without relying on dynamic or NaN-based conventions.

---

## Why Nivara

Most DataFrame-style libraries trade correctness and type safety for convenience. Nivara takes a different approach:

- **Strong typing end-to-end** — column types are explicit and enforced
- **Explicit null handling** — no NaN-based semantics or hidden behavior
- **Immutable data model** — operations return new data structures
- **Vectorized execution where it matters** — SIMD via `System.Numerics`
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

## Quick Example

```csharp
using Nivara;

var column = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 });

var doubled = column * 2;

Console.WriteLine(doubled[0]); // 2
```

---

## Core Concepts

### Typed Columns

Nivara columns are strongly typed and immutable:

```csharp
var ages = NivaraColumn<int>.Create(new[] { 25, 30, 35 });
Console.WriteLine(ages.Length); // 3
```

### Explicit Null Semantics

Nulls are tracked explicitly using validity masks, not sentinel values:

```csharp
var data = new int?[] { 1, null, 3 };
var column = NivaraColumn<int>.CreateFromNullable(data);

Console.WriteLine(column.HasNulls);   // True
Console.WriteLine(column.NullCount);  // 1
```

Null-aware operations behave predictably:

```csharp
var filled = column.FillNull(0);  // [1, 0, 3]
var dropped = column.DropNulls(); // [1, 3]
```

---

## Vectorized Operations

For vectorizable types, Nivara automatically uses SIMD-accelerated kernels backed by `System.Numerics.Tensors`:

```csharp
var a = NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0 });
var b = a * 1.5;
var c = a + b;
```

Nivara automatically selects the optimal storage backend:
- **TensorStorage**: For vectorizable types (`int`, `float`, `double`, `bool`) using `System.Numerics.Tensors`
- **MemoryStorage**: For non-vectorizable types (`string`, `Guid`, reference types) using `Memory<T>`

Vectorization is applied opportunistically and safely, with scalar fallbacks when required.

---

## Frames and Schemas

Multiple columns can be combined into a schema-aware frame:

```csharp
var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.Create(new[] { "Alice", "Bob" })),
    ("Age",  NivaraColumn<int>.Create(new[] { 30, 40 }))
);

Console.WriteLine(frame.RowCount);    // 2
Console.WriteLine(frame.ColumnCount); // 2
```

Schemas are immutable and validated on every transformation.

---

## Lazy Query API

Nivara provides a fluent, LINQ-like query API with lazy execution:

```csharp
using Nivara.Expressions;

var result = frame.AsQueryFrame()
    .Filter(ColumnExpressions.Col("Age") > 30)
    .Select("Name")
    .Collect();
```

Queries are planned and validated before execution, allowing early detection of schema and type errors.

---

## Lazy Data Sources (CSV / JSON)

File-based data sources support lazy scanning with automatic schema inference:

```csharp
using Nivara.IO;

// Create a QueryFrame that scans the CSV lazily
var query = Csv.ScanAsQueryFrame("employees.csv")
    .Filter(ColumnExpressions.Col("Salary") > 70000)
    .Select("Name", "Salary");

var result = query.Collect();
```

File I/O is deferred until `Collect()` is called.

---

## Extensions & Integrations

Additional I/O adapters and integrations are provided in the separate `Nivara.Extensions` package (install when you need these features):

```bash
dotnet add package Nivara.Extensions
```

The integrations are shipped as a separate package so the core Nivara runtime remains small and focused.

---

## Current Capabilities

Nivara currently supports:

- Typed, immutable columns and frames with automatic storage selection
- Explicit null handling with fill and drop operations
- Vectorized arithmetic and comparisons using `System.Numerics.Tensors`
- High-performance tensor-backed storage for numeric types
- Query diagnostics and plan inspection
- Schema-aware lazy query construction
- CSV and JSON lazy data sources
- Parquet read/write (via `Nivara.Extensions`)
- Apache Arrow interoperability (via `Nivara.Extensions`)
- ML.NET integration helpers (via `Nivara.Extensions`)

---

## Roadmap (Planned)

The following features are still planned or in-progress:

- Aggregations (Sum, Min, Max, Mean)
- Join operations
- GroupBy aggregations
- Streaming and parallel execution strategies

---

## Stability

Nivara is under active development.

- Core concepts are stabilizing
- APIs may evolve as additional features are added
- Feedback and experimentation are encouraged

---

## Documentation

- [**ARCHITECTURE**](ARCHITECTURE.md) — design and internal architecture
- [**CONTRIBUTING**](CONTRIBUTING.md) — how to contribute to the project
- [**GUIDELINES**](GUIDELINES.md) — architectural rationale, lessons learned, and known gotchas

---

Nivara aims to bring **predictable, high-performance data processing** to the .NET ecosystem — without sacrificing correctness or clarity.

