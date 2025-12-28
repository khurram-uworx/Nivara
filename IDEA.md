# Nivara

**A modern, columnar data processing engine for .NET**

Nivara is a high‑performance, strongly‑typed, columnar data processing library for .NET. It draws conceptual inspiration from pandas and Polars, while being deeply idiomatic to C# and the .NET ecosystem.

The goal is to provide:
- Polars‑like performance
- Pandas‑like ergonomics
- LINQ‑native developer experience
- First‑class SIMD and tensor acceleration via `System.Numerics`

---

## 1. Core Philosophy

### 1.1 Columnar, Not Row‑Based
- All data is stored and processed **column‑wise**
- Rows exist only at the **type system level**, never as runtime objects
- Execution is vectorized and batch‑oriented

### 1.2 Strong Typing Without Friction
- No `object` columns
- No dynamic typing at runtime
- Nullable semantics are explicit (`T?`)

### 1.3 Functional Core, Friendly to OOP
- Core execution is functional and immutable
- Public API remains approachable for OOP developers
- Fluent APIs and LINQ coexist with a functional execution engine

### 1.4 LINQ Is Syntax, Not Execution
- LINQ expression trees are accepted
- LINQ delegates are **never executed row‑by‑row**
- LINQ queries are translated into columnar execution plans

---

## 2. Core Types

### 2.1 `NivaraColumn<T>`

Represents a typed, immutable column backed by tensor storage.

Responsibilities:
- Own or reference tensor‑backed data
- Track null masks
- Expose vectorized operations

```csharp
NivaraColumn<double> salary;
var raised = salary * 1.1;
```

---

### 2.2 `NivaraSeries<T>`

A labeled column, optionally indexed.

Use cases:
- Time‑series
- Indexed analytics
- Alignment and joins

```csharp
NivaraSeries<double> prices;
```

---

### 2.3 `NivaraFrame`

A collection of named columns sharing the same length.

Responsibilities:
- Schema management
- Query construction
- Lazy execution

```csharp
var frame = NivaraFrame.ScanCsv("sales.csv");
```

---

## 3. Execution Model

### 3.1 Lazy by Default

- `ScanXxx()` → lazy
- `ReadXxx()` → eager
- `Collect()` → execution barrier

```csharp
var result =
    NivaraFrame.ScanCsv("data.csv")
        .Filter(Col("Age") > 30)
        .GroupBy("City")
        .Agg(Col("Salary").Mean())
        .Collect();
```

---

### 3.2 Expression‑Based API

All operations build an internal expression graph.

```csharp
Col("Salary") * 1.1 + 1000
```

This enables:
- Query optimization
- Predicate pushdown
- SIMD execution

---

## 4. Execution Engine Architecture

Nivara exposes **one logical execution engine** with multiple physical execution paths. Users never choose an engine explicitly; execution strategies are selected automatically based on data characteristics and operation semantics.

> **There is no “Tensor engine” vs “Memory engine” in the public API.**
> Nivara is adaptive by default.

---

### 4.1 Adaptive Execution Model

For each operation, Nivara dynamically selects:
- Vectorized (SIMD / tensor-based) kernels
- Scalar or memory-based kernels

Selection is based on:
- Data type
- Column storage kind
- Data size
- Null density
- Operation semantics

This allows Nivara to evolve its execution strategies without breaking user code.

---

### 4.2 System.Numerics as the Numeric Kernel Backend

System.Numerics APIs are used as **internal execution primitives**, not as part of the public API.


### 4.3 Tensors as Numeric Storage

- Numeric columns may be backed by `System.Numerics.Tensors.Tensor<T>`
- Tensor-backed storage is preferred for:
  - Numeric primitives
  - Boolean masks
  - Vectorizable operations
- Slicing and views are preferred over copying


- Columns are backed by `System.Numerics.Tensors.Tensor<T>`
- Slicing and views are preferred over copying

### 4.4 Vectorized Kernels

Execution relies on:
- `System.Numerics.Vectors`
- `System.Numerics.Tensors.TensorPrimitives`

Used for:
- Arithmetic
- Comparisons
- Reductions
- Masked operations

All tensor kernels are wrapped behind internal abstractions to isolate the rest of the engine from API changes in `System.Numerics`.


Execution relies on:
- `System.Numerics.Vectors`
- `System.Numerics.Tensors.TensorPrimitives`

Used for:
- Arithmetic
- Comparisons
- Reductions
- Masked operations

Numerics APIs are **internal only** and never leak into public APIs.

---

## 5. Storage Abstraction and Data Types

### 5.1 Unified Public Column Model

Public APIs always expose:
- `NivaraColumn<T>`
- `NivaraSeries<T>`

There are **no string-specific or type-specific public column classes**.
Execution strategy and storage layout are internal concerns.

---

### 5.2 Internal Storage Abstraction

Nivara uses an internal storage abstraction to support multiple physical layouts without exposing them publicly.

```csharp
internal interface IColumnStorage<T>
{
    int Length { get; }
    bool IsVectorizable { get; }
}
```

Two storage implementations are provided out of the box:

#### Tensor-Based Storage
- Backed by `System.Numerics.Tensors.Tensor<T>`
- Used for numeric and vectorizable types

#### Memory-Based Storage
- Backed by `Memory<T>` / `ReadOnlyMemory<T>`
- Used for strings, reference types, and non-vectorizable data

Additional storage implementations (e.g., dictionary-encoded, compressed, GPU-backed) may be added internally or via extensions.

---

### 5.3 Vectorizable vs Non-Vectorizable Types

Vectorizable types include:
- Numeric primitives (`int`, `long`, `float`, `double`)
- `bool`
- Temporal types (via numeric representation)

Non-vectorizable types include:
- `string`
- `Guid`
- Reference types

This distinction affects execution strategy only, not the public API.

---

### 5.4 String Columns

String columns:
- Use memory-based storage
- Are first-class citizens in the data model
- Do not use SIMD acceleration

Optimizations for string columns include:
- Hash caching
- Dictionary encoding (future)
- Batched scalar execution

---

## 6. Null Handling

- Nulls are tracked using boolean masks
- No NaN-based null semantics
- All operations are mask-aware


- Nulls are tracked using boolean tensor masks
- No NaN‑based null semantics
- All operations are mask‑aware and vectorized

---

## 7. LINQ Integration

---

## 6.5 Data Types, Storage, and Non-Vectorizable Columns

### 6.5.1 Vectorizable vs Non-Vectorizable Types

Not all data types benefit from SIMD or tensor-based execution. Nivara explicitly distinguishes between:

**Vectorizable types**
- Numeric primitives (`int`, `long`, `float`, `double`)
- `bool`
- `DateTime` / `DateOnly` / `TimeOnly` (as underlying numeric representations)

**Non-vectorizable types**
- `string`
- `Guid`
- `Complex` (unless custom kernels are provided)
- Arbitrary user-defined reference types

This distinction is an execution concern, not an API concern.

---

### 6.5.2 Unified Column Model (No String-Specific Public Types)

Nivara does **not** introduce public types like `NivaraStringSeries`.

Instead:

- All columns are represented as `NivaraColumn<T>` / `NivaraSeries<T>`
- Execution strategy is selected internally based on column capabilities

```csharp
NivaraColumn<string> names;
NivaraColumn<double> salary;
```

This avoids API fragmentation while keeping performance optimizations internal.

---

### 6.5.3 Storage Abstraction (Pluggable by Design)

Nivara introduces an internal storage abstraction:

```csharp
internal interface IColumnStorage<T>
{
    int Length { get; }
    bool IsVectorizable { get; }
}
```

Two primary storage implementations are provided out of the box:

#### 1. Tensor-Based Storage (Default for Numerics)

- Backed by `System.Numerics.Tensors.Tensor<T>`
- Used when:
  - `T` is unmanaged
  - Vectorized kernels exist

```csharp
internal sealed class TensorStorage<T> : IColumnStorage<T>
    where T : unmanaged
```

#### 2. Memory-Based Storage (Default for Strings and Objects)

- Backed by `Memory<T>` / `ReadOnlyMemory<T>`
- Optimized for:
  - Strings
  - Reference types
  - Dictionary-based operations

```csharp
internal sealed class MemoryStorage<T> : IColumnStorage<T>
```

---

### 6.5.4 Execution Strategy Selection

Execution is chosen per operation, per column:

| Operation | Numeric Column | String Column |
|--------|---------------|---------------|
| Filter | Tensor kernel | Scalar loop |
| Compare | SIMD | Ordinal / culture-aware |
| GroupBy | Hash + SIMD agg | Hash only |
| Join | Hash | Hash |

The user does not choose the strategy — the engine does.

---

### 6.5.5 Strings as First-Class Citizens (But Not SIMD)

String columns:
- Use `Memory<string>` storage
- Support:
  - Equality / inequality
  - Prefix / suffix
  - Contains
  - Regex (extension)

String-heavy operations are:
- Explicitly non-vectorized
- Clearly documented as such
- Optimized via batching and caching, not SIMD

---

### 6.5.6 Custom Storage and Custom Columns

Advanced users may plug in custom storage:

- GPU-backed storage
- Dictionary-encoded columns
- Compressed columns

```csharp
internal sealed class DictionaryEncodedStorage<T>
    : IColumnStorage<T>
```

Public APIs remain unchanged.

---

### 6.5.7 Design Rule

> **Vectorization is an optimization, not a requirement.**

Nivara correctness never depends on SIMD availability.

---

### 6.5.8 Why Not Expose Storage Publicly?

- Prevents leaky abstractions
- Avoids premature coupling
- Keeps API stable while internals evolve

Storage is an internal concern and may change across versions without breaking users.

---



### 6.1 Supported Model

```csharp
record Person(int Age, string City, double Salary);

var query =
    frame.Query<Person>()
         .Where(p => p.Age > 30)
         .GroupBy(p => p.City)
         .Select(g => new
         {
             g.Key,
             AvgSalary = g.Average(p => p.Salary)
         });
```

### 6.2 LINQ Rules

Allowed:
- Property access
- Arithmetic
- Comparisons
- Boolean logic
- Aggregates

Rejected:
- Loops
- Method calls
- Closures
- Side effects

Unsupported expressions fail fast with clear diagnostics.

---

## 8. Benchmarking, Diagnostics, and Monitoring

### 8.1 Benchmarking API

Nivara provides a built-in benchmarking facility to help users understand performance on their own data.

```csharp
var report = frame.Benchmark(options =>
{
    options.SampleFraction = 0.05;
    options.WarmupRuns = 3;
    options.MeasureAllocations = true;
});
```

Benchmark reports include:
- Logical and physical execution plans
- Operator-level timings
- Kernel selection (vectorized vs scalar)
- Memory allocations
- SIMD usage and fallback reasons

Benchmarking uses the same adaptive engine as production execution.

---

### 8.2 Diagnostics and Observability

Nivara includes first-class observability support:

- Execution metrics
- Operator timings
- Memory usage
- Kernel decisions

These are exposed via:
- Observer hooks
- `ActivitySource` / OpenTelemetry integration
- ETW-compatible events

```csharp
frame
  .WithObserver(new NivaraMetricsObserver())
  .Collect();
```

Diagnostics are designed to be safe for production use.

---

## 9. IO and Extensibility

### 7.1 Core Library IO

Supported in core:
- CSV
- JSON

Core IO is designed to be:
- Streaming‑friendly
- Schema‑aware
- Extensible

---

### 7.2 Extension Packages

Non‑core formats are delivered as extensions:

- Apache Arrow
- Parquet
- Future formats (ORC, IPC, etc.)

```text
Nivara.Core
Nivara.Extensions.Arrow
Nivara.Extensions.Parquet
```

---

### 7.3 Custom Extensions

The core supports extensibility beyond IO:

- Custom column types (e.g., `Complex`, `Decimal128`)
- Custom expressions
- Custom filters
- Domain‑specific kernels

---

## 10. Apache Arrow Integration

Arrow is the primary interoperability format.

Capabilities:
- Zero‑copy column exchange
- Shared memory pipelines
- Interop with Python / Rust ecosystems

```csharp
var arrowTable = frame.ToArrow();
var frame2 = NivaraFrame.FromArrow(arrowTable);
```

---

## 11. ML.NET Interop

Nivara integrates with ML.NET as a first‑class citizen.

### 9.1 DataView Support

```csharp
IDataView view = frame.ToDataView();
NivaraFrame frame = NivaraFrame.FromDataView(view);
```

### 9.2 Use Cases

- Feature engineering in Nivara
- Model training in ML.NET
- Prediction results returned to NivaraFrame

---

## 12. What Nivara Explicitly Avoids

- Row‑by‑row execution
- Mutable DataFrames
- `object` columns
- Silent type coercion
- Hidden performance cliffs

---

## 13. Roadmap (High‑Level)

### Phase 1 – Foundation
- NivaraColumn / Series / Frame
- Expressions
- CSV & JSON
- Tensor‑based arithmetic

### Phase 2 – Performance & Scale
- Query optimizer
- Joins
- Window functions
- Arrow & Parquet extensions

### Phase 3 – Ecosystem
- ML.NET deep integration
- Streaming execution
- GPU offload (when available)

---

## 14. Guiding Principle

> **Feels like LINQ. Executes like a vectorized engine. Scales like a data system.**

Nivara is not a pandas clone.
It is a **native .NET data engine**, built for the next decade.

