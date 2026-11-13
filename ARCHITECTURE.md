# Nivara Architecture & Design Decisions

This document describes the **internal architecture and design decisions** of Nivara. It is intended for contributors, advanced users, and AI agents working on the codebase.

> **Scope**
> - Explains *how* Nivara is implemented and *why* it was designed this way
> - Describes major subsystems, their responsibilities, and the decisions behind them
> - References [**GUIDELINES**](GUIDELINES.md) for rationale, trade-offs, and learned constraints
>
> **Out of scope**
> - How to build or contribute (see CONTRIBUTING.md)
> - User-facing tutorials (see README.md)

---

## Design Principles

Nivara is guided by a small set of non-negotiable principles:

1. **Type safety over convenience**
2. **Explicitness over implicit behavior**
3. **Immutability by default**
4. **Performance where it is provable and measurable**
5. **Early failure over late surprises**

Many implementation choices make more sense when viewed through this lens. These principles emerged from real constraints: incomplete information, time and complexity limits, and trade-offs between correctness, performance, and extensibility.

---

## High-Level Architecture

At a high level, Nivara consists of:

- **Columnar storage abstractions** (`Nivara.Storage.ColumnStorage<T>` — a single unified storage class for all element types)
- **Schema-aware frames** (`NivaraFrame`, `Nivara.Query.QueryFrame`)
- **A lazy query layer** with optimization engine (`Nivara.Query`, `Nivara.Optimization`)
- **Fused expression engine** (SIMD-accelerated `FusedExpressionEvaluator` with `TensorPrimitives` backend, flat IR fallback, chunked streaming execution)
- **Execution strategies** (`Nivara.Execution` - lazy, eager, streaming, parallel)
- **Operation implementations** (`Nivara.Operations` - joins, aggregations, filtering, sorting, window functions)
- **Diagnostics and planning infrastructure** (`Nivara.Diagnostics`)
- **Interop boundaries** (`Nivara.Tensors`, Arrow/Parquet/CSV/JSON extensions)

```
Data Source (CSV/Parquet/Arrow)
        ↓
Nivara.Query.QueryFrame (lazy construction)
        ↓
Logical Plan (operations + schema)
        ↓
Query Optimization (Nivara.Optimization - rule-based)
        ↓
Physical Plan (Nivara.Execution strategy + kernels)
        ↓
Execution Engine (scalar/vectorized kernels)
        ↓
Materialized NivaraFrame
```

---

## Column Storage Model

### NivaraColumn<T>

`NivaraColumn<T>` is the fundamental data container implemented using a pluggable storage architecture.

Key characteristics:

- **Strongly typed** (`T` is never erased at the API level)
- **Immutable** (operations return new columns)
- **Unified storage backend** (single `ColumnStorage<T>`; kernel dispatch via `KernelSelector`)
- **Explicit null handling** via optional boolean masks

Each column delegates to an `IColumnStorage<T>` implementation:

- **Data storage** (`Nivara.Storage.ColumnStorage<T>` — a sole-owner `T[]` plus an optional `bool[]` null mask)
- **Null mask** (when nulls are present)
- **Kernel selection** (scalar vs vectorized decided by `KernelSelector`, never by the storage class)

### Storage Backend Selection

All columns use a single storage class — `ColumnStorage<T>`. There is no Memory-vs-Tensor split at the storage layer; whether a column dispatches to vectorized `TensorPrimitives` kernels is decided at runtime by `KernelSelector.DetermineKernelType()` (which considers `ColumnStorageFactory.IsVectorizable<T>()`, `Vector.IsHardwareAccelerated`, and a length threshold).

- **Unified `T[]` backing** with an optional `bool[]` null mask, shared by all types
- **Kernel dispatch** (scalar vs vectorized) is a runtime decision, not a storage-type decision
- `ColumnStorageFactory.IsVectorizable<T>()` reports whether `T` supports SIMD (`int`, `float`, `double`, `long`, `short`, `byte`, `bool`, etc.)

This selection is **transparent** to users and happens at column creation time. It is an internal storage and execution choice, not a promise that Nivara owns tensor math APIs.

### Storage Layout and Span Access

```
NivaraColumn<T> → IColumnStorage<T> → ColumnStorage<T> (sole-owner T[] + optional bool[] null mask)
  TryGetSpan   → zero-copy VIEW over T[] when no nulls; returns false when nulls present
  Slice        → zero-copy shared-buffer view (mutations to the source array are visible)
  AsTensorView → lazy zero-copy Tensor<T> view for unmanaged element types (check HasNulls first)
```

Span-access semantics are uniform across all types because there is a single storage class:

- **`TryGetSpan`** returns a true zero-copy read-only view over the backing `T[]` when the
  column has no nulls; it returns `false` (no span) when a null mask is present so callers
  take the explicit-fill path.
- **`Slice`** returns a zero-copy shared-buffer view — it shares the source array and null
  mask, so mutations to the source are visible through the slice.
- **`AsTensorView()`** exposes a lazy zero-copy `Tensor<T>` view over the same array for
  unmanaged element types (cached after first access); callers must check `HasNulls` first
  since null positions hold `default(T)`.

### Null Representation Decision

**Decision**: Use **explicit validity masks** to represent nulls instead of sentinel values, and
represent that mask as a **byte-wise `ReadOnlySpan<bool>`** — one byte per element — rather than as
a bit-packed bitmap. This is a deliberate design decision (ADR-007), not an accident of
implementation.

**Context**: The system operates on strongly typed, columnar data and must support both numeric and
non-numeric types consistently.

**Alternatives Considered**:
- Sentinel values (e.g. `NaN`)
- Type-specific nullable encodings
- A bit-packed validity bitmap (1 bit per element, the Apache Arrow representation) as the primary
  in-core form

**Why This Was Chosen**:
- Sentinel values do not generalize beyond floating-point types
- They leak into user-visible results and complicate semantics
- A separate validity mask keeps data and nullness orthogonal
- Byte-wise keeps the mask in the shape every null-aware kernel already consumes. `nullMask` is
  exposed as `ReadOnlySpan<bool>` through `IColumnStorage<T>` and `ColumnStorage<T>`, and the fused
  expression engine ORs an output mask from leaf masks in the same pass it computes values. A
  bit-packed primary form would charge unpack-on-entry and pack-on-exit to **every** kernel in order
  to save mask memory that is small relative to the payload it annotates

**Canonical representation**:

- Storage holds a sole-owner `bool[]` null mask alongside the data buffer, sliced together with it so
  the mask stays aligned one-to-one with the values in every view.
- **`true` means null.** The polarity is part of the contract and is never inverted inside the engine.
- **An absent mask means no nulls.** `HasNulls` is true only when a mask exists; an all-false mask is
  never materialised, and an explicit empty mask is normalised to absent. A column with no nulls
  therefore pays nothing for validity.

**Arrow boundary**:

Apache Arrow requires a **bit-packed** validity buffer — 1 bit per element, least-significant-bit
numbering, with a set bit meaning *valid* and a zero bit meaning *null* — so the representation must
change at the interop boundary. That is a boundary concern, never a storage concern, and it does two
things rather than one:

- **Packing and unpacking**, because Arrow is bit-dense and Nivara is byte-wise.
- **Polarity inversion**, because Nivara's mask is true-at-null while Arrow's bit is true-at-valid.

Both sides also have an "absent" convention, with the same meaning but different mechanics: Nivara
omits the mask entirely, and Arrow permits eliding the validity buffer when the null count is zero.
The conversion belongs solely to the Arrow interop layer.

**How that conversion works today — and its limit.** It is element-wise in both directions: export
walks each value through the Apache.Arrow builders (which maintain the bitmap internally), apart from
the fixed-size-binary path, which writes the bitmap directly; import reads each slot and materialises
boxed values. A buffer-level mask-to-bitmap conversion — the one that would make the boundary cheap
and non-allocating — **exists as an internal, unit-tested helper** (`Nivara.IO.ArrowValidityMask`),
but it is not wired into the interop path, so the conversion described above is still element-wise.

**Consequences**:
- 1 byte per element against Arrow's 1 bit: **8× the validity memory, only when nulls are present**,
  additive to the data payload. This is the deliberate cost of the fast path, not an oversight
- Much clearer and more predictable null propagation
- No pack/unpack cost at any kernel boundary, including the fused single-pass mask OR
- Interop pays an element-wise packing cost today, because the buffer-level helper is not wired in yet

**Revisit when**: validity becomes a dominant memory consumer (very wide, densely-null, narrow-element
frames), or when the Arrow roadmap needs a buffer-level validity handoff for real zero-copy. At that
point the packing cost is paid once at the boundary, which is the trade this decision already assumes.

Nulls are **not** encoded using sentinel values such as `NaN`. Null handling is explicit and first-class:

- Presence of nulls is tracked separately from data
- Operations propagate nulls deterministically
- Null-aware kernels are selected when required

Null-related behavior is a major source of subtle bugs.

> Detailed rules, edge cases, and LLM-specific guidance are documented in [GUIDELINES](GUIDELINES.md)

---

## Storage & Kernel Selection

Nivara uses a single storage class with runtime kernel dispatch:

### ColumnStorage<T>

All columns use `ColumnStorage<T>` — a sole-owner contiguous `T[]` plus an optional `bool[]` null mask.

**Characteristics:**
- One storage implementation for every element type (vectorizable or not)
- Slices are zero-copy shared-buffer views
- `AsTensorView()` provides a lazy zero-copy `Tensor<T>` view for unmanaged types
- Whether operations dispatch to SIMD kernels is decided by `KernelSelector`, never by the storage class

### Kernel Selection

`KernelSelector.DetermineKernelType()` decides scalar vs vectorized execution at runtime:

**Vectorized when:**
- `T` is vectorizable per `ColumnStorageFactory.IsVectorizable<T>()` (`int`, `float`, `double`, `long`, `short`, `byte`, `bool`, etc.)
- `Vector.IsHardwareAccelerated`
- Length exceeds the vectorization threshold (`Length >= vectorSize * 4`)

**Scalar otherwise.**

### Automatic Selection

```csharp
// Kernel selection is automatic and transparent to users:
var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 });                    // vectorizable
var stringColumn = NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c" }); // scalar
```

This distinction is **internal** and transparent to users.

### Tensor Interop Boundary

Nivara's public tensor boundary is conversion and metadata preservation:

- `ToTensor()` exports null-free columns, series, and homogeneous frames to `Tensor<T>`
- `ToNullableTensor()` exports data plus an explicit `Tensor<bool>` null mask
- `FromTensor()` / `FromMatrix()` ingest 2D tensors as schema-aware frames
- `FromRows()` ingests labeled row vectors as a label column plus feature columns

Nivara does not implement a competing tensor math layer. Callers should use `System.Numerics.Tensors.TensorPrimitives`, `Tensor<T>`, `ReadOnlySpan<T>`, and Microsoft AI abstractions for numerical kernels, embeddings, and model-facing APIs. Nivara owns the tabular concerns around those APIs: typed columns, schemas, labels, query planning, and explicit null masks.

### Vectorization Strategy Decision

**Decision**: Apply vectorization selectively and only where semantics are trivial.

**Context**: SIMD execution can provide performance gains but complicates control flow and debugging.

**Alternatives Considered**:
- Aggressive vectorization throughout
- Pure scalar execution

**Why This Was Chosen**:
- Premature vectorization obscured correctness issues
- Many operations are not safely vectorizable

**Consequences**:
- Mixed execution paths
- Easier correctness validation

---

## Null Handling

Null handling is explicit and first-class:

- Presence of nulls is tracked separately from data
- Operations propagate nulls deterministically
- Null-aware kernels are selected when required

Null-related behavior is a major source of subtle bugs.

> Detailed rules, edge cases, and LLM-specific guidance are documented in [GUIDELINES](GUIDELINES.md)

---

## Frames and Schemas

### NivaraFrame

A `NivaraFrame` is a collection of named columns governed by a schema.

Ownership and disposal:

- A frame owns only the columns it created. Columns borrowed from another frame
  (via `SelectColumns`, `WithoutColumn`, `RenameColumn`, `Skip(0)`, identity
  `ReorderByIndices`, and carried columns in `Normalize`) are shared and are NOT
  disposed when the borrowing frame is disposed.
- `NivaraFrame.Create` and the public constructor own every column they receive.
- Disposing an owning frame disposes its owned columns; any borrower of those
  columns becomes invalid.
- Disposing a borrowing frame leaves its source fully usable.
- Lazy query sources own one file handle released through `QuerySourceHandle`; see
  `## File I/O Architecture` → `### Lazy source ownership and release` (#1189).

Schema responsibilities:

- Column name resolution
- Type consistency enforcement
- Validation of transformations

### Schema Enforcement Decision

**Decision**: Make schemas **explicit, immutable, and mandatory** for all frame operations.

**Context**: Dynamic column manipulation is error-prone and often fails late.

**Alternatives Considered**:
- Implicit schema inference
- Mutable schemas

**Why This Was Chosen**:
- Early schema validation catches many classes of errors
- Immutability simplifies reasoning about transformations

**Consequences**:
- More upfront validation work
- Fewer runtime surprises

Schemas are:

- Immutable
- Validated eagerly
- Required for all frame operations

This allows many errors to surface during query construction rather than execution.

---

## Expression System

Nivara queries are built using a strongly typed expression tree system.

Expression responsibilities:

- Represent column references
- Encode arithmetic and comparison operations
- Preserve type information

Expressions are **pure** and side-effect free.

### Fused Expression Engine

The `FusedExpressionEvaluator` (`src/Nivara/Expressions/FusedExpressionEvaluator.cs`) lowers a validated `ColumnExpression` AST into a single-pass kernel over the whole column with no intermediate columns and no per-element `object?` boxing:

- **SIMD backend** (primary path): Compiles the expression tree with `System.Linq.Expressions` into a cached delegate over typed leaf arrays. The JIT-emitted straight-line loop body auto-vectorizes for numeric leaves. No intermediate columns, no boxing.
- **Flat IR span interpreter** (fallback): A generic node-tree kernel (`FusedKernel`) handles expressions that cannot be compiled to delegates (e.g., null-bearing columns where masks must be OR'd from leaf masks in a separate pass).
- **Chunked streaming**: Expressions are evaluated per-chunk in streaming mode, keeping memory bounded.
- **Ternary `?:`** is supported alongside arithmetic and comparisons — compiled to conditional branches in the delegate path.
- Expressions that cannot be fused at all throw (no boxed fallback). Comparisons produce masked-false at nulls (SQL-like semantics).

---

## Lazy Query Engine

### QueryFrame

`Nivara.Query.QueryFrame` represents a lazily constructed computation graph.

**Characteristics:**
- **Lazy construction** - no data processing until `Collect()`
- **Operation chaining** - builds logical plans through method calls
- **Schema validation** - validates operations before execution
- **Optimization opportunities** - plans can be analyzed and optimized

### Execution Strategies

Nivara supports multiple execution strategies via the `Nivara.Execution.ExecutionStrategy` enum. All strategies share a common `ExecutionStrategyBase` class that eliminates boilerplate (null checks, cancellation, error wrapping). All four strategies are fully implemented — Parallel and Streaming are no longer placeholders.

#### 1. Lazy (Default)
- Builds complete query plan before execution
- Applies optimizations transparently
- Best memory efficiency through deferred evaluation
- Optimal for complex multi-operation queries

#### 2. Eager
- Executes operations immediately without planning
- Simpler debugging with predictable execution order
- Good for simple, single-operation queries
- No optimization overhead

#### 3. Streaming
- Processes data in configurable chunks using memory budget
- Controlled memory usage for large datasets
- Automatic chunk size calculation
- Suitable for datasets larger than available memory
- Falls back to lazy for operations requiring full data (Sort, GroupBy, Join)
- **Public API**: `QueryFrame.AsStream(chunkSize, ct)` yields one `NivaraFrame` per source chunk via `IAsyncEnumerable<T>`; `NivaraQuery<T>.AsStream` passthrough. Lazy query-frame factories (`Csv.ScanAsQueryFrame`, `Json.ScanAsQueryFrame`, `NivaraParquetReader.ScanAsQueryFrame`) open streaming directly from files.

#### 4. Parallel
- Uses multiple threads for CPU-intensive operations
- Automatic detection of parallelizable operations via `IParallelSortOperation`, `IParallelGroupByOperation`, `IParallelJoinOperation`, `IParallelConcatenationOperation`
- Configurable degree of parallelism
- Best for compute-heavy workloads on multi-core systems

#### Execution Context Configuration

```csharp
var context = new Nivara.Execution.ExecutionContext
{
    Strategy = Nivara.Execution.ExecutionStrategy.Parallel,
    MaxDegreeOfParallelism = Environment.ProcessorCount,
    MemoryBudget = 1024 * 1024 * 1024, // 1GB
    CancellationToken = cancellationToken,
    Progress = progressReporter,
    ExecutionDiagnostics = new ExecutionDiagnostics()
};

// After execution, inspect diagnostics
var engine = new ExecutionEngine();
var result = engine.Execute(plan, context);
Console.WriteLine(engine.LastDiagnostics?.GenerateReport());
```

### Planning Stages

1. **Logical Plan Construction**
   - Operation validation and type checking
   - Schema transformation and compatibility verification
   - Dependency analysis for optimization safety

2. **Query Optimization** (Lazy strategy only)
   - Rule-based optimization engine
   - Conservative transformations that preserve correctness
   - Predicate pushdown, projection pruning, operation fusion

3. **Physical Plan Generation**
   - Execution strategy selection
   - Kernel selection (scalar vs vectorized)
   - Resource allocation and memory planning

4. **Execution**
   - Triggered by `Collect()` or materialization methods
   - Error handling with structured context
   - Progress reporting and cancellation support

Execution begins only when `Collect()` or similar materialization methods are invoked.

### Public Streaming API

`QueryFrame.AsStream(chunkSize, ct)` yields one `NivaraFrame` per source chunk via `IAsyncEnumerable<T>`. Fully-streamable plans (Filter, Select, Slice, SelectRows) process each chunk independently; non-streamable boundaries (Sort, GroupBy, Join, Rolling, Cumulative, Shift, Rank) or window expressions fall back to a single merged frame identical to `CollectAsync()`.

Lazy query-frame factories open streaming directly from files — each takes a path and an
options object, and no chunk size:
- `Csv.ScanAsQueryFrame(string filePath, CsvOptions? options = null)` — `src/Nivara.Extensions/IO/CsvExtensions.cs`
- `Json.ScanAsQueryFrame(string filePath, JsonOptions? options = null)` — `src/Nivara/IO/JsonExtensions.cs`
- `NivaraParquetReader.ScanAsQueryFrame(string filePath, ParquetReadOptions? options = null)` — `src/Nivara.Extensions/IO/NivaraParquetReader.cs`

There is no `Parquet` factory class and no `chunkSize` parameter on any of the three; the
chunk size is supplied when streaming, by `AsStream(chunkSize:)`. It is honored by
row-oriented sources and advisory (row-group aligned) for Parquet; when unset it is derived
from the memory budget (`clamp(budget/10 ÷ 100 bytes/row, 1000, 100000)`). Full contract in
`docs/STREAMING.md`.

## Query Optimization Engine

### OptimizationEngine Architecture

The `Nivara.Optimization.OptimizationEngine` applies rule-based optimizations to query plans:

**Core Components:**
- **Nivara.Optimization.OptimizationRule** base class for implementing optimization strategies
- **Rule prioritization** system for applying optimizations in correct order
- **Statistics tracking** for performance analysis and debugging
- **Conservative approach** that preserves correctness over aggressive optimization

### Built-in Optimization Rules

#### 1. PredicatePushdownRule
- **Purpose**: Moves filter operations closer to data sources
- **Benefit**: Reduces data movement and processing overhead
- **Safety**: Only applied when predicate dependencies are verified
- **Location**: `Nivara.Optimization.PredicatePushdownRule`

#### 2. ProjectionPushdownRule  
- **Purpose**: Eliminates unused columns early in the pipeline
- **Benefit**: Reduces memory usage and I/O overhead
- **Safety**: Analyzes column usage across entire query plan
- **Location**: `Nivara.Optimization.ProjectionPushdownRule`

#### 3. ColumnEliminationRule
- **Purpose**: Removes columns not needed by subsequent operations
- **Benefit**: Minimizes memory footprint and processing time
- **Safety**: Performs dependency analysis to ensure required columns are preserved
- **Location**: `Nivara.Optimization.ColumnEliminationRule`

#### 4. OperationFusionRule
- **Purpose**: Combines compatible operations (multiple filters, projections)
- **Benefit**: Reduces operation overhead and improves cache locality
- **Safety**: Only fuses operations with identical semantics
- **Location**: `Nivara.Optimization.OperationFusionRule`

### Optimization Process

```csharp
var optimizer = new Nivara.Query.QueryOptimizer();
var optimizedPlan = optimizer.Optimize(originalPlan);

// Access optimization statistics
var stats = optimizer.Engine.GetStatistics();
foreach (var stat in stats)
{
    Console.WriteLine($"{stat.RuleName}: {stat.EstimatedImprovementPercent:F1}% improvement");
}
```

### Safety Guarantees

- **Correctness Preserved**: Optimizations never change query semantics or results
- **Conservative Approach**: When optimization safety cannot be proven, the rule is skipped
- **Graceful Degradation**: Failed optimizations don't break queries - original plan is used
- **Schema Validation**: All optimizations respect type safety and schema constraints
- **Dependency Analysis**: Rules analyze operation dependencies before applying transformations

---

## DataFrame Operations Architecture

### Operation Implementation Pattern

All DataFrame operations implement the `Nivara.Query.IQueryOperation` interface. Operation types are referenced via `OperationType` constants (`OperationType.Filter`, `OperationType.Select`, etc.) instead of magic strings:

```csharp
public interface IQueryOperation
{
    string OperationType { get; }
    Schema TransformSchema(Schema inputSchema);
    IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input);
}
```

### Core Operations

#### Filtering Operations
- **FilterOperation**: Applies predicates to filter rows (`Nivara.Operations.FilterOperation`)
- **SliceOperation**: Extracts row ranges (Take, Skip, Slice) (`Nivara.Operations.SliceOperation`)
- **SelectOperation**: Applies complex selection criteria (`Nivara.Operations.SelectOperation`)

#### Transformation Operations  
- **ProjectionOperation**: Selects and renames columns (`Nivara.Operations.ProjectionOperation`)
- **SortOperation**: Multi-column sorting with null ordering control (`Nivara.Operations.SortOperation`)
- **GroupByOperation**: Groups data by key columns with aggregation support (`Nivara.Operations.GroupByOperation`)

#### Combination Operations
- **JoinOperation**: Inner, left, right, and full outer joins with configurable conflict resolution (`Nivara.Operations.JoinOperation`)
- **ConcatenationOperation**: Vertical and horizontal DataFrame combination (`Nivara.Operations.ConcatenationOperation`)

#### Aggregation Operations
- **AggregationFunction**: Base class for Sum, Count, Mean, Min, Max, Quantile, Median, StdDev, Variance (`Nivara.Operations.AggregationFunction`)
- **Vectorized aggregations**: Automatic SIMD acceleration for numeric types
- **Null-aware processing**: Proper null handling in all aggregation functions

#### Window Functions
- **`Over()` / `WindowSpec` builder** (#162): SQL-style partitioned windows on both `NivaraFrame` (eager) and `QueryFrame` (lazy). `Over()` returns an immutable `WindowSpec` with `PartitionBy(params string[])` and `OrderBy` overloads (ascending / NULLS LAST by default).
- **Rolling windows**: `Sum`, `Mean`, `Min`, `Max` with configurable window size
- **Cumulative windows**: `Sum`, `Max`, `Min`, `Product`, `Count`
- **Shift / Lead**: Look ahead or behind within partitions
- **Rank family**: `RowNumber`, `Rank`, `DenseRank`, `PercentRank` — preserving existing null-order-key semantics
- **Execution**: Partition → sort → compute → scatter via the shared `PartitionedWindowEngine` (`src/Nivara/Tensors/PartitionedWindowEngine.cs`); partition and order keys are validated up front.

#### Conditional Expressions
- **Ternary `?:` support** in the LINQ expression DSL: `frame.Where(row => row.GetValue<int>("x") > 0 ? row.GetValue<double>("a") : row.GetValue<double>("b"))`. The fused expression engine compiles ternaries into straight-line JIT code alongside arithmetic and comparisons.

### Operation Integration

Operations integrate seamlessly with the query system:

1. **Schema Validation**: `TransformSchema()` validates compatibility before execution
2. **Lazy Evaluation**: Operations build query plans in `Nivara.Query.QueryFrame`
3. **Optimization**: Operations participate in query optimization rules
4. **Error Handling**: Structured exceptions with operation context

### Fluent API Integration

Extension methods provide fluent APIs that delegate to core operations:

```csharp
// Extension method delegates to FilterOperation; the predicate receives a typed NivaraRow
var filtered = frame.Where(row => row.GetValue<int>("Age") > 25);

// Extension method delegates to SortOperation  
var sorted = frame.OrderBy("Name").ThenByDescending("Age");

// Extension method delegates to JoinOperation
var joined = left.Join(right, "Id", JoinType.Inner);
```

### Type Erasure in Execution Engine Decision

**Decision**: Introduce a **non-generic execution interface** beneath generic, user-facing APIs.

**Context**: Execution and planning layers must operate on collections of columns with unknown concrete types at runtime.

**Alternatives Considered**:
- Purely generic designs
- Reflection-based dispatch

**Why This Was Chosen**:
- Pure generics cannot be stored or dispatched uniformly
- Reflection proved fragile and difficult to optimize

**Consequences**:
- Slightly more complex abstraction layers
- Clear separation between compile-time and runtime concerns

---

## Diagnostics and Introspection

Nivara includes diagnostic tooling to aid development and optimization:

- Query plan inspection
- Execution path tracing
- Performance diagnostics

### Diagnostics and Introspection Decision

**Decision**: Expose **explicit diagnostics** for query planning and execution.

**Context**: Invisible optimizations make systems hard to trust and tune.

**Alternatives Considered**:
- Minimal diagnostics
- Debug-only internal tracing

**Why This Was Chosen**:
- Users need to understand performance behavior
- Debugging without introspection proved costly

**Consequences**:
- Additional implementation complexity
- Much improved debuggability and trust

Diagnostics are intentionally explicit and opt-in.

```csharp
var diagnostics = new Nivara.Diagnostics.ExecutionDiagnostics();

var result = Nivara.Diagnostics.DiagnosticHelper.ExecuteWithDiagnostics(
    diagnostics,
    "ComplexQuery",
    () => /* operation */,
    rowCount);

Console.WriteLine(diagnostics.GenerateReport());
```

---

## File I/O Architecture

### Lazy Data Sources

File-based data sources (CSV, JSON) integrate with the query engine:

- Schema inference occurs lazily
- Data is read in chunks
- Filters and projections are pushed down where possible

This enables scalable processing of large files without eager loading.

### Lazy source ownership and release (#1189)

A lazy source owns a single file handle. Release is one-shot through `QuerySourceHandle`
(`src/Nivara/Query/QuerySourceHandle.cs`): disposing any frame in a chain releases the
source for the whole chain, and survivors throw `ObjectDisposedException`.

- **Parquet** holds its reused reader (and so the file stream) until `Dispose()`; its
  stream is opened synchronously, so a dropped scan releases the handle at finalization.
- **CSV / JSON** open their chunk reader lazily. CSV uses a synchronous stream in the
  synchronous read path; JSON always uses `useAsync: true` streams and performs async
  reads, which bind a `ThreadPoolBoundHandle` that finalization alone does *not* unbind —
  an explicit `Dispose()` is required to close the handle.
- **Decision (recorded here, per issue #1189):** `JsonLazySource` and `CsvLazySource`
  carry a finalizer that disposes their reader, so a dropped scan releases within one
  collection cycle. Parquet needs no finalizer (its stream is synchronous and finalizes
  correctly). We do not add a finalizer to Parquet, and we do not keep a
  process-global safety net (removed in #1168): finalizers are a last resort, and here
  they exist only because async I/O binding makes finalization insufficient. Disposing
  the scan you created remains the whole contract; the finalizer is the safety net for
  the forgotten case.

---

## Error Handling Philosophy

Nivara prefers:

- Compile-time errors over runtime errors
- Validation-time errors over execution-time failures
- Explicit exceptions over silent coercion

Many design choices exist solely to surface errors as early as possible.

---

## Testing Strategy Decision

**Decision**: Rely heavily on **property-based testing** for core logic.

**Context**: Traditional unit tests failed to cover edge cases in expression evaluation and optimization.

**Alternatives Considered**:
- Example-based tests only
- Exhaustive hand-written test cases

**Why This Was Chosen**:
- Property-based testing exposed subtle semantic bugs early
- It scales better with system complexity

**Consequences**:
- Steeper learning curve for contributors
- Stronger correctness guarantees


## Automatic Differentiation Engine

Nivara provides a full reverse-mode and forward-mode automatic differentiation (AutoDiff) subsystem as a core component. Moved from Extensions to core in P0–P6, it powers gradient-based optimization and machine learning workloads directly on DataFrame-backed data.

> **For detailed documentation** — operations, modules, optimizers, forward-mode AD, DataFrame integration, and null-handling semantics — see [`docs/AUTODIFF.md`](docs/AUTODIFF.md).

### Summary

```
DataFrame → ReverseGradTensor<T> → Computation Graph (GradNode DAG)
                                        ↓
                                 Backward() pass
                                        ↓
                              Gradients on Parameters
                                        ↓
                           Optimizer.Step() (SGD/Adam/AdamW)
                                        ↓
                                 Updated Parameters
```

**Key components:**
- **`ReverseGradTensor<T>`** / **`ForwardGradTensor<T>`** — autograd tensors for reverse and forward modes
- **`ReverseGradOperations`** / **`ForwardGradOperations`** — static factories of all differentiable operations, fully span-ified (no `NivaraColumn.Data` access; `Span<T>` + `TensorPrimitives` directly)
- **Module system** (`Nn/`) — `Linear`, `Sequential`, `Embedding`, `SparseEmbedding`, `Conv1d`, `Conv2d`, `ConvTranspose2d`, `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `DepthwiseSeparableConv2d`, `TransformerBlock`, `MultiheadAttention`, `ConvVAE`, `VAE` (optional conditioning), `TextClassifierModel`, `TokenClassifierModel`, `MaxPool2d`, `AdaptiveAvgPool2d`, `TextTokenizer`, `Sampler`
- **Optimizers** (`Optimizer/`) — `SGD`, `Adam`, `AdamW`, all with `TensorPrimitives` SIMD chains and `ArrayPool`-backed state
- **Training** (`Training/`) — `TrainingLoop`, `DataParallelTrainer`
- **Serialization** (`Serialization/`) — `ModelSerializer` (JSON save/load), `StateDict()` / `LoadStateDict()`

**Type constraint:** AutoDiff tensors and modules are constrained to `IFloatingPointIeee754<T>`, a narrower but more precise bound than the previous `INumber<T>`. This enables `Half`/F16 and BFloat16 runtime validation alongside `float`/`double`. The `TypeValidator` runtime gatekeeper was removed in favor of the compile-time constraint plus a single supported-type check.

**Inference graph isolation:** Graph construction is strictly scoped to `GradientUtils.Grad()`. All operations call `GradientUtils.ShouldTrackGrad()` to decide whether to attach `OpNode` history; `ComputationGraph.AddNode()` has a `Debug.Assert` regression guard, and a dedicated test suite (`InferenceGraphTests`) verifies that plain forward calls never build a graph.

**Null handling:** Per [ADR-001](docs/adr/001-autodiff-nonnullable-domain.md), AutoDiff is a non-nullable domain — null boundary enforced at `NivaraColumn<T>` → `ReverseGradTensor<T>` conversion. Null-handling branches have been removed from hot paths (`AccumulateGradient`, `BroadcastGradient`, KL/sample ops, `AdamW`) for single-path SIMD execution via `TensorPrimitives`.

**BCE backward fix:** `BCEWithLogitsLoss` backward was rewritten from a multi-op decomposition (Relu + Abs + SoftPlus) to a single fused `OpNode<T>` computing `sigmoid(x) - z` directly. The original decomposition produced incorrect gradients at x=0 where Relu and Abs subgradients are both 0. Forward values unchanged. `LeakyRelu` default slope corrected from 0 to 0.01.

### Convolution Kernel Strategy

Nivara's convolution layers use specialized kernels optimized for common cases while maintaining full autograd correctness.

#### Conv2d: Tiled im2col → Dot

The `Conv2d<T>` kernel follows a tiled im2col pipeline: gather input patches into contiguous buffers, then use `TensorPrimitives.Dot` per output channel. Key optimizations:

- **PatchLocation lookup table**: `PatchLocation` struct precomputes `(Batch, OH, OW)` per-tile, eliminating 4 integer divisions per position from all hot loops
- **ConvForward1x1**: bypasses im2col entirely for 1×1 kernels (stride=1, padding=0). Gathers input channels into pooled buffer, then `Dot` per output channel
- **InputGrad specializations**: `InputGrad1x1` (direct MultiplyAdd), `InputGrad3x3` (bounds-checked 9-tap scatter), `InputGradGeneric` (nested loops) — eliminates im2col for the backward input path
- **Span access via TryGetSpan**: `NivaraColumn.TryGetSpan` returns a true zero-copy read-only view over the backing array when the column has no nulls, and returns `false` when a null mask is present (callers take the explicit-fill path). `AsTensorView()` exposes a lazy zero-copy `Tensor<T>` view for unmanaged element types
- **Grouped convolution**: `groups` parameter splits input/output channels. For `groups=1` (common path), zero overhead — full buffers passed directly. For `groups>1`, gather/scatter per group with NCHW layout

#### Conv1d: 1D im2col → Dot

The `Conv1d<T>` kernel mirrors Conv2d's approach in 1D: tiled `Im2Col1DTile` materializes contiguous channel-major patches, `TensorPrimitives.Dot` per output channel. Specialized `Conv1dForward1x1` path skips im2col for 1×1 kernels. Weight grad uses im2col + `TensorPrimitives.MultiplyAdd`; input grad is a direct scatter loop. Weight layout is PyTorch-compatible `[outChannels, inChannels, kernelSize]`. Kaiming-Uniform initialization.

#### ConvTranspose2d: Direct Scatter

`ConvTranspose2d<T>` uses a direct scatter kernel (`Col2ImForward`) rather than im2col. The forward pass scatters input values across the output grid, while backward uses `ConvTransposeInputGradKernel` (scatter with stride check) and `ConvTransposeWeightGradKernel` (reduction over spatial dimensions). No grouped convolution support.

#### BatchNorm/LayerNorm: Fused Span Kernels

`BatchNormKernel<T>` and `LayerNormKernel<T>` use `TensorPrimitives` for forward and backward — single `OpNode` per call with no expanded tensors. LayerNorm uses `TensorPrimitives.Dot` for SIMD-accelerated sum-of-squares. Both accept `Span<T>`/`ReadOnlySpan<T>` for composability with grouped slicing.

#### RMSNormKernel: Shared Per-Row Normalization

`RMSNormKernel<T>` consolidates the duplicated `PerRowRMSNorm` forward and backward logic into a single shared kernel used by `ReverseGradOperations.PerRowRMSNorm` and `TransformerBlock`. Its backward gradient is SIMD-accelerated via `TensorPrimitives.Dot`. `PerRowLayerNorm` delegates to `LayerNormKernel<T>`.

### Cross-Framework Validation (NivaraTorch)

Nivara's AutoDiff correctness is validated against PyTorch via the `NivaraTorch` test suite (`tests/Nivara.Tests/NivaraTorch/`) — **55 functional tests** asserting forward and backward values against tensors produced by `gen_reference.py`. Layer fixtures cover 21+ layer types (Linear, Conv1d/2d, BatchNorm1d/2d, LayerNorm, Embedding, TransformerBlock, MultiheadAttention, GELU, pooling, etc.) and are regenerated on demand from Python. A cross-framework parity exercise trained an identical 3-layer MLP in both frameworks and measured **<0.04% relative loss-curve divergence over 50 epochs** (see `samples/README.md`).

### Model Interop (SafeTensors)

The `SafeTensorsLoader` (in `samples/Nivara.Samples`) reads Hugging Face `.safetensors` weights with a generic, dtype-aware `Read<T>()` — supporting I32, I64, F16, BF16, and F32 storage converted to the target element type. It powers the `NivaraInference` (MobileNetV2 / ResNet-18) and `NivaraFineTuning` (DistilBERT / SST-2) samples.

---

## Testing Strategy

**Decision**: Rely heavily on **property-based testing** for core logic.

**Rationale**: Traditional unit tests failed to cover edge cases in expression evaluation and optimization. Property-based testing exposed subtle semantic bugs early and scales better with system complexity.

**Implementation**:
- Property-based tests for all core operations
- Invariant checking across transformations
- Randomized input generation with edge case coverage
- Performance regression detection through benchmarking

---

## Extension Architecture

### Promoted extension pieces (v1.5)

- **Silk (OpenCL GPU backend)** — `src/Nivara.Extensions/Silk/` (`namespace Nivara.Silk`): `SilkRuntime`, `SilkLauncher` (internal typed dispatch), `SilkKernels` (internal), `WeightLayout` (`Transposed`/`QkvPacked`/`Req`), the three runners and two sessions. Silk is the only shipped GPU path; ILGPU remains historical reference in `tests/Nivara.GpuProbe` and `docs/ILGPU.md`. See [SILK.md](SILK.md).
- **AI / model layer (BERT family)** — `src/Nivara.Extensions/AI/` (`namespace Nivara.AI`): `BertModel`, `ModernBertModel`, Laya (`LayaHeadModel`, `LayaHeadScoring`, `LayaPromptBuilder`, `LayaCalibration`), `Gpt2BpeTokenizer`, `SafeTensorsLoader`, `StateDictLoader`, and the GPU seam (`GpuSeam.cs`).
- **Embeddings convention** — `NivaraEmbeddingExtensions`: `float[]` column ↔ flat `E0..E{n-1}` frame convention, Microsoft.Extensions.AI conversion in both directions, caller-supplied null policies. See [EMBEDDINGS.md](EMBEDDINGS.md).
- **ONNX interop** — `src/Nivara.Extensions/Onnx/`, consumed by NivaraInference/NivaraClassifier for parity checking.

### Arrow boundary (ADR-007)

- Opt-in zero-copy export via `ArrowConversionOptions.UseZeroCopy` aliases flat primitive column buffers; null-bearing columns rebuild the validity bitmap. Import and Parquet still copy. Buffer-level `ArrowValidityMask` helper exists and is unit-tested but is not yet wired into `ArrowInterop`.

### Modular Design

Core Nivara runtime remains focused and lightweight:
- Essential DataFrame operations in core library
- Optional features in `Nivara.Extensions` package
- Clear separation of concerns
- Minimal dependencies in core package

### Extension Points

- **Custom aggregation functions** through base class extension
- **Custom data sources** via interface implementation
- **Custom execution strategies** through strategy pattern
- **Custom optimization rules** via rule registration

---

This architecture enables Nivara to provide high performance and type safety while maintaining clear separation of concerns and extensibility for future enhancements.

---

## Appendix: Quick Reference — Architecture Decisions

This appendix summarizes key architecture decisions in concise form. See the relevant sections above for detailed rationale.

### Storage & Vectorization

- **Storage strategy**: Single `ColumnStorage<T>` for all types (sole-owner `T[]` + optional `bool[]` null mask); vectorization decided by `KernelSelector` at runtime, never by the storage class.
  - Rationale: one storage path simplifies span/slice semantics; generic static constraints are limiting (CS0080); runtime dispatch is more flexible.
- **Vectorization detection**: Use factory-level checks (`IsVectorizable<T>()`) plus `KernelSelector` heuristics.
  - Rationale: kernel selection is orthogonal to storage; `KernelSelector` centralizes vectorization decisions.
- **Null semantics**: Explicit boolean masks, never NaN-based.
  - Rationale: predictable across all types; NaN only works reliably for IEEE floats.
- **Comparisons**: Return `NivaraColumn<bool>` with SQL-like null propagation (null compared to anything → null).
  - Rationale: avoids surprising booleans, aligns with common DB semantics.

### Series & Column Design

- **Series indexing**: Both position-based (`this[int]`) and label-based (`this[object]`) indexers. Boxed ints route to label indexer.
  - Rationale: preserves both semantics while allowing explicit disambiguation.
- **Kernel selection**: Centralize in a single `DetermineKernelType()` method checking vectorizability, hardware acceleration, and size threshold.

### Query Engine & Optimization

- **Query engine**: Expose generic `IQueryOperation<T>` and public `QueryPlan` to enable external extensions; use immutable `Schema` and `QueryPlan` objects.
- **Query optimization**: Rule-based engine with conservative approach.
  - `OptimizationRule` base class with priority system.
  - Operation type strings for dispatch when internal details inaccessible.
  - Visitor pattern for plan analysis and transformation.
  - Comprehensive statistics tracking and reporting.
- **Optimization safety rules**:
  - Do not change semantics to achieve performance.
  - If uncertain, skip the optimization for that path.
- **Immutable plans**: `QueryPlan` is immutable; optimization passes produce a new `QueryPlan`.
- **Always validate schema transformations**; if optimization fails, fall back to original plan.
- **Use expression analysis (visitor)** to discover referenced columns for column elimination and predicate pushdown.

### DataFrame Operations

- **Pattern**: Implement as `IQueryOperation` with schema validation in `TransformSchema` phase.
- **Concatenation**: Configurable mismatch handling (`Error`, `FillWithNulls`) for vertical; always `Error` for horizontal.
- **Joins**: Hash-based with composite keys, exclude nulls from matching, coalesce join keys for outer joins.
- **Sorting**: Sort indices first, then reorder columns; explicit null ordering (`NullsFirst`/`NullsLast`).
- **Grouping**: Dedicated composite key class with proper equality/hashing; handle nulls explicitly.
- **Window functions**: `Over()` / `WindowSpec` builder for SQL-style partitioned windows. Immutable spec with `PartitionBy` / `OrderBy`. Execution via `PartitionedWindowEngine`: partition → sort → compute → scatter. Supports rolling, cumulative, shift/lead, and rank family.
- **Conditional expressions**: Ternary `?:` in the LINQ DSL, compiled alongside arithmetic and comparisons by the fused expression engine.
- **Fluent API**: Extension methods over core operations. `Where()` predicates receive a typed `NivaraRow` (readonly struct with `GetValue<T>`/`TryGetValue<T>`/`IsNull` accessors). Clear execution semantics: immediate on DataFrame, lazy via `AsQueryFrame()`.

### Streaming API

- **Public API**: `QueryFrame.AsStream(chunkSize, ct)` yields one `NivaraFrame` per source chunk via `IAsyncEnumerable<T>`. `NivaraQuery<T>.AsStream` passthrough.
- **Lazy factories**: `Csv.ScanAsQueryFrame`, `Json.ScanAsQueryFrame`, `NivaraParquetReader.ScanAsQueryFrame` open streaming directly from files.
- **Streamable plans**: Filter, Select, Slice, SelectRows process each chunk independently. Non-streamable boundaries (Sort, GroupBy, Join, windows) fall back to merged frame.
- **Chunk size**: Honored by row-oriented sources; advisory (row-group aligned) for Parquet. Derived from memory budget when unset.
- **Contract**: Full details in `docs/STREAMING.md`.

### Fused Expression Engine

- **SIMD backend**: `FusedExpressionEvaluator` compiles `ColumnExpression` AST to cached delegates over typed leaf arrays — no intermediate columns, no boxing. JIT auto-vectorizes for numeric leaves.
- **Null handling**: Null masks OR'd from leaf masks in a separate pass; comparisons produce masked-false at nulls.
- **Fallback**: `FusedKernel` node-tree interpreter for expressions that cannot compile to delegates. No boxed fallback — non-fusible expressions throw.
- **Chunked evaluation**: Per-chunk in streaming mode for memory-bounded execution.

### Execution & Error Handling

- **Execution strategies**: Pluggable strategy pattern (lazy, eager, streaming, parallel) via `IExecutionStrategy` + `ExecutionStrategyBase` base class eliminating boilerplate.
  - All four strategies are fully implemented (Parallel and Streaming are no longer no-ops).
  - Parallel execution uses dedicated interfaces (`IParallelSortOperation`, `IParallelGroupByOperation`, `IParallelJoinOperation`, `IParallelConcatenationOperation`) for type-safe dispatch.
  - Streaming execution processes data in chunks using memory budget, falling back to lazy for full-data operations (Sort, GroupBy, Join).
  - Cost estimation with operation-specific weights.
  - Async-first design with sync wrappers.
  - Intelligent parallel execution heuristics.
- **Diagnostics integration**: `ExecutionDiagnostics` is wired into the execution pipeline. `NivaraExecutionContext.ExecutionDiagnostics` enables per-execution tracking. `ExecutionEngine.LastDiagnostics` provides post-execution access.
- **Operation types**: Use `OperationType` constants class instead of magic strings across all operations.
- **Error handling**: Structured exception hierarchy with query context.
  - Include failed plan and operation references in exceptions.
  - Specific exception types for common failure modes.

### Target Framework

- **.NET 11** with latest C# features, matching the `net11.0` target of `src/Nivara/Nivara.csproj` and `src/Nivara.Extensions/Nivara.Extensions.csproj`.
  - Rationale: leverage latest performance improvements; `System.Numerics.Tensors` 11.x requires modern .NET.
