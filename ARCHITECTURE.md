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

- **Columnar storage abstractions** (`Nivara.Storage.MemoryStorage<T>`, `Nivara.Storage.TensorStorage<T>`)
- **Schema-aware frames** (`NivaraFrame`, `Nivara.Query.QueryFrame`)
- **A lazy query layer** with optimization engine (`Nivara.Query`, `Nivara.Optimization`)
- **Execution strategies** (`Nivara.Execution` - lazy, eager, streaming, parallel)
- **Operation implementations** (`Nivara.Operations` - joins, aggregations, filtering, sorting)
- **Diagnostics and planning infrastructure** (`Nivara.Diagnostics`)

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
- **Pluggable storage backend** (Memory vs Tensor storage)
- **Explicit null handling** via optional boolean masks

Each column delegates to an `IColumnStorage<T>` implementation:

- **Data storage** (`Nivara.Storage.MemoryStorage<T>` or `Nivara.Storage.TensorStorage<T>`)
- **Null mask** (when nulls are present)
- **Type-specific operations** (scalar vs vectorized)

### Storage Backend Selection

The `ColumnStorageFactory` automatically selects the optimal storage backend:

#### MemoryStorage<T>
- Used for non-vectorizable types (strings, objects, complex types)
- Simple `Memory<T>` backing with scalar operations
- Predictable performance characteristics
- Lower memory overhead

#### TensorStorage<T>
- Used for vectorizable types (`int`, `float`, `double`, `bool`)
- Backed by `System.Numerics.Tensors`
- Automatic SIMD acceleration via `TensorPrimitives`
- Falls back to scalar operations when needed

This selection is **transparent** to users and happens at column creation time.

### Null Representation Decision

**Decision**: Use **explicit validity masks** to represent nulls instead of sentinel values.

**Context**: The system operates on strongly typed, columnar data and must support both numeric and non-numeric types consistently.

**Alternatives Considered**:
- Sentinel values (e.g. `NaN`)
- Type-specific nullable encodings

**Why This Was Chosen**:
- Sentinel values do not generalize beyond floating-point types
- They leak into user-visible results and complicate semantics
- A separate validity mask keeps data and nullness orthogonal

**Consequences**:
- Slight memory overhead
- Much clearer and more predictable null propagation

Nulls are **not** encoded using sentinel values such as `NaN`. Null handling is explicit and first-class:

- Presence of nulls is tracked separately from data
- Operations propagate nulls deterministically
- Null-aware kernels are selected when required

Null-related behavior is a major source of subtle bugs.

> Detailed rules, edge cases, and LLM-specific guidance are documented in [GUIDELINES](GUIDELINES.md)

---

## Memory vs Tensor Storage

Nivara uses a factory-based approach to select optimal storage:

### MemoryStorage<T>

**Used when:**
- `T` is not vectorizable (strings, objects, DateTime, etc.)
- Operations cannot benefit from SIMD acceleration

**Characteristics:**
- Simple `Memory<T>` layout
- Scalar execution paths only
- Predictable semantics and performance
- Lower memory overhead
- Faster for non-numeric operations

### TensorStorage<T>

**Used when:**
- `T` supports SIMD operations (`int`, `float`, `double`, `bool`)
- Operations can be safely vectorized

**Characteristics:**
- Uses `System.Numerics.Tensors` for storage
- Automatic SIMD acceleration via `TensorPrimitives`
- Falls back to scalar paths when vectorization isn't beneficial
- Higher performance for mathematical operations
- Slightly higher memory overhead

### Automatic Selection

The `Nivara.Storage.ColumnStorageFactory.IsVectorizable<T>()` method determines storage type:

```csharp
// Automatically selects TensorStorage<int>
var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 });

// Automatically selects MemoryStorage<string>
var stringColumn = NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c" });
```

This distinction is **internal** and transparent to users.

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

Nivara supports multiple execution strategies via the `Nivara.Execution.ExecutionStrategy` enum:

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
- Processes data in configurable chunks
- Controlled memory usage for large datasets
- Automatic chunk size calculation
- Suitable for datasets larger than available memory

#### 4. Parallel
- Uses multiple threads for CPU-intensive operations
- Automatic detection of parallelizable operations
- Configurable degree of parallelism
- Best for compute-heavy workloads on multi-core systems

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

All DataFrame operations implement the `Nivara.Query.IQueryOperation` interface:

```csharp
public interface IQueryOperation
{
    string OperationType { get; }
    Schema TransformSchema(Schema inputSchema);
    NivaraFrame Execute(NivaraFrame input);
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
- **AggregationFunction**: Base class for Sum, Count, Mean, Min, Max, etc. (`Nivara.Operations.AggregationFunction`)
- **Vectorized aggregations**: Automatic SIMD acceleration for numeric types
- **Null-aware processing**: Proper null handling in all aggregation functions

### Operation Integration

Operations integrate seamlessly with the query system:

1. **Schema Validation**: `TransformSchema()` validates compatibility before execution
2. **Lazy Evaluation**: Operations build query plans in `Nivara.Query.QueryFrame`
3. **Optimization**: Operations participate in query optimization rules
4. **Error Handling**: Structured exceptions with operation context

### Fluent API Integration

Extension methods provide fluent APIs that delegate to core operations:

```csharp
// Extension method delegates to FilterOperation
var filtered = frame.Where(row => row.Age > 25);

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

---

## File I/O Architecture

### Lazy Data Sources

File-based data sources (CSV, JSON) integrate with the query engine:

- Schema inference occurs lazily
- Data is read in chunks
- Filters and projections are pushed down where possible

This enables scalable processing of large files without eager loading.

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

---

## Storage Architecture

### Column Storage Model

`NivaraColumn<T>` is the fundamental data container with these characteristics:

- Strongly typed (`T` is never erased)
- Immutable by design
- Backed by contiguous memory
- Optional null mask for explicit null tracking

Each column owns:
- A data buffer (`T[]`, `Memory<T>`, or tensor-backed storage)
- A validity bitmap (when nulls are present)

### Memory vs Tensor Storage Strategy

Nivara distinguishes between two storage strategies based on data type characteristics:

#### Memory Storage
Used when:
- `T` is not vectorizable (strings, objects, complex types)
- Operations cannot be safely SIMD-accelerated

Characteristics:
- Simple memory layout using `Memory<T>`
- Scalar execution paths
- Predictable semantics
- Lower memory overhead

#### Tensor Storage
Used when:
- `T` supports SIMD operations (`int`, `float`, `double`, `bool`)
- Operations are safe to vectorize

Characteristics:
- Uses `System.Numerics.Tensors` for storage
- Automatic SIMD acceleration via `TensorPrimitives`
- Falls back to scalar paths when required
- Higher performance for mathematical operations

This distinction is **internal** and transparent to users. The system automatically selects the optimal storage backend based on type analysis.

### Vectorization Strategy

**Decision**: Apply vectorization selectively and only where semantics are trivial.

**Rationale**: SIMD execution provides performance gains but complicates control flow and debugging. Premature vectorization obscured correctness issues during development.

**Implementation**:
- Vectorization is applied opportunistically
- Automatic fallback to scalar operations when vectorization isn't safe
- Mixed execution paths with clear boundaries
- Easier correctness validation through separation of concerns

---

## Query Engine Architecture

### Lazy Query Construction

The query engine uses a **lazy evaluation model** with explicit execution boundaries:

```
Data Source → QueryFrame (lazy) → Logical Plan → Physical Plan → Execution Kernels → Results
```

#### Planning Stages

1. **Logical Plan Construction**
   - Operation ordering and validation
   - Predicate composition and simplification
   - Projection pruning and column elimination
   - Schema validation and type checking

2. **Physical Plan Generation**
   - Kernel selection (scalar vs vectorized)
   - Vectorization decisions based on data types
   - Null-aware execution path selection
   - Resource allocation planning

3. **Execution**
   - Begins only when `Collect()` is invoked
   - Applies optimizations transparently
   - Handles errors with structured context

### Query Optimization Engine

The optimizer applies conservative, provably safe transformations:

#### Optimization Strategies

- **Predicate Pushdown**: Moves filter operations closer to data sources to reduce data movement
- **Projection Pushdown**: Eliminates unused columns early in the pipeline to reduce memory usage  
- **Operation Fusion**: Combines compatible operations (multiple filters, projections) to reduce overhead
- **Column Elimination**: Removes columns that aren't needed by subsequent operations

#### Safety Guarantees

- **Correctness Preserved**: Optimizations never change query results
- **Conservative Approach**: When in doubt, chooses correctness over performance
- **Graceful Degradation**: Failed optimizations don't break queries
- **Schema Validation**: All optimizations respect type safety and schema constraints

### Execution Strategies

Multiple execution strategies optimize performance for different use cases:

#### Strategy Types

1. **Lazy Execution (Default)**
   - Builds query plans and optimizes before execution
   - Optimal memory usage through deferred evaluation
   - Best for complex queries with multiple operations

2. **Eager Execution**
   - Executes operations immediately without optimization
   - Easier debugging with predictable execution order
   - Good for simple, single-operation queries

3. **Streaming Execution**
   - Processes data in chunks for large datasets
   - Controlled memory usage with configurable budgets
   - Automatic chunk size calculation

4. **Parallel Execution**
   - Uses multiple threads for CPU-intensive operations
   - Automatic detection of parallelizable operations
   - Configurable degree of parallelism

#### Execution Context Configuration

```csharp
var context = new Nivara.Execution.ExecutionContext
{
    Strategy = Nivara.Execution.ExecutionStrategy.Parallel,
    MaxDegreeOfParallelism = Environment.ProcessorCount,
    MemoryBudget = 1024 * 1024 * 1024, // 1GB
    CancellationToken = cancellationToken,
    Progress = progressReporter
};
```

---

## Error Handling Architecture

### Structured Exception Hierarchy

Nivara uses a structured exception hierarchy that provides detailed context about failures:

#### Exception Types

- **JoinException**: Join operation failures with key and type information
- **SchemaValidationException**: Schema mismatch details with specific error locations
- **QueryExecutionException**: Query execution failures with plan context
- **DataFrameOperationException**: General DataFrame operation errors

#### Context Preservation

Each exception includes:
- Operation context and parameters
- Query plan information (when applicable)
- Schema details and type information
- Suggested remediation steps

### Performance Diagnostics

#### Execution Diagnostics

Track performance metrics and identify optimization opportunities:

```csharp
var diagnostics = new Nivara.Diagnostics.ExecutionDiagnostics();

var result = Nivara.Diagnostics.DiagnosticHelper.ExecuteWithDiagnostics(
    diagnostics,
    "ComplexQuery",
    () => /* operation */,
    rowCount);

Console.WriteLine(diagnostics.GenerateReport());
```

#### Diagnostic Features

- **Operation-level timing** with nested scope support
- **Memory usage tracking** with allocation patterns
- **Automatic warning detection** for common performance issues
- **Optimization tracking** with estimated performance improvements
- **Comprehensive reporting** with human-readable analysis

---

## Type System Architecture

### Type Erasure Strategy

**Decision**: Introduce a **non-generic execution interface** beneath generic, user-facing APIs.

**Context**: Execution and planning layers must operate on collections of columns with unknown concrete types at runtime.

**Implementation**:
- Generic APIs (`NivaraColumn<T>`, `NivaraFrame`) for compile-time safety
- Non-generic interfaces (`IColumn`, `IFrame`) for runtime operations
- Clear separation between compile-time and runtime concerns
- Efficient dispatch without reflection overhead

### Schema Enforcement

**Decision**: Make schemas **explicit, immutable, and mandatory** for all frame operations.

**Benefits**:
- Early schema validation catches many classes of errors
- Immutability simplifies reasoning about transformations
- Explicit validation prevents runtime surprises

**Implementation**:
- Schema validation occurs at construction time
- All transformations produce new schemas
- Type compatibility checked before operations
- Clear error messages for schema violations

---

## File I/O Architecture

### Lazy Data Sources

File-based data sources integrate seamlessly with the query engine:

#### CSV/JSON Integration
- Schema inference occurs lazily during query planning
- Data is read in chunks during execution
- Filters and projections are pushed down where possible
- Automatic type detection with configurable overrides

#### Parquet Integration (via Extensions)
- Columnar format alignment with Nivara's architecture
- Zero-copy operations where possible
- Configurable compression and row group sizes
- Streaming support for large files

### Arrow Interoperability (via Extensions)

#### Conversion Strategy
- Bidirectional conversion between Nivara and Arrow formats
- Type mapping with validation and error handling
- Zero-copy optimization when memory layouts align
- Proper null semantics preservation

---

## Memory Management

### Buffer Management

- **Pooled allocations** for frequently used buffer sizes
- **Automatic cleanup** with deterministic disposal patterns
- **Memory budget enforcement** in streaming scenarios
- **Garbage collection optimization** through object lifetime management

### Resource Management

```csharp
public class ResourceManager : IDisposable
{
    // Manages buffer pools, memory budgets, and cleanup
    // Provides centralized resource allocation and tracking
    // Enables memory pressure monitoring and response
}
```

**Location**: `Nivara.Helpers.ResourceManager`

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

- **Storage strategy**: Default to `MemoryStorage` for non-vectorizable types and `TensorStorage` for vectorizable types. Use `ColumnStorageFactory` with runtime type checks.
  - Rationale: generic static constraints are limiting (CS0080); runtime dispatch is more flexible.
- **Vectorization detection**: Use factory-level checks (`IsVectorizable<T>()`) rather than instance flags.
  - Rationale: storage selection vs arithmetic vectorization have different concerns; factory centralizes decisions.
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
- **Fluent API**: Extension methods over core operations. Use `ExpandoObject` for `Where()` predicates. Clear execution semantics: immediate on DataFrame, lazy via `AsQueryFrame()`.

### Execution & Error Handling

- **Execution strategies**: Pluggable strategy pattern (lazy, eager, streaming, parallel).
  - Cost estimation with operation-specific weights.
  - Async-first design with sync wrappers.
  - Intelligent parallel execution heuristics.
- **Error handling**: Structured exception hierarchy with query context.
  - Include failed plan and operation references in exceptions.
  - Specific exception types for common failure modes.

### Target Framework

- **.NET 10.0** with latest C# features.
  - Rationale: leverage latest performance improvements; `System.Numerics.Tensors` requires modern .NET.
