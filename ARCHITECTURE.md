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

- Columnar storage abstractions
- Schema-aware frames
- A lazy query layer
- Execution kernels (scalar and vectorized)
- Diagnostics and planning infrastructure

```
Data Source (CSV / JSON)
        ↓
QueryFrame (lazy)
        ↓
Logical Plan
        ↓
Physical Plan
        ↓
Execution Kernels
        ↓
Materialized Frame / Column
```

---

## Column Storage Model

### NivaraColumn<T>

`NivaraColumn<T>` is the fundamental data container.

Key characteristics:

- Strongly typed (`T` is never erased)
- Immutable
- Backed by contiguous memory
- Optional null mask

Each column owns:

- A data buffer (`T[]`, `Memory<T>`, or tensor-backed storage)
- A validity bitmap (when nulls are present)

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

Nivara distinguishes between two storage strategies:

### Memory Storage

Used when:

- `T` is not vectorizable
- Operations cannot be safely SIMD-accelerated

Characteristics:

- Simple memory layout
- Scalar execution
- Predictable semantics

### Tensor Storage

Used when:

- `T` supports SIMD operations
- Operations are safe to vectorize

Characteristics:

- Uses `System.Numerics.Vector<T>`
- Automatically selected
- Falls back to scalar paths when required

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

> Detailed rules, edge cases, and LLM-specific guidance are documented in (GUIDELINES)[GUIDELINES.md]

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

`QueryFrame` represents a lazily constructed computation.

Characteristics:

- No data is read or processed eagerly
- Operations build a logical plan
- Plans are validated before execution

### Lazy Execution Model Decision

**Decision**: Adopt a **lazy query construction model** with explicit execution boundaries.

**Context**: Eager execution limits optimization opportunities and makes composition difficult.

**Alternatives Considered**:
- Fully eager execution
- Partial eagerness with implicit triggers

**Why This Was Chosen**:
- Lazy plans allow validation and optimization before execution
- Explicit execution points make behavior predictable

**Consequences**:
- Errors may surface later than construction time
- Requires careful validation to avoid deferred failures

### Planning Stages

1. **Logical Plan**
   - Operation ordering
   - Predicate composition
   - Projection pruning

2. **Physical Plan**
   - Kernel selection
   - Vectorization decisions
   - Null-aware execution paths

Execution begins only when `Collect()` is invoked.

### Query Optimization Scope Decision

**Decision**: Restrict optimizations to **conservative, provably safe transformations**.

**Context**: Aggressive query reordering risks changing semantics in the presence of dependencies.

**Alternatives Considered**:
- Cost-based aggressive optimization
- Heuristic reordering without dependency analysis

**Why This Was Chosen**:
- Correctness regressions are extremely difficult to debug
- Many optimizations provide marginal gains compared to their risk

**Consequences**:
- Some performance opportunities are intentionally skipped
- System behavior remains predictable and correct

---

## Execution Kernels

Execution kernels are responsible for applying operations to data.

Types of kernels:

- Scalar kernels
- Vectorized (SIMD) kernels
- Null-aware variants

Kernel selection is driven by:

- Data type
- Presence of nulls
- Operation semantics

Kernels are designed to be:

- Side-effect free
- Easily testable
- Benchmarkable in isolation

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
var context = new ExecutionContext
{
    Strategy = ExecutionStrategy.Parallel,
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
var diagnostics = new ExecutionDiagnostics();

var result = DiagnosticHelper.ExecuteWithDiagnostics(
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

