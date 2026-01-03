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

## Notes on Reconsideration

Decisions in this document are not immutable.

However, revisiting a decision should:
- Reference this document explicitly
- Identify which constraints have changed
- Explain why previous trade-offs no longer apply

If a reconsideration yields a **general lesson**, that lesson should be added to `GUIDELINES.md` — not duplicated here.

---

## Where to Put New Architecture Knowledge

- **Reusable rules, invariants, and pitfalls** → [GUIDELINES](GUIDELINES.md)
- **User-facing explanations** → [README](README.md)
- **Workflow and process** → [CONTRIBUTING](CONTRIBUTING.md)

When in doubt, do not duplicate — reference instead.

---

This document should evolve as Nivara grows, but its purpose should remain stable: **explain how the system works, not how to use it**.

