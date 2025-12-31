# Nivara Architecture

This document describes the **internal architecture and design decisions** of Nivara. It is intended for contributors, advanced users, and AI agents working on the codebase.

> **Scope**
> - Explains *how* Nivara is implemented
> - Describes major subsystems and their responsibilities
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

Many implementation choices make more sense when viewed through this lens.

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

Nulls are **not** encoded using sentinel values such as `NaN`.

> See (GUIDELINES)[GUIDELINES.md] for null-handling constraints and common pitfalls.

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

---

## Diagnostics and Introspection

Nivara includes diagnostic tooling to aid development and optimization:

- Query plan inspection
- Execution path tracing
- Performance diagnostics

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

## Where to Put New Architecture Knowledge

- **Reusable rules, invariants, and pitfalls** → [GUIDELINES](GUIDELINES.md)
- **User-facing explanations** → [README](README.md)
- **Workflow and process** → [CONTRIBUTING](CONTRIBUTING.md)

When in doubt, do not duplicate — reference instead.

---

This document should evolve as Nivara grows, but its purpose should remain stable: **explain how the system works, not how to use it**.

