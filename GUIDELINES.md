# GUIDELINES

## Purpose

This document captures **transferable engineering lessons** discovered while building data-intensive, lazy-execution systems. It is intentionally **not project-specific** and avoids recording historical implementation details.

Its primary audience is:
- Future humans working on similar systems
- AI agents assisting with design, refactoring, and optimization

The goal is to **prevent repeated exploration of known dead ends** and to encode patterns that reliably worked under real constraints.

> Focus on **why certain approaches failed or succeeded**, not on *what* was implemented in a specific codebase.

---

## How to Use This Document

Each section is written using the same structure:

- **Problem** – A recurring engineering challenge
- **Constraint** – Why the obvious solution fails
- **Pattern That Worked** – A reusable solution
- **Negative Rule** – What to avoid next time
- **Outcome** – Why this pattern generalizes

When extending this document:
- Avoid project names and concrete APIs
- Prefer abstraction over narration
- Capture failure modes aggressively

---

## Core Engineering Principles

### Correctness Over Performance

**Problem**  
Performance optimizations often introduce subtle semantic bugs.

**Constraint**  
Many optimizations are only conditionally safe and require deep dependency analysis.

**Pattern That Worked**  
Adopt a correctness-first mindset: a slow but correct system is preferable to a fast incorrect one.

**Negative Rule**  
Never apply an optimization unless its semantic safety can be proven.

**Outcome**  
This dramatically reduces hard-to-debug correctness issues and builds trust in the system.

---

### Explicit Over Implicit Behavior

**Problem**  
Implicit behavior (auto-coercion, hidden execution, silent fallbacks) obscures system behavior.

**Constraint**  
Implicit systems are hard to reason about, test, and optimize.

**Pattern That Worked**  
Make execution boundaries, error conditions, and type behavior explicit.

**Negative Rule**  
Avoid “magic” behavior that changes execution strategy without visibility.

**Outcome**  
Systems become easier to debug, explain, and extend.

---

## Data & Null Semantics

### Avoid NaN-Based Null Semantics

**Problem**  
Representing missing data in numeric systems is non-trivial.

**Constraint**  
NaN-based null semantics:
- Do not work for integer types
- Leak into user-visible results
- Behave inconsistently across operations

**Pattern That Worked**  
Track nulls explicitly using boolean masks that participate in all operations.

**Negative Rule**  
Do not encode nulls using sentinel numeric values.

**Outcome**  
Null behavior becomes predictable, type-agnostic, and testable.

---

### Null Propagation Rules

**Problem**  
Operations involving null values often produce ambiguous results.

**Constraint**  
Implicit null handling leads to silent data corruption.

**Pattern That Worked**  
Define a strict rule: *any operation involving a null produces a null*.

**Negative Rule**  
Never “fill” or ignore nulls implicitly during computation.

**Outcome**  
Data integrity is preserved across all transformations.

---

## Generic & Type System Design

### Type Erasure for Runtime Systems

**Problem**  
Query engines and expression evaluators often need to operate on values of unknown generic types at runtime.

**Constraint**  
Purely generic designs break down when types must be stored, inspected, or dispatched dynamically.

**Pattern That Worked**  
Introduce a type-erased interface for runtime operations, layered beneath a generic, type-safe API.

**Negative Rule**  
Avoid reflection-heavy designs for core execution paths.

**Outcome**  
Runtime flexibility is achieved without sacrificing compile-time safety for user code.

---

### Over-Constraining Generics

**Problem**  
Strong generic constraints appear attractive for numeric and algebraic systems.

**Constraint**  
Compile-time generic constraints quickly become unmanageable for:
- Operator overloading
- Mixed-type operations
- Extensible execution engines

**Pattern That Worked**  
Prefer runtime type inspection and dispatch in complex generic scenarios.

**Negative Rule**  
Do not attempt to encode all numeric rules into generic constraints.

**Outcome**  
The system remains flexible and evolvable.

---

## Lazy Execution & Deferred Errors

### Deferred Error Handling Trade-offs

**Problem**  
Lazy systems defer work — and therefore errors — until execution.

**Constraint**  
Deferring errors can:
- Mask failures during query construction
- Cause cascading validation issues later

**Pattern That Worked**  
Defer only *unavoidable* errors while validating structure and schema eagerly.

**Negative Rule**  
Do not defer errors that invalidate the logical correctness of a query.

**Outcome**  
Lazy semantics are preserved without sacrificing debuggability.

---

## Query Planning & Optimization

### Conservative Optimization Strategy

**Problem**  
Query reordering and optimization can dramatically improve performance.

**Constraint**  
Many optimizations subtly change semantics when dependencies exist.

**Pattern That Worked**  
Apply only conservative optimizations unless full dependency analysis proves safety.

**Negative Rule**  
If optimization safety is uncertain, skip the optimization.

**Outcome**  
Correctness is preserved while still allowing meaningful performance gains.

---

### Dependency Awareness

**Problem**  
Optimizations often assume independence between operations.

**Constraint**  
Expressions may reference overlapping data, making reordering unsafe.

**Pattern That Worked**  
Analyze dependencies explicitly before transforming execution plans.

**Negative Rule**  
Never reorder operations blindly.

**Outcome**  
Optimization remains predictable and correct.

---

## Interoperability & External APIs

### Zero-Copy Aspirations vs Reality

**Problem**  
High-performance systems often promise zero-copy operations for interoperability.

**Constraint**  
True zero-copy requires:
- Shared memory layout assumptions
- Compatible lifetime management
- Exposed internal data structures

**Pattern That Worked**  
Design for zero-copy but implement with copying initially. Optimize to true zero-copy only when:
- Performance profiling proves it's necessary
- Internal APIs can safely expose underlying data
- Memory layout compatibility is guaranteed

**Negative Rule**  
Do not compromise internal API design for theoretical zero-copy benefits.

**Outcome**  
Systems remain flexible and correct while preserving optimization opportunities.

---

### Method Overload Resolution in Generic Contexts

**Problem**  
Generic methods with similar signatures create ambiguous overload resolution.

**Constraint**  
Type inference often fails when:
- Multiple generic methods match the same call pattern
- Optional parameters create multiple valid resolutions
- Return types differ but parameters are identical

**Pattern That Worked**  
Design method signatures to be unambiguous:
- Use different parameter counts or types
- Provide explicit disambiguation parameters
- Consider separate method names for fundamentally different operations

**Negative Rule**  
Do not rely on return type differences alone to distinguish overloads.

**Outcome**  
API calls become predictable and less error-prone.

---

### Make Performance Explainable

**Problem**  
Users struggle to trust systems that optimize invisibly.

**Constraint**  
Silent optimizations make debugging and tuning difficult.

**Pattern That Worked**  
Expose diagnostic information explaining:
- Execution strategy
- Optimization decisions
- Performance characteristics

**Negative Rule**  
Do not hide execution behavior from users.

**Outcome**  
Users gain insight and confidence without needing internal knowledge.

---

## Type System & Platform Compatibility

### Native Integer Types in Cross-Platform APIs

**Problem**  
Modern .NET uses native-sized integers (nint) for performance-critical operations.

**Constraint**  
Native integers create compatibility issues:
- Test assertions expect specific integer types
- Serialization and interop may assume fixed sizes
- Generic constraints become more complex

**Pattern That Worked**  
Handle native integers explicitly:
- Cast to expected types in test assertions
- Document when APIs return native vs fixed-size integers
- Provide conversion helpers when needed

**Negative Rule**  
Do not assume nint and int are interchangeable in all contexts.

**Outcome**  
Code works correctly across different architectures and runtime versions.

---

## Aggregation & Grouping Operations

### Composite Key Design for Grouping

**Problem**  
Grouping operations need to handle multiple columns with different types and null values efficiently.

**Constraint**  
Simple string concatenation for composite keys:
- Fails with null values
- Creates ambiguous keys (e.g., "AB" + "C" vs "A" + "BC")
- Poor performance for numeric types
- Inconsistent hash distribution

**Pattern That Worked**  
Create a dedicated composite key class with proper equality, hashing, and null handling:
- Use object arrays to preserve type information
- Implement proper GetHashCode using HashCode.Add for each component
- Handle nulls explicitly in equality comparisons
- Provide meaningful ToString for debugging

**Negative Rule**  
Never use string concatenation or simple tuple hashing for composite keys in production grouping operations.

**Outcome**  
Grouping becomes reliable, performant, and debuggable across all data types.

---

### Nullable Type Handling in Generic Operations

**Problem**  
Runtime type dispatch fails when dealing with nullable value types (e.g., `int?` vs `int`).

**Constraint**  
Type switch expressions don't automatically handle nullable variants:
- `typeof(int?)` doesn't match `typeof(int)` patterns
- Aggregation functions need to work on both nullable and non-nullable columns
- Type validation becomes complex with nullable generics

**Pattern That Worked**  
Use `Nullable.GetUnderlyingType()` to normalize types before dispatch:
- Extract underlying type for nullable types, use original type for non-nullable
- Apply type-specific logic to the underlying type
- Handle null extraction at the value level, not the type level

**Negative Rule**  
Do not create separate code paths for nullable and non-nullable variants of the same underlying type.

**Outcome**  
Single code path handles both nullable and non-nullable types correctly, reducing complexity and maintenance burden.

---

### Vectorization Strategy for Aggregations

**Problem**  
Aggregation functions need to balance performance with correctness across diverse data types.

**Constraint**  
TensorPrimitives only supports specific types (float, double) and requires contiguous memory:
- Not all numeric types are vectorizable
- Null values break vectorization assumptions
- Mixed-type operations complicate vectorization

**Pattern That Worked**  
Implement a fallback hierarchy:
1. Use TensorPrimitives for supported types (float, double) when no nulls present
2. Fall back to scalar operations for unsupported types or when nulls exist
3. Extract valid values first, then apply vectorized operations to clean data
4. Always provide scalar implementations as the baseline

**Negative Rule**  
Never assume vectorization is always faster - measure and provide fallbacks.

**Outcome**  
Aggregations are both fast (when vectorizable) and correct (always), with predictable behavior across all scenarios.

---

### Property-Based Testing

**Problem**  
Traditional unit tests fail to cover combinatorial edge cases.

**Constraint**  
Complex systems exhibit failures only under unexpected input combinations.

**Pattern That Worked**  
Use property-based tests to validate invariants across wide input spaces.

**Negative Rule**  
Do not rely solely on example-based tests for core logic.

**Outcome**  
Subtle correctness issues are caught early.

---

### Testing Strategy for Complex Operations

**Problem**  
Complex operations like grouping and aggregation have many edge cases and type combinations that are difficult to test comprehensively.

**Constraint**  
Manual test case enumeration:
- Misses edge cases with null values
- Doesn't cover all type combinations
- Becomes unmaintainable as operations grow
- Fails to catch subtle correctness issues

**Pattern That Worked**  
Combine comprehensive unit testing with systematic edge case coverage:
- Test each operation with multiple data types (int, double, string, nullable types)
- Explicitly test null handling scenarios
- Test empty inputs and single-element inputs
- Test error conditions with descriptive assertions
- Group related tests in nested test classes for organization

**Negative Rule**  
Do not rely on "happy path" testing alone for complex data operations.

**Outcome**  
High confidence in correctness across all supported scenarios, with clear test organization that makes maintenance easier.

---

## What Didn’t Work (High-Value Failures)

### Reflection-Heavy Designs

- Fragile
- Hard to optimize
- Difficult to reason about

### Premature Vectorization

- Increases complexity
- Often provides marginal gains
- Obscures correctness issues

### Over-Specification

- Encoding too many rules upfront reduces flexibility
- Systems evolve faster than rigid designs

---

## AI-Specific Guidance

### Where Tokens Are Commonly Wasted

- Re-deriving known null-handling strategies
- Over-engineering generic constraints
- Attempting unsafe query optimizations

### Preferred Defaults for AI Agents

- Start with correctness
- Choose explicit designs
- Avoid cleverness unless justified

---

## Closing Note

This document is intentionally **opinionated and distilled**. It should evolve slowly and only when a *new, broadly applicable lesson* is learned.

If a lesson only applies to a specific implementation, it does **not** belong here.

