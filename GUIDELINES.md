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

---

## DataFrame Concatenation Operations

### Schema Compatibility Handling in Concatenation

**Problem**  
DataFrame concatenation operations need to handle schema mismatches between sources while maintaining data integrity.

**Constraint**  
- Vertical concatenation may encounter DataFrames with different column sets
- Horizontal concatenation requires identical row counts and unique column names
- Users need control over how mismatches are handled (error vs fill with nulls)

**Pattern That Worked**  
Implement configurable mismatch handling strategies:
- Provide explicit enum options (Error, FillWithNulls) for vertical concatenation
- Always use Error strategy for horizontal concatenation (column conflicts are unambiguous)
- Validate schemas during TransformSchema phase to fail fast
- Create null columns of appropriate types when filling missing columns

**Negative Rule**  
Do not silently ignore schema mismatches or apply implicit type conversions during concatenation.

**Outcome**  
Concatenation operations are predictable and user-controllable, with clear error messages when operations cannot proceed.

---

### Type-Safe Column Concatenation

**Problem**  
Concatenating columns requires handling different data types while preserving null semantics and type safety.

**Constraint**  
- Must work across all supported column types (value types, reference types, nullable types)
- Must preserve null values correctly during concatenation
- Must validate type compatibility between columns being concatenated
- Must handle empty DataFrames gracefully

**Pattern That Worked**  
Use type dispatch with proper null handling:
- Validate all columns have identical types before concatenation
- For value types: create nullable arrays and use CreateFromNullable
- For reference types: create typed arrays and use CreateForReferenceType
- Handle empty sources by filtering them out before processing
- Use reflection-based type dispatch for unknown types with object fallback

**Negative Rule**  
Never attempt to concatenate columns of different types - always validate type compatibility first.

**Outcome**  
Column concatenation works reliably across all data types with proper null preservation and clear error reporting for incompatible operations.

---

### Public API Design for Complex Operations

**Problem**  
Complex operations like concatenation need both low-level control and high-level convenience methods.

**Constraint**  
- Internal operation classes should remain internal for implementation flexibility
- Public enums and types need to be accessible from extension methods
- Users want both explicit control and convenient defaults
- Multiple overloads should provide different levels of control

**Pattern That Worked**  
Layer the API with public enums and internal operations:
- Define public enums (ConcatenationDirection, ConcatenationMismatchHandling) at namespace level
- Keep operation classes internal but expose functionality through extension methods
- Provide both single-frame and multi-frame overloads
- Include convenient aliases (Append, Combine) for common use cases
- Use sensible defaults (FillWithNulls for vertical, Error for horizontal)

**Negative Rule**  
Do not expose internal operation classes directly - always provide a clean public API through extension methods.

**Outcome**  
Users get a clean, discoverable API with appropriate levels of control while implementation details remain flexible.

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

## DataFrame Column Transformations

### Type-Safe Column Transformations

**Problem**  
DataFrame operations need to transform column data while preserving type safety and null handling.

**Constraint**  
- Must work across all column types (value types, reference types, nullable types)
- Must preserve null values correctly during transformations
- Must handle transformation exceptions gracefully
- Must maintain performance for large datasets

**Pattern That Worked**  
Implement transformations at the column level with proper null propagation:
- Add `Transform<TResult>` method to `NivaraColumn<T>` that applies function element-wise
- Check for null values before applying transformation function
- Propagate nulls through transformations (null input → null output)
- Handle exceptions during transformation with clear error context
- Use appropriate column creation methods based on result type (value vs reference types)

**Negative Rule**  
Do not apply transformations without checking for null values first - this leads to unexpected exceptions.

**Outcome**  
Column transformations work reliably across all data types with predictable null handling and clear error reporting.

---

### DataFrame Projection Operations

**Problem**  
DataFrame operations need to select and rename columns while maintaining schema consistency.

**Constraint**  
- Must validate that source columns exist before projection
- Must handle column renaming without name conflicts
- Must preserve column data and types during projection
- Must integrate with existing query operation infrastructure

**Pattern That Worked**  
Implement projection as a query operation that works on column dictionaries:
- Create `ProjectionOperation` that implements `IQueryOperation` interface
- Validate column existence during schema transformation phase
- Use column mappings (original name → new name) to handle renaming
- Preserve original column objects during projection (no data copying)
- Validate result column names are unique to prevent conflicts

**Negative Rule**  
Do not inherit from `DataFrameOperation` for simple operations - implement `IQueryOperation` directly for better flexibility.

**Outcome**  
Projection operations integrate cleanly with the query system while providing efficient column selection and renaming.

---

### Extension Method Design for DataFrames

**Problem**  
DataFrame operations need fluent API support while maintaining type safety and performance.

**Constraint**  
- Must provide intuitive method names and overloads
- Must handle edge cases (empty selections, name conflicts) gracefully
- Must integrate with existing DataFrame methods
- Must support both simple and complex transformation scenarios

**Pattern That Worked**  
Create extension methods that delegate to core DataFrame functionality:
- Provide multiple overloads for different use cases (single column, multiple columns, etc.)
- Use descriptive parameter names and comprehensive validation
- Delegate to existing DataFrame methods when possible (avoid duplication)
- Handle complex scenarios (multi-column computations) with proper null checking
- Use reflection judiciously for generic scenarios while providing typed overloads for common cases

**Negative Rule**  
Do not duplicate DataFrame functionality in extension methods - delegate to core methods to maintain consistency.

**Outcome**  
Extension methods provide a rich, fluent API while maintaining the reliability and performance of core DataFrame operations.


---

## DataFrame Row Operations

### Filtering and Slicing Implementation

**Problem**  
DataFrame operations need efficient row-level filtering and slicing while preserving schema and handling null values correctly.

**Constraint**  
- Must work across all column types (numeric, string, nullable)
- Must preserve null masks during operations
- Must handle edge cases (empty results, out-of-bounds parameters)
- Must maintain schema consistency

**Pattern That Worked**  
Implement operations at the NivaraFrame level using:
- `FilterByMask()` for boolean mask-based filtering
- `Take(n)` for getting first n rows
- `Skip(n)` for skipping first n rows  
- `Slice(start, length)` for arbitrary row ranges
- Delegate to column-level `Slice()` methods when available
- Use reflection fallback for type-agnostic column operations

**Negative Rule**  
Do not implement filtering logic separately for each column type - use unified approach with type dispatch.

**Outcome**  
Row operations work consistently across all data types with proper null handling and schema preservation.

---

### Column Type Dispatch for Operations

**Problem**  
Operations need to work on columns of unknown types at runtime while maintaining type safety.

**Constraint**  
Generic constraints become unwieldy for operations that must work on any column type.

**Pattern That Worked**  
Use pattern matching on `Type` objects with fallback to generic object handling:
- Match common types explicitly (int, double, string, bool, etc.)
- Use reflection to call typed methods when available
- Provide object-based fallback for unknown types
- Handle value types vs reference types appropriately

**Negative Rule**  
Do not attempt to constrain all operations with generic type parameters - runtime dispatch is more flexible.

**Outcome**  
Operations work reliably across all supported types with good performance for common cases.

---

## DataFrame Sorting Operations

### Multi-Column Sorting Implementation

**Problem**  
DataFrame sorting needs to handle multiple sort keys with different directions and null ordering strategies while maintaining performance and correctness.

**Constraint**  
- Must work across all column types (numeric, string, nullable)
- Must preserve stable sort semantics when requested
- Must handle null values according to specified ordering (nulls first/last)
- Must support both ascending and descending directions per column
- Must validate that columns are comparable before attempting sort

**Pattern That Worked**  
Implement sorting using a multi-step approach:
- Create `SortKey` class to encapsulate column name, direction, and null ordering
- Use `MultiColumnComparer` that implements `IComparer<int>` to compare row indices
- Compute sort indices first, then reorder all columns using the same indices
- Use stable sorting (OrderBy) by default, with option for unstable (Array.Sort) for performance
- Validate column types implement `IComparable` or `IComparable<T>` before sorting

**Negative Rule**  
Do not attempt to sort columns directly - always sort indices and then reorder all columns consistently.

**Outcome**  
Sorting works reliably across all data types with proper null handling and maintains referential integrity between columns.

---

### Column Reordering Strategy

**Problem**  
Reordering DataFrame rows requires efficiently rearranging all columns while preserving data integrity and null masks.

**Constraint**  
- Must work with any column type (value types, reference types, nullable types)
- Must preserve null values and their positions correctly
- Must validate indices are within bounds and handle edge cases
- Must maintain column type information during reordering

**Pattern That Worked**  
Implement `ReorderByIndices` method that:
- Validates indices array length matches row count
- Validates all indices are within valid range [0, rowCount)
- Uses type dispatch to call appropriate reordering method for each column type
- For value types: creates nullable arrays and uses `CreateFromNullable`
- For reference types: creates typed arrays and uses `CreateForReferenceType`
- Handles identity reordering efficiently (when indices are [0,1,2,...])

**Negative Rule**  
Do not use reflection for column reordering in performance-critical paths - use type dispatch with explicit type handling.

**Outcome**  
Row reordering is efficient and type-safe, with proper null preservation across all supported column types.

---

### Sorting Null Value Handling

**Problem**  
Null values in sortable columns need consistent ordering behavior that users can control.

**Constraint**  
- Different databases and systems have different default null ordering
- Users need control over whether nulls appear first or last
- Null comparison must be consistent across all column types
- Must work with both nullable value types and reference types

**Pattern That Worked**  
Implement explicit null ordering in the comparer:
- Check for null values before attempting comparison
- Apply null ordering strategy (NullsFirst/NullsLast) consistently
- Handle null-to-null comparisons as equal (return 0)
- Only compare non-null values using `IComparable.CompareTo`
- Provide sensible defaults (NullsLast) while allowing user control

**Negative Rule**  
Never rely on default null comparison behavior - always handle nulls explicitly in custom comparers.

**Outcome**  
Null handling in sorting is predictable and user-controllable, working consistently across all column types.

---

## DataFrame Join Operations

### Join Key Coalescing for Outer Joins

**Problem**  
Outer joins (right and full outer) need to handle join key values correctly when one side has no match.

**Constraint**  
- Left join keys may be null when right rows have no left match
- Right join keys may be null when left rows have no right match
- Users expect to see the actual join key values, not nulls, in the result
- Must work across all data types (value types, reference types, nullable types)

**Pattern That Worked**  
Implement coalesced join key columns that prefer left values when available, otherwise use right values:
- Create `CreateCoalescedJoinKeyColumn` method that handles type dispatch
- For each result row, use left join key value if left index >= 0, otherwise use right join key value
- Handle both nullable value types and reference types appropriately
- Use proper column creation methods based on the underlying type

**Negative Rule**  
Do not simply use left join key columns for outer joins - this loses information when left side has no match.

**Outcome**  
Outer join results show meaningful join key values instead of nulls, making the results more useful and intuitive.

---

### Join Index Computation Strategy

**Problem**  
Join operations need efficient algorithms to match rows between DataFrames while handling different join types and null values.

**Constraint**  
- Must handle multiple join keys with composite key matching
- Must support all join types (inner, left, right, full outer)
- Must handle null values correctly (nulls don't match other nulls)
- Must be efficient for large datasets

**Pattern That Worked**  
Use hash-based join algorithm with proper null handling:
- Build hash map for right DataFrame using composite keys
- Exclude null values from hash map (nulls never match)
- Iterate through left DataFrame and probe hash map for matches
- Track matched right indices to identify unmatched rows for outer joins
- Use separate index arrays for left and right sides with -1 indicating no match

**Negative Rule**  
Do not use nested loop joins as the primary algorithm - hash joins are much more efficient for most cases.

**Outcome**  
Join operations scale well with data size and handle all join types correctly with proper null semantics.

---

### Column Name Disambiguation in Joins

**Problem**  
Join operations often encounter column name conflicts between left and right DataFrames that must be resolved consistently.

**Constraint**  
- Users need control over how conflicts are resolved
- Resolution strategy must be applied consistently across all conflicting columns
- Must preserve original column names when no conflicts exist
- Must provide clear error messages when conflicts cannot be resolved

**Pattern That Worked**  
Implement configurable disambiguation strategies:
- `Prefix`: Add prefixes to conflicting columns (left_column, right_column)
- `Suffix`: Add suffixes to conflicting columns (column_left, column_right)  
- `Error`: Throw exception when conflicts are detected
- Always exclude right join key columns from results (left join keys are kept)
- Apply disambiguation only to columns that actually conflict

**Negative Rule**  
Do not automatically rename all columns - only rename when actual conflicts exist to preserve user expectations.

**Outcome**  
Join results have predictable column names with user control over conflict resolution, avoiding silent data loss or confusion.

---

### Join Type Validation and Error Handling

**Problem**  
Join operations have many potential failure modes that need clear, actionable error messages.

**Constraint**  
- Join key columns must exist in both DataFrames
- Join key types must be compatible between left and right sides
- Schema validation should happen before expensive computation
- Error messages must be specific enough for users to fix issues

**Pattern That Worked**  
Implement comprehensive validation with specific error messages:
- Validate join key existence in both schemas during `TransformSchema`
- Check type compatibility using exact type matching (no implicit conversions)
- Provide detailed error messages listing available columns when keys are missing
- Wrap execution exceptions in `QueryExecutionException` with context
- Validate schemas before computing join indices to fail fast

**Negative Rule**  
Do not allow implicit type conversions in join keys - require exact type matches to avoid subtle bugs.

**Outcome**  
Join operations fail fast with clear error messages, making debugging and fixing issues straightforward for users.

---

## Closing Note

This document is intentionally **opinionated and distilled**. It should evolve slowly and only when a *new, broadly applicable lesson* is learned.

If a lesson only applies to a specific implementation, it does **not** belong here.

---

## Query Optimization Engine Implementation

### Optimization Rule Architecture

**Problem**  
Query optimization systems need to be extensible and maintainable while ensuring correctness.

**Constraint**  
- Optimization rules must be composable and independent
- Failed optimizations should not break valid queries
- Internal operation classes limit access to optimization details
- Rules need to be prioritized and applied in correct order

**Pattern That Worked**  
Implement a rule-based optimization engine with:
- Abstract `OptimizationRule` base class with priority system
- `OptimizationEngine` that applies rules in priority order with error handling
- Conservative optimization approach when internal details are inaccessible
- Comprehensive statistics tracking and reporting for debugging

**Negative Rule**  
Never let optimization failures break query execution - always fall back to original plan.

**Outcome**  
Optimization system is extensible, safe, and provides detailed feedback for debugging and performance analysis.

---

### Working with Internal Operation Classes

**Problem**  
Query optimization requires analyzing operation details, but operation classes are internal for encapsulation.

**Constraint**  
- Cannot access internal properties like filter conditions or selected columns
- Must work with public interfaces only (`IQueryOperation`, `OperationType`)
- Need to maintain type safety while being generic

**Pattern That Worked**  
Use operation type strings and conservative analysis:
- Dispatch on `OperationType` property instead of concrete types
- Be conservative when internal details are inaccessible
- Use schema transformation methods available on public interfaces
- Provide fallback behavior that assumes worst-case scenarios

**Negative Rule**  
Do not attempt to use reflection or other mechanisms to access internal state - this breaks encapsulation and is fragile.

**Outcome**  
Optimization works safely with public interfaces while maintaining encapsulation boundaries.

---

### Visitor Pattern for Query Plans

**Problem**  
Query plan analysis and transformation requires traversing complex operation hierarchies.

**Constraint**  
- Need both read-only analysis and transformation capabilities
- Must work with internal operation types through public interfaces
- Should be extensible for new operation types

**Pattern That Worked**  
Implement visitor pattern with operation type dispatch:
- Base visitor classes that dispatch on `OperationType` strings
- Separate interfaces for analysis (`IQueryPlanVisitor`) and transformation (`IQueryPlanVisitor<T>`)
- Default implementations that handle unknown operation types gracefully
- Concrete visitors for specific tasks (statistics, validation)

**Negative Rule**  
Do not create visitors that depend on accessing internal operation state - use public interfaces only.

**Outcome**  
Flexible visitor system that works with encapsulated operations and is easily extensible.

---

### Optimization Statistics and Reporting

**Problem**  
Query optimization decisions need to be transparent and debuggable for performance tuning.

**Constraint**  
- Users need to understand what optimizations were applied
- Failed optimizations should be reported with context
- Performance impact estimates help guide optimization priorities

**Pattern That Worked**  
Comprehensive statistics tracking:
- `OptimizationStatistics` class with timing, improvement estimates, and metadata
- `OptimizationResult` that includes both successful and failed rule applications
- Detailed reporting with human-readable explanations
- Error context preservation for debugging failed optimizations

**Negative Rule**  
Do not hide optimization decisions from users - transparency builds trust and enables debugging.

**Outcome**  
Users can understand and debug query performance issues with clear optimization feedback.

---

### Conservative Optimization Strategy

**Problem**  
Query optimizations must preserve correctness while improving performance.

**Constraint**  
- Cannot access internal operation details for dependency analysis
- Must work across different operation types safely
- Optimization failures should not break queries

**Pattern That Worked**  
Be conservative when in doubt:
- Only apply optimizations when safety can be verified through public interfaces
- Use schema transformation methods to validate compatibility
- Provide fallback behavior that assumes all columns/operations are needed
- Wrap optimization attempts in try-catch with graceful degradation

**Negative Rule**  
Never apply an optimization unless its safety can be verified through available public interfaces.

**Outcome**  
Optimization system is safe and reliable, preferring correctness over aggressive optimization.
