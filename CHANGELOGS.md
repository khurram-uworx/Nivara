# Nivara Development Changelogs

Note: Please read CONTRIBUTING.md first for repository contribution rules.

Purpose
- A living, concise reference for maintainers and AI agents explaining why design decisions were made, common pitfalls encountered, and recommended patterns to avoid repeating past mistakes.
- Keep it short and actionable: later entries override earlier conflicting ones; complementary historical context may be merged where helpful.

How to use this document
- Quick lookups for "why" (decisions & rationale), "what not to do" (gotchas), and "how" (patterns/snippets).
- Prefer this document over ad-hoc chat or resurrecting old code comments — it exists to reduce repeated investigation and wasted tokens/time.

Contents (high level)
- Architecture decisions (concise)
- Implementation gotchas and safe patterns (grouped by topic)
- Testing lessons and patterns
- I/O (Arrow / Parquet / CSV) and interop guidance
- Performance patterns and thresholds
- Diagnostics, resource management, and tooling notes
- Known issues and follow-ups
- Quick reference (supported/vectorizable types, thresholds, deps)

--------------------------------------------------------------------------------
ARCHITECTURE DECISIONS (concise)
- Storage strategy: default to MemoryStorage for non-vectorizable types and TensorStorage for vectorizable types. Use a factory (ColumnStorageFactory) and runtime type checks for selection.
  - Rationale: generic static constraints are limiting (CS0080); runtime dispatch is more flexible.

- Vectorization detection: use factory-level checks such as ColumnStorageFactory.IsVectorizable<T>() rather than instance flags.
  - Rationale: selection of storage vs arithmetic vectorization have different concerns; factory approach centralizes decisions.

- Null semantics: explicit null masks (boolean masks) rather than NaN-based null semantics.
  - Rationale: predictable across types; NaN only works reliably for IEEE floats.

- Comparisons: All comparisons return NivaraColumn<bool> with SQL-like null propagation (null compared to anything → null).
  - Rationale: avoids surprising booleans and aligns with common DB semantics.

- Series indexing: Provide both position-based `this[int position]` and label-based `this[object label]` indexers. Boxed ints route to label indexer.
  - Rationale: preserves both semantics while allowing disambiguation when needed.

- Diagnostics and kernel selection:
  - Use ColumnDiagnostics for storage-level info and DiagnosticsTracker for operation-level tracking.
  - Centralize kernel selection logic in a single DetermineKernelType() method that checks vectorizability, hardware acceleration, and a size threshold.

- Query engine: expose generic IQueryOperation<T> and public QueryPlan to enable external extensions and optimizers; use immutable Schema and QueryPlan objects.

- DataFrame operations: Implement as query operations (IQueryOperation) with schema validation in TransformSchema phase
  - Concatenation: Use configurable mismatch handling (Error, FillWithNulls) for vertical, always Error for horizontal
  - Joins: Hash-based algorithm with composite keys, exclude nulls from matching, coalesce join keys for outer joins
  - Sorting: Sort indices first, then reorder all columns consistently, explicit null ordering (NullsFirst/NullsLast)
  - Grouping: Use dedicated composite key class with proper equality/hashing, handle nulls explicitly

- Fluent API design: Layer extension methods over core operations, delegate to existing infrastructure
  - Use dynamic row objects (ExpandoObject) for predicate evaluation in Where() methods
  - Maintain consistent method signatures between DataFrame and QueryFrame APIs
  - Clear execution semantics: immediate execution on DataFrames, lazy via AsQueryFrame()

- Query optimization: Rule-based engine with conservative approach
  - Abstract OptimizationRule base class with priority system
  - Use operation type strings for dispatch when internal details inaccessible
  - Visitor pattern for query plan analysis and transformation
  - Comprehensive statistics tracking and reporting

- Execution strategies: Pluggable strategy pattern (lazy, eager, streaming, parallel)
  - Cost estimation with operation-specific weights
  - Async-first design with sync wrappers using GetAwaiter().GetResult()
  - Intelligent parallel execution with heuristics for data size/operation type

- Error handling: Structured exception hierarchy with query context
  - Include failed query plan and operation references in exceptions
  - Specific exception types for common failure modes
  - Comprehensive validation with detailed error messages

- Target framework: .NET 10.0 with latest C# language features
  - Rationale: leverage latest performance improvements and language features; System.Numerics.Tensors requires modern .NET

--------------------------------------------------------------------------------
IMPLEMENTATION GOTCHAS & PATTERNS (grouped, one representative snippet per concept)

Null handling (core rules)
- Store nulls as explicit boolean masks.
- Propagate nulls in arithmetic/comparison operations via OR of null masks.
- Provide APIs: HasNulls, NullCount, IsNull(index), GetNullIndices(), FillNull(value), FillNullForward(), FillNullBackward(), DropNulls().

ReadOnlyMemory<T>? detection gotcha
- Empty ReadOnlyMemory<T> has HasValue = true — always check length.
```csharp
// Correct: check both HasValue and Length > 0
public bool HasNulls => nullMask.HasValue && nullMask.Value.Length > 0;
```

Slicing empty memory gotcha
```csharp
if (nullMask.HasValue && nullMask.Value.Length > 0)
    slicedNullMask = nullMask.Value.Slice(start, length);
```

Nullable generics & static constraints (CS0080)
- Avoid adding `where T : struct` to static methods in generic classes. Instead, validate at runtime and throw clear exceptions when unsupported.
```csharp
public static NivaraColumn<T> CreateFromNullable(T?[] values)
{
    if (!typeof(T).IsValueType)
        throw new InvalidOperationException("Method only supports value types");
    // process...
}
```

Safe conversion for tensor primitives / MemoryMarshal limitations
- MemoryMarshal.Cast requires unmanaged constraints; use explicit type switch and (T)(object) casting when necessary.
```csharp
if (typeof(T) == typeof(int))
{
    var intValues = values.ToArray();
    var intStorage = new TensorStorage<int>((ReadOnlySpan<int>)(object)intValues.AsSpan());
    return (IColumnStorage<T>)(object)intStorage;
}
```

Tensor usage patterns
- Use System.Numerics.Tensors.Tensor.Create from arrays; use FlattenTo for reads; create tensors from sliced arrays for slicing.
```csharp
var dataBuffer = new T[data.FlattenedLength];
data.FlattenTo(dataBuffer);
var sliced = Tensor.Create(dataBuffer.AsSpan(start, length).ToArray(), new[] { length });
```
- For empty tensors use Array.Empty<T>().

Tensor interop gotchas
- Tensor.Lengths returns nint[], not int[] — cast to int for test assertions: `(int)tensor.Lengths[0]`.
- Method overload resolution: disambiguate 1D vs 2D tensor methods with explicit parameters: `FromTensor<T>(tensor, null)` for 2D.
- Zero-copy limitations: NivaraColumn doesn't expose underlying data as Span; tensor interop requires copying data element-by-element.
- Empty frame construction: NivaraFrame requires at least one column; create minimal empty column for empty tensor cases.

Kernel selection
- DetermineKernelType() pattern:
  - If not vectorizable → Scalar
  - If Vector.IsHardwareAccelerated = false → Scalar
  - If Length < vectorSize * threshold → Scalar
  - Else → Vectorized

Series indexer ambiguity
- Prefer explicit casts or GetByLabel to disambiguate integer label vs position:
```csharp
var positionValue = series[42];          // position
var labelValue = series[(object)42];     // label
var labelValue2 = series.GetByLabel(42); // explicit
```

Expression and operator overloading patterns
- Implement ColumnExpression hierarchy (ColumnReference, BinaryExpression, LiteralExpression, ComparisonExpression).
- Override Equals/GetHashCode when adding custom equality operators.

Diagnostics & operation tracking
- ColumnDiagnostics: storage type, recommended kernel.
- DiagnosticsTracker: enable/disable and collect operation-level metrics.

Memory & resource management
- Implement IDisposable consistently for frames, columns, and data sources.
- Provide ResourceManager for tracking with WeakReference; provide cleanup methods and conservative memory estimates.

DataFrame operation patterns
- Concatenation: Type dispatch with proper null handling, validate type compatibility, filter empty sources
- Joins: Hash-based with composite keys, exclude nulls from matching, coalesce keys for outer joins
- Sorting: Multi-step approach with index sorting then column reordering, explicit null ordering
- Grouping: Composite key class with proper equality/hashing, handle nulls in keys
- Filtering: Boolean mask-based with null propagation
- Projection: Column dictionary operations with existence validation

Query optimization patterns
- Conservative approach: only optimize when safety verifiable through public interfaces
- Operation type dispatch when internal details inaccessible
- Visitor pattern for plan analysis with graceful handling of unknown types
- Statistics tracking for transparency and debugging

Execution strategy patterns
- Strategy pattern with pluggable implementations
- Cost estimation based on operation complexity and data size
- Async-first with proper cancellation token propagation
- Parallel execution with intelligent heuristics for overhead vs benefit

Error handling patterns
- Structured exceptions with query context and operation references
- Schema validation in TransformSchema phase to fail fast
- Detailed error messages with available options listed
- Wrap execution exceptions with additional context

--------------------------------------------------------------------------------
TESTING LESSONS & PATTERNS
- Avoid TestCase with null arrays; use regular [Test] with inline arrays.
- For complex anonymous-type arrays, prefer explicit typed tests or separate focused tests per type.
- Reflection cannot pass Span<T> in MethodInfo.Invoke; convert to array first.
- Test for key phrases in error messages rather than exact message strings.
- Property-like tests can be implemented with parameterized test suites (NUnit) rather than full FsCheck.
- Native integer types (nint): use nint for test assertions when comparing tensor dimensions.
- Method overload disambiguation: use explicit parameters to resolve ambiguous generic method calls in tests.
- Property-based testing approach: Use descriptive test names with "Property" prefix and feature categories for organization.

Representative testing pattern for null handling
```csharp
[Test]
public void NullMaskMaintenance_ArithmeticOperations_PreservesNullPositions()
{
    var testCases = new[] { new int?[] { 1, null, 3 } };
    foreach (var values in testCases)
    {
        var column = NivaraColumn<int>.CreateFromNullable(values);
        var result = column.Multiply(5);
        for (int i = 0; i < values.Length; i++)
            Assert.That(result.IsNull(i), Is.EqualTo(values[i] == null));
    }
}
```

Property-based test pattern
```csharp
[Test]
[Category("Feature: nivara-frame, Property 13: Type compatibility validation")]
public void Property_ArithmeticCompatibility_ValidatesCorrectly()
{
    // Property: Compatible numeric types should validate successfully
    foreach (var (leftType, rightType) in compatiblePairs)
    {
        Assert.DoesNotThrow(() => 
            TypeCompatibilityValidator.ValidateArithmeticCompatibility(leftType, rightType, "test"));
    }
}
```

--------------------------------------------------------------------------------
I/O: Arrow, Parquet, CSV (principles & gotchas)

General
- Keep third-party dependencies in Nivara.Extensions; core stays dependency-free.
- Type mapping: map CLR ↔ Arrow ↔ Parquet with explicit dictionaries and fallback suggestions for unsupported types.
- Handle nullable value types by extracting underlying types via Nullable.GetUnderlyingType().

Arrow interoperability
- Build Arrow arrays using builders and individual Append/AppendNull calls.
- Convert DateTime to UTC (or configured timezone) and use DateTimeOffset for Timestamp arrays.
- Handle chunked arrays by iterating chunkedArray.ArrayCount and extracting each chunk.
- Create valid empty schemas/record batches for empty tables rather than returning null.

Parquet read/write
- For reading: validate schema first, then reconstruct Nivara columns:
  - For value types: build nullable arrays, use CreateFromNullable.
  - For reference types: build arrays preserving nulls.
- For writing: Parquet.Net DataColumn expects non-nullable arrays matching DataField<T> generic type; pass default(T) for nulls and set field as nullable. Preserve string nulls as null, not empty string.
- Empty frame handling: if Parquet requires fields and frame is empty, write a dummy "empty" column.

Representative Parquet write pattern
```csharp
var values = new T[column.Length];
for (int i = 0; i < column.Length; i++)
    values[i] = column.IsNull(i) ? default(T) : column[i];
var dataColumn = new DataColumn(field, values);
await rowGroupWriter.WriteColumnAsync(dataColumn);
```

Known Parquet gotcha (documented)
- Parquet round-trip may convert nullable value types' nulls to default values if null masks are not correctly preserved — treat this as high-priority investigation when writing large nullable datasets.

CSV/JSON lazy vs eager sources
- Lazy sources: IsLazy = true, do schema inference from samples (e.g., 100 rows).
- Eager sources wrap lazy ones and materialize immediately.
- For CSV inference, prefer conservative detection (int → double → string) and fallback to string in ambiguous cases.

Current dependency versions (Extensions only)
- CsvHelper 33.1.0 (updated from 33.0.1)
- Apache.Arrow 22.1.0
- Parquet.Net 5.4.0
- Microsoft.ML 5.0.0 (for ML.NET integration)

--------------------------------------------------------------------------------
PERFORMANCE & OPTIMIZATION PATTERNS
- Vectorization: Check Vector.IsHardwareAccelerated and type vectorizability before using SIMD kernels.
- Overhead threshold: prefer vectorization only when Length >= vectorSize * 4 (adjustable heuristics).
- Buffer pooling: rent large arrays (>1024 elements) from BufferPool (in Extensions) to reduce GC pressure.
- FlattenTo: cache flattened tensor data if multiple accesses are needed; otherwise use FlattenTo for single access.
- StreamingBufferManager: use a bounded buffer manager (in Extensions) for large dataset streaming with memory budgets and GC triggers.

Thresholds & defaults used
- Buffer pooling threshold: 1024 elements (configurable)
- Default memory budget for streaming: 256 MB (example configuration)
- Vectorization overhead threshold: ~4 * vectorSize (heuristic)
- Unmanaged constraint: TensorStorage requires unmanaged types (int, float, double, long, bool)

--------------------------------------------------------------------------------
QUERY ENGINE & OPTIMIZATIONS
- QueryPlan is immutable; optimization passes transform into a new QueryPlan.
- Optimizations applied conservatively: predicate pushdown, operation fusion, column elimination, and reordering when safe.
- Always validate schema transformations; if optimization fails, fall back to original plan.
- Use expression analysis (visitor) to discover referenced columns for column elimination and predicate pushdown.

Optimization safety rules
- Do not change semantics to achieve performance.
- If uncertain, skip the optimization for that path.

--------------------------------------------------------------------------------
RESOURCE MANAGEMENT & DIAGNOSTICS
- ResourceManager tracks frames/columns via WeakReference and offers ForceCleanup() for tests.
- Implement proper IDisposable and object disposed guards.
- Provide diagnostic modes (None, Basic, Detailed, Performance, Comprehensive) with QueryDiagnostics for analysis output.
- Deferred error handling in lazy sources: collect errors at scan time and throw during Collect().

Testing resource management
- Force multiple GC cycles in tests to account for nondeterministic GC behavior.

--------------------------------------------------------------------------------
KNOWN ISSUES & TODO (prioritized)
- Parquet round-trip: investigate nullable value type null preservation (high priority).
- Zero-copy Arrow arrays: placeholder implementation; real zero-copy requires exposing underlying buffer ownership — consider as advanced optimization.
- Improve column creation dynamic dispatch coverage for less common CLR types.
- Internal Span access: consider adding internal AsSpan() methods to NivaraColumn for zero-copy tensor interop scenarios.
- Tensor interop: investigate more efficient conversion patterns for large datasets.

--------------------------------------------------------------------------------
QUICK REFERENCE
- Vectorizable types (confirmed): int, float, double, long, bool (requires unmanaged constraint)
- Target framework: .NET 10.0 with System.Numerics.Tensors 10.0.1
- Common dependencies (Extensions only): CsvHelper 33.1.0, Apache.Arrow 22.1.0, Parquet.Net 5.4.0, Microsoft.ML 5.0.0
- Useful helpers:
  - ColumnDiagnostics, DiagnosticsTracker
  - ColumnStorageFactory.IsVectorizable<T>()
  - NivaraColumn<T>.CreateFromNullable(T?[] values)
  - Tensor.Create(array) + FlattenTo(buffer)
- Storage selection: TensorStorage for vectorizable unmanaged types, MemoryStorage for others
- Null handling: explicit boolean masks, no NaN-based semantics
- Query execution: lazy by default, multiple strategies available (eager, streaming, parallel)

--------------------------------------------------------------------------------
MAINTENANCE GUIDELINES
- Keep decisions short: decision → rationale → link to example code if needed.
- New discoveries: add a single bullet under the relevant category and mark date + author; avoid appending chronological dumps.
- When changing a decision, update the earlier entry by marking it superseded with a one-line reason and link to the new rule.
