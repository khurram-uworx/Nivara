# Guidance for AI-assisted coding

## Shell environment (Windows with GNU coreutils)

This environment has GNU coreutils at `C:\Program Files\coreutils\bin\` on PATH. Most Linux commands work directly (`grep`, `find`, `touch`, `sort`, `head`, `tail`, `wc`, `cat`, `ls`, `rm`, `mv`, `cp`). PowerShell aliases map `rm`/`mv`/`cp`/`cat`/`ls` to their PowerShell cmdlet equivalents, which behave similarly for basic file operations. Use normal command syntax — avoid verbose PowerShell idioms like `Remove-Item -LiteralPath`.

Purpose
- Concise, machine-friendly rules and locations to guide automated code generation and human edits that use System.Numerics.Tensors opportunistically.
- Designed to be consumed by AI assistants when producing or refactoring tensor-aware code.

Principles (high level)
- Use tensor-backed storage for vectorizable, unmanaged types; use memory-backed storage otherwise.
- Prefer zero-copy tensor/span paths when the data contains no nulls and a TensorSpan/AsTensorSpan is available.
- Use System.Numerics.Tensors `Tensor<T>`, `TensorSpan<T>`, and `TensorPrimitives` for float/double kernels; provide safe scalar fallbacks for other types.
- Preserve explicit null semantics: null masks are authoritative and must be propagated (mask OR semantics) in arithmetic and comparison results.
- Minimize allocations on hot paths: avoid repeated FlattenTo allocations, rent large buffers, and cache flattened buffers when safe.

Where to look (implementation map)
- Storage and selection
  - `src/Nivara/Storage/ColumnStorageFactory.cs` — runtime selection: `IsVectorizable<T>()` and `Create<T>(ReadOnlySpan<T>)`.
  - `src/Nivara/Storage/TensorStorage.cs` — tensor-backed storage implementation.
  - `src/Nivara/Storage/MemoryStorage.cs` — memory-backed storage and null-mask representation.

- Column kernels and high-level ops
  - `src/Nivara/NivaraColumn.cs` — arithmetic, comparison, mask propagation, and use of `TensorPrimitives` for `float`/`double`.

- Tensor helpers & interop
  - `src/Nivara/Tensors/TensorInteropExtensions.cs` — conversions `Series/Frame <-> Tensor`, `TensorSpan` utilities, reshape/flatten helpers.

- Kernel selection & diagnostics
  - `src/Nivara/KernelSelector.cs` — centralized `DetermineKernelType` heuristics (used by `NivaraColumn` and `ColumnDiagnostics`).

- Execution engine
  - `src/Nivara/Execution/ExecutionEngine.cs` — `ExecutionStrategy` enum, `IExecutionStrategy` interface, strategy routing, `LastDiagnostics`.
  - `src/Nivara/Execution/ExecutionStrategyBase.cs` — shared base class eliminating boilerplate across all four strategies.
  - `src/Nivara/Execution/LazyExecutionStrategy.cs` — deferred plan execution with optimization.
  - `src/Nivara/Execution/EagerExecutionStrategy.cs` — immediate execution.
  - `src/Nivara/Execution/StreamingExecutionStrategy.cs` — chunk-based streaming with memory budget.
  - `src/Nivara/Execution/ParallelExecutionStrategy.cs` — multi-threaded dispatch with parallel operation interfaces.
  - `src/Nivara/Execution/ParallelExecutionHelper.cs` — chunking, parallel processing, aggregation utilities.

- Interfaces & query contracts
  - `src/Nivara/Interfaces.cs` — consolidated `IColumn`, `IColumn<T>`, `IColumnStorage<T>`, `IFrame`.
  - `src/Nivara/Query/IQueryInterfaces.cs` — consolidated `IQueryOperation<T>`, `IQueryOperation`, `IQuerySource`.

- Operation type constants
  - `src/Nivara/Query/OperationType.cs` — `OperationType.Filter`, `.Select`, `.Sort`, `.GroupBy`, `.Join`, etc.

- Frame-level batch ops
  - `src/Nivara/NivaraFrame.cs` — `Dot`, `CosineSimilarity`, `ColumnNorms`, `RowNorms` with null propagation and `OperationDiagnostics` recording.

- AutoDiff optimizer
  - `src/Nivara.Extensions/AutoDiff/Optimizer/SgdOptimizer.cs` — SGD update with null-skip semantics.

- Factory & utilities
  - `src/Nivara/Storage/ColumnStorageFactory.cs` — runtime switch for creating `Nivara.Storage.TensorStorage<T>` vs `Nivara.Storage.MemoryStorage<T>`.
  - `src/Nivara/Tensors/TensorExtensions.cs` / `TensorInteropExtensions.cs` — tensor helpers used across codebase.

Key rules for AI Agents to follow when generating tensor-aware code
1. Storage selection
   - Call `Nivara.Storage.ColumnStorageFactory.IsVectorizable<T>()` first to decide tensor usage.
   - Only instantiate `Nivara.Storage.TensorStorage<T>` when `T` is unmanaged and vectorizable.

2. Zero-copy preference
   - If `Nivara.Storage.TensorStorage` exposes `AsTensorSpan()` and the null mask is empty, prefer `TensorSpan<T>` kernels to avoid allocations.
   - When creating `TensorSpan` from an existing span, ensure there are no nulls; otherwise throw or fallback to copy.

3. Kernel use
   - Use `TensorPrimitives` for `float` and `double` arithmetic/comparisons. Prefer generic `TensorPrimitives` overloads (e.g., `TensorPrimitives.CosineSimilarity<T>`) when available for type flexibility without branching.
   - Provide robust scalar fallbacks using `Span<T>` loops or `INumber<T>` where available.

4. Null-mask semantics
   - Null propagation rule: resultNullMask = leftNullMask OR rightNullMask (or leftNullMask for scalar op)
   - Where result is boolean (comparisons), null positions should be represented in the result null mask and the boolean output at those positions should be false (SQL-like semantics). Always keep the mask.

5. Minimize allocations
   - Avoid calling `Tensor.FlattenTo` repeatedly in hot loops.
   - For temporary buffers larger than 1024 elements, rent arrays from `ArrayPool<T>.Shared` (core) or `BufferPool` (Extensions) and return promptly.
   - Consider caching the flattened buffer inside `Nivara.Storage.TensorStorage` behind an `internal` API if multiple accesses are expected.

6. Kernel selection heuristics
   - Implement or reuse `DetermineKernelType()` that considers:
     - `IsVectorizable` (type level)
     - `Vector.IsHardwareAccelerated`
     - `Length >= vectorSize * 4` (heuristic threshold, configurable)
   - If kernel selection resolves to scalar, avoid preparing tensor copies.

7. Safe type dispatch
   - When converting spans/arrays to typed `Nivara.Storage.TensorStorage<T>`, convert to arrays first and then call the `TensorStorage<T>` constructor. Avoid `MemoryMarshal.Cast` unless `T` is unmanaged.
   - Keep explicit type-switch branches for each supported primitive (int, float, double, long, short, byte, bool, etc.).

8. Testing & diagnostics
   - Add unit tests covering null-mask propagation across arithmetic and comparisons.
   - Validate tensor conversions keep correct shape (`tensor.Lengths` is `nint[]`) and use casts `(int)tensor.Lengths[0]` in tests.
   - Record `Nivara.Diagnostics.OperationDiagnostics` for kernel selection and include in performance tests.
   - Use `ColumnDiagnostics`, `DiagnosticsTracker`, and `QueryDiagnostics` when changing kernel selection, query execution, or optimization behavior.

Suggested small, safe improvements to implement (prioritized)
- ✓ Cache flattened buffer in `Nivara.Storage.TensorStorage` (internal, lazy) — DONE via `GetFlattenedSpan()` in Phase 0.
- ✓ Add internal `AsTensorSpanIfNoNulls()` to `Nivara.Storage.TensorStorage` — DONE (Phase 0).
- ✓ Add `BufferPool.Rent(int size)` usage in `NivaraColumn` heavy paths — DONE (Phase 0).
- ✓ Implement `DetermineKernelType` central helper — DONE as `KernelSelector.DetermineKernelType()` (Phase 1).
- ✓ Add `RowNorms`/`ColumnNorms`/`TopKDescending` on `NivaraFrame` — DONE via per-row loop (Phase 3).
- Add batch `TensorPrimitives` kernel for `RowNorms` (not yet implemented — currently uses per-row `ArrayPool` loop).

Common gotchas (use these as lint-like checks in generated code)
- `ReadOnlyMemory<T>?` has `HasValue == true` for empty memory; always check `.Length > 0` to decide if mask exists.
- Slicing null masks: always check `.Length > 0` before slicing to avoid invalid operation on empty memory.
- `Tensor.Create(..., [length])` in codebase should use `new nint[] { length }` or `new ReadOnlySpan<nint>(new nint[] { length })` for dimensions; ensure creation uses correct API overloads.
- `Tensor.Lengths` is `nint[]`, not `int[]`.
- Avoid reflection/emits that attempt to pass `Span<T>` to `MethodInfo.Invoke` — convert to arrays first in tests and generated helpers.
- Nullable generics & static constraints (CS0080): avoid `where T : struct` on static methods in generic classes. Validate at runtime and throw clear exceptions.
- MemoryMarshal.Cast requires unmanaged constraints; use explicit type switch with `(T)(object)` casting for safe conversion.
- Tensor interop: zero-copy is limited — `NivaraColumn` doesn't expose underlying data as `Span`; interop requires element-by-element copying.
- Series indexer ambiguity: boxed `int` routes to label indexer. Use explicit casts or `GetByLabel()` to disambiguate integer labels vs positions.
- Method overload resolution: disambiguate 1D vs 2D tensor methods with explicit parameters (e.g., `FromTensor<T>(tensor, null)` for 2D).
- Expression Equals/GetHashCode: always override when adding custom equality operators to expression types.
- `Memory<T>` disposal: implement `IDisposable` consistently for frames, columns, and data sources.
- `NivaraColumn.TryGetSpan` returns `ReadOnlySpan<T>` (immutable guarantee), diverging from BCL's `Tensor<T>.TryGetSpan` which returns `Span<T>` (mutable). This is deliberate — Nivara columns are immutable. Use `CopyTo(Span<T>, T)` for the explicit-fill path when nulls are present or mutation is needed.
- `DataFrameOperation` no longer has strategy-switch dispatch or `Strategy` property — it was simplified to a single `Execute()` abstract method. Strategy dispatch is the `ExecutionEngine`'s responsibility via `IExecutionStrategy`.

Example patterns (pseudocode for AI Agent to reuse)

- Zero-copy tensor kernel (safe path)

```csharp
// Precondition: tensorStorage.HasNulls == false
var span = tensorStorage.AsTensorSpan(); // returns TensorSpan<T>
// Call kernel that accepts TensorSpan<T> directly
MyKernels.AddTensorSpan(span, otherSpan, destinationSpan);
```

- Safe tensor creation from nullable values

```csharp
var data = new T[len];
var nullMask = new bool[len];
for (int i=0;i<len;i++) {
  if (values[i].HasValue) data[i] = values[i].Value; else { data[i] = default; nullMask[i] = true; }
}
var tensor = Tensor.Create(data, new nint[] { len });
var nullTensor = hasNulls ? Tensor.Create(nullMask, new nint[] { len }) : null;
```

How AI Assistant should use this file
- Prefer deterministic, explicit code that follows rules above.
- Emit checks and fallbacks rather than optimistic, zero-check assumptions.
- When suggesting performance changes, include a small test that validates correctness (null mask and value equality).

Testing & diagnostics patterns
- Avoid `[TestCase]` with null arrays; use regular `[Test]` with inline arrays.
- For complex anonymous-type arrays, prefer explicit typed tests or separate focused tests per type.
- Reflection cannot pass `Span<T>` via `MethodInfo.Invoke` — convert to array first.
- Test for key phrases in error messages rather than exact message strings.
- Property-like tests: implement with parameterized NUnit test suites rather than full FsCheck.
  - FsCheck has limited visibility in mainstream C#; AI agents struggle to produce correct FsCheck code in C#.
- Native integer types (`nint`): use `nint` for test assertions when comparing tensor dimensions.
- Method overload disambiguation: use explicit parameters to resolve ambiguous generic method calls in tests.
- Property-based test naming: use descriptive names with "Property" prefix and feature categories.
- Resource-management tests that depend on weak-reference cleanup may force multiple GC cycles; avoid GC forcing in normal code paths.

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
    foreach (var (leftType, rightType) in compatiblePairs)
    {
        Assert.DoesNotThrow(() =>
            TypeCompatibilityValidator.ValidateArithmeticCompatibility(leftType, rightType, "test"));
    }
}
```

---

## I/O & Interop Guidance

### General Principles
- Keep third-party dependencies in `Nivara.Extensions`; core stays dependency-free.
- Map CLR ↔ Arrow ↔ Parquet with explicit dictionaries and fallback suggestions.
- Handle nullable value types by extracting underlying types via `Nullable.GetUnderlyingType()`.

### Arrow Interoperability
- Build Arrow arrays using builders and individual `Append`/`AppendNull` calls.
- Convert `DateTime` to UTC (or configured timezone) and use `DateTimeOffset` for Timestamp arrays.
- Handle chunked arrays by iterating `chunkedArray.ArrayCount` and extracting each chunk.
- Create valid empty schemas/record batches for empty tables rather than returning null.

### Parquet Read/Write
- **Reading**: validate schema first, then reconstruct columns — use `CreateFromNullable` for value types, build arrays preserving nulls for reference types.
- **Writing**: `Parquet.Net DataColumn` expects non-nullable arrays matching `DataField<T>` generic type; pass `default(T)` for nulls and set field as nullable. Preserve string nulls as null, not empty string.
- **Empty frames**: if Parquet requires fields and frame is empty, write a dummy "empty" column.

### CSV/JSON Sources
- Lazy sources: `IsLazy = true`, infer schema from samples (e.g., 100 rows).
- Eager sources wrap lazy ones and materialize immediately.
- Conservative type detection: int → double → string; fallback to string in ambiguous cases.
- Lazy sources should validate structure/schema early, collect scan errors while traversing data, and throw during `Collect()` with source and operation context.

### Current Dependency Versions (Extensions only)
- CsvHelper 33.1.0, Apache.Arrow 23.0.0, Parquet.Net 6.0.3, Microsoft.ML 5.0.0, System.Numerics.Tensors 10.0.8
- Treat these versions as a snapshot; check the relevant `.csproj` before API-sensitive work.

---

## Performance & Optimization Thresholds

- **Buffer pooling threshold**: rent arrays >1024 elements from `BufferPool` (in Extensions).
- **Default memory budget for streaming**: 256 MB (configurable).
- **Vectorization overhead threshold**: prefer only when `Length >= vectorSize * 4` (heuristic).
- **FlattenTo**: cache flattened tensor data if multiple accesses needed; use single `FlattenTo` for one-time access.
- **StreamingBufferManager**: use bounded buffer manager (in Extensions) for large datasets with memory budgets and GC triggers.
- **Vectorization checks**: verify `Vector.IsHardwareAccelerated` and type vectorizability before using SIMD kernels.
- **Unmanaged constraint**: `TensorStorage<T>` requires unmanaged types (`int`, `float`, `double`, `long`, `bool`).
- **Resource management**: implement object-disposed guards and dispose frames, columns, and data sources consistently.
- **Diagnostics**: preserve diagnostic context when wrapping kernel, query, optimization, and I/O failures.

---

## Known Issues & Follow-ups

- **Parquet round-trip**: nullable value type null preservation may degrade — investigate (high priority).
- **Zero-copy Arrow arrays**: placeholder implementation; real zero-copy requires exposing underlying buffer ownership.
- **Column creation dynamic dispatch**: improve coverage for less common CLR types.
- **Internal Span access**: consider adding `internal AsSpan()` methods to `NivaraColumn` for zero-copy tensor interop.
- **Tensor interop**: investigate more efficient conversion patterns for large datasets.
- **NivaraFrame TopKDescending**: added in Phase 3, returns labeled results with null-propagating scores; threshold-based optimization not yet implemented.
- **NivaraFrame RowNorms/ColumnNorms**: added in Phase 3, null-propagating; currently uses per-row `ArrayPool` loop — batch `TensorPrimitives` kernel not yet implemented.
- **Phase D complete**: Execution engine overhauled — Pattern B (`DataFrameOperation` strategy dispatch) eliminated, real parallel and streaming implementations, diagnostics integration across all strategies, `OperationType` constants replacing magic strings, 1216 tests passing.

---

## Quick Reference

- **Vectorizable types (confirmed)**: `int`, `float`, `double`, `long`, `short`, `byte`, `uint`, `ulong`, `ushort`, `sbyte`, `bool` (requires unmanaged constraint)
- **Target framework**: .NET 10.0 with System.Numerics.Tensors 10.0.8
- **Common deps (Extensions only)**: CsvHelper 33.1.0, Apache.Arrow 23.0.0, Parquet.Net 6.0.3, Microsoft.ML 5.0.0, System.Numerics.Tensors 10.0.8
- **Useful helpers**: `ColumnDiagnostics`, `DiagnosticsTracker`, `ColumnStorageFactory.IsVectorizable<T>()`, `NivaraColumn<T>.CreateFromNullable(T?[])`, `Tensor.Create(array)` + `FlattenTo(buffer)`, `KernelSelector.DetermineKernelType()`, `SgdOptimizer.SgdUpdate<T>()`
- **Storage**: `TensorStorage` for vectorizable unmanaged types, `MemoryStorage` for others
- **Null handling**: explicit boolean masks, no NaN-based semantics
- **Query execution**: lazy by default, multiple strategies (eager, streaming, parallel)

References (implementations to inspect)
- `src/Nivara/Storage/ColumnStorageFactory.cs`
- `src/Nivara/Storage/TensorStorage.cs`
- `src/Nivara/NivaraColumn.cs`
- `src/Nivara/Tensors/TensorInteropExtensions.cs`
- `src/Nivara/KernelSelector.cs`
- `src/Nivara/NivaraFrame.cs`
- `src/Nivara/Storage/MemoryStorage.cs`
- `src/Nivara/Interfaces.cs`
- `src/Nivara/Query/IQueryInterfaces.cs`
- `src/Nivara/Execution/ExecutionEngine.cs`
- `src/Nivara/Execution/ExecutionStrategyBase.cs`
- `src/Nivara/Execution/ParallelExecutionHelper.cs`
