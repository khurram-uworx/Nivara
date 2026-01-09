# TENSORS — Guidance for AI-assisted coding

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
  - `src/Nivara/Storage/ColumnStorageFactory.cs` — runtime selection: `IsVectorizable<T()` and `Create<T>(ReadOnlySpan<T>)`.
  - `src/Nivara/Storage/TensorStorage.cs` — tensor-backed storage implementation.
  - `src/Nivara/Storage/MemoryStorage.cs` — memory-backed storage and null-mask representation.

- Column kernels and high-level ops
  - `src/Nivara/NivaraColumn.cs` — arithmetic, comparison, mask propagation, and use of `TensorPrimitives` for `float`/`double`.

- Tensor helpers & interop
  - `src/Nivara/Tensors/TensorInterop.cs` — conversions `Series/Frame <-> Tensor`, `TensorSpan` utilities, reshape/flatten helpers.

- Factory & utilities
  - `src/Nivara/Storage/ColumnStorageFactory.cs` — runtime switch for creating `TensorStorage<T>` vs `MemoryStorage<T>`.
  - `src/Nivara/Tensors/TensorExtensions.cs` / `TensorInterop.cs` — tensor helpers used across codebase.

Key rules for AI Agents to follow when generating tensor-aware code
1. Storage selection
   - Call `ColumnStorageFactory.IsVectorizable<T>()` first to decide tensor usage.
   - Only instantiate `TensorStorage<T>` when `T` is unmanaged and vectorizable.

2. Zero-copy preference
   - If `TensorStorage` exposes `AsTensorSpan()` and the null mask is empty, prefer `TensorSpan<T>` kernels to avoid allocations.
   - When creating `TensorSpan` from an existing span, ensure there are no nulls; otherwise throw or fallback to copy.

3. Kernel use
   - Use `TensorPrimitives` for `float` and `double` arithmetic/comparisons.
   - Provide robust scalar fallbacks using `Span<T>` loops or `INumber<T>` where available.

4. Null-mask semantics
   - Null propagation rule: resultNullMask = leftNullMask OR rightNullMask (or leftNullMask for scalar op)
   - Where result is boolean (comparisons), null positions should be represented in the result null mask and the boolean output at those positions should be false (SQL-like semantics). Always keep the mask.

5. Minimize allocations
   - Avoid calling `Tensor.FlattenTo` repeatedly in hot loops.
   - For temporary buffers larger than 1024 elements, rent arrays from a `BufferPool` (see `src/Nivara.Extensions/IO/BufferPool.cs`) and return promptly.
   - Consider caching the flattened buffer inside `TensorStorage` behind an `internal` API if multiple accesses are expected.

6. Kernel selection heuristics
   - Implement or reuse `DetermineKernelType()` that considers:
     - `IsVectorizable` (type level)
     - `Vector.IsHardwareAccelerated`
     - `Length >= vectorSize * 4` (heuristic threshold, configurable)
   - If kernel selection resolves to scalar, avoid preparing tensor copies.

7. Safe type dispatch
   - When converting spans/arrays to typed `TensorStorage<T>`, convert to arrays first and then call the `TensorStorage<T>` constructor. Avoid `MemoryMarshal.Cast` unless `T` is unmanaged.
   - Keep explicit type-switch branches for each supported primitive (int, float, double, long, short, byte, bool, etc.).

8. Testing & diagnostics
   - Add unit tests covering null-mask propagation across arithmetic and comparisons.
   - Validate tensor conversions keep correct shape (`tensor.Lengths` is `nint[]`) and use casts `(int)tensor.Lengths[0]` in tests.
   - Record `OperationDiagnostics` for kernel selection and include in performance tests.

Suggested small, safe improvements to implement (prioritized)
- Cache flattened buffer in `TensorStorage` (internal, lazy) to avoid repeated `FlattenTo` allocations for multi-read workloads.
  - Provide `internal ReadOnlySpan<T> GetFlattenedSpan()` returning a span over a cached array; clear on `Dispose()`.
- Add internal `AsTensorSpanIfNoNulls()` to `TensorStorage` to get zero-copy `TensorSpan<T>` when `HasNulls == false`.
- Add `BufferPool.Rent(int size)` usage in `NivaraColumn`'s heavy kernel pre-copy paths (threshold >= 1024).
- Implement `DetermineKernelType(int length, bool isVectorizable)` central helper and use it in `NivaraColumn` for arithmetic/comparisons.

Common gotchas (use these as lint-like checks in generated code)
- `ReadOnlyMemory<T>?` has `HasValue == true` for empty memory; always check `.Length > 0` to decide if mask exists.
- `Tensor.Create(..., [length])` in codebase should use `new nint[] { length }` or `new ReadOnlySpan<nint>(new nint[] { length })` for dimensions; ensure creation uses correct API overloads.
- `Tensor.Lengths` is `nint[]`, not `int[]`.
- Avoid reflection/emits that attempt to pass `Span<T>` to `MethodInfo.Invoke` — convert to arrays first in tests and generated helpers.

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

References (implementations to inspect)
- `src/Nivara/Storage/ColumnStorageFactory.cs`
- `src/Nivara/Storage/TensorStorage.cs`
- `src/Nivara/NivaraColumn.cs`
- `src/Nivara/Tensors/TensorInterop.cs`
- `src/Nivara/Storage/MemoryStorage.cs`
