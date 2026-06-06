# Tensor Improvements Roadmap

## Tomorrow: What The Improved Tensor Experience Should Look Like

These examples are proposed API shapes for discussion with ML and AI engineers. They are not current runnable code. The corresponding current-state examples live in `EXAMPLES.md`.

### 1. Product Ranking With Embedding Similarity

The same product-ranking workflow from the review should become label-aware, null-aware, and explicit about whether candidates are stored in columns or rows.

```csharp
using Nivara;

var user = NivaraSeries<float>.Create(new[] { 0.8f, 0.1f, 0.6f, 0.3f });

using var products = NivaraFrame.Create(
    ("ProductA", NivaraColumn<float>.Create(new[] { 0.9f, 0.2f, 0.5f, 0.4f })),
    ("ProductB", NivaraColumn<float>.Create(new[] { 0.1f, 0.9f, 0.2f, 0.7f })),
    ("ProductC", NivaraColumn<float>.Create(new[] { 0.7f, 0.1f, 0.8f, 0.2f }))
);

var ranking = products
    .ColumnCosineSimilarity(user)
    .TopKDescending(3);

foreach (var item in ranking)
{
    Console.WriteLine($"{item.Label}: {item.Score:0.000}");
}
```

What this improves:

- No manual score array.
- No positional name mapping after `ArgSortDescending`.
- The operation name makes the column-wise candidate orientation explicit.
- Nulls can be represented in the score result instead of failing the entire ranking.
- The no-null fast path can still use `TensorPrimitives.CosineSimilarity`.

### 2. Semantic Search Over Document Embeddings

This is the row-wise version of the same idea: each row is a document, each numeric embedding column is a feature. This is closer to how many ML engineers expect embedding tables to look.

```csharp
using Nivara;

var query = NivaraSeries<float>.Create(new[] { 0.8f, 0.1f, 0.6f, 0.3f });

using var documents = NivaraFrame.Create(
    ("DocumentId", NivaraColumn<string>.CreateForReferenceType(new[] { "doc-101", "doc-102", "doc-103" })),
    ("e0", NivaraColumn<float>.Create(new[] { 0.9f, 0.1f, 0.7f })),
    ("e1", NivaraColumn<float>.Create(new[] { 0.2f, 0.9f, 0.1f })),
    ("e2", NivaraColumn<float>.Create(new[] { 0.5f, 0.2f, 0.8f })),
    ("e3", NivaraColumn<float>.Create(new[] { 0.4f, 0.7f, 0.2f }))
);

var topMatches = documents
    .SelectColumns("e0", "e1", "e2", "e3")
    .RowCosineSimilarity(query, labels: documents.GetColumn<string>("DocumentId"))
    .TopKDescending(2);

foreach (var match in topMatches)
{
    Console.WriteLine($"{match.Label}: {match.Score:0.000}");
}
```

What this improves:

- Rows can be candidates without transposing data or writing manual loops.
- Feature columns are selected explicitly.
- Metadata labels can flow into the ranking result.
- The API is honest about row-wise execution instead of hiding it behind ambiguous `axis: 1` numbers.
- The implementation can use row buffers, pooled temporaries, or optimized tensor paths without changing user code.

## Purpose

This document captures the useful signal from an external ML-oriented review of Nivara's tensor experience.

The review is not the product spec and is not treated as a gold standard. It is valuable feedback from a user perspective: Nivara should remain a strongly typed, columnar DataFrame library with explicit null semantics, while making common tensor and ML workflows less manual.

`Review.md` may be deleted later. The essentials we want to keep are here.

## Current Direction

Nivara's tensor work should stay aligned with the library's existing principles:

- Preserve explicit null masks. Do not use `NaN` as null semantics.
- Keep public operations predictable and type-safe.
- Preserve immutability at the public API boundary.
- Prefer span-backed `System.Numerics.Tensors` kernels when data is contiguous and null-free.
- Fall back to scalar or null-aware paths when tensor kernels would obscure semantics.
- Keep third-party and optional ML integrations outside core unless the API is fundamental.

Microsoft's tensor APIs support this direction:

- `TensorPrimitives` performs primitive tensor operations over spans of memory and is SIMD-optimized in modern .NET.
- `TensorPrimitives.Dot`, `Norm`, and `CosineSimilarity` are first-class APIs for vector kernels.
- `Tensor<T>.AsTensorSpan()` returns a tensor span over the same backing memory, so internal zero-copy paths must be treated as mutable access unless exposed as read-only.
- `TensorPrimitives.CosineSimilarity` requires non-empty equal-length inputs and returns `NaN` for IEEE special values, which is separate from Nivara's null-mask semantics.

References:

- https://learn.microsoft.com/dotnet/api/system.numerics.tensors.tensorprimitives
- https://learn.microsoft.com/dotnet/api/system.numerics.tensors.tensorprimitives.dot
- https://learn.microsoft.com/dotnet/api/system.numerics.tensors.tensorprimitives.norm
- https://learn.microsoft.com/dotnet/api/system.numerics.tensors.tensorprimitives.cosinesimilarity
- https://learn.microsoft.com/dotnet/api/system.numerics.tensors.tensor-1.astensorspan

## Essential Feedback From The Review

The ML workflow that exposed the gap was product ranking by cosine similarity:

```csharp
var scores = products.CosineSimilarity(user);
var ranking = scores.ArgSortDescending();
```

Before the recent tensor commit, the same workflow required manual loops across frame columns and LINQ sorting to recover ranking indices. The review's core point was not "make Nivara a tensor library"; it was that common ML-shaped work should not require users to drop into repetitive plumbing when Nivara already owns typed columns, tensor-backed storage, and vectorized kernels.

The useful product gaps are:

- Batch operations across frame columns: dot product, norm, cosine similarity.
- Ranking helpers such as `ArgSort` / `ArgSortDescending`.
- Named conversion between Nivara types and `Tensor<T>` with explicit null policy.
- Clear shape conventions when a `NivaraFrame` is treated as a 2D tensor.
- Better matrix multiplication implementation before operator syntax is considered.
- Fluent arithmetic helpers only after semantics around nulls, alignment, and broadcasting are explicit.

## What The Last Commit Improved

The last tensor commit made real progress:

- Added `NivaraColumn<T>.ToTensor()` and `ToTensor(T defaultValue)`.
- Added `NivaraSeries<T>.ToTensor()` and `ToTensor(T defaultValue)`.
- Added `NivaraFrame.ToTensors<T>()`.
- Added frame-level `Dot<T>(NivaraSeries<T>)`.
- Added frame-level `CosineSimilarity<T>(NivaraSeries<T>)`.
- Added `NivaraSeries<T>.ArgSort()` and `ArgSortDescending()`.
- Added cached flattened buffers and internal `AsTensorSpanIfNoNulls()` in `TensorStorage<T>`.
- Added tests for conversion null policy, batch dot/cosine, and arg-sort null ordering.

This meaningfully improves the review example. Users can now compute one score per product column and rank the result without hand-written loops.

## Significant Gaps To Fix Before Expanding Features

### 1. Null Semantics In Batch Tensor Operations

Current `NivaraFrame.Dot` and `CosineSimilarity` throw when the vector or any frame column has nulls. That is simple, but it does not match Nivara's broader rule that null masks are authoritative and should be propagated.

Recommended next step:

- Decide and document reduction null semantics for batch operations.
- Prefer returning a result series where a score is null if the input vector or that source column contains any null used by the reduction.
- Keep a no-null fast path using `TensorPrimitives`.
- Add null-aware scalar fallback tests for dot, norm, and cosine.

### 2. Mutable TensorSpan Exposure

`Tensor<T>.AsTensorSpan()` shares backing memory. Nivara currently has internal paths that expose mutable `TensorSpan<T>` or manufacture a writable span from `ReadOnlySpan<T>`.

This is a correctness risk because Nivara presents immutable columns. If internal code mutates through a tensor span after a flattened cache has been populated, cached data can also become stale.

Recommended next step:

- Prefer `ReadOnlyTensorSpan<T>` or read-only span APIs for public and internal read paths.
- If mutable tensor spans are needed internally, keep them tightly scoped to construction or owned temporary buffers.
- Add cache invalidation rules or remove writable span exposure from immutable storage paths.
- Fix `TensorInterop.ToTensorSpan`: `MemoryMarshal.CreateSpan` on a `ReadOnlySpan<T>` creates a writable span from potentially immutable memory. Replace with a copy-based path or a true read-only path.

### 3. Conversion API Consistency

The new named conversion APIs are useful, but the API surface is inconsistent:

- `NivaraFrame.ToTensors<T>()` requires `where T : unmanaged`.
- `NivaraSeries<T>.ToTensor()` and `NivaraColumn<T>.ToTensor()` do not have matching public constraints.
- `TensorInterop.ToTensor(series)` changed from default-filling nulls to throwing via `series.Values.ToTensor()`.

Recommended next step:

- Define one public conversion policy and document it everywhere:
  - `ToTensor()` throws on nulls.
  - `ToTensor(defaultValue)` replaces nulls.
  - Future nullable-preserving conversion should use an explicit type or result object, not hidden sentinel behavior.
- Update XML docs and tests for both instance methods and `TensorInterop` extension methods.
- Decide whether frame conversion should allow every `INumber<T>` or only unmanaged tensor-backed types.

### 4. Shape And Axis Semantics

Nivara is columnar. ML embeddings are often reasoned about as a matrix where rows are samples and columns are features. The reviewed example used frame columns as products, with rows as embedding dimensions.

Recommended next step:

- Document frame-as-tensor conventions explicitly.
- Keep `NivaraFrame.ToTensor<T>()` as `[rows, columns]`.
- Keep `NivaraFrame.ToTensors<T>()` as one 1D tensor per column.
- Add named APIs instead of ambiguous axis integers where possible:
  - `ColumnNorms<T>()`
  - `RowNorms<T>()`
  - `DotColumns<T>(NivaraSeries<T> vector)`
- If an `axis` enum is added, use `Axis.Rows` / `Axis.Columns`, not magic numbers.

### 5. Matrix Multiplication Is Still Naive

`TensorExtensions.MatrixMultiply<T>()` converts frames to tensors, then uses a triple nested loop. That is correct for small inputs but does not meet the expectation created by tensor-oriented APIs.

Recommended next step:

- Do not add matrix multiplication operators yet.
- Keep the method but label performance expectations honestly.
- Investigate a row/column dot implementation using `TensorPrimitives.Dot` where practical.
- Add benchmarks before changing the implementation.
- For larger optimized BLAS-style multiplication, decide whether it belongs in core or an optional extension.

## Prioritized Next Steps

### P0 - Correctness And API Contract

- Fix or document null behavior for `Dot`, `CosineSimilarity`, and future `Norm` batch operations.
- Fix mutable tensor span exposure and align it with Nivara immutability. Replace `MemoryMarshal.CreateSpan` in `TensorInterop.ToTensorSpan` which creates a writable span from read-only memory.
- Keep `CosineSimilaritySpans` limited to floating/root-capable numeric types. Do not add an `INumber<T>` fallback for integer types because cosine similarity requires square root and fractional results.
- Normalize conversion APIs and XML docs around null policy.
- Add tests for null propagation, empty inputs, length mismatch, mixed column types, and disposed objects.

### P1 - Complete The Reviewed Workflow

- Add batch norms:
  - `ColumnNorms<T>()` for one norm per column.
  - `RowNorms<T>()` — row-wise ops are motivated by the Section 2 (Semantic Search) example.
- Add `CosineSimilarity` tests that combine ranking:
  - `products.CosineSimilarity(user).ArgSortDescending()`.
- Add `TopKDescending(int count)` on `NivaraSeries<T>` that combines `ArgSortDescending` with label resolution, returning `(string Label, T Score)[]`.
- Add allocation-aware tests or benchmarks for repeated tensor conversion and repeated column reads.

### P2 - Performance And Diagnostics

- Centralize kernel selection so frame, column, and tensor-extension operations use the same threshold logic.
- Record `OperationDiagnostics` for batch dot, norm, cosine, and matrix multiply.
- Use `TensorPrimitives` generic overloads where the target `System.Numerics.Tensors` version supports them, while keeping explicit float/double paths if required by `10.0.8`.
- Avoid preparing tensor copies when kernel selection resolves to scalar.

### P3 - Ergonomics After Semantics Are Stable

- Add fluent helpers such as `Divide`, `Multiply`, and `Normalize` on series only after null and index-alignment semantics are explicit.
- Consider broadcasting only as an explicit design item, not a side effect of arithmetic overloads.
- Consider operator syntax only after the named method behavior is stable and benchmarked.

## Suggested Acceptance Tests

- `ToTensor_WithoutNulls_ReturnsCopyWithExpectedShape`
- `ToTensor_WithNulls_ThrowsAndMentionsDefaultReplacement`
- `ToTensor_WithDefault_ReplacesOnlyNullPositions`
- `FrameToTensors_MixedColumnType_ThrowsWithColumnName`
- `FrameDot_NoNulls_ReturnsOneScorePerColumn`
- `FrameDot_WithNullInOneColumn_ReturnsNullOnlyForThatColumn` or documents the chosen throw behavior
- `FrameCosineSimilarity_WithNullVector_PropagatesOrThrowsByDocumentedPolicy`
- `ColumnNorms_MatchesTensorPrimitivesNormForFloatAndDouble`
- `ArgSortDescending_IsStableAndPlacesNullsLast`
- `CosineSimilarity_WithUnsupportedIntegerType_ThrowsClearMessage`
- `CosineSimilarity_WithSupportedRootCapableType_DoesNotThrow` if broader generic support is added
- `TopKDescending_ReturnsLabeledResultsInOrder`
- `ToTensorSpan_DoesNotExposeWritableSpanOverReadOnlyMemory`
- `TensorSpanAccess_DoesNotMutateImmutableColumnOrStaleFlattenedCache`

### Performance Benchmarks

Benchmark scenarios to validate performance characteristics of tensor operations:

- `BatchDot_1000Columns_512Dimensions` — batch dot product throughput
- `RepeatedColumnToTensor_NoAllocsAfterFirst` — flattened cache effectiveness
- `CosineSimilarityRanking_EndToEnd` — end-to-end ranking pipeline throughput
- `FrameToTensors_AllColumns` — batch conversion overhead

## Non-Goals For The Next Pass

- Do not turn Nivara into a general-purpose tensor framework.
- Do not make broadcasting implicit until alignment and null rules are designed.
- Do not add matrix multiplication operators while the implementation is still naive.
- Do not use `NaN` to encode missing values.
- Do not treat the external review as a binding spec.

## Summary

The review identified a real usability gap: Nivara already has tensor-aware internals, but common ML-style workflows still felt too manual. The last commit improved that by adding conversion, batch dot/cosine, and arg-sort APIs.

The next pass should be disciplined: first make null semantics, tensor span mutability, and conversion policy correct and documented; then complete the ranking workflow with norms and diagnostics; then consider fluency, broadcasting, and operators.
