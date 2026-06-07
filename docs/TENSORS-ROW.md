# Row-Oriented Tensor Workflows

> **⚠️ Pre-decision draft.** This document proposes row-wise tensor math APIs
> (`RowDot`, `RowCosineSimilarity`) that the project has since decided *not* to
> pursue. See [`TENSORS.md`](TENSORS.md) for the current direction — Nivara
> focuses on tabular data and hands tensor math to the BCL.

## Purpose

Nivara should remain a column-first library.

The row-first interest from AI/ML reviewers is primarily about feature-vector
workflows, where each row represents an embedding, sample, item, or observation
and columns represent features. The solution is to add ergonomic row-wise tensor
operations on top of the existing column-backed frame model, not to introduce a
second primary storage layout.

## Design Direction

Keep column storage canonical.

Add row-vector APIs that materialize temporary row-major buffers only when a
row-wise tensor kernel needs contiguous row spans. This preserves Nivara's
current storage and query model while making common ML workflows practical.

The existing tensor convention remains:

- 2D frame tensors are shaped as `[rows, columns]`.
- Frame storage remains column-backed.
- Row-major tensor buffers are temporary execution artifacts.

## Proposed API Additions

Add row-wise frame operations:

- `NivaraFrame.RowDot<T>(NivaraSeries<T> query, IColumn? labels = null)`
- `NivaraFrame.RowCosineSimilarity<T>(NivaraSeries<T> query, IColumn? labels = null)`

For these APIs:

- `query.Length` must equal `ColumnCount`.
- The result length is `RowCount`.
- Optional `labels` provide result labels and must have length `RowCount`.
- Nulls in a row make only that row's result null.
- Nulls in the query make all row results null.
- Null result values should be `default` while the null mask remains
  authoritative.

Add ingestion helpers for common matrix and embedding-table shapes:

- `NivaraFrame.FromRows<T>(IEnumerable<(string Label, T[] Vector)> rows)`
- `TensorInteropExtensions.FromTensor<T>(Tensor<T> matrix, string[]? columnNames = null, object[]? rowLabels = null)`

These helpers should stay scoped to ingestion and interop. They should not turn
Nivara into a general matrix library.

## Implementation Approach

Add an internal row-major materialization helper for homogeneous numeric frames.

The helper should:

- Validate all selected columns are type `T`.
- Validate `T` follows existing tensor/vectorizable constraints.
- Fill a row-major `T[]` buffer as `data[row * ColumnCount + col]`.
- Build or reuse a row null mask using OR semantics across feature columns.
- Rent temporary buffers from `ArrayPool<T>.Shared` when the total element count
  exceeds the existing pooling threshold.

Use the helper in:

- `RowDot<T>`
- `RowCosineSimilarity<T>`
- a future optimized `RowNorms<T>` implementation

Use `TensorPrimitives` over row slices where possible:

- `TensorPrimitives.Dot`
- `TensorPrimitives.CosineSimilarity`
- `TensorPrimitives.Norm`

Keep the existing column-wise APIs unchanged:

- `Dot`
- `CosineSimilarity`
- `ColumnNorms`
- `RowNorms` public contract

## Null Semantics

Null masks remain authoritative.

For row-wise operations:

- A row result is null if any feature value in that row is null.
- A row result is null if any query value is null.
- Boolean or numeric placeholder values at null positions must not be treated as
  valid output.

This mirrors the existing arithmetic and comparison rule:

`resultNullMask = leftNullMask OR rightNullMask`

## Testing

Add tests for row-wise operations:

- `RowDot` returns one score per row.
- `RowCosineSimilarity` matches `TensorPrimitives.CosineSimilarity` for each
  row.
- Query length mismatch throws a clear `ArgumentException`.
- Optional labels are preserved in result order.
- Null in one row affects only that row.
- Null in query marks all row scores null.
- Empty row frames return empty results where mathematically valid.

Update `RowNorms` tests after optimization to ensure:

- Existing values remain unchanged.
- Existing null-mask behavior remains unchanged.
- Wide frames still avoid per-row allocations where possible.

Add benchmark coverage for:

- Current per-row temporary buffer behavior.
- Row-major materialization plus row-slice tensor kernels.
- Allocation pressure for repeated row-wise scoring.

## Non-Goals

Do not add an `Axis` enum yet. Named methods such as `RowDot` and
`RowCosineSimilarity` are clearer while the API surface is still small.

Do not add a nullable-preserving tensor conversion result yet. Nivara result
series can already preserve nulls with explicit masks.

Do not change the core column-first storage model.

Do not add BLAS-level matrix multiplication to the core package as part of this
work. If needed later, that belongs in `Nivara.Extensions`.

## Summary

The practical solution is a row-wise tensor execution layer over column-backed
frames.

This gives AI/ML users the row-vector ergonomics they expect while preserving
Nivara's column-first identity, explicit null semantics, and current storage
model.
