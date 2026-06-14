# Future Work

This document collects open ideas, unresolved topics, and uncommitted ambitions.
Items here are not roadmap commitments and may conflict with current project
direction until they are revisited and promoted into a concrete plan.

## FrameExtensions - Data-Prep

Future frame-level data-prep helpers:

- `Split(double trainRatio, int? seed = null)` returns train and test frames using a deterministic Fisher-Yates shuffle when a seed is supplied.
- `Normalize(params string[] columnNames)` applies z-score normalization to selected numeric columns and skips null values.
- `Standardize(params string[] columnNames)` is an alias for `Normalize`.

These helpers belong to the frame/data-prep surface, not AutoDiff or LINQ integration. They should be planned and tested as preprocessing conveniences, with no use in AutoDiff training hot paths.

## Tensors

Future tensor-oriented ambitions that are not committed product direction:

- Row-wise frame scoring APIs such as `RowDot<T>(NivaraSeries<T> query, IColumn? labels = null)` and `RowCosineSimilarity<T>(NivaraSeries<T> query, IColumn? labels = null)` for feature-vector workflows where rows represent embeddings, samples, items, or observations.
- Matrix and embedding-table ingestion helpers such as `NivaraFrame.FromRows<T>(IEnumerable<(string Label, T[] Vector)> rows)` and richer `FromTensor<T>(Tensor<T> matrix, string[]? columnNames = null, object[]? rowLabels = null)` overloads.
- Internal row-major materialization helpers for homogeneous numeric frames, using pooled buffers for large data and preserving the `[rows, columns]` tensor convention.
- Potential reuse of the row-major helper for optimized `RowNorms<T>()`, `TensorPrimitives.Dot`, `TensorPrimitives.CosineSimilarity`, and `TensorPrimitives.Norm` row-slice kernels.
- Null semantics for any row-wise result should remain mask-first: nulls in a row make only that row's score null, nulls in the query make all scores null, and placeholder values at null positions are not valid output.
- Benchmark coverage should compare current per-row temporary buffer behavior, row-major materialization, row-slice tensor kernels, and allocation pressure for repeated row-wise scoring.

Keep these ideas scoped to ergonomic interop and optional execution helpers. Do not change the column-first storage model, add an `Axis` enum, introduce nullable-preserving tensor conversion types, or add BLAS-level matrix multiplication to core unless a later plan explicitly commits to that direction.
