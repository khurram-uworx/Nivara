# Tensor Improvements — Suggestions For Future Passes

Items deferred beyond the TENSORS-IMPROVEMENTS roadmap for separate discussion.

These should come after the correctness and API-contract work in
`docs/TENSORS-IMPROVEMENTS.md`. They are useful ideas, but none of them should
force Nivara away from explicit null semantics, immutability, or a columnar data
model.

## Collection Expressions Support

`NivaraSeries<float> s = [1f, 2f, 3f]` would match C# 12 idiomatic syntax.

This is viable, but it requires proper collection-expression opt-in for
`NivaraSeries<T>`, not just an existing `Create()` method. The likely shape is:

- Add `[CollectionBuilder(typeof(NivaraSeriesBuilder), "Create")]` to
  `NivaraSeries<T>`.
- Add an accessible static builder method returning `NivaraSeries<T>` with a
  final `ReadOnlySpan<T>` parameter.
- Keep nullable creation separate; collection expressions should not hide null
  policy.

(This was P1 #7 in the original Review.md.)

## BufferPool Usage In Tensor Paths

AGENTS.md guidance: `BufferPool.Rent` for buffers > 1024 elements.

Do not apply pooling blindly to public conversion APIs. `NivaraColumn.ToTensor()`
returns a `Tensor<T>` that owns or retains the backing data, so a rented buffer
cannot be returned promptly and is the wrong ownership model for that method.

Better candidates:

- Internal temporary buffers in hot arithmetic/comparison pre-copy paths.
- Row-wise similarity implementations that need one temporary row vector at a
  time.
- Benchmark-only experiments that prove allocation pressure before adding
  pooling complexity.

Also note package boundaries: the current `BufferPool` lives in
`Nivara.Extensions`, while core should remain dependency-light. If core tensor
paths need pooling, either move a minimal pool abstraction into core or keep the
optimization inside extensions.

Before adding pooling, measure whether `TensorStorage`'s flattened cache already
removes the repeated allocation in the target scenario.

## Shape-Oriented Constructors

Once row-wise tensor operations are designed, consider convenience constructors
for embedding tables:

- `NivaraFrame.FromRows<T>(IEnumerable<(string Label, T[] Vector)> rows)`
- `NivaraFrame.FromMatrix<T>(Tensor<T> matrix, string[]? rowLabels, string[]? columnNames)`

These should be evaluated carefully because they could make Nivara feel like a
tensor matrix library. The goal would be ergonomic data ingestion, not replacing
`System.Numerics.Tensors`.

## Ranking Result Type

`TopKDescending` could initially return simple tuples, but a named result type
may be better once ranking APIs grow:

```csharp
public readonly record struct RankedValue<T>(
    int Position,
    object? Label,
    T Score,
    bool IsNull);
```

This would preserve Nivara's null semantics better than returning only
`(string Label, T Score)[]`.

## Future Ergonomics & Optimizations

- Fluent arithmetic such as `scores.Divide(norms).Divide(userNorm)` after
  alignment and null rules are stable.
- Broadcasting only after explicit row/column orientation rules exist.
- Operator syntax only after named methods are stable and benchmarked.
