# ADR-009: Mergeable Accumulators for Streaming and Partitioned Metrics

**Status:** Accepted  
**Date:** 2026-11-09  
**Issue:** #1064

## Context

Every metric in `Nivara.Metrics` today (`NivaraMetrics.ConfusionMatrixFrame`,
`PerClassMetricsFrame`, `RegressionSummaryFrame`) requires two fully materialised
`IColumn` of equal length. The classification half of the computation is a monoid:
a label to index map plus a `long[k * k]` count matrix, where merge is key-union
followed by cell-wise addition. Nothing in the repository exposes that shape:
`AggregationFunction` (`src/Nivara/Operations/AggregationFunction.cs`) is
`Apply(column, indices)` with no accumulate or merge contract.

The streaming machinery already exists: `QueryFrame.AsStream` and its context
overloads (`src/Nivara/Query/QueryFrame.cs`), `StreamingExecutionStrategy`, and
`ParallelExecutionStrategy`. Reusing it for metrics requires a mergeable
accumulator and four decisions, recorded below.

## Decision

### 1. Public shape: a public, stateful accumulator per metric family

**Adopt a public accumulator type rather than an internal-only overload.**

The stateful API is the point: an accumulator is exactly the object that lets a
training loop carry state across epochs and lets a parallel runner fold partitions,
neither of which a stateless `ConfusionMatrixFrame(...)` overload can do without
re-reading the data. Exposing `ConfusionMatrixFrame(IAsyncEnumerable<NivaraFrame>)`
only would hide the state in a helper and leave the cross-epoch case unserved.

- `Nivara.Metrics.ConfusionMatrixAccumulator<TLabel>` is public and sealed, generic
  over the label type (`TLabel : notnull` comparable), constructed through
  `NivaraMetrics.CreateConfusionMatrixAccumulator<TLabel>()`.
- Members: `Add(IColumn predicted, IColumn actual)`,
  `Merge(ConfusionMatrixAccumulator<TLabel> other)`, `ToFrame()`, and
  `ToPerClassFrame()` (per-class counts and rates are a pure function of the
  accumulated matrix, so they need no separate accumulator).
- `Nivara.Metrics.RegressionAccumulator` is public and sealed, carrying the
  regression sufficient statistics described in decision 3;
  `Add(IColumn predicted, IColumn actual)`, `Merge(RegressionAccumulator other)`,
  `ToFrame()` (a one-row `RegressionSummaryFrame` shape: `Count`, `Mae`, `Mse`,
  `Rmse`, `R2`).
- `ToFrame()` never mutates the accumulator, so a training loop can query an
  intermediate frame without losing state.
- A convenience `NivaraMetrics.ConfusionMatrixFrame(IAsyncEnumerable<NivaraFrame>)`
  (and the regression equivalent) may be added later as a thin driver over the
  accumulator, but it is not the primitive: the accumulator is. Until that ships,
  callers drive accumulation themselves.
- Accumulators are not thread-safe. Parallel code folds with `Merge`; concurrent
  `Add` on one instance is a caller error, matching how `StringBuilder` behaves.

The internal state for confusion matrices is exactly the monoid identified in the
issue: a sorted `TLabel[]` label map plus a `long[k * k]` dense count matrix, with
`ValidPairs` and `TotalPairs` counters. Merging two instances builds the union of
their label sets, allocates one matrix sized to the union, and adds the old cells
into their new positions. No other representation is accepted: a sparse dictionary
would defer the O(k²) cost to `ToFrame()` and make two different accumulator shapes
agree only at finalisation.

### 2. Determinism: fixed chunk index, not completion order

**Parallel and streaming runs reduce in ascending chunk index.**

Integer counts are commutative, so the matrix is bit-exact for any reduction order,
including completion order; but the label set is sorted and the floating regression
statistics (`n`, `Σ`, and the second moment) are not commutative in finite
precision. A run that folds chunks as they complete can therefore disagree, in the
last bits, with the sequential run even though both are correct.

Rule: every driver that folds partitions or chunks keeps a chunk/partition index and
calls `Merge` in strictly ascending index order, buffering out-of-order results
until the next contiguous index is available. Completion order may be used only to
schedule work; it may never be used to reduce it. With this rule a parallel run and
its sequential single-shot equivalent agree bit-for-bit for counts and within the
regression bound stated in decision 3.

### 3. Regression: accept the mergeable one-pass form, state the precision change

**Refusing to stream regression was rejected.** Requiring materialisation would keep
`R2` byte-identical to the single-shot path, but it would also defeat the reason to
have an accumulator: a dataset that does not fit in memory is exactly the case where
a regression summary is wanted and exactly the case where materialisation is
impossible.

Instead the accumulator keeps, per chunk, the sufficient statistics
`(n, Σ residual, Σ |residual|, Σ residual², mean accumulator for actual)`. The mean
of `actual` needed for `SS_tot` is carried in mergeable form rather than recomputed
over the whole population; merging two partitions combines their means and second
moments using Chan's parallel formula (the numerically stable variant of the
one-pass `(n, Σx, Σx²)` form named in the issue). The acceptable consequence is
stated as a contract:

- Classification is bit-exact across single-shot, sequential-chunked, and parallel
  paths for every count and every derived rate (`Precision`, `Recall`, `F1`).
- Regression agrees within a stated bound. A chunked or parallel regression result
  matches the single-shot two-pass result (`TensorPrimitives.SumOfSquares` over the
  residual from the mean) to within `1 ulp` of the stored value, and no worse than
  `1e-12` relative or `1e-9` absolute, whichever is larger. Implementations must
  use `double` throughout, as the current kernel does. Both paths run on the same
  machine, so a cross-architecture difference in `TensorPrimitives` does not enter
  this comparison.
- `SS_res` is accumulated as `Σ (predicted - actual)²` per chunk and summed; because
  that sum is of non-negative terms it does not suffer catastrophic cancellation.
- A regression accumulator whose `Count` is `0` throws on `ToFrame()`, matching the
  single-shot rule that zero valid pairs is an error.
- Non-finite input still raises `ArgumentException` naming the row and column at
  `Add` time (row numbers are relative to the chunk being added), so a chunked run
  fails as early as a single-shot run rather than producing a `NaN` frame.

The difference between one-pass and two-pass `SS_tot` is the entire content of the
allowed tolerance; everything else (`Mae`, `Mse`, `Rmse`, `Count`) is bit-exact
because it is a linear or root reduction of non-negative per-chunk sums.

### 4. Diagnostics: one operation per finalisation

**Record one diagnostic per accumulator finalisation, none per `Add`.**

A chunked run may call `Add` thousands of times; recording per chunk would flood the
tracker and would make the notes field's `"{validPairs} of {total} pairs used"`
meaningless (every chunk would claim its own small fraction). Instead:

- `Add`, `Merge`, and construction record nothing.
- Each finalising call (`ToFrame`, `ToPerClassFrame`, and `ToFrame` on the
  regression accumulator) records exactly one operation:
  - `Metrics.ConfusionMatrix` with `KernelType.Scalar`, total input rows accumulated,
    accumulated valid pairs, element type, and the aggregated null flag;
  - `Metrics.RegressionSummary` with
    `KernelSelector.DetermineKernelType<double>(validPairs)`, as the single-shot
    path does.
- The notes field reports the aggregate `"{validPairs} of {totalPairs} pairs used"`
  over the whole accumulated population, overriding the earlier per-chunk reading.
- A driver that finalises the same accumulator more than once (for example, to log
  every epoch) records once per finalisation, which is the intended semantics: each
  frame is a distinct metric observation.

## Consequences

**Positive:**

- Classification and regression gain a memory-bounded, incremental path without
  giving up determinism: counts stay bit-exact and regression stays within a stated,
  testable tolerance.
- Training loops can keep an accumulator alive across epochs and query it per epoch,
  which the stateless API could not express without re-running the predictions.
- Diagnostics volume stays proportional to metric observations, not to chunk count,
  and the notes field keeps a single meaningful population ratio.
- The fixed chunk-index reduction rule also applies to any future monoid-shaped
  metric, so the determinism contract does not have to be rediscovered.

**Negative / limits:**

- Merging two confusion matrices is O(k²) in the union label count and allocates a
  new matrix; that cost is paid on every `Merge`, which is the price of exact
  agreement without a compaction pass. Very high-cardinality labels remain out of
  scope for the same reason they are out of scope for the batch API.
- Regression tolerance is a new contract surface: a caller comparing a streamed
  result to a hand-computed single-shot reference must use the stated bound, not
  `==`. The single-shot path itself is unchanged, so batch users see no difference.
- Public accumulators widen the supported-API surface and must carry the same
  validation (element type, comparability, finiteness, null pairing,
  zero-valid-pairs) as the existing static methods; a mismatch between the two is a
  bug, and both are pinned by the same tests.
- No public streaming API (`...Frame(IAsyncEnumerable<...>)` convenience) ships with
  this ADR; only the accumulator primitive. That overload remains a separate, later
  issue.
- Determinism under parallel execution assumes partitions are identifiable by index
  and that the driver respects the reduction order in decision 2. An out-of-order
  `Merge` is not detected at runtime; it is a contract violation by the driver, not
  by the accumulator.
