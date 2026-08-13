# Plan: Boxing/reflection purge (#198, #199, #200)

Branch: `khurram/issues`

## Problem

Three GitHub issues share one theme: needless boxing / reflection on hot paths.

- **#198** — `Adam.ApplyAdamToSpan` / `AdamW.ApplyAdamWToSpan` box 7 generic `T` values
  per kernel call via `(float)(object)lr!` etc. (18 call sites in `Adam.cs` / `AdamW.cs`).
  `SGD.cs` already uses generic `TensorPrimitives` and is unaffected.
- **#199** — `OpNode<T>.Inputs` is `IReadOnlyList<object>`, boxing every
  `ReverseGradTensor<T>` operand at op creation. Consumed by `ComputationGraph`
  (`BuildBackwardPlan`, `ZeroGrad`), which unboxes via `is ReverseGradTensor<T>`.
- **#200** — `NivaraColumn.Transform` / `TransformNonNull` run full reflection
  (`MakeGenericType` + `Array.CreateInstance` + `Activator.CreateInstance` +
  `GetMethod` + `MethodInfo.Invoke`) on every call that produces a null-valued
  value-type `TResult`.

In-scope duplicates of the same pattern (user-confirmed):

- `NivaraFrameExtensions` (compute helper, ~line 443) — identical reflection block
  to #200.
- `NivaraSeries.divideByCount` (~line 101) — identical `(float)(object)sum!` boxing
  to #198.

## Proposed changes

### 1. #198 — Adam/AdamW kernel boxing

Replace each boxed cast with the generic `CreateChecked` static-interface member
(grounded in MS Learn: `INumberBase<TSelf>.CreateChecked<TOther>` requires
`TOther : INumberBase<TOther>`, satisfied by `IFloatingPointIeee754<T>`):

```csharp
// before
n, (float)(object)lr!, (float)(object)wd!, (float)(object)biasCorr1!, ...
// after
n, float.CreateChecked(lr), float.CreateChecked(wd), float.CreateChecked(biasCorr1), ...
```

- `Adam.cs`: lines 73-75, 87-89, 101-103.
- `AdamW.cs`: lines 59-61, 73-75, 87-89.
- Same idiom as existing `T.CreateChecked(beta1)` in these files.

### 2. #199 — `OpNode<T>.Inputs` typed

- `OpNode.cs:8,13` — change `Inputs` and ctor param to `IReadOnlyList<ReverseGradTensor<T>>`.
- `ComputationGraph.cs:91,109,158` — drop the `input is ReverseGradTensor<T>`
  unboxing filters; iterate `ReverseGradTensor<T>` directly.
- ~53 construction sites. **Scalar entries are dead data** (only consumed by the
  removed `is` filters; closures capture the scalars), so trim them:
  - `Clip` `{a,min,max}` → `[a]`; `LeakyRelu` → `[a]`; `Pow` (double exponent) → `[a]`;
    `RMSNorm` (double eps) → `[a]`.
  - `MultiHeadAttention`/`BatchedMultiHeadAttention` — retype local `inputs` to
    `ReverseGradTensor<T>[]`, `mask` is `ReverseGradTensor<T>?` → use `mask!`.
  - `Concat` — `tensors` already `ReverseGradTensor<T>[]`, unchanged.
  - All other tensor-only sites: `new object[] { a, b }` → `[a, b]` / `[a]`.
  - `Nn/` files: Conv1d/Conv2d/ConvTranspose2d (`{ input, _weight.Tensor }`),
    BatchNorm (`{ input }`), LayerNorm, MaxPool2d, AdaptiveAvgPool2d,
    BCEWithLogitsLoss, TransformerBlock.

### 3. #200 — Transform/TransformNonNull reflection

Replace the reflection block (both `NivaraColumn.cs:2370-2395` and `:2470-2493`)
with the existing fully-generic public helper. At masked positions `result[i]` is
already `default(TResult)!`, so semantics are identical to the old
`CreateFromNullable` path:

```csharp
return NivaraColumn<TResult>.CreateFromSpans(result, resultNullMask);
```

### 4. #200 duplicate — `NivaraFrameExtensions` compute helper

Same replacement at `NivaraFrameExtensions.cs:449-468`:

```csharp
resultColumn = NivaraColumn<TResult>.CreateFromSpans(result, resultNullMask);
```

### 5. #198 duplicate — `NivaraSeries.divideByCount`

Collapse the 17-branch type switch to generic arithmetic (equivalent for every
enumerated type; count is always small, no overflow risk):

```csharp
static T divideByCount(T sum, int count) => sum / T.CreateChecked(count);
```

## Blast radius

- **#198** → `AutoDiff/Optimizer/Adam.cs`, `AdamW.cs`. Downstream: `Optimizer<T>`
  base, `TrainingLoop`, `DataParallelTrainer`, all optimizer-using samples.
  Tests: `OptimizerTests` (float/double/Half + hand-computed references),
  `TrainingTests.TrainingLoop_WithAdam_Converges`.
- **#199** → `AutoDiff/OpNode.cs`, `AutoDiff/ComputationGraph.cs`,
  `AutoDiff/Operations/ReverseGradOperations.cs`, 8 files under
  `AutoDiff/Nn/`. Downstream: the entire graph forward/backward/zero-grad path.
  Tests: `GradOperationsTests`, `InferenceGraphTests`, `TrainingTests`,
  `OptimizerTests`, NivaraTorch functional suite.
- **#200** → `NivaraColumn.cs`, `NivaraFrameExtensions.cs`, `NivaraSeries.cs`.
  Downstream: `Transform`, `TransformNonNull`, frame compute, series `Average`.
  Tests: `ColumnTransformationTests`, `NivaraSeriesAggregateTests`,
  `NivaraFrameTests`.

## Planned commit list

1. `docs: plan boxing/reflection purge (#198-#200) in TODO.md`
2. `perf(autodiff): remove boxing in Adam/AdamW SIMD kernels (#198)`
3. `perf(autodiff): type OpNode<T>.Inputs as ReverseGradTensor<T> (#199)`
4. `perf: drop reflection in NivaraColumn.Transform/TransformNonNull (#200)`
5. `perf: drop reflection in NivaraFrameExtensions compute helper (#200)`
6. `perf(series): remove boxing in NivaraSeries.divideByCount`

## Verification

- `dotnet build Nivara.slnx` after each step.
- Targeted tests (ask human before running):
  `ColumnTransformationTests`, `AutoDiff/OptimizerTests`, `AutoDiff/TrainingTests`,
  `AutoDiff/GradOperationsTests`, `InferenceGraphTests`, `NivaraSeriesAggregateTests`,
  `NivaraFrameTests`.

## GitHub issues log

- [ ] — ColumnFilterHelper (`src/Nivara/Helpers/ColumnFilterHelper.cs`) still uses
  the `MakeGenericType`+`GetMethod`+`Invoke` reflection pattern (lines ~90-233);
  candidate follow-up issue to create during execution if confirmed on a hot path.

Reminder: as each task executes, if you find deferred work or a concern, create a
GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record
its number here — don't rely on memory or wait until the end of the plan.
