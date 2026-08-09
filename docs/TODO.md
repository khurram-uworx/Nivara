# Plan: Fix Silent API lies in AutoDiff (Issue #179)

Branch: `khurram/179` (created from updated `main`; `khurram/173` merged and deleted).

## Problem

Several AutoDiff APIs silently do something other than their signature promises
(issue #179, originally REVIEW finding #8). Each compiles, runs, and produces
subtly wrong/surprising results without an error.

1. **`Softmax<T>.dim` / `LogSoftmax<T>.dim` is a dead parameter** — stored, never
   read. `Forward` always calls the dim-less op; the raw op uses `shape[1]`
   (dim=1) for rank>=2, not true last-dim for rank>=3. `dim: 0` on 2D silently
   gives last-dim semantics.
2. **`Half` admitted but not fully served** — `TypeValidator` accepts Half, but
   conversion surface has `ToFloat`/`ToDouble`/`ConvertTo` but no `ToHalf`; doc
   comments on `GradTensor`/`ReverseGradTensor`/`ForwardGradTensor` still say
   `INumber<T>`; several kernels special-case float/double and scalar-fallback
   for Half.
3. **`GradientUtils.CanBackward` understates capability** — returns
   `RequiresGrad && Length == 1`, but `Backward()` supports non-scalar when a
   gradient seed is supplied.
4. **`BatchNorm*.RunningMean/RunningVar/NumBatchesTracked` NRE** — non-nullable
   properties backed by `!`; accessing with `trackRunningStats: false` throws
   NRE instead of a clear error.
5. **`NivaraAutoGradExtensions.BatchBackward(tensors, loss)` ignores `tensors`**
   — validates the dict, then calls only `loss.Backward()`.

## Decisions (confirmed with human)

| Finding | Direction |
|---|---|
| 1 Softmax.dim | Implement real dim support (strided kernel, dim=-1 = true last dim) |
| 2 Half | Fully support Half (ToHalf + doc fixes + fast paths) |
| 3 CanBackward | Add gradient-seed overload; keep scalar helper |
| 4 BatchNorm NRE | ADR-001 non-null domain: keep non-nullable surface, add boundary throw |
| 5 BatchBackward | Enforce per-tensor gradients (throw when a listed requires-grad tensor got no grad) |

## Changes

### 1. Softmax/LogSoftmax real dim support

Grounding: `System.Numerics.Tensors.TensorPrimitives` is generic for
`IFloatingPointIeee754<T>` types (float/double/Half) in .NET 9/10 — Half is a
first-class generic target; existing kernels already call
`TensorPrimitives.SoftMax`-style row ops.

- **`src/Nivara/AutoDiff/Operations/GradKernels.cs`** — add strided dim kernels:
  - `SoftmaxDim<T>(ReadOnlySpan<T> input, Span<T> output, int outer, int classCount, int inner)`
  - `LogSoftmaxDim<T>(...)`
  - `SoftmaxDimGradient<T>(ReadOnlySpan<T> softmax, ReadOnlySpan<T> gradOutput, Span<T> output, int outer, int classCount, int inner)`
  - `LogSoftmaxDimGradient<T>(ReadOnlySpan<T> input, ReadOnlySpan<T> gradOutput, Span<T> output, int outer, int classCount, int inner)`
  - Slice math: for resolved dim `d`, `outer = Π shape[0..d)`,
    `classCount = shape[d]`, `inner = Π shape[d+1..)`. Each of `outer * inner`
    slices is `classCount` elements spaced `inner` apart at
    `base = b * classCount * inner + o` (`b` in `[0, outer)`, `o` in `[0, inner)`).
  - Numerically stable (subtract max) per slice; when `inner == 1` the existing
    contiguous kernels remain the fast path (functional wrappers dispatch).
  - Temp slice buffer rented from `ArrayPool<T>.Shared` when `classCount > 1024`.
- **`src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`** — add
  `int dim = -1` to `Softmax`/`LogSoftmax`. Normalize negative dims
  (`dim < 0 ? dim + Rank : dim`), validate range with
  `ArgumentOutOfRangeException`. Resolve `outer/classCount/inner` from `a.shape`;
  dispatch contiguous fast path when `inner == 1`. Backward closure captures the
  resolved dims. Default `-1` → true last dim (`shape[Rank-1]`) — identical to
  current behavior for rank 1/2 (all existing callers unaffected:
  `CrossEntropyLoss.cs:14`, `samples/MicroGpt/Program.cs:131`,
  `tests/NivaraTorch/OperationTests.cs` 2D fixtures).
- **`src/Nivara/AutoDiff/Nn/Functional/Softmax.cs` / `LogSoftmax.cs`** — pass
  `dim` through to the op; add null-input guard already present; keep the
  `dim = -1` PyTorch-compatible surface.

### 2. Half fully served

- **`src/Nivara/AutoDiff/Utilities/TypeConverter.cs`** — add
  `ToHalf<T>(ReverseGradTensor<T> source, bool? requiresGrad = null)` →
  `Convert<T, Half>`.
- **`src/Nivara/AutoDiff/ReverseGradTensor.cs`** — add
  `ToHalf(bool? requiresGrad = null)`.
- **Doc fixes** — `GradTensor.cs:13`, `ReverseGradTensor.cs:13`,
  `ForwardGradTensor.cs:13`: `<typeparam name="T">...implements INumber<T>...`
  → `...implements IFloatingPointIeee754<T>...`.
- **Half fast paths** (correct scalar fallbacks already exist; add SIMD paths):
  - `src/Nivara/AutoDiff/Nn/RMSNormKernel.cs` — add `typeof(T) == typeof(Half)`
    branch using generic `TensorPrimitives<Half>` (or widen to float and narrow
    via `TensorPrimitives.ConvertToSingle`/`ConvertToHalf`).
  - `src/Nivara/AutoDiff/Optimizer/Adam.cs`, `AdamW.cs` — add Half branch
    mirroring the float/double pattern.
  - BatchNorm/LayerNorm keep scalar fallback (functionally correct; documented
    slower — not a correctness lie).

### 3. CanBackward

- **`src/Nivara/AutoDiff/Utilities/GradientUtils.cs`** — keep
  `CanBackward(tensor)` (scalar rule) and add overload
  `CanBackward<T>(ReverseGradTensor<T> tensor, ReverseGradTensor<T> gradient)` →
  `tensor.RequiresGrad && tensor.Length == gradient.Length && ShapeEquals(...)`.
  Update xmldocs to state the real rule. Update `DescribeTensor` to include a
  backward-readiness line.

### 4. BatchNorm running-stats boundary (ADR-001)

- **`src/Nivara/AutoDiff/Nn/BatchNorm.cs`** (both 1d and 2d): keep the
  non-nullable property surface; each accessor throws
  `InvalidOperationException` with a clear message when
  `trackRunningStats: false` (backing field null). Add public
  `bool TrackRunningStats => _trackRunningStats;`.
- `Forward`/`StateDict`/`LoadStateDict` unchanged (already null-guard).

### 5. BatchBackward per-tensor enforcement

- **`src/Nivara/AutoDiff/Extensions/NivaraAutoGradExtensions.cs:153-167`** —
  after `loss.Backward()`, throw when any tensor with `RequiresGrad == true` has
  `Grad == null`, listing the offending keys. Constants exempt. Update xmldoc.
- Keep `ToGradientFrame` returning `NivaraFrame?` (test asserts null); update
  xmldoc to state the intentional asymmetry vs `ToFrame`.

### Docs

- `docs/AUTODIFF.md`: Type Conversion table (+`ToHalf`), `CanBackward` row,
  BatchBackward examples, Softmax dim-aware description.
- `docs/REVIEW.md`: mark finding #8 resolved.
- `CHANGELOG.md` (Unreleased): entry for #179.

## Verification

- `dotnet build Nivara.slnx` after each logical change (ask before `dotnet test`).
- Targeted tests: `GradKernelsTests`, `LossTests`, `NnTests`, `GradientUtilsTests`,
  `NivaraIntegrationTests`, `TypeSafetyTests`, `OperationTests` (NivaraTorch).
- Confirm existing rank-2 Softmax/LogSoftmax PyTorch fixtures still pass.

## Planned commits (one logical unit each)

1. `docs: plan #179 AutoDiff API lies in TODO.md`
2. `feat(autodiff): implement dim-aware Softmax/LogSoftmax (strided kernels)`
3. `feat(autodiff): add ToHalf conversion + fix stale INumber<T> docs`
4. `perf(autodiff): add Half fast paths to RMSNorm/Adam/AdamW kernels`
5. `feat(autodiff): add CanBackward gradient-seed overload + DescribeTensor`
6. `fix(autodiff): BatchNorm running-stats access throws clear error when untracked`
7. `fix(autodiff): BatchBackward enforces per-tensor gradients`
8. `docs: update AUTODIFF.md, REVIEW.md, CHANGELOG for #179`
9. `docs: remove TODO.md — #179 plan executed`

## Blast radius

- **Softmax dim**: `GradKernels`, `ReverseGradOperations`, functional
  `Softmax`/`LogSoftmax`. Downstream: `CrossEntropyLoss`, `MicroGpt` sample,
  `NivaraTorch.OperationTests`, `LossTests`. Default `-1` keeps rank-1/2
  behavior identical — no behavioral change for existing callers.
- **Half**: `TypeConverter`, `ReverseGradTensor`, `RMSNormKernel`, `Adam`,
  `AdamW`, doc comments only. Downstream: `TypeSafetyTests`, optimizer tests,
  `NivaraAutoGradExtensions.ToReverseGradTensorsAuto` (already handles Half).
- **CanBackward**: `GradientUtils` (+ overload). Downstream:
  `GradientUtilsTests`, `samples/Nivara.SampleApp/AutoDiffExample.cs:218` (prints
  result — unchanged semantics for the no-seed overload).
- **BatchNorm**: `BatchNorm.cs` properties. Downstream: `NnTests` running-stats
  tests (tracking enabled — unaffected); `StateDict`/`LoadStateDict` untouched.
  `MobileNetV2`/`ResNet18` samples load running stats (tracking path — unaffected).
- **BatchBackward**: `NivaraAutoGradExtensions`. Downstream: none in production
  (docs only); `NivaraIntegrationTests` add coverage.
- **Docs**: `AUTODIFF.md`, `REVIEW.md`, `CHANGELOG.md` — no code impact.

## GitHub issues log

- [ ] #179 — Fix Silent API lies in AutoDiff (Softmax.dim, Half, CanBackward, BatchNorm NRE, BatchBackward). **This plan.**
- [ ] (placeholder) — if any deferred work/concern is discovered during
      execution, create a GitHub issue immediately via
      `gh issue create --repo khurram-uworx/Nivara` and record its number here;
      never hold it in memory.

## Reminder

As each task executes, if you find deferred work or a concern (known
limitations, follow-ups, refactors) outside the current plan, create a GitHub
issue immediately and record its number in the GitHub issues log — don't rely on
memory or wait until the plan finishes, as compaction can lose it.
