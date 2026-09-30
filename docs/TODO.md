# #448 — attention mask as a conditional select, not an unconditional add

## Problem

The attention mask is applied as an **additive** mask. Suppressed scores are `-inf` and the
kernel computes `score + (-inf)`. That works until the score is already non-finite, because
`NaN + (-inf) = NaN` and `(+inf) + (-inf) = NaN`. A poisoned score therefore escapes suppression
entirely and poisons every query row in the frame on the next layer.

This is latent, not live in-tree today. It is filed to remove the *class* of failure.

`docs/MODERNBERT.md:29` scopes the residual gap precisely: "a `NaN` already present in **q/k/v**
is still not suppressed". The gap is a NaN in the *score*, not a NaN in the *mask*.

### Where the add actually is

The issue title says `GradKernels`, but `GradKernels.cs` holds only the softmax clamp
(`SoftmaxSingle:487`, `SoftmaxSingleStrided:612`). The additive mask lives in four sites:

| File | Line | Code |
|---|---|---|
| `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` | 569 | `TensorPrimitives.Add(scoresSpan, maskSpan, scoresSpan)` |
| `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` | 807 | `TensorPrimitives.Add(scoresSpan, maskSpan.Slice(b * scoreLen, scoreLen), scoresSpan)` |
| `src/Nivara/AutoDiff/Operations/ForwardGradOperations.cs` | 1617 | `TensorPrimitives.Add(scoresSpan, maskSpan, scoresSpan)` |
| `src/Nivara/AutoDiff/Operations/ForwardGradOperations.cs` | 1820 | `TensorPrimitives.Add(scoresSpan, maskSpan.Slice(b * scoreLen, scoreLen), scoresSpan)` |

### Why the issue's two proposals do not work as written

1. **The "cheaper middle ground" is arithmetically inert.** `output = (score + mask) * keepMask`
   cannot suppress a NaN: IEEE 754 gives `0 * NaN = NaN`. The NaN is already in the accumulator
   before the multiply, so an extra FMA buys nothing. Comment 1's claim that it "ignores the NaN in
   suppressed positions" is incorrect. Only a genuine branch discards the NaN, because it never
   adds to it.

2. **The literal select breaks the fixtures acceptance says not to touch.**
   `keep = mask > -inf; output = keep ? score : -inf` keeps the boolean but discards the mask's
   *magnitude*:
   - `BandedAttention_FinfoMinFill_Forward_MatchesPyTorch` — HuggingFace's `finfo.min` fill stops
     saturating, so the fully-masked row becomes a raw softmax instead of the uniform average of V
     the fixture asserts.
   - `BandedAttention_NonFiniteMaskCells_PropagateAsNaNRows_MatchingPyTorch` — a NaN *mask* cell
     reads as `keep == false` (because `NaN > -inf` is false), so the row no longer poisons. That is
     a PyTorch-parity fixture, and parity is a stronger commitment than the fixture list.

### Already landed, not in scope

Comment 3 on the issue reports the GPU path unguarded. That is **stale**: #447 (merged as #475)
landed the guard at `samples/Nivara.Samples/Gpu/AttentionKernels.cs:86`, covered by
`tests/Nivara.Tests/Gpu/GpuAttentionBandTests.cs:318`
(`BatchedAttention_RowOutsideTheBand_WritesZerosAndNotNaN`). The CPU causal write at
`src/Nivara/AutoDiff/Operations/AttentionKernels.cs:317` is already a select
(`scores[rowStart + j] = T.NegativeInfinity`) and is therefore already immune — it is the in-tree
precedent for this fix.

## Grounding (G1)

Via the microsoft-learn MCP server and code-memory MCP:

| Dependency | Grounded in | Verdict |
|---|---|---|
| `T.NegativeInfinity` on `T : IFloatingPointIeee754<T>` | `IFloatingPointIeee754<T>.NegativeInfinity` is a `static abstract` member, reachable as `T.NegativeInfinity` through the constraint | works; already the form `GradKernels.cs:487,612` uses |
| `NaN == -inf` is false, so a NaN mask cell stays additive | IEEE 754: "comparisons EQ… when either or both operands is NaN return **FALSE**" | the linchpin holds |
| `NaN + (-inf) = NaN` (the bug) | IEEE 754: "NaN (any OP) any-value = NaN", "INF - INF = NaN" | confirmed |
| `0 x NaN = NaN` (kills the issue's post-multiply) | IEEE 754: "(+/-)INF * 0 = NaN" | the issue's cheaper variant is confirmed inert |
| A vectorised select *is* available | `Vector128/256/512.ConditionalSelect<T>` | **not taken, by repo convention** |

`TensorsHelper.cs:107` states the repo deliberately takes the "BCL path instead of a hand-rolled
`Vector{T}` SIMD kernel", and `ConditionalSelect<T>` would not cover `Half`/`BFloat16` under the
`IFloatingPointIeee754<T>` constraint anyway. So `ApplyMask` is a scalar loop, consistent with
existing style. The vectorised option is filed as a follow-up to be revisited only if measured.

**Blast radius addition found during grounding:** `TransformerBlock.CausalMaskSlice`
(`Nn/TransformerBlock.cs:84`) is a fifth mask source — a sub-slice of `CreateCausalMask`, therefore
still `{0, -inf}`, no delta.

## Proposed change

Promote the mask application to one authoritative helper (AGENTS.md rule 8 — it is copied four
times today) in `src/Nivara/AutoDiff/Operations/AttentionKernels.cs`, and route all four sites
through it.

```csharp
// AttentionKernels<T>
public static void ApplyMask(Span<T> scores, ReadOnlySpan<T> mask)
{
    for (int i = 0; i < scores.Length; i++)
    {
        T m = mask[i];
        scores[i] = m == T.NegativeInfinity ? T.NegativeInfinity : scores[i] + m;
    }
}
```

Rationale, in the code comment: a suppressed cell is **assigned** `-inf` rather than summed into,
because summing cannot suppress a non-finite score. Only the `-inf` cell is a suppression signal;
a `NaN` or `+inf` mask cell stays additive so it still propagates (PyTorch parity), and a finite
fill keeps its magnitude so it still saturates (HuggingFace's `finfo.min` convention).

Call sites:

| File | Line | Becomes |
|---|---|---|
| `ReverseGradOperations.cs` | 569 | `AttentionKernels<T>.ApplyMask(scoresSpan, maskSpan)` |
| `ReverseGradOperations.cs` | 807 | `AttentionKernels<T>.ApplyMask(scoresSpan, maskSpan.Slice(b * scoreLen, scoreLen))` |
| `ForwardGradOperations.cs` | 1617 | `AttentionKernels<T>.ApplyMask(scoresSpan, maskSpan)` |
| `ForwardGradOperations.cs` | 1820 | `AttentionKernels<T>.ApplyMask(scoresSpan, maskSpan.Slice(b * scoreLen, scoreLen))` |

The `!maskSpan.IsEmpty` guard stays at each site — "is there a mask" is the caller's decision, not
the kernel's.

### Why this form preserves every existing fixture

The delta against today's behaviour is confined to exactly the two cells that are the bug:

| mask | score | today | after | change |
|---|---|---|---|---|
| `0` | finite | `s` | `s` | — |
| `-inf` | finite | `-inf` | `-inf` | — |
| `-inf` | `-inf` | `-inf` | `-inf` | — |
| `finfo.min` | any | `s + min` (saturates) | `s + min` | — |
| `NaN` | any | `NaN` | `NaN` | — |
| `+inf` | any | `+inf` | `+inf` | — |
| `-inf` | `NaN` | `NaN` | **`-inf`** | the fix |
| `-inf` | `+inf` | `NaN` | **`-inf`** | the fix |

That zero-delta property is the invariant to assert. It is what makes acceptance criterion 3 true
by construction rather than by inspection.

### No backward change needed

The backward reads `savedWeights`. Under the fix a suppressed cell's probability is `0` rather than
`NaN`, so `SoftmaxBackwardRows` contributes `0` to dK/dV where it previously contributed `NaN`.
The forward fix contains the backward automatically.

## Tests

New file `tests/Nivara.Tests/AutoDiff/AttentionKernelsTests.cs` (one test class per source class,
per repo convention). `Nivara.Tests` has `InternalsVisibleTo` (`src/Nivara/Nivara.csproj:18`), so
the `internal` generic `AttentionKernels<T>` is directly testable.

1. `ApplyMask_SuppressedCell_NaNScoreBecomesNegativeInfinity` — the acceptance case.
2. `ApplyMask_SuppressedCell_PositiveInfinityScoreBecomesNegativeInfinity` — the twin cell.
3. `ApplyMask_NaNScoreInKeptCell_PropagatesNaN` — an *unsuppressed* NaN must still propagate. This
   is the guard against over-suppression, and it is the case comment 1 of the issue argued for.
4. `ApplyMask_NaNMaskCell_StillPropagatesNaN` — pins the PyTorch parity that
   `BandedAttention_NonFiniteMaskCells_PropagateAsNaNRows_MatchingPyTorch` depends on.
5. `ApplyMask_FiniteFillMagnitude_Preserved` — `finfo.min` still saturates, pinning the HF convention.
6. `ApplyMask_OrdinaryZeroAndNegInfMask_MatchesAdditiveAdd` — bit-for-bit equality against
   `TensorPrimitives.Add` over a `{0, -inf}` mask, which is what every in-tree mask builder produces.
7. `ApplyMask_PerBatchSlice_BatchedMaskSuppressesIndependently` — a `[B, qLen, kvLen]` mask applied
   via per-batch slices, batch 0 fully masked and batch 1 not, so a slice-offset bug cannot hide.
8. `ApplyMask_ComposedWithSoftmax_SuppressedNaNCellYieldsZeroProbability` — the end-to-end form of
   acceptance criterion 1: mask then `GradKernels.Softmax`, assert `0` and no NaN.

### Deviation from the literal acceptance criteria (deliberate)

Criterion 2 asks for "the same for the strided `SoftmaxDim` path (`[outer][classCount][inner]`)".
`SoftmaxDim` and `SoftmaxSingleStrided` take **no mask parameter** — they only ever see post-mask
values, so there is no suppressed slot at that level, and no strided mask application exists
anywhere in the codebase. The batched per-batch slice (test 7) is the real non-contiguous
analogue. An `ApplyMask` strided overload would have no caller and is deliberately not added.

## Verification

- `dotnet build Nivara.slnx`
- `dotnet test` — **requires explicit human go-ahead** (AGENTS.md). Report the exit status of
  `dotnet test` itself, not of a filter.
- Gates to watch: `BandedAttentionTests` (all 9, especially `FinfoMinFill_Forward` and
  `NonFiniteMaskCells`), `GradKernelsTests` (the 5 clamp tests must not move),
  `ForwardParityTests`, `BatchedMultiHeadAttentionTests`, `LlamaCausalAttentionTests`,
  `ModernBertMaskTests`, and the NivaraTorch attention suites.
- **No fixture should need editing.** If one does, the zero-delta property has been broken.

## Blast radius

| Symbol | File | Downstream callers |
|---|---|---|
| `AttentionKernels<T>.ApplyMask` (new) | `src/Nivara/AutoDiff/Operations/AttentionKernels.cs` | the 4 sites below |
| `ReverseGradOperations.MultiHeadAttention<T>` | `ReverseGradOperations.cs:496` | `MultiheadAttention.ComputeAttention:173`, `LlamaCausalAttention:190,277`, `samples/Nivara.Samples/BertModel.cs:140` |
| `ReverseGradOperations.BatchedMultiHeadAttention<T>` | `ReverseGradOperations.cs:703` | `TransformerBlock.cs:137`, sample batched models |
| `ForwardGradOperations.MultiHeadAttention<T>` | `ForwardGradOperations.cs:1546` | forward-mode paths, `ForwardParityTests` |
| `ForwardGradOperations.BatchedMultiHeadAttention<T>` | `ForwardGradOperations.cs:1729` | forward-mode batched paths |

Mask builders feeding these sites — all produce exactly `{0, -inf}`, so behaviour is unchanged:

- `ModuleHelpers<T>.CreateCausalMask` (`Nn/ModuleHelpers.cs:67`) — used by `MultiheadAttention`
  and `LlamaCausalAttention:189`
- `ModuleHelpers<T>.CreateBlockDiagonalMask` (`Nn/ModuleHelpers.cs:21`)
- `MultiheadAttention<T>.CreatePaddingMask` (`Nn/MultiheadAttention.cs:176`)
- `ModernBertMasks.Build<T>` (`samples/Nivara.Samples/ModernBertModel.cs:233`)
- `BertModel` mask builders (`samples/Nivara.Samples/BertModel.cs:90,119`)

Public API surface is unchanged: `ApplyMask` is a new `internal` member on an already-`internal`
class, and the four attention signatures keep their `mask` parameter and its documented additive
meaning.

## Planned commits

1. `docs: plan #448 in TODO.md`
2. `fix(autodiff): suppress attention mask cells by select, not by add` — kernel + 4 call sites + tests
3. `docs: record that #448 closes the NaN-escapes-suppression gap` — `docs/MODERNBERT.md`,
   `docs/ACCELERATION.md` (×2), `docs/LAYA.md`
4. `docs: remove TODO.md - plan executed`

The issue reply is posted via `gh`, not committed.

## Follow-ups to raise as issues during execution

- The select cannot be a single `TensorPrimitives.Add`, so masking becomes a scalar loop over
  `[qLen, kvLen]` per head where it was a vectorized add. AGENTS.md records attention at ~0.13% of
  Laya's MACs so the absolute cost should be small, but it is **unmeasured**. Measure in
  `tests/Nivara.PerformanceTests` only if it ever shows up.
- Pre-existing: `mask` is listed in the `OpNode` inputs of both attention ops
  (`ReverseGradOperations.cs:587,832`) but `AccumulateGradient` is never called for it, so a mask
  that did require grad silently receives nothing. Out of scope here.

## GitHub issues log

- [x] #448 — mask-as-select (this branch)
- [x] #480 — measure `ApplyMask`'s scalar cost against the `TensorPrimitives.Add` it replaced
- [x] #481 — `mask` is an `OpNode` input but never receives a gradient

> As each task executes, if you find deferred work or a concern outside this plan, create a tracked
> issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number in the log
> above. Do not rely on memory or wait until the end of the plan — compaction can lose it.