# Plan: P2 — Fuse final RMSNorm + tied LM head into fused decode/prefill (#413)

Branch: `khurram/413` (off `khurram/411`). Issue: https://github.com/khurram-uworx/Nivara/issues/413

## Problem

After #404 (per-token fused decoder block), `Qwen decode fwd [1 step]` is ~619,631 B/op and
almost all of it is the irreducible `[1, 151,936]` float logits array (~608,373 B). The ~11 KB/op
residual comes from the final `RMSNorm.Forward` + tied-LM-head `MatMulTransposedB` still running
as boxed per-op `ReverseGradTensor` calls after the 24 fused layers:

- `ForwardCachedFusedCore` (`samples/Nivara.Samples/LlamaForCausalLM.cs:205-210`): per step it
  allocates `cur.AsSpan().ToArray()` (hidden copy), `finalNorm.Forward`'s src copy + result array
  (2 × hidden), and the per-op matmul wrapper boxes.
- `ForwardPrefillFusedCore` (`:232-245`) has the same boxed pattern over `[L, hidden]` plus a
  `lastRow` copy (in scope per user decision).

Goal: run the final RMSNorm (gamma multiply) and the single-row head GEMV into span/workspace
buffers so the only per-step allocation is the returned `[1, vocab]` logits buffer.

## Proposed changes

### 1. Core — public fused RMSNorm entry (`src/Nivara/AutoDiff/Nn/LlamaFusedKernels.cs`)

`RMSNormKernel<T>` is `internal`; `LlamaForCausalLM<T>` lives in samples and cannot reach it
(the #404 constraint). Add a public helper to the existing public `LlamaFusedKernels` class
(user choice), with `rows` generalized so decode (1) and prefill (L) share it:

```csharp
public static void RMSNormForwardInPlace<T>(T[] buffer, ReadOnlySpan<T> gamma, int rows, int cols, double eps)
    where T : struct, IFloatingPointIeee754<T>
```

- Validates `buffer`/`gamma` lengths (ArgumentNullException/ArgumentException).
- `RMSNormKernel<T>.PerRowRMSNormForwardKernel(buffer, buffer, rows, cols, eps)` — in-place is
  already the decoder-block pattern (`LlamaDecoderBlock.ForwardCachedFused` lines 231-235, 274-276).
- `for rows: TensorPrimitives.Multiply(buffer.AsSpan(i*cols, cols), gamma, buffer.AsSpan(i*cols, cols))`
  — the exact gamma multiply `RMSNorm<T>.ForwardInference` applies (RMSNorm.cs:148-152), so the
  result is bit-identical to the per-op path.
- Needs `using System.Numerics;` + `using System.Numerics.Tensors;`.

`RMSNorm<T>.ForwardInference` = `ToArray(input)` copy → `PerRowRMSNormForwardKernel(src, y, ...)` →
per-row `TensorPrimitives.Multiply`. The fused version skips the input copies by running the same two
steps in place; elementwise math is identical → bit-identical output for the same hidden row.

### 2. Samples — fused final stages (`samples/Nivara.Samples/LlamaForCausalLM.cs`)

`ForwardCachedFusedCore` — replace the `hTensor`/`finalNorm.Forward`/per-op matmul block:

```csharp
finalNorm.Weight!.Tensor.Data.TryGetSpan(out var finalGamma);
LlamaFusedKernels.RMSNormForwardInPlace(cur, finalGamma, 1, hidden, double.CreateChecked(finalNorm.Eps));
var logits = new T[vocabSize];
GradKernels.MatMulTransposedB(cur, embedSpan, logits, 1, hidden, vocabSize);   // reuse line-194 embedSpan
return ReverseGradTensor<T>.FromMatrix(logits, 1, vocabSize, requiresGrad: false);
```

`ForwardPrefillFusedCore` — replace the chunk copy → `finalNorm.Forward` → lastRow copy → per-op head:

```csharp
finalNorm.Weight!.Tensor.Data.TryGetSpan(out var finalGamma);
LlamaFusedKernels.RMSNormForwardInPlace(cur, finalGamma, L, hidden, double.CreateChecked(finalNorm.Eps));
var logits = new T[vocabSize];
GradKernels.MatMulTransposedB(cur.AsSpan((L - 1) * hidden, hidden), embedSpan, logits, 1, hidden, vocabSize);
return ReverseGradTensor<T>.FromMatrix(logits, 1, vocabSize, requiresGrad: false);
```

Bit-identity of the head: `ReverseGradOperations.MatMulTransposedB` (per-op) calls the same public
`GradKernels.MatMulTransposedB(aSpan, bSpan, resultArr, aRows, aCols, bCols)`; for `aRows == 1` +
transposed B, `TensorsHelper.MultiplyCore` takes the allocation-free BLAS2 GEMV fast path
(`TensorPrimitives.Dot` per output column, lines 124-132) — zero allocs, bit-identical reduction.

Allocation after (decode): `new T[vocab]` (607,744 B, LOH → gen0 0) + FromMatrix int[2]/boxes
(~100-300 B) → ≈ 608 KB + <1 KB residual. Prefill additionally drops ~3×L×hidden×4 B/op.

### 3. Tests (`tests/Nivara.Tests/AutoDiff/LlamaForCausalLMPrefillTests.cs`)

Reuse `RangeBitEqual`/`AssertClose` helpers and the existing `Nivara.Samples` reference. Add:

- `FinalStage_NormAndHead_FusedMatchesPerOpLogits_BitExact` — single row: fixed `T[hidden]` row +
  fixed head weights; per-op `norm.Forward(FromMatrix(row,1,hidden,false))` →
  `ReverseGradOperations.MatMulTransposedB(h, head)`; fused `RMSNormForwardInPlace(clone, gamma, 1, hidden, eps)`
  + `GradKernels.MatMulTransposedB(clone, headSpan, fused, 1, hidden, vocab)`; assert bit-equal.
- `FinalStage_MultiRow_FusedMatchesPerOpLastRowLogits_BitExact` — rows=4 chunk: per-op full
  `[4, hidden]` norm + last-row slice + head; fused in-place chunk norm + last-row slice + head;
  assert bit-equal (pins the prefill destination math).

Existing `ForwardCached_FusedMatchesPerOp_Within1e5` and prefill parity tests stay green unchanged
(they now exercise the new fused final stage; 1e-5 tolerance ≫ bit-identical final stage).

## Verification

1. `dotnet build Nivara.slnx`.
2. Filtered NUnit: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~LlamaForCausalLMPrefillTests"` (new tests + existing fused/per-op parity + prefill/decode).
3. Perf harness A/B (user approved both unit + perf verification):
   - Before (on `khurram/411`): `--only "Qwen decode fwd" --runs 3 --json decode-before.json` → expect ≈ 619,631 B/op, gen0 0.
   - After: full run `--json decode-after.json` → decode row ≈ 608 KB + <1 KB residual, gen0 0; prefill rows drop ~3×L×hidden×4 B/op.
   - Gate: `--compare qwen-decode-block-baseline.json` on the after run → 16/16 PASS (B/op only decreases; ops only improves; gen0 unchanged 0 ≤ base+0.05).

## Planned commits

1. `docs: plan P2 fused final norm + tied head (#413) in TODO.md`
2. `perf: add public LlamaFusedKernels.RMSNormForwardInPlace fused norm entry`
3. `perf: fuse final RMSNorm + tied LM head into fused decode/prefill final stage`
4. `test: add bit-exact fused-vs-perop final norm+head tests (#413)`
5. `docs: remove TODO.md — issue #413 executed` (after G2)

## Blast radius

- `src/Nivara/AutoDiff/Nn/LlamaFusedKernels.cs` — one new public static method on an existing
  public class; additive, no behavior change.
- `samples/Nivara.Samples/LlamaForCausalLM.cs` — only the two `*FusedCore` final stages; per-op
  (`DecoderBlockFused=false`) paths, `Embed`, `finalNorm` wiring, `ForwardCached`/`ForwardPrefill`
  are untouched. Return contract (`ReverseGradTensor<T>` `[1, vocab]`, requiresGrad false) unchanged.
- `tests/Nivara.Tests/AutoDiff/LlamaForCausalLMPrefillTests.cs` — two additive tests only.
- Downstream: `Qwen.cs`/`GenerateCore` callers (decode/prefill) and `--compare` gate rows
  (`Qwen decode fwd [1 step]`, prefill seed rows) are the only consumers of the fused paths.

## GitHub issues log

- [ ] #418 — vectorize RMSNorm backward grad loops (out of scope here; tracked from #411). Confirm still relevant at G2.