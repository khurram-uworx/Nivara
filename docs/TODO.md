# Plan: GradKernels bypass sweep + softmax consolidation (issue: two stragglers)

## Problem

A prior review round (ADR-002 / F1) migrated AutoDiff ops onto the
`GradKernels` span-kernel facade, but two stragglers remained:

1. **ReverseGradOperations.Sigmoid/Tanh** (ReverseGradOperations.cs:1255-1309)
   call `TensorPrimitives` inline and use scalar backward loops, bypassing the
   `GradKernels.Sigmoid/SigmoidGradient/Tanh/TanhGradient` SIMD kernels that
   `ForwardGradOperations` already routes through — the same bypass-asymmetry
   F1 fixed for the matrix calls. On closer inspection **Relu, Gelu, LeakyRelu,
   Negate, Abs, Clip, Exp, Log** bypass GradKernels too (ForwardGrad uses the
   kernels for all of them).

2. **AttentionKernels.SoftmaxRows** is a third softmax. Commit b351fdb moved the
   MHA forward softmax into `GradKernels.SoftmaxRowsInPlace`, but it is still a
   second implementation duplicating `SoftmaxSingle`'s math, and
   `AttentionKernels.SoftmaxBackwardRows` (AttentionKernels.cs:61-82) duplicates
   `GradKernels.SoftmaxGradient`.

## Changes

### Part A — Route ReverseGradOperations ops through GradKernels

All in `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`. Forward calls
replace inline `TensorPrimitives`/scalar loops; backward loops replace with the
matching gradient kernel. Op labels, `ResultTensor`, graph-node wiring unchanged.
`using System.Numerics.Tensors` stays (remaining inline calls are legitimate).

| Op | Forward | Backward |
|---|---|---|
| Sigmoid (1255) | `GradKernels.Sigmoid(a.AsSpan(), resultArr)` | `GradKernels.SigmoidGradient(resultArr, gSpan, gradArr)` |
| Tanh (1283) | `GradKernels.Tanh(...)` | `GradKernels.TanhGradient(...)` |
| Relu (1156) | `GradKernels.Relu(...)` | `GradKernels.ReluGradient(...)` |
| Gelu (1184) | `GradKernels.Gelu(...)` | `GradKernels.GeluGradient(...)` |
| LeakyRelu (1389) | `GradKernels.LeakyRelu(a.AsSpan(), negativeSlope, resultArr)` | `GradKernels.LeakyReluGradient(...)` |
| Negate (1311) | `GradKernels.Negate(...)` | `GradKernels.Negate(...)` |
| Abs (1336) | `GradKernels.Abs(...)` | `GradKernels.AbsGradient(a.AsSpan(), gSpan, gradArr)` |
| Clip (1362) | `GradKernels.Clamp(...)` | `GradKernels.ClipGradient(...)` |
| Exp (1421) | `GradKernels.Exp(...)` | keep inline `Multiply(g, resultArr)` — no kernel, matches ForwardGrad |
| Log (1446) | `GradKernels.Log(...)` | `GradKernels.LogGradient(a.AsSpan(), gSpan, gradArr)` |

Behavioral deltas (all align ReverseGrad with ForwardGrad + PyTorch; kernels
already covered by `GradKernelsTests`):
- LeakyRelu at x==0: backward becomes `slope·g` (was `g`).
- Clip at x==min/max: backward now flows (inclusive `[min,max]`, was strict → 0).
- Gelu: numerics switch from double-intermediate scalar to the SIMD T-precision
  kernel already used by ForwardGrad and validated against PyTorch fixtures.

### Part B — Collapse softmax to one forward + one backward primitive

- `src/Nivara/AutoDiff/Operations/GradKernels.cs`: rewrite `SoftmaxRowsInPlace`
  (314-319) as a per-row loop calling `SoftmaxSingle(row, row)` (in-place safe:
  max scan precedes mutation; TensorPrimitives already does in-place at line 329).
  Delete private `SoftmaxRowInPlace` (321-351). float/double stay bit-identical;
  Half switches to the generic TensorPrimitives path the Softmax op already uses.
- `src/Nivara/AutoDiff/Operations/AttentionKernels.cs`: `SoftmaxBackwardRows`
  (61-82) becomes a one-line delegate
  `GradKernels.SoftmaxGradient(weights, dS, dS, cols)` (`SoftmaxGradientSingle`
  computes the per-row dot before writing, so the dP-as-both-input-and-output
  alias is safe). `SoftmaxRows` delegate unchanged.
- MHA callsites (ReverseGradOperations.cs:489, 548, 727, 824) untouched.

Net: single softmax kernel (`SoftmaxSingle`) + single softmax-gradient kernel
(`SoftmaxGradientSingle`) in GradKernels; AttentionKernels keeps only
gather/scatter/pack + thin delegates.

## Verification

- `dotnet build Nivara.slnx`
- `dotnet test` (requires human confirmation per AGENTS.md), filtered to:
  `GradOperationsTests`, `ForwardParityTests`, `GradKernelsTests`,
  `BatchedMultiHeadAttentionTests`, `MultiHeadAttentionTests`, NivaraTorch
  `ActivationTests` + attention suites.

## Planned commits

1. `docs: plan GradKernels bypass sweep in TODO.md`
2. `refactor: route activation ops through GradKernels in ReverseGradOperations`
   (Sigmoid, Tanh, Relu, Gelu, LeakyRelu)
3. `refactor: route elementwise ops through GradKernels in ReverseGradOperations`
   (Negate, Abs, Clip, Exp, Log)
4. `refactor: consolidate MHA softmax onto GradKernels primitives`
   (SoftmaxRowsInPlace → SoftmaxSingle; SoftmaxBackwardRows → SoftmaxGradient)
5. `docs: remove TODO.md — plan executed`
6. offer push + PR (human-confirmed)

## Follow-ups (not in scope)

- `ReverseGradOperations.Divide` forward still calls `TensorPrimitives.Divide`
  inline; its backward is a compound quotient-rule op with no single GradKernels
  counterpart. Left as-is to keep the change surgical.
