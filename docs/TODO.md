# Plan: Route reverse-mode MatMul/Transpose through GradKernels

## Problem

The GradKernels bypass sweep (PR #150) routed activations and elementwise ops in
`ReverseGradOperations` through the `GradKernels` span-kernel facade, matching
`ForwardGradOperations`. One inconsistency remains: the reverse-mode **matrix
ops and attention kernels** still call `TensorsHelper.MultiplyCore` /
`TensorsHelper.Transpose` directly (24 call sites), while forward mode routes
`MatMul`/`Transpose` through `GradKernels` (ForwardGradOperations.cs:220-286).

Note on Sigmoid/Tanh/SoftMax: the historic TensorsHelper↔GradKernels duplication
was already removed (commit f6e8754); those live only in `GradKernels` now. The
only AutoDiff bypass of `GradKernels` left is the MatMul/Transpose family.

## Changes

All in `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`. Every call is
a drop-in facade swap — `GradKernels.MatMul` / `MatMulTransposedB` / `Transpose`
(GradKernels.cs:458-468) are identical-signature delegates to the same
`TensorsHelper` kernels, and every call site already passes a `T[]` output as
`GradKernels.MatMul` requires. Behavior is bit-identical; op labels, graph-node
wiring, and diagnostics are untouched.

| Site (line) | Now | Replace with |
|---|---|---|
| MatMul fwd (265) | `TensorsHelper.MultiplyCore(a,b,arr,R,K,C)` | `GradKernels.MatMul(...)` |
| MatMul bwd (282, 291) | `TensorsHelper.Transpose(...)` | `GradKernels.Transpose(...)` |
| MatMul bwd (285, 294) | `TensorsHelper.MultiplyCore(...)` | `GradKernels.MatMul(...)` |
| MatMulTransposedB bwd (366, 379) | `TensorsHelper.MultiplyCore(...)` | `GradKernels.MatMul(...)` |
| MatMulTransposedB bwd (377) | `TensorsHelper.Transpose(...)` | `GradKernels.Transpose(...)` |
| Transpose fwd (893) / bwd (904) | `TensorsHelper.Transpose(...)` | `GradKernels.Transpose(...)` |
| MHA fwd (494, 732) | `TensorsHelper.MultiplyCore(...)` | `GradKernels.MatMul(...)` |
| MHA bwd (547, 823) | `TensorsHelper.MultiplyCore(..., bTransposed:true)` | `GradKernels.MatMulTransposedB(...)` |
| MHA bwd (555, 557, 559, 831, 833, 835) | `TensorsHelper.MultiplyCore(...)` | `GradKernels.MatMul(...)` |
| MHA bwd (556, 558, 832, 834) | `TensorsHelper.Transpose(...)` | `GradKernels.Transpose(...)` |

Mapping:
- `TensorsHelper.MultiplyCore(a, b, out, R, K, C)` → `GradKernels.MatMul(a, b, out, R, K, C)`
- `TensorsHelper.MultiplyCore(a, b, out, R, K, C, bTransposed: true)` → `GradKernels.MatMulTransposedB(a, b, out, R, K, C)`
- `TensorsHelper.Transpose(src, dst, R, C)` → `GradKernels.Transpose(src, dst, R, C)`

Cleanup:
- `using Nivara.Tensors;` (line 3) becomes unused — the file's only
  `Nivara.Tensors` type is `TensorsHelper` (`NivaraColumn` resolves via the
  parent `Nivara` namespace). Drop it.
- Fix the `MultiHeadAttention` doc comment (line 401) that references
  `TensorsHelper.MultiplyCore` → `GradKernels.MatMulTransposedB`.
- `using System.Numerics.Tensors` stays (inline `TensorPrimitives` calls remain).
- `Divide` forward stays inline (no GradKernels counterpart — prior plan's
  documented follow-up).

## Verification

- `dotnet build Nivara.slnx`
- `dotnet test` (requires human confirmation per AGENTS.md), filtered to:
  `GradOperationsTests`, `ForwardParityTests`, `LinearTransposedWeightCacheTests`,
  `LinearInferenceTests`, `MultiHeadAttentionTests`, `BatchedMultiHeadAttentionTests`.
- Optional guard: `rg "TensorsHelper" src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`
  should return no code references (doc comment updated).

## Planned commits

1. `docs: plan GradKernels MatMul/Transpose sweep in TODO.md`
2. `refactor: route reverse MatMul/MatMulTransposedB/Transpose through GradKernels`
3. `refactor: route attention kernels through GradKernels in ReverseGradOperations`
4. `chore: drop unused using and update MHA doc comment`
5. `docs: remove TODO.md — plan executed`
6. offer push + PR (human-confirmed)

## Follow-ups (not in scope)

- `ReverseGradOperations.Divide` forward still calls `TensorPrimitives.Divide`
  inline; its backward is a compound quotient-rule op with no single GradKernels
  counterpart. Left as-is to keep the change surgical.
