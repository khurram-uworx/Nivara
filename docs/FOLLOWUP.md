# Follow-up: .NET 11+ API Tracking for AutoDiff

Tracks `System.Numerics.Tensors` APIs that don't exist in .NET 10 but might ship later.
When upgrading the target framework, check these items and update `AUTODIFF-PLAN.md`.

## Confirmed Gaps (not in .NET 10 or .NET 11 preview)

| API | Status | Affected Plan Section | What to do when it ships |
|---|---|---|---|
| `Tensor.MatrixMultiply<T>(Tensor<T>, Tensor<T>)` | ❌ NOT in .NET 10 or .NET 11 preview | P0a — MatMul SIMD | Replace body of `MatMulHelper.Multiply<T>(Tensor<T>, ...)` in `src/Nivara/Tensors/MatMulHelper.cs:20` to call it directly. No callers change. |
| Full GEMM (batched matmul, BLAS3) | ❌ Not in either release | P0a, P6 | Revisit data-parallel training batching. |

## What .NET 10 Gives Us (confirmed)

| API | Used By |
|---|---|
| `TensorPrimitives.Dot<T>(ReadOnlySpan<T>, ReadOnlySpan<T>)` | P0a MatMul inner kernel (via `MatMulHelper`) |
| `TensorPrimitives.Multiply<T>` / `Subtract<T>` / `Add<T>` | P0b optimizer kernels |
| `TensorPrimitives.Clamp<T>` | P0c gradient clipping |
| `TensorPrimitives.Norm<T>` | P0c gradient norm |
| `Tensor<T>.FlattenTo(Span<T>)` | P0a extracting tensor data |
| `Tensor.Transpose<T>(Tensor<T>)` | Not currently used; possible optimization for MatMul |
| `Tensor.Create<T>(T[], nint[])` | P0a wrapping result data |
| `TensorPrimitives` generic overloads | All of P0b, P0c |

## .NET 11 Preview Status

All APIs verified against `System.Numerics.Tensors` v11.0.0-preview.4.26230.115 (same as v10.0.9 stable). **No new matmul/GEMM APIs added.** The `TensorPrimitives.Dot` and `Tensor.Transpose` APIs are unchanged.

## When to Revisit

- **.NET 11 stable ships**: re-verify `Tensor.MatrixMultiply` — it may appear without preview notice
- **A new `System.Numerics.Tensors` NuGet package ships**: check release notes for `MatrixMultiply` or `Gemm`
- **P0a is implemented and benchmarked**: capture baseline perf numbers so the swap impact is measurable
