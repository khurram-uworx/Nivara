# Plan: Issue #1133 — BF16 backward GEMM kernel variants

## Problem

The Silk DistilBERT trainer's true BF16 path is blocked on backward GEMM kernels
that can read BF16 (`ushort`) weight buffers. Forward BF16 GEMMs already exist
(`gemm_row4_bf16`, etc.), but the backward chain uses F32-only `gemm_row4_nt`,
`gemm_row4_tn`, and `gemm_row4_nt_residual`.

Scope (issue #1133): add BF16 backward GEMM variants where the **weight operand
(B) is `__global const ushort*`** and activations/gradients stay `float`, matching
the existing Phase 1 narrow lane (weights BF16, activations F32):

- `gemm_row4_nt_bf16` (`GemmNtBf16`)
- `gemm_row4_tn_bf16` (`GemmTnBf16`)
- `gemm_row4_nt_residual_bf16` (`GemmNtResidualBf16`)

## Proposed changes

### `src/Nivara.Extensions/Silk/SilkKernels.cs`

1. Add kernel name constants:
   - `GemmNtBf16 = "gemm_row4_nt_bf16"`
   - `GemmNtResidualBf16 = "gemm_row4_nt_residual_bf16"`
   - `GemmTnBf16 = "gemm_row4_tn_bf16"`
2. Register all three in `KernelNames`.
3. Add macros `GEMM_NT_ACCUMULATE_BF16` and `GEMM_TN_ACCUMULATE_BF16` (B loads via
   `BF16_TO_F32`; A loads unchanged).
4. Add `__kernel void` definitions for the three variants (B declared
   `__global const ushort*`).

### `src/Nivara.Extensions/Silk/SilkLauncher.cs`

5. Add `SilkLauncher.GemmNtBf16`, `GemmTnBf16`, `GemmNtResidualBf16` mirroring
   the existing F32 launchers.

F16 symmetry variants are optional and out of scope for this issue.

## Blast radius

- `SilkKernels.KernelNames` drives `SilkRuntime.BuildProgram`, which compiles every
  listed kernel at runtime; the three new kernels must be syntactically valid OpenCL C.
- `SilkLauncher` gains three public (internal) methods; no existing callers change.
- No trainer or runner changes in this issue; callers remain F32. Follow-up work
  (issue #1103) will route the trainer to the new kernels.
- Tests: `SilkBackwardKernelTests` (Gpu category) and any Silk runtime construction
  exercise the program build.

## Verification steps

- `dotnet build src/Nivara.Extensions/Nivara.Extensions.csproj -c Release`
- On a machine with OpenCL, run `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkBackwardKernelTests"` to confirm program build + existing backward kernels.
- A small parity probe dispatching each new kernel with BF16 weights (uploaded via
  `AllocateUshort`/`Upload(ReadOnlySpan<ushort>)`) should match the F32 kernel within
  the documented narrow BF16 tolerance.

## Planned commits

1. `docs: plan issue #1133 in TODO.md`
2. `feat(silk): add BF16 backward GEMM kernel name constants and macros`
3. `feat(silk): add gemm_row4_nt/tm/nt_residual _bf16 kernel definitions`
4. `feat(silk): register BF16 backward GEMM kernels in KernelNames`
5. `feat(silk): add SilkLauncher GemmNtBf16/GemmTnBf16/GemmNtResidualBf16`
6. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none yet)
