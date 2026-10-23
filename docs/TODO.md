# Plan: issue #822 — B-tile vload4 alignment guard in Silk Row4 macro

## Problem

Follow-up to #807. PR #821 dropped the overly conservative A-tile `vload4` guard `(aBase % 4u) == 0u` from `SilkKernels.cs` because OpenCL `vload4` requires alignment only to the element type, which `a + aBase` always has. The same reasoning applies to the **B-tile** staging load in the same `GEMM_ROW4_ACCUMULATE` macro, where the guard is still present:

```c
if (bRow < aCols && (outCol0 + 3u) < bCols && (bBase % 4u) == 0u) {
    ... vload4(0, b + bBase) ...
}
```

## Proposed changes

### (1) `src/Nivara.Extensions/Silk/SilkKernels.cs`
Drop the `(bBase % 4u) == 0u` conjunct at the B-staging `vload4(0, b + bBase)` site in **both** copies of `GEMM_ROW4_ACCUMULATE` (f32 and BF16), keeping the in-range guard `bRow < aCols && (outCol0 + 3u) < bCols`:

```diff
- if (bRow < aCols && (outCol0 + 3u) < bCols && (bBase % 4u) == 0u) { \
+ if (bRow < aCols && (outCol0 + 3u) < bCols) { \
```

Two occurrences (lines ~440 and ~500 in the current tree). `b` is `__global const float*` (f32 route) or `__global const ushort*` (BF16 route), so `b + bBase` is always element-aligned by construction; the scalar tail still covers partial edges.

### (2) `docs/SILK.md` §11.3
Add a B-tile parity note alongside the existing A-tile alignment note, stating that the same element-alignment reasoning applies to `vload4(0, b + bBase)` and that the guard has been dropped while the in-range guard and scalar tail remain. Byte-identity (including `#pragma OPENCL FP_CONTRACT OFF`) is unchanged.

## Out of scope

- `samples/Nivara.Samples/Gpu/GemmKernels.cs` (ILGPU being removed).
- No performance claims.

## Blast radius

- Source: `SilkKernels.cs` internal (OpenCL C embedded string) — two macro sites only. `SilkKernels.Source` and `SilkKernels.KernelNames` unchanged; `SilkKernelsCatalogTests` unaffected.
- Consumers: kernels `gemm_row4`, `gemm_row4_bias`, `gemm_row4_gelu`, `gemm_row4_relu`, `gemm_row4_qkv` (f32) and `gemm_row4_bf16` (BF16) share the macro — all 50 `--silk-kernels` cells exercise it. No public C# signature changes. No docs count/pin changes expected (no new files or citations).
- Tests: `--silk-kernels` gate (~50 cells, byte-identity on row4 family). Device-free `SilkKernelsCatalogTests` stays green.

## Verification steps

1. `dotnet build Nivara.slnx -c Release`.
2. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
3. Device-free: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkKernelsCatalogTests"`.
4. Docs: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`.
5. Human-run (OpenCL GPU): `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels` → expect Gate PASS, all 50 cells, byte-identity on the four row4 kernels, exit 0.

## Planned commits

1. `docs: plan #822 in TODO.md`
2. `perf(silk): drop overly conservative B-tile vload4 alignment guard`
3. `docs(SILK): note B-tile vload alignment parity for #822`
4. `docs: remove TODO.md — plan executed` (after G2 review clears)

## GitHub issues log

- [ ] #822 — hygiene: drop overly conservative B-tile vload4 alignment guard in Silk Row4 macro (this issue)