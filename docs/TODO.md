# Plan: #785 — vectorized float4 loads in GEMM tile (Silk/ILGPU)

## Problem
GEMM hot path stages A and B element-by-element into shared memory. B's four output
columns per thread are contiguous and can be loaded as one `float4`, reducing load
instructions and improving coalescing on the iGPU.

## Proposed changes
- OpenCL `GEMM_ROW4_ACCUMULATE` macro (f32 variant) in `src/Nivara.Extensions/Silk/SilkKernels.cs`: replace B staging loop with guarded `vload4`, scalar fallback for partial/out-of-range/unaligned.
- ILGPU `TiledGemmKernelRow4` + bias/gelu/relu/qkv variants in `samples/Nivara.Samples/Gpu/GemmKernels.cs`: same B staging change via `b.Cast<ILGPU.Util.Float4>()`, scalar fallback. Add `using ILGPU.Util;`.
- A stays scalar (not contiguous per thread).
- BF16 path untouched.

## Blast radius
- OpenCL kernel string (compile-time macro) — all gemm_row4* f32 kernels.
- ILGPU sample kernels used by `--silk-kernels` parity gate.
- Parity: byte-identity must hold (accumulation order unchanged).

## Verification
- Build samples + extensions.
- `--silk-kernels` parity gate (human-run).
- Targeted `SilkGemmGemvAddTests`.

## Planned commits
1. docs: plan #785
2. feat: vectorized float4 B-load in OpenCL GEMM macro
3. feat: vectorized float4 B-load in ILGPU GEMM kernels
4. docs: remove TODO.md

## GitHub issues log
- (none yet)
