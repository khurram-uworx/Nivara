# Plan: Issue #748 — Silk/GPU device-side BF16 weights

Branch: khurram/748 (off khurram/747). Target PR base: khurram/747.

## Problem

Silk/OpenCL GPU path is F32 throughout: GEMM/GEMV/fused kernels declare weight params as
`__global const float*`, runners accept `Dictionary<string,(float[],int[])>` and upload F32,
and the sample rejects `--silk --precision bf16`. Issue #387 already keeps BF16 weights raw on
CPU (widen-in-kernel). Need mirror that on GPU: upload raw `ushort` BF16 weights and widen
per-lane on device, halving device weight memory and upload bandwidth.

## Proposed changes

1. `src/Nivara.Extensions/Silk/SilkKernels.cs`: add BF16-weight variants
   (`gemm_row4_bf16`, bias/gelu/relu/qkv/add, `rms_norm_lm_head_bf16`, `gemm_gemv_add_bf16`,
   `rms_norm_gemv_bf16`, `mlp_block_bf16`, `decode_gqa_attention_oproj_fused_bf16`,
   `gather_rows_bf16`) with `b`/weight params typed `__global const ushort*` and tile-load
   conversion `BF16_TO_F32(x) = (float)(((uint)x) << 16)`. Register names in `KernelNames`.
2. `SilkLauncher.cs`: matching typed dispatch helpers targeting the `_bf16` kernel names.
3. `WeightLayout.cs`: generic `Transposed<T>`/`Req<T>` (element copy), reusable for BF16.
4. `SilkRuntime`: `Upload(nint, ReadOnlySpan<ushort>)`.
5. `SilkQwenGpuRunner`: add `WeightPrecision` discriminator + BF16 constructor; branch kernel
   calls per precision; widen small tensors (norms/biases) to float at upload. `SilkQwenSession`
   exposes `Open(..., bf16Weights)`.
6. Samples: enable qwen synthetic/benchmark Silk BF16 paths; keep tools/distill F32-only.
7. Tests: parity tests comparing Silk F32-upcast vs Silk BF16-raw; GPU-guarded.

Blast radius: fused OpenCL kernel source, catalog tests, all three Qwen forward paths, launch
helpers, host weight plumbing, sample CLI. Qwen first; other runners follow same pattern in later PRs.

## Verification steps

- `dotnet build Nivara.slnx` (core + Extensions + samples when possible)
- `SilkKernelsCatalogTests` (device-free)
- Targeted GPU parity tests on a machine with OpenCL
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commit list

1. docs: plan #748 in TODO.md
2. feat(Silk): on-device BF16 widen helper + BF16-weight kernel variants
3. feat(Silk): launch helpers for BF16-weight kernels
4. feat(Silk): generic WeightLayout + ushort Upload
5. feat(Silk): SilkQwenGpuRunner BF16-weight path
6. feat(sample): allow Silk BF16 qwen synthetic/benchmark
7. test(Qwen): Silk BF16 parity vs F32-upcast
8. docs: revise DocsCoveragePins (if needed)

## GitHub issues log

- [ ] #756 — Silk extend device-side BF16 weights to remaining models + sample CLI wiring (created while working on #748)
