# Plan: Fix Silk SmolLM BF16 (issue #766)

## Problem

SmolLM Silk BF16 run produces all-zero generated tokens and fails the reference-sequence parity gate.

Repro: `dotnet run --project samples/NivaraInference -c Release -- smollm --silk --precision bf16`

Root causes:
1. `SilkKernels.Source` emits the BF16 kernel block (`gemm_row4_bf16`, `gemm_gemv_add_bf16`, `rms_norm_gemv_bf16`, `mlp_block_bf16`) twice, causing `clBuildProgram` redefinition errors (or wrong-copy usage on lenient drivers).
2. `rms_norm_gemv_bf16` declares its weight pointer as `const float*` instead of `const ushort*`, so `BF16_TO_F32` reads 4 bytes per BF16 slot.
3. LM-head buffers (`tokEmb`/`tokEmbTransposed`/`finalNormW`) are widened to `float` in the BF16 constructor, but LM-head calls route through `Gemm`/`RmsNormGemv`, which in the BF16 path dispatch to `GemmBf16`/`RmsNormGemvBf16` and reinterpret the float buffer as `ushort*`.

F32 Silk passes; CPU BF16 passes. Defect is specific to the true BF16 Silk path.

## Proposed changes

- `src/Nivara.Extensions/Silk/SilkKernels.cs`: remove duplicate BF16 kernel block; fix `rms_norm_gemv_bf16` weight pointer type to `ushort*`.
- `src/Nivara.Extensions/Silk/SilkSmolLMGpuRunner.cs`: route LM-head `Gemm`/`RmsNormGemv` calls that use `tokEmbTransposed` (a float buffer) directly to `SilkLauncher.Gemm`/`SilkLauncher.RmsNormGemv` (float kernels), bypassing `bf16Weights` dispatch. Layer-weight calls keep BF16 kernels.

## Blast radius

- Kernel source consumed by `SilkRuntime` (all Silk runners). Removing duplicate definitions affects program build for every runner but changes no kernel semantics.
- `SilkSmolLMGpuRunner` LM-head path only; layer paths unchanged.
- No public API changes.

## Verification steps

- Build `Nivara.Extensions` and `NivaraInference`.
- Run repro; expect parity gate PASS.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- Targeted tests in `Nivara.Tests` (if any Silk/SmolLM tests exist) — none found; run Docs snippet gate if docs touched (not here).

## Planned commits

1. `docs: plan issue 766 fix in TODO.md`
2. `fix(silk): remove duplicate BF16 kernel definitions and fix ushort pointer type`
3. `fix(silk): route SmolLM BF16 LM-head through float kernels`
4. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none yet)
