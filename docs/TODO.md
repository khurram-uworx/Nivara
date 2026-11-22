# Plan: #1453 Silk SmolLM/Qwen decode fusion — MLP occupancy fix

## Problem

Issue: https://github.com/khurram-uworx/Nivara/issues/1453
Branch: `khurram/silk-fusion` (off `khurram/1453`, PR targets `khurram/1453`)

First decode fusion (`rotary_qk`) landed: 2 rotary dispatches/layer/token → 1, dispatches/token -16.4% (SmolLM) / -16.3% (Qwen). Wall-time impact is small because rotary is ~2 ms/token while **MLP dominates**: 135.9 ms/token (73%, SmolLM) / 499.2 ms/token (87%, Qwen) in the profiled synthetic BF16 decode run on Intel(R) Graphics.

Root cause: `MlpBlock` / `mlp_block_bf16` is already fused (RMSNorm + gate/up + SwiGLU + down + residual) but dispatches as **a single workgroup of 256 threads** (`SilkLauncher.MlpBlock`, `SilkKernels.cs:3650-3699`). That serializes the MLP on an iGPU. By contrast `RmsNormGemv` and `gemm_gemv_add` dispatch multi-WG.

Measurement caveat: profile mode inserts `runtime.Finish()` after each phase for attribution (`SilkSmolLMGpuRunner.cs:428+`, `SilkQwenGpuRunner.cs:419+`), so profiled ms/token is not steady-state. Steady-state and profiled attribution must be reported separately.

## Goals

- Make `tok/s` the primary reported metric (industry standard), alongside ms/token.
- Fix MLP decode occupancy with a parity-safe multi-WG path.
- Keep profiling/dispatch evidence reproducible via synthetic-weight profile command.

## Grounding (G1 - before implementation)

- [ ] microsoft-learn MCP: confirm OpenCL NDRange/work-group sizing guidance and .NET timing/Stopwatch guidance used by profiling; verify no BCL API assumptions.
- [ ] code-memory MCP: map `MlpBlock` call sites, LoRA branches, BF16/F32 variants, `RmsNormGemv` slot behavior, `SwiGlu`/`gemm_gemv_add` kernels, and tests pinning MLP parity.
- [ ] Runtime probe/inspection: confirm `LinearGroupSize==256`, `MlpBlock` global/local dispatch shape, `RmsNormGemv` multi-WG dispatch, and LoRA fallback behavior.
- [ ] Blast radius stated below. Any decision/red flag escalated to human before implementation.

## Blast radius

- `src/Nivara.Extensions/Silk/SilkKernels.cs` - no new kernel expected for Option A; possible minor cleanup only if needed. MLP kernels remain as reference/fallback.
- `src/Nivara.Extensions/Silk/SilkLauncher.cs` - may add helper dispatches only if needed; existing `RmsNormGemv`, `SwiGlu`/elementwise swiglu path, `GemmGemvAdd*` reused.
- `src/Nivara.Extensions/Silk/SilkSmolLMGpuRunner.cs` - decode MLP path: replace single-WG `MlpBlock` with multi-WG decomposition when safe (no active LoRA on the MLP slots; BF16/F32 supported). Keep LoRA path correct.
- `src/Nivara.Extensions/Silk/SilkQwenGpuRunner.cs` - same change.
- `src/Nivara.Extensions/Silk/SilkTinyShakespeareGpuRunner.cs` - check for same MLP shape; apply only if equivalent and parity-covered.
- `samples/NivaraInference/Program.cs`, `samples/NivaraInference/Qwen.cs` - report `tok/s` prominently in profile/benchmark output; keep existing fields.
- `tests/Nivara.Tests/Gpu/*` - add parity tests for decomposed MLP vs existing fused MLP/CPU reference.
- `docs/SILK.md`, `docs/QWEN.md`, `docs/TODO.md` - evidence and profiling semantics.
- No core `src/Nivara` changes expected. No public API break intended.

## Proposed changes

### P0 - Baseline hygiene

- Add steady-state summary alongside profiled phase attribution in synthetic profile output. Profile mode keeps per-phase Finish for attribution; also report end-to-end decode tok/s/ms-per-token without per-phase sync (or clearly label profiled totals as profiled).
- Report `tok/s` as primary metric in SmolLM/Qwen synthetic profile/benchmark output.
- Record dispatch shape caveat: MlpBlock single WG 256 threads; RmsNormGemv/GemmGemvAdd multi-WG.

### P1 - MLP occupancy fix (Option A, preferred first)

Decompose decode MLP into existing multi-WG kernels when no active MLP LoRA is present:

1. `RmsNormGemv(x, postNormW, [gateOut, upOut], hasBias=false, aRows=1, aCols=hidden, eps)` - reuse existing fused norm+2 GEMVs.
2. SwiGLU elementwise: `silu(gateOut) * upOut -> interOut` into global scratch buffer. Reuse existing `SwiGlu` kernel/launcher if shape matches; otherwise add a small dedicated elementwise kernel (multi-WG, simple, parity-tested).
3. Down projection + residual: `gemm_gemv_add`/BF16 variant (`GemmGemvAdd*`) with residual=x (pre-MLP residual input as current MlpBlock does). Confirm residual semantics match current `mlp_block`.

Implementation notes:
- Allocate/reuse scratch buffers for gateOut, upOut, interOut in workspace (Ensure-sized, no per-token alloc).
- BF16 path: use `RmsNormGemvBf16` + BF16-compatible swiglu/down path as needed; match existing BF16 weight storage.
- LoRA: if any of gate/up/down LoRA active, keep current correct path (existing MlpBlock or existing unfused LoRA-aware path) - do not regress LoRA.
- Keep `MlpBlock` kernel available for reference/fallback; do not delete unless proven unnecessary.
- Apply to SmolLM and Qwen decode paths; TinyShakespeare only if it uses same MLP shape and is covered.

### P2 - Follow-ups (issues, not this PR unless trivial)

- GEMV efficiency pass: vectorized loads, local-mem staging of normed x, subgroup reductions where available; measure achieved GB/s vs copy peak.
- Attention/O-proj shape: revisit `decode_gqa_attention_oproj_fused` single-WG at long kvLen / after MLP fixed.
- Host/pipeline cleanup: per-token upload/readback audit, CL_QUEUE_PROFILING_ENABLE diagnostic mode.
- Narrower weights/int4/int8 dequant (separate scope).

## Verification

- `dotnet build Nivara.slnx -c Release`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Ask before running tests. Targeted:
  - `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkRmsNormGemv|FullyQualifiedName~Silk|FullyQualifiedName~SmolLM|FullyQualifiedName~Qwen"`
  - `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"`
  - `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` when docs change
- Human-approved synthetic benchmarks (agent got explicit approval to run synthetic benchmarks in this session/issue context; run sequentially, wait for idle):
  - `dotnet run -c Release --project samples/NivaraInference -- smollm --synthetic-weights profile --silk`
  - `dotnet run -c Release --project samples/NivaraInference -- qwen --synthetic-weights profile --silk`
- Record before/after tok/s, ms/token, dispatches/token, phase attribution. Do not claim win without steady-state evidence.

## Planned commits

1. `docs: plan MLP occupancy decode fusion follow-up in TODO.md`
2. `feat(silk): add steady-state tok/s decode profile reporting`
3. `perf(silk): decompose decode MLP into multi-WG kernels with parity`
4. `test/docs: cover MLP decomposition parity and record evidence`
5. `docs: remove TODO.md - plan executed` (only after G2 clears)

## GitHub issues log

- [ ] #1453 — Silk SmolLM/Qwen decode fusion using synthetic-weight profiling (this branch continues it)
- [ ] #1457 — Silk: chunked linear+cross-entropy for SmolLM/Qwen training (deferred)
- [ ] #1458 — Silk: flash-style attention forward+backward for SmolLM/Qwen training (deferred)
- [ ] #1459 — Silk: fuse SwiGLU forward+backward for SmolLM/Qwen (deferred)
- [ ] #1460 — Silk: fuse RMSNorm+residual forward+backward for SmolLM/Qwen (deferred)
- [ ] #1461 — Silk: fuse RoPE into QKV path for SmolLM/Qwen (deferred)
- [ ] #1462 — Silk: fuse LoRA epilogue base(x)+(x*A)*B*scale for SmolLM/Qwen (deferred)
- [ ] #1463 — Silk: fuse bias/activation/residual matmul epilogues for SmolLM/Qwen (deferred)

## G1 grounding notes

- Grounded code paths: `SilkLauncher.MlpBlock` dispatch is single-WG (`global/local = LinearGroupSize=256`, `SilkLauncher.cs:740-761`); `RmsNormGemv` is multi-WG (`GemvGlobalCols/GemvGroupSize`); `GemmGemvAdd` is multi-WG. Existing no-LoRA LoRA-path already decomposes MLP into multi-WG ops; no-LoRA path used fused single-WG kernel.
- Blast radius: SmolLM/Qwen decode `MlpBlock` no-LoRA path only; BF16/F32 supported; LoRA active path unchanged. Fused `mlp_block`/`mlp_block_bf16` kernels left in place.
- Decision: use Option A decomposition (RmsNormGemv gate+up + SwiGlu + GemmGemvAdd down+residual) for decode rows=1. RmsNormGemv requires aRows==1; all decode call sites are rows=1. No red flag requiring escalation.
- microsoft-learn search was not specific to OpenCL NDRange (returned generic Windows results); grounding relied on existing runtime invariants (`SilkRuntime.Run` requires global multiple of local) and in-repo kernel/launcher contracts.

## Evidence (synthetic Silk, Intel(R) Graphics, .NET 11, Release)

### Baseline (khurram/1453, before this branch, 2026-05-23)

SmolLM-135M (30L, 576 hidden, 9/3 heads, 1536 inter, vocab 49152, BF16, 64 prompt/32 generated):
- Profile run: 6058 ms, 32 tokens, prefill 112 ms, decode 5946 ms, **5.38 tok/s** (profiled, per-phase Finish)
- Dispatches: 4896 (153.00/token), copies: 1920
- MLP 135.888 ms/token (73%)

Qwen2.5-0.5B (24L, 896 hidden, 14/2 heads, 4864 inter, vocab 151936, BF16, 64 prompt/32 generated):
- Profile run: 18494 ms, 32 tokens, prefill 221 ms, decode 18273 ms, **1.75 tok/s** (profiled, per-phase Finish)
- Dispatches: 3936 (123.00/token), copies: 1536
- MLP 499.207 ms/token (87%)

### After P1 MLP decomposition (this branch, 2026-05-23)

Commands (human-approved, run sequentially):
- `dotnet run -c Release --project samples/NivaraInference -- smollm --synthetic-weights profile --silk`
- `dotnet run -c Release --project samples/NivaraInference -- qwen --synthetic-weights profile --silk`

SmolLM-135M (same shape):
- Steady-state warmup (no per-phase sync): **11.80 tok/s**, 11.80 ms/token decode, 2845 ms total (32 tokens, prefill 134 ms, decode 2711 ms)
- Profiled run (per-phase Finish sync, attribution only): **10.84 tok/s**, 92.28 ms/token decode, 3052 ms total (32 tokens, prefill 99 ms, decode 2953 ms)
- Dispatches: 6816 (213.00/token), copies: 1920 (+60 dispatches/token = 2 extra/layer * 30 layers, expected)
- MLP 50.752 ms/token (55%), down from 135.888 ms/token (**2.67x faster**)
- Total profiled 2948.33 ms, down from ~5946 ms (**~2.02x**)

Qwen2.5-0.5B (same shape):
- Steady-state warmup (no per-phase sync): **6.29 tok/s**, 159.09 ms/token decode, 5338 ms total (32 tokens, prefill 247 ms, decode 5091 ms)
- Profiled run (per-phase Finish sync, attribution only): **5.83 tok/s**, 171.50 ms/token decode, 5695 ms total (32 tokens, prefill 207 ms, decode 5488 ms)
- Dispatches: 5472 (171.00/token), copies: 1536 (+48 dispatches/token = 2 extra/layer * 24 layers, expected)
- MLP 113.193 ms/token (66%), down from 499.207 ms/token (**4.41x faster**)
- Total profiled 5484.13 ms, down from ~18273 ms (**~3.33x**)

Notes: synthetic weights (shape-driven), same device/driver session, profiled run includes per-phase `Finish()` sync for attribution; steady-state warmup is the better wall-clock throughput indicator. Dispatch increase is expected and accepted given wall-time win.
