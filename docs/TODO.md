# TODO: #1470 — evaluate 2-dispatch MLP fusion (Option B)

## Problem

Follow-up to #1453 / PR #1465. P1 replaced single-WG fused decode MLP (`mlp_block`/`mlp_block_bf16`) with Option A (3 dispatches): `RmsNormGemv`(gate+up) + `SwiGlu` + `GemmGemvAdd`(down+residual). Option A removed iGPU serialization and gave large wins (SmolLM MLP 2.67x, Qwen MLP 4.41x per issue), but still writes `gate`, `up`, and `mlp/intermediate` through global buffers and issues 3 dispatches.

Issue #1470 asks to evaluate **Option B**: fuse norm+gate+up+swiglu into one multi-WG kernel writing only `intermediate` to global, then keep down+residual as a second multi-WG kernel. Prototype only if profiling shows intermediate/global traffic or dispatch overhead is material; keep Option A as reference/fallback. User confirmed: "Option B i agreed".

## Grounding (G1)

- **Codebase (code-memory/grep):** Current no-LoRA decode MLP is in `SilkSmolLMGpuRunner.MlpBlock` (`src/Nivara.Extensions/Silk/SilkSmolLMGpuRunner.cs:774-811`) and `SilkQwenGpuRunner.MlpBlock` (`src/Nivara.Extensions/Silk/SilkQwenGpuRunner.cs:810-847`). SmolLM uses base F32/BF16 variants; Qwen uses subgroup F32/BF16 variants. LoRA-active path already bypasses fused decomposition (`RmsNorm + GemmWithLora + SwiGlu + GemmWithLora + Add`).
- **Existing kernels:** `RmsNormGemv*` family is multi-slot (up to 3) multi-WG GEMV with shared norm staging (`SilkKernels.cs:3548` F32, `4285` BF16, subgroup variants at 5109/5283). `SwiGlu` is elementwise (`SilkKernels.cs:1709`). `GemmGemvAdd*` is second dispatch and remains unchanged. Old single-WG `mlp_block`/`mlp_block_bf16` still exist in kernels/launcher and are used by perf gate + `SilkMlpBlockDecompositionTests` as reference.
- **microsoft-learn:** Not directly applicable to Silk.NET/OpenCL kernel fusion internals; grounded in existing repo patterns (RmsNormGemv multi-slot staging, SwiGLU formula `g/(1+exp(-g))*u`, GEMV dispatch geometry `GEMVCOLS/GEMVLANES/BLOCK`). No external API assumption changed.
- **Decision (user-confirmed):** Implement Option B. Keep Option A as reference/fallback. Preserve LoRA semantics by bypassing fusion when LoRA active (already the case).

## Blast radius

| Area | Files/symbols | Risk |
|---|---:|---|
| Kernel defs | `src/Nivara.Extensions/Silk/SilkKernels.cs` - add `RmsNormGemvSwiGlu*` kernel names + OpenCL bodies (F32, BF16, subgroup F32, subgroup BF16) | Medium - new kernels, no existing behavior change |
| Launcher | `src/Nivara.Extensions/Silk/SilkLauncher.cs` - add `RmsNormGemvSwiGlu*` launch methods | Low - additive |
| SmolLM runner | `SilkSmolLMGpuRunner.MlpBlock` no-LoRA path | Medium - decode behavior; parity tests required |
| Qwen runner | `SilkQwenGpuRunner.MlpBlock` no-LoRA path | Medium - uses subgroup variants |
| Tests | New GPU parity test for Option B vs Option A/decomposition; existing `SilkMlpBlockDecompositionTests` remains reference | Low/positive |
| Perf gate | `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` - add Option B measurement (optional but useful) | Low |
| Docs | `docs/SILK.md` update if behavior/perf claim changes; `DocsCoveragePins` only if citations/counts change | Low |

No public API changes. No LoRA behavior change. Old fused `mlp_block` kernels remain.

## Proposed changes

1. **Capture baseline numbers** in `release/vCurrent/` before code changes (user-requested): silk-kernels gate output and, if feasible, SmolLM/Qwen synthetic profile. Record files in plan/PR.
2. **Add Option B kernels** in `SilkKernels.cs`: fused `rms_norm_gemv_swiglu` family that computes RMSNorm(x), gate/up projections for same intermediate column block, applies SwiGLU, writes only `intermediate`. Structure mirrors `RmsNormGemv` staged path but takes two weight matrices (gate,up) and one output (intermediate), no bias/out slots. Provide F32, BF16, subgroup F32, subgroup BF16 to match runner coverage.
3. **Add launcher methods** in `SilkLauncher.cs`: `RmsNormGemvSwiGlu`, `RmsNormGemvSwiGluBf16`, subgroup variants/capability-gated fallbacks.
4. **Wire runners**: in no-LoRA decode `MlpBlock`, replace 3-dispatch Option A with 2-dispatch Option B (`RmsNormGemvSwiGlu*` + `GemmGemvAdd*`), keep LoRA bypass unchanged, keep Option A easy to restore (comment/reference).
5. **Tests**: add GPU parity test asserting Option B output matches Option A decomposition bit-for-bit (F32) / within tolerance as appropriate for BF16, on SmolLM and Qwen shapes. Keep existing `SilkMlpBlockDecompositionTests` passing.
6. **After-implementation numbers**: rerun same commands, store in `release/vCurrent/`, compare before/after.

## Verification

- [ ] `dotnet build Nivara.slnx -c Release`
- [ ] Targeted GPU tests: `SilkMlpBlockDecompositionTests`, new Option B parity test (ask before running / user already authorized test run)
- [ ] `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- [ ] If docs changed: Docs namespace tests as needed
- [ ] Baseline and after numbers captured in `release/vCurrent/`

## Planned commits

1. `docs: plan #1470 Option B MLP fusion evaluation`
2. `perf(silk): add 2-dispatch norm+gate+up+swiglu fused kernels`
3. `test(gpu): add Option B MLP parity coverage`
4. `docs: record #1470 before/after numbers and Silk notes` (if needed)
5. `docs: remove TODO.md — plan executed` (after G2 clear)

## GitHub issues log

- [ ] #1470 — evaluate 2-dispatch MLP fusion (this work)

## Deferred work / concerns

- If profiling shows Option B not worth it, close #1470 with evidence instead of merging prototype (user said Option B agreed; still record evidence).
- Subgroup variant complexity: if subgroup fusion is disproportionate, document and gate fallback carefully.
- Perf gate currently measures old single-WG `mlp_block`; consider adding Option B measurement for repeatability.