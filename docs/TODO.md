# Plan: Issue #1481 — Silk: multi-slot biased rms_norm_gemv decode GB/s measurement

## Problem

Follow-up to #1473. Real cached decode uses the fused multi-slot `rms_norm_gemv` (2 or 3 slots,
`hasBias`), not the single-slot no-bias shape currently measured for GB/s. Qwen uses biased Q/K/V
projections; SmolLM uses bias-free. The current GB/s report only covers the no-bias single-slot
case. Acceptance: reproducible GB/s numbers for the real multi-slot biased/bias-free decode
projection shapes.

## Blast radius

- `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` — add a new measurement method
  `MeasureDecodeShapeMultiSlotRmsNormGemv` and call it from the gate flow. Timing-only; no
  correctness changes. Existing correctness tests (`SilkRmsNormGemvTests`) already cover multi-slot
  biased/bias-free behavior.
- `docs/SILK.md` — extend §14a with a multi-slot addendum (#1481).
- No production code (`src/`) changes. No storage/column/query changes.
- No new test files. No DocsCoveragePins changes expected (no new docs/*.md file; TODO.md is
  excluded from the scan).

## Proposed changes

### 1. New perf-gate method

File: `tests/Nivara.PerformanceTests/SilkKernelsGate.cs`, new method following the style of
`MeasureDecodeShapeAbBf16` / `MeasureDecodeShapeGemvAddF32`:

- const `warmups = 5`, `rounds = 10`
- header: "decode-shape kernel GB/s: F32 rms_norm_gemv multi-slot (biased/bias-free) (#1481)"
- print the bytes-moved formula and caveats (same as §14a, plus optional bias bytes)

Cases (hidden, slotCols[], hasBias, label):

| label | hidden | slots | hasBias | source |
|---|---:|---|---:|---|
| qwen qkv | 896 | [896, 128, 128] | true | Qwen decode QKV, `SilkQwenGpuRunner.cs:658-665`, `QwenSilkConfig` |
| smollm qkv | 576 | [576, 192, 192] | false | SmolLM decode QKV, `SilkSmolLMGpuRunner.cs:434-441`, `SmolLMConfig.AttentionBias=false` |
| smollm gate/up | 576 | [1536, 1536] | false | SmolLM decode MLP decomposition, `SilkSmolLMGpuRunner.cs:775-788` |

Per case:
- allocate/upload `x[hidden]`, `gamma[hidden]`
- for each slot: allocate/upload `weight[hidden*cols]`, `output[cols]`, and `bias[cols]` only when
  `hasBias` (else bias handle 0)
- build `GemvSlot[]` with real bias handle when biased, 0 when not
- run `SilkLauncher.RmsNormGemv(silk, sx, sg, slots, hasBias, aRows:1, aCols:hidden, eps)`
- report median ms, GB/s, bytes, slot widths, hasBias, label

### 2. Bytes-moved model (logical)

```
bytes = sizeof(float) * (
    hidden                      // x
  + hidden                      // gamma
  + sum_i(hidden * cols_i)      // weights
  + sum_i(hasBias ? cols_i : 0) // biases, only when hasBias
  + sum_i(cols_i)               // outputs
)
GB/s = bytes / elapsed_seconds / 1e9
```

Caveats (consistent with §14a): logical lower-bound model; `rms_norm_gemv` reads `x` twice
internally and `gamma` once, formula counts once; cache effects may reduce actual traffic; compare
against copy-bandwidth probe. `hasBias` is a single flag for all slots in `SilkLauncher.RmsNormGemv`.

### 3. Wire into gate flow

Call `MeasureDecodeShapeMultiSlotRmsNormGemv(silk, buffers)` from `SilkKernelsGate` flow
(`SilkKernelsGate.cs:113-116`), placed after `MeasureDecodeShapeAb` and before `MeasureCopyBandwidth`
so the copy-bandwidth reference probe stays last.

### 4. Docs

Extend `docs/SILK.md` §14a with a #1481 addendum documenting the three shapes, the multi-slot
bytes formula including optional bias bytes, the single `hasBias` flag semantics, and the same
command. Keep existing #1473 single-slot formula intact.

## Planned commits

1. `docs: plan Silk multi-slot rms_norm_gemv decode GB/s measurement (#1481) in TODO.md`
2. `perf: add multi-slot biased/bias-free rms_norm_gemv decode GB/s measurement (#1481)` — gate method + wiring
3. `docs: document multi-slot rms_norm_gemv decode GB/s measurement (#1481)` — SILK.md addendum
4. `docs: remove TODO.md — plan executed (#1481)`

## Verification

- Build: `dotnet build Nivara.slnx -c Release`
- Targeted tests (ask before running): `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkRmsNormGemv|FullyQualifiedName~SilkGemvGeometry"`
- Formatting: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Acceptance evidence (GPU, human-run): `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels`

## GitHub issues log

(none yet)
