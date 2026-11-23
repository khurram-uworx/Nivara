# Plan: Silk decode GEMV GB/s measurement guidance (#1473)

## Problem

Issue [#1473](https://github.com/khurram-uworx/Nivara/issues/1473) (follow-up to #1466): the decode GEMV efficiency pass needs a reproducible way to measure achieved GB/s vs device copy/bandwidth peak. Today the `--silk-kernels` perf gate (`tests/Nivara.PerformanceTests/SilkKernelsGate.cs`) reports timings/ratios but not GB/s, and NivaraInference reports tok/s + phase attribution. Before/after comparisons for future decode GEMV changes are not comparable without an explicit bytes-moved model and a device bandwidth proxy.

## Engineering note (user guidance)

Where GB/s or probe code already exists in one sample/probe, prefer promoting it to a shared location (`Nivara.Samples` or a shared helper) rather than duplicating it. GB/s calculation patterns currently exist inline in `ApplyMaskProbe.cs` and `ScalarKernelProbe.cs`; if the implementation adds a new reusable GB/s helper, promote it to a shared location so future probes can reuse it.

## Acceptance criteria (from issue)

- Reproducible GB/s numbers for SmolLM/Qwen decode GEMV shapes before/after future changes.

## Proposed changes

### 1. Add logical bytes-moved GB/s to decode-shape A/B gate

**File:** `tests/Nivara.PerformanceTests/SilkKernelsGate.cs`

Extend `MeasureDecodeShapeAb` (currently line ~1334, called from `Run` at line ~113) to report GB/s for both the tile (`rms_norm_lm_head`) and flat (`rms_norm_gemv`) paths.

**Bytes-moved model (logical, documented):**

```
bytes = weight_bytes_read + x_bytes_read + gamma_bytes_read + output_bytes_written [+ bias_bytes_read if hasBias]
GB/s  = bytes / elapsed_seconds / 1e9
```

For the current one-slot no-bias `rms_norm_gemv` A/B:

- `x`: `hidden * sizeof(float)`
- `gamma`: `hidden * sizeof(float)`
- `weight`: `hidden * width * sizeof(float)`
- `output`: `width * sizeof(float)`
- `bias`: `0` (A/B uses `hasBias: false`)

Caveats to document and print: logical lower-bound traffic model for comparability. `rms_norm_gemv` reads `x` twice (ssq pass + compute pass) and `gamma` once in compute; caches may reduce actual DRAM traffic. The formula is the comparison contract - keep it stable across before/after runs.

**Shape coverage (reproducible SmolLM/Qwen decode GEMV shapes):**

Keep existing shapes and add representative SmolLM/Qwen decode projections:

| Model | Shape (hidden,width) | Meaning |
|---|---:|---|
| Qwen2.5-0.5B | (896, 151936) | LM head / vocab projection (existing) |
| Qwen2.5-0.5B | (896, 1152) | QKV out (14 heads * 64 + 2*2 heads * 64) |
| Qwen2.5-0.5B | (896, 4864) | gate/up intermediate projection |
| Qwen2.5-0.5B | (896, 896) | O/down hidden projection |
| SmolLM-135M | (576, 49152) | LM head / vocab projection |
| SmolLM-135M | (576, 960) | QKV out (9 heads * 64 + 2*3 heads * 64) |
| SmolLM-135M | (576, 1536) | gate/up intermediate projection |
| SmolLM-135M | (576, 576) | O/down hidden projection |

Verify exact widths against config records before finalizing: `SmolLMConfig` (`src/Nivara.Extensions/Silk/SilkSmolLMGpuRunner.cs:1011`) and Qwen2.5-0.5B config (`docs/QWEN.md`, NivaraInference Qwen config).

Output format: keep median ratio/ms, add logical bytes, tile GB/s, flat GB/s. Keep interleaved warmup/round discipline.

### 2. Add device copy-bandwidth probe

**File:** `tests/Nivara.PerformanceTests/SilkKernelsGate.cs`

Add a small device-to-device copy-bandwidth probe (same gate run, or clearly labelled section) using `SilkRuntime.CopyBuffer` + `Finish()` + `Stopwatch`. 

- Allocate two device buffers (large enough to be bandwidth-bound, e.g. 64 MiB / 16M floats; report exact size).
- Warm up, then interleaved/repeated timed copies.
- Report median GB/s = `bytes_copied / elapsed_seconds / 1e9`, min/max or rounds for context.
- Label as proxy: `clEnqueueCopyBuffer` device-to-device bandwidth, not kernel-achieved DRAM bandwidth; useful as an upper-bound reference point.

### 3. Document methodology in docs/SILK.md

**File:** `docs/SILK.md`

Add a "Decode GEMV GB/s measurement" subsection (profiling/measurement guidance). Include:

- Command: `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels`
- Shape table (SmolLM/Qwen decode GEMV shapes) and what each shape represents.
- Bytes-moved formula and explicit caveats (logical model, x read twice in rms_norm_gemv, gamma once in compute, cache effects, bias optional, BF16 halves weight bytes).
- Copy-bandwidth probe meaning and limitations (D2D CopyBuffer proxy, not DRAM peak/kernel bandwidth).
- Comparability rules: same device, Release build, note power state (`GemmBenchmark.PrintPowerState()` already prints it), use median of N rounds, keep formula stable.

## Blast radius

- `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` - adds GB/s reporting/probe; no existing Leg correctness gate behavior changes. DecodeShapeAb output format changes (additive columns/lines).
- `docs/SILK.md` - documentation only.
- No `src/` runtime behavior changes expected; `SilkRuntime.CopyBuffer` already exists.
- No public API changes; no package/version impact.
- Docs pins: adding `docs/SILK.md` text only does not change document count; citations only change if we add `File.cs::Symbol` references - avoid adding new citations unless needed. Check `DocsCoveragePins` before finalizing.

## Verification

- `dotnet build tests/Nivara.PerformanceTests/Nivara.PerformanceTests.csproj -c Release`
- Targeted tests for changed area as appropriate (PerformanceTests is a harness; may run specific mode only with human confirmation where long-running/hardware-dependent). At minimum compile cleanly.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- If docs changed: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (ask before running - long-running relative? AGENTS says 66 tests ~25s, ask before running full suite; targeted docs namespace is borderline but I'll ask/confirm per user instruction flow).
- Manual/human-run perf gate for actual GB/s numbers: `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels` (requires Silk/OpenCL device). Do not run as agent without confirmation; capture output for PR test plan.

## Planned commits

1. `docs: plan Silk decode GEMV GB/s measurement guidance (#1473)`
2. `perf: report GB/s in Silk decode-shape GEMV A/B gate`
3. `perf: add Silk device copy-bandwidth probe for decode GEMV comparison`
4. `docs: document decode GEMV GB/s measurement methodology`
5. `docs: remove TODO.md — plan executed` (only after G2 reviews clear)

## GitHub issues log

- [ ] #1473 — Silk: add achieved GB/s measurement guidance for decode GEMV (this plan)

## Reminder

As each task executes, if you find deferred work or a concern outside this plan, create a tracked issue immediately via `gh issue create --repo khurram-uworx/Nivara` and record its number in the GitHub issues log. Don't rely on memory or wait until the plan finishes.
