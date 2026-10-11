# Plan: Silk.NET OpenCL MiniLM GPU path (kernels first, then sample)

Branch: `khurram/silk-minilm` (off `main`).

## Problem

ILGPU is the incumbent GPU backend for the BERT-family encoders (`samples/Nivara.Samples/Gpu/`),
measured and gated (`docs/ACCELERATION.md`). The `tests/Nivara.GpuProbe` Silk.NET OpenCL leg
(`OpenCl/SilkLeg.cs`, `SILK.md`) already proves a raw-OpenCL path is the fastest hand-authored
leg and the only one that reaches native FP16 (`cl_khr_fp16`) — but it runs only *naive*
reference kernels (dot16/silu/gemv), which `ACCELERATION.md` explicitly says are "not a
backend-selection argument". Nothing has ever measured whether a hand-authored OpenCL C
**production** kernel stack (Row4-equivalent tiled GEMM, fused attention, elementwise) matches
ILGPU's, nor whether a full encoder forward runs correctly through Silk.

Decision (human-approved, 2026-10-11): evaluate Silk vs ILGPU by porting **MiniLM first**,
kernels-first — author and gate the six MiniLM kernels before touching the sample runner.
CLI: rename `--gpu` → `--ilgpu` now; add `--silk` for the new path.

Deliberate scope decisions from the human:
- **Correctness and "good enough" are the acceptance bar** — an informal same-session timing
  band of ±10–20% vs ILGPU is sufficient; AC power is not required for this comparison.
- **Do NOT document performance figures** anywhere (no ACCELERATION/README benchmark rows,
  no measured claims). Documentation updates are flag/usage-only.
- Keep `--gpu` semantics intact, renamed to `--ilgpu`; no other behavior change to ILGPU path.
- `--silk` initially supports `minilm` only (default / `benchmark` / `compare`). Other models
  reject it with the requirement named, same validation discipline as today.
- `--silk` is F32-only; `--silk --precision bf16|fp16` is rejected (probe FP16 leg is
  diagnostic only, never a production path, per `SILK.md`).
- `--ilgpu --silk` together is rejected (one backend per run).
- No `src/Nivara`, `Nivara.Extensions`, or `Nivara.Tests` changes. Sample + probe-harness only.

## Changes

### 1. CLI rename `--gpu` → `--ilgpu` (mechanical; no behavior change)

- `samples/NivaraInference/Program.cs`: parse `--ilgpu`; update help text, `GpuModes`
  validation errors, the F32-only reject message, `--gpu requires...` strings.
- `samples/Nivara.Samples/Gpu/IlgpuRuntime.cs` + `GpuBuffers.Req` error strings mention `--gpu`
  → `--ilgpu`.
- Docs sweep (usage/runbook invocations): `samples/NivaraInference/README.md`,
  `docs/ACCELERATION.md`, `docs/LAYA.md`, `docs/MODERNBERT.md`, `docs/ROADMAP-SUGGESTION.md`
  (and `docs/DISTELBERT.md` if it carries invocations). Historical verbatim run records that
  used `--gpu` are updated to `--ilgpu` where the flag is named as the current CLI; prose that
  refers to the flag generically is left as-is or reworded minimally.

### 2. Silk kernels for MiniLM — authored once, gated + measured before any runner

New `samples/Nivara.Samples/Gpu/SilkKernels.cs`: the OpenCL C source strings + thin host
dispatch helpers, backend `Silk.NET.OpenCL 2.23.0` (same pin as the probe; added to
`Nivara.Samples.csproj`). Only the six kernels MiniLM needs, each ported from its ILGPU
reference with the same accumulation order:

| kernel (OpenCL C) | ILGPU reference | gate |
|---|---|---|
| `embedSum` | `ElementwiseKernels.EmbeddingSum` | numeric vs ILGPU output |
| `layerNormResidual` | `ElementwiseKernels.LayerNormResidual1D` | numeric vs ILGPU output |
| `qkvGemm` (with `bias` epilogue) | `TiledGemmKernelRow4Qkv` | **byte-identity** vs Row4 family |
| `attention` | `AttentionKernels.BatchedAttention` (band = −1) | numeric vs ILGPU output, 1e-3 bound |
| `gemmBias` | `TiledGemmKernelRow4Bias` | **byte-identity** |
| `gemmGelu` | `TiledGemmKernelRow4Gelu` (`GeluExact` A–S 7.1.26) | **byte-identity** |

GEMM-port rules (from `GemmKernels.cs`): 16×16 group, single shared tile (A cols [0,16),
B cols [16, 16+64), per-thread 4 accumulators, k0-step 16, strictly ascending-K, `mul`+`add`
with `#pragma OPENCL FP_CONTRACT OFF` (probe's own setting) so the summation order matches
Row4 and byte-identity is achievable; if the byte gate flags a mismatch, flip the inner
product to `fma(...)` — one of the two matches ILGPU's lowered schedule, and the gate decides.
Write guards mirror Row4's edge handling (partial tiles zero; per-column `col < bCols`).

Runner-level port rules: Silent-arg-hazard statuses checked where produced (never OR'd),
`CL_PROGRAM_BUILD_LOG` decoded on build failure, device selection = Intel/AMD OpenCL GPU
first, `CL_DEVICE_TYPE_GPU` asserted — no CPU fallback (probe contract, `SILK.md` §8).

### 3. Kernel gates + informal A/B — in `tests/Nivara.PerformanceTests`

New `--silk-kernels` mode (new file, e.g. `SilkKernelsGate.cs`; `Silk.NET.OpenCL 2.23.0`
added to that csproj), following the existing `GemmBenchmark`/`GemmFingerprint` pattern:

- **GEMM byte-identity**: run the ILGPU Row4 family in-process (already accessible via the
  `Nivara.Samples` reference + IVT), run the Silk GEMMs on the same MiniLM shapes
  (128×384×1152 qkv / 128×384×384 o_proj,fc2 / 128×384×1536 fc1), assert identical f32 bits.
- **Numeric gates**: attention / `layerNormResidual` / `embedSum` vs the ILGPU kernel outputs
  on the same inputs, at the model gate's `≤ 1e-3·(1+|ref|)` bound (exp is the known ulp
  diverger: ILGPU `XMath.Exp` vs OpenCL `exp`).
- **Informal timing band**: per-kernel best-of-N (probe discipline) for Silk vs ILGPU on the
  same shapes, printed but *not* committed to any doc. Exit code = unexpected kernel failures
  only. If anything lands outside the ±10–20% sanity band, stop and report to the human —
  that is the plan's explicit red flag.

### 4. MiniLM sample runner — `--silk` path

- New `samples/Nivara.Samples/Gpu/SilkRuntime.cs` — mirror of `IlgpuRuntime`'s contract
  (device select/assert, in-order context+queue, in-process `clBuildProgram`, explicit-byte
  upload/download helpers, `Ensure`-only buffer growth). Raw `clEnqueueWriteBuffer` takes
  explicit payload bytes, so the ILGPU `CopyFromCPU` whole-view gotcha (`ACCELERATION.md`
  lesson 1) does not exist here.
- New `samples/Nivara.Samples/Gpu/SilkBertEncoderRunner.cs` — MiniLM path only; same public
  surface as `BertEncoderGpuRunner.Forward(...) → BertEncoderGpuResult`, sharing
  `BertGpuNaming`/`BertGpuKeys`/`BertConfig`/`BertEncoderGpuResult` unchanged. Persistent
  cap-sized workspace (batch ≤ 8, seqLen ≤ 128), weights transposed once at upload, one
  device sync before readbacks.
- `Program.cs`: `--silk` dispatch for `minilm` — default / `benchmark` / `compare`, reusing
  the existing compare body (GPU hidden vs in-process CPU hidden, pooled-embedding cosine,
  PyTorch fixture when present) and the 3-warmup/10-pass benchmark protocol.
- Gate: `minilm --silk compare` exit 0 (hidden `maxRel ≤ 1e-3`, cosine ≥ 0.9999). Requires
  the `samples/data/minilm` weights — check presence at that commit; if absent, report to the
  human (may need `hf download`).

### 5. Docs (flag/usage only — no perf figures)

`samples/NivaraInference/README.md`: renamed `--ilgpu` invocations + `--silk` rows for
`minilm`. `docs/ACCELERATION.md`: Silk status line + flag usage; explicitly no measured rows.

## Verification

- `dotnet build Nivara.slnx -c Release` after each commit.
- `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels` — exit 0
  (byte-identity + numeric gates pass); informal A/B within ±10–20% of ILGPU per kernel.
- `dotnet run -c Release --project samples/NivaraInference -- minilm --silk compare` — exit 0.
- Targeted test run before the final commit review (ask the human first): GPU-adjacent
  fixtures, e.g. `--filter "FullyQualifiedName~Gpu"` in `tests/Nivara.Tests` (Release).
- No full-suite run without asking.

## Planned commits

1. `docs: plan silk-minilm in TODO.md`
2. `sample: rename --gpu to --ilgpu in NivaraInference and docs`
3. `sample: add SilkKernels (OpenCL C) for MiniLM to Nivara.Samples`
4. `tests: gate Silk kernels vs ILGPU in Nivara.PerformanceTests (--silk-kernels)`
5. `sample: add SilkRuntime + SilkBertEncoderRunner and wire --silk for minilm`
6. `docs: update NivaraInference README and ACCELERATION for --ilgpu/--silk`

Push and PR are human-confirmed at the end (G2 review first).

## GitHub issues log

- [ ] (none yet — fill as deferred work is discovered)

## Blast radius

- `samples/Nivara.Samples/Gpu/` — new `SilkKernels.cs`, later `SilkRuntime.cs` +
  `SilkBertEncoderRunner.cs`; `Nivara.Samples.csproj` gains `Silk.NET.OpenCL 2.23.0`. Existing
  ILGPU runner files are untouched except `--gpu`/`--ilgpu` strings.
- `samples/NivaraInference/Program.cs` — CLI parse/validation/dispatch for `minilm`.
- `tests/Nivara.PerformanceTests/` — new mode + Silk.NET dep. No changes to existing gates.
- Docs: `samples/NivaraInference/README.md`, `docs/ACCELERATION.md`, `docs/LAYA.md`,
  `docs/MODERNBERT.md`, `docs/ROADMAP-SUGGESTION.md` (flag sweep).
- No `src/` changes; no public API changes; nothing pushed until the human confirms.
- Downstream callers of the sample CLI: docs and the release-benchmark runbook
  (`RELEASING.md` may reference `--gpu` — swept in commit 2 if so).
- Tests covering the touched paths: `tests/Nivara.Tests/Gpu/` (kernel-shape constants,
  `SharedMemoryAllocationTests`), the `--gemm`/`--gemm-legs` gates, `minilm --gpu compare`.