# TODO: GPU acceleration for SmolLM-135M — probe-first investigation (`khurram/gpu-probe`)

Status: **In progress — plan committed, investigation running. No production code.**

## Problem

SmolLM-135M-instruct inference in `samples/NivaraInference` is CPU-only and its
`smollm` mode still decodes **cache-free** (`model.Forward(sequence)` per token).
We want GPU branches in the fused inference kernels to accelerate decode and
prefill. Before choosing a backend (Level Zero SPIR-V vs DX12/ComputeSharp) and
a precision strategy (F32 / native BF16 / widened BF16), we must **verify on the
actual Arc 140T iGPU and current driver** what the Level Zero stack can really do.

`tests/Nivara.GpuProbe` already proves real iGPU compute and documents a wall:
the current Intel driver's IGC (OpenCL/Level Zero frontend) crashes on
`OpAccessChain` (IGCIT #1144 bug class) and mis-links private variables, which
today rules out indexed tensor kernels (GEMV/GEMM/attention). Whether that still
holds, whether BF16 is natively usable, and what the fallback options are — those
are the questions this branch answers with the probe itself.

## Investigation questions (the probe-first deliverables)

### Q1 — Current driver state: is indexed access still broken?
- Re-run the existing `run` bisection + `l0` enumeration as committed.
- Record: driverVersion, L0 API version, extension list, SPIR-V max version,
  fp16/fp32/fp64/DP4A module caps.
- Attempt a minimal GEMV-shaped kernel (weight row × input row → output) to
  (re)confirm the access-chain crash boundary — reported as expected driver
  diagnostic if it fires.
- **Expected outcome**: on this driver build, indexed access remains broken via
  Level Zero (OpenCL frontend); only the safe subset works.

### Q2 — Does BF16 work natively on this GPU (Level Zero)?
- Report the `ZE_extension_bfloat16_conversions` extension version + presence
  (via the existing extension enumeration) and any BF16-related module caps.
- Probe a **native-conversion module**: SPIR-V declaring the
  `Bfloat16ConversionINTEL` capability (and 16-bit float storage where legal)
  that converts a scalar BF16 → F32 — does IGC build it?
- Probe an **emulated-conversion kernel** (safe subset): BF16 held as `ushort`,
  converted with shift/&/exponent math (same bit-arithmetic as the CPU
  `WidenBf16ToF32` kernel), accumulated in F32 through an OpPhi loop, atomic
  aggregation via 1-lane workgroups, host-verified exact.
- **Expected outcome**: native conversion may or may not build (capability
  gate); emulated conversion proves BF16→F32 math is GPU-correct but needs
  indexed access (Q1 result) to become a real dot-product path.

### Q3 — If BF16 is not natively usable, what are the options?
Produce a documented recommendation (in this file / `docs/BFLOAT16-GPU.md`):
- **(a) Driver update + re-probe** — IGC fix is upstream (IGCIT #1144 class);
  `run` bisects the whole surface in seconds, so re-validation is cheap.
- **(b) BF16-emulated on GPU** — store BF16 weights on device, widen in-register
  per dot (the CPU `UseWidenSimd` pattern); viable only once indexed access
  works (Q1) since it needs real GEMV kernels.
- **(c) F32 GPU path only** — weights widened to F32 at upload (2× memory, 513 MB
  for SmolLM); BF16 stays CPU-only. Simplest, works where kernels can run at all.
- **(d) DX12 / ComputeSharp** — Intel's **DX12 driver is a separate path** from
  the IGC OpenCL frontend; the probe's driver bugs don't apply there, and a
  managed backend would run real GEMM/attention kernels today. Cost: a NuGet
  dependency in a new project (core stays dependency-free). This is the likely
  landing spot if Q1 comes back "still broken" and a driver bump is not
  available/practical.

## Scope / non-goals for this branch

- **No production GPU code**: this branch extends `tests/Nivara.GpuProbe` and
  updates docs only. The production plan (`-gpu` flag, `src/Nivara.Gpu`,
  `LlamaFusedKernels` GPU branches, SmolLM KV cache in NivaraInference) is
  recorded below for the follow-up branch, gated on the probe answers.
- No changes to `src/Nivara`, `samples/NivaraInference`, or `NivaraChat`.

## Probe extensions (planned commits)

1. `docs: plan SmolLM GPU probe-first investigation` (this file; branch
   `khurram/gpu-probe`)
2. `probe: baseline re-run + driver/BF16 capability report` — run the probe as
   committed; add explicit `bfloat16_conversions` extension + BF16-relevant cap
   lines to `L0Probe` output; add a GEMV-shaped kernel to `SpvKernels` +
   `L0Run` (access-chain re-validation; expected diagnostic)
3. `probe: BF16 conversion kernels` — native-capability module (build-only) +
   safe-subset emulated `bf16_to_f32` accumulation kernel with host-verified
   results; wire into `run`
4. `docs: record probe answers + BF16 options recommendation` — Q1/Q2/Q3 writeup
   in `docs/SMOLLM-GPU.md` + `docs/BFLOAT16-GPU.md` (if warranted)

## Verification steps

- `dotnet run -c Release --project tests/Nivara.GpuProbe -- run` passes its
  real gates (add_parallel, add_loop) and reports the expected diagnostics
  without crashing the process.
- New BF16 kernels print host-verified results (exact float where the subset
  allows) and are clearly reported as PASS / expected-diagnostic.
- All existing probe modes (`list`/`run`/`spv`/`ocl`) keep working.

## Blast radius

- `tests/Nivara.GpuProbe/LevelZero/SpvKernels.cs`, `L0Run.cs`, `L0Probe.cs`,
  `README.md`, `tests/Nivara.GpuProbe/Program.cs` — probe-only additions.
- `docs/SMOLLM-GPU.md`, `docs/BFLOAT16-GPU.md` — documentation.
- No core library, no samples, no Nivara.Tests, no package changes.

## Follow-up (next branch, gated on probe answers)

- SmolLM KV-cached decode in `NivaraInference` (mirror Qwen), then `-gpu`:
  new `src/Nivara.Gpu` (DX12/ComputeSharp or updated-L0 backend per Q1–Q3),
  GPU branches at `LlamaFusedKernels` / `LlamaDecoderBlock` hooks, `GpuMatMul`,
  `GpuDecodeAttention`, `GpuRmsNorm`, `GpuSoftmax`, `GpuRoPE`, `GpuSilu`,
  `smollm -gpu [benchmark]`, parity + PyTorch-fixture validation.

## GitHub issues log

- (empty — no issues created yet)