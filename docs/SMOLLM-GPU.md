# TODO: GPU acceleration for SmolLM-135M — probe-first investigation (`khurram/gpu-probe`)

Status: **Completed — probe answers recorded. No production code. Push/PR pending.**

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

**Answer (probe findings):** driver `0x010393E2`, L0 API 1.15, SPIR-V max 1.0 —
indexed access remains **broken**. All 8 access-chain variants (OpAccessChain /
InBounds / PtrAccessChain, u32/i32/u64 index, param/global base, Restrict/NoAlias)
trigger `IGC: Internal Compiler Error: Access violation` at compile time. Same bug
class as IGCIT #1144 (Blender AV on Arc B580). A minimal GEMV-shaped kernel
(`c[i] = a[i] + b[i]`) crashes identically. **The safe working subset** (direct
loads/stores, OpPhi loops, 1-lane atomics, BF16 conversion ops) compiles and runs.
IGC bug #4 confirmed: LocalSize ≥ 8 atomics drop the upper SIMD half; localSize=1
(simd1) passes.

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

**Answer (probe findings):** `ZE_extension_bfloat16_conversions` is present. `BFloat16`
round-trip is **fully proven on this iGPU**:
- **Native path**: `OpConvertBF16ToFINTEL` (capability 6115) — IGC honors the
  contract; BF16 values pass through losslessly to `f32` registers; the widened
  `f32`'s upper 16 bits map back to the original `BFloat16` exactly.
- **Safe-subset emulation**: `OpUConvert` + `<<16` shift — same result, no extension
  required (future-proof if the extension disappears).
- **1M-iteration accumulation**: exact (1,000,000.0) — the BF16→F32→f32-accumulate
  pipeline works end-to-end on device.
- **Host edge is pure `BFloat16`**: no `ushort`, no `ToSingle`, no host bit
  arithmetic; the raw 16-bit `BFloat16` layout IS the on-device storage pattern.
  Zero widening on the input path; the `f32` accumulator is the hardware's native
  BF16 compute model (Xe2 DPAS also accumulates BF16 products in `f32` — `f32`
  accumulation is required for precision, not an added widening).
The BF16 compute path is ready. The blocker is the access-chain ICE (Q1).

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

**Answer (probe findings):**
- **(a) Driver update + re-probe** — still recommended. The IGC access-chain ICE
  is the same class as IGCIT #1144 (Arc B580), fixed upstream. Re-run `run`
  after any driver bump (seconds of work).
- **(b) BF16 on GPU** — already proven (see Q2); the blocker is indexed access,
  not BF16 itself. Once Q1 clears, the native `OpConvertBF16ToFINTEL` path is
  ready as-is.
- **(c) F32-only GPU path** — still viable as fallback (2× weight memory,
  513 MB for SmolLM); BF16 stays CPU-only until the access-chain fix lands.
- **(d) DX12 / ComputeSharp** — **the Arc 140T reaches D3D12 FL 12_2 + shader
  model 6.8** (`D3d12Check.cs`). A managed DX12 compute backend targets a
  separate shader compilation path (HLSL → DXIL / usc) that bypasses the IGC
  OpenCL frontend entirely. This is the **immediate viable fallback** for real
  GEMM/attention kernels on the iGPU while the Level Zero access-chain fix is
  pending. Cost: a NuGet dependency in a new project; core stays dependency-free.
  **Recommendation: pursue (d) DX12/ComputeSharp in parallel with (a) driver
  re-probe on next driver update.**

## Scope / non-goals for this branch

- **No production GPU code**: this branch extends `tests/Nivara.GpuProbe` and
  updates docs only. The production plan (`-gpu` flag, `src/Nivara.Gpu`,
  `LlamaFusedKernels` GPU branches, SmolLM KV cache in NivaraInference) is
  recorded below for the follow-up branch, gated on the probe answers.
- No changes to `src/Nivara`, `samples/NivaraInference`, or `NivaraChat`.

## Probe extensions (planned commits)

1. ✓ `8c0c46d` — `docs: plan SmolLM GPU probe-first investigation` (this file)
2. ✓ `a70fc79` — baseline re-run recorded (driver `0x010393E2`, L0 API 1.15,
   SPIR-V max 1.0); `ZE_extension_bfloat16_conversions` + BF16-relevant module
   caps summarized in `L0Probe` output; the minimal GEMV-shaped access check is
   covered by the bisection's `straight` variant (`c[0] = a[0] + b[0]`) and
   still triggers the access-chain ICE
3. ✓ `a70fc79` + `027c80b` — `bf16_native` / `bf16_emul` / `bf16_native_acc`
   kernels built, launched, host-verified exact; host edge refined to a pure
   `BFloat16` round trip (no `ushort`/`ToSingle`)
4. ✓ `0393130` + this file — `dx12` availability probe committed; Q1/Q2/Q3
   answer blocks + BF16/DX12 recommendation recorded in `docs/SMOLLM-GPU.md`
   (`docs/BFLOAT16-GPU.md` not needed separately — the answers live here)

## Verification steps

- `dotnet run -c Release --project tests/Nivara.GpuProbe -- run` passes its
  real gates (add_parallel, add_loop, bf16_native, bf16_emul, bf16_native_acc)
  — exit 0; the expected IGC-bug diagnostics (access-chain ICE, 15) are
  reported separately and do not fail the run.
- New BF16 kernels print host-verified results (exact f32 where the subset
  allows) and are clearly reported as PASS.
- `dx12` mode passes on this machine: Arc 140T FL 12_2 / shader model 6.8.
- All probe modes (`list`/`run`/`spv`/`ocl`/`dx12`) keep working.

## Blast radius

- `tests/Nivara.GpuProbe/LevelZero/SpvKernels.cs`, `L0Run.cs`, `L0Probe.cs`,
  `README.md`, `tests/Nivara.GpuProbe/Program.cs`, `tests/Nivara.GpuProbe/D3d12Check.cs`
  — probe-only additions.
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