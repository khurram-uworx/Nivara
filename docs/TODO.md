# GPU Kernel Probe Phase: BF16 Dot / GEMV / SiLU — three-way (CPU · SYCL/oneAPI · DX12)

Branch: `khurram/gpu-probe` (off `main`). Probe-first, like the previous phase — no change to
`src/Nivara` code, `samples/` code, or any NUnit test project. The probe *consumes* existing
Nivara/Nivara.Samples kernels read-only via project references.

## Problem

`main` now carries the L0/BF16/DX12 availability probe (PR #425). It proved:

- The Arc 140T iGPU runs hand-authored SPIR-V (OpenCL kernel model, SPIR-V 1.0) BF16 compute:
  native `OpConvertBF16ToFINTEL` widen + safe-subset `<<16` widen both exact; 1M-iteration f32
  accumulate exact; **pure `System.Numerics.BFloat16` end-to-end**.
- **Indexed memory access is impossible on this driver** (any access-chain opcode → IGC access
  violation; 17-variant bisection). So a real GEMV (`y[i] = Σⱼ W[i,j]·x[j]`) is not expressible on
  L0 today; the L0 safe subset is: direct loads/stores through kernel-arg pointers, OpPhi loops,
  1-lane atomics, BF16 ops.
- **NEW (commit 4, decisive): this driver also cannot multiply or divide floats in the OpenCL
  kernel model** — IGC executes `OpFMul`(131) as `OpFSub`(130) and `OpFDiv`(132) as `OpFMul`(131)
  (proven by exact-input single-op probes; `OpFAdd` and `OpenCL.std exp` are correct). Dot = Σ(a−b),
  SiLU = x·(1+exp(−x)), both mirror-confirmed. **Hand-authored SPIR-V on L0 is therefore a dead
  end for any real math**, and proof-of-correctness pivots to the **oneAPI SYCL/DPC++ toolchain**
  (`icpx`-produced SPIR-V — the same path llama.cpp's SYCL backend runs on Arrow Lake Arc iGPUs) per human decision — details below.
- DX12 is available (FL 12_2 / SM 6.8, verified via `D3d12Check.cs`) and targets a DXIL/DXBC
  shader path that bypasses the buggy IGC OpenCL frontend — the compiler-backed fallback.

Before committing to a GPU inference backend for SmolLM we must prove the actual compute kernels
correct on both GPU paths, apples-to-apples against the CPU. This phase implements **exactly two
kernels SmolLM actually needs**, on **three backends**, with **correctness verification first
(against the production Nivara kernels — the CPU leg is the gold target), then performance** —
all inside GpuProbe.

## Kernel choices (decided with the human)

1. **Kernel A — BF16 dot / GEMV** (the workhorse: every SmolLM decode token is dominated by
   linears; 135M params ≈ all in matmuls). Exercises the exact compute model we care about:
   BF16→f32 widen-in-register, f32 accumulate, reduction-order differences across vectors.
   - **SYCL leg (replaces the hand-authored L0 leg after commit 4):** K=16 dot via a SYCL kernel
     written in DPC++ and compiled with `icpx` to SPIR-V, then loaded/launched through the
     probe's proven L0 harness (`zeModuleCreate` + `zeKernelCreate`, which work end-to-end).
     This answers the decisive question left by commit 4: does IGC execute FP mul/div correctly
     when the bytecode is *toolchain-produced*? Native + emul widen parity with the old kernels.
   - **DX12 leg:** full GEMV with real indexing (M×576, SmolLM-shaped), plus the same K=16 dot
     for the strict three-way apple.
   - **CPU leg:** production kernels consumed as-is — widen at load, then the real SmolLM dot /
     GEMV kernels (no hand-written CPU kernels, see *Proposed changes*).
2. **Kernel B — SiLU** (`x·σ(x)`, SwiGLU FFN activation). Deliberately the other side of the
   envelope: **elementwise** (fully expressible on all three legs) and it
   exercises **transcendentals (`exp`)** which neither GPU leg has touched yet (everything so far
   is FAdd/FMul). Validates math-library correctness on both SPIR-V/IGC and DX12.
   - SYCL: `exp` from the device math library (compiler-imported).
   - DX12/CPU: same formula.

Left out (phase 2+): RMSNorm (reduce → indexed → L0-degenerate), RoPE, attention QK^T/softmax/PV,
LM-head GEMV at full 49152×576 (≈54 MiB fixture; sliceable later). Phase 2 gates on this
foundation proving out.

## Host/device data contract (BFloat16 everywhere on the wire)

- **All fixture buffers are `System.Numerics.BFloat16`** (net11 in-box) — .NET type end-to-end,
  no `ushort`/`nint` middle layers, no host-side widening, no host-side bit-juggling.
- **GPU legs send raw BF16 patterns as-is** and **widen in-register on the device** (L0: native
  `OpConvertBF16ToFINTEL` and emul `<<16` variants; DX12 HLSL: `asfloat(uint << 16)`). The GPU
  owns any widening, exactly like production would.
- **Kernel outputs are the f32 computation results** (dot/GEMV f32 accumulator per row, SiLU f32
  value) — read back as f32 and compared against the CPU-leg production kernels. Outputting f32
  is the GPU's natural accumulator precision, not a host-side semantic conversion. CPU leg widens
  once at load (`SafeTensorsLoader.WidenBf16ToF32`, SIMD) then runs existing f32 kernels.

## Synthetic data (self-contained; embedded SmolLM constants)

The probe embeds the verified SmolLM-135M shapes (hidden 576, heads 9, kv 3, intermediate 1536,
silu, vocab 49152) instead of depending on the gitignored model download; reading
`samples/data/smollm-135m/config.json` is an optional override only. Fixtures are generated
in-memory once per run, byte-identical across all three legs, from a small mirror of the Qwen
synthetic generator (`samples/NivaraInference/Qwen.cs`): deterministic xorshift RNG (seed
`0x9E3779B9`), values `(rng/uint.Max - 0.5f) * 0.1f`, narrowed to `BFloat16` via
`BFloat16.CreateChecked`.

- `dot16`: two BF16 vectors len 16 (the strict three-way apple)
- `dot576`: two BF16 vectors len 576 (DX12 vs CPU)
- `gemv`: X ∈ BF16[576], W ∈ BF16[M×576] with **M = 1536** (SmolLM `intermediate_size`, the real
  `gate_proj`/`up_proj` row count)
- `silu`: one BF16 vector len 576

The Qwen-bias wrinkle (issue #426) doesn't affect this phase — the probe needs kernel-shaped
fixtures, not full-model tensors.

## Proposed changes (all inside `tests/Nivara.GpuProbe/` + docs)

New files:

- `Kernels/KernelFixtures.cs` — fixture generator (embedded SmolLM shapes, RNG mirror, BF16
  buffers) + shared host read/write helpers (write `BFloat16` to native buffer, read f32 back).
- `Kernels/CpuLeg.cs` — the **CPU leg = the production Nivara kernels exactly as
  `NivaraInference` calls them for SmolLM**, consumed read-only via public API (gold target
  every GPU leg is gated against): dot via `LlamaFusedKernels.MatMulTransposedB<float>` with
  `bCols=1`, GEMV via `MatMulTransposedB<float>` (aRows=1 → the allocation-free BLAS2 GEMV path
  the fused Llama head runs), SiLU via `Activation.Silu` (`GradKernels.Silu` — **sigmoid-then-
  multiply**, NOT the `x/(1+exp(−x))` formula this plan text previously described). Carries the
  gate helpers (`WithinTolerance`, `UlpDistance`, `GateAbs`/`GateRel`) — no double-precision
  oracle exists or is maintained (the double golden was deleted once the production kernels were
  wired as CPU).
- `Kernels/KernelGate.cs` — the N-way compare harness: runs each leg over the same fixture,
  asserts `|leg − cpuNivara| ≤ 1e-6 + 1e-5·|cpuNivara|`, reports per-leg worst ULP as a
  diagnostic (against the CPU-leg production-kernel output),
  then timings (warmup once + median-of-N, mirroring the `add_loop` timing pattern in
  `L0Run.cs`). Dot16 is launch-bound (µs-scale) — its timing row is an availability signal; gemv
  and silu are the meaningful perf rows. Legs: CPU, SYCL (oneAPI), DX12.
- `D3d12/D3d12Compute.cs` (new, largest unit) — hand-rolled D3D12 compute path, pure P/Invoke
  against inbox `d3d12.dll`/`dxgi.dll`/`d3dcompiler_47.dll` (no NuGet, same ethos as `L0Loader`):
  CreateDevice (FL 12_2, skip WARP) → queue → command allocator/list → **D3DCompile HLSL→DXBC
  `cs_5_1`** (SM 5.1 DXBC is loadable by D3D12; d3dcompiler_47 is inbox on Win10+) → root
  signature → descriptor heaps → `CreateCommittedResource` default+upload heaps →
  `CopyBufferRegion` upload→default → resource barrier to `D3D12_RESOURCE_STATE_UNORDERED_ACCESS`
  → `CreateComputePipelineState` → draw-less dispatch → fence (`Signal`/wait) → readback via
  `CopyBufferRegion` into a readback heap. Skips WAVE_MMA / SM 6.8 for now (needs DXIL+HLSL 2021;
  recorded as follow-up) — SM5.1 DXBC gives us the *full-GEMV indexed* kernel, which is the point.
- `D3d12/GemvKernels.cs` (new) — HLSL source strings (`cs_5_1`) for `dot16`, `gemv` (thread-per
  output row, serial K loop, `asfloat(uint<<16)` widen — BF16 stored raw as `uint`/`uint16_t` in
  a `StructuredBuffer<uint>`, matching the .NET `BFloat16` layout), `silu` (`exp`). No-FMA
  guarantees are no longer load-bearing (gate is tolerance-based, see below), but the K loops stay
  simple serial order for run-to-run determinism.
- `Sycl/` (new after commit 5 bootstrap) — SYCL/DPC++ kernel sources (`dot16`, `gemv`, `silu` in
  C++ with `sycl::queue` + USM or buffer/accessor), compiled with `icpx -fsycl` to SPIR-V; loaded
  either through the existing L0 harness (`zeModuleCreate` on compiler-produced SPIR-V) or via a
  small SYCL-runtime shim executable/dll P/Invoked from the probe. Decision recorded in the
  commit-5 README notes.

Modified files:

- `Nivara.GpuProbe.csproj` — already on disk (uncommitted): `ProjectReference` to
  `src/Nivara` and `samples/Nivara.Samples` (read-only consumption; `samples/Nivara.Samples` is
  net11.0, exposes the public `SafeTensorsLoader.WidenBf16ToF32` SIMD widen) + explicit
  `System.Numerics.Tensors 11.0.0-rc.1.26425.128` package ref (matches the solution-wide refresh;
  for `TensorPrimitives` on the CPU leg).
- `LevelZero/SpvKernels.cs` — add:
  - `SpvWriter.ExtInstImport(uint resultId, string name)` — `OpExtInstImport` (opcode 11, word
    count = string words + 2), required for the `OpenCL.std` extended instruction set; emitted
    before `OpMemoryModel` per the SPIR-V 2.4 logical layout.
  - `Bf16DotK16(bool nativeWiden, uint spirvVersion = Version10)` — SPIR-V 1.0, **33 kernel-arg
    pointers**: 32 × `ptr<CrossWorkgroup,u16>` (a[0..15] = ids 11..26, b[0..15] = ids 27..42)
    + f32 out at arg-index 32 (id 43); straight-line unrolled dot: per i, `OpLoad` a/b ×2, widen
    (native `OpConvertBF16ToFINTEL` with capability 6115 / emul `OpUConvert`(113) →
    `OpShiftLeftLogical`(196) → `OpBitcast`(124)), `OpFMul`(131), then a **fixed serial FAdd
    chain** (129, first product as acc init) → `OpStore` out. Kernel names
    `bf16_dot_k16_native`/`bf16_dot_k16_emul`. Bound = running id counter.
  - `SiLUBf16(uint spirvVersion = Version10)` — emul widen (`OpUConvert` + `<<16`, no native
    param), `w.ExtInstImport(10, "OpenCL.std")` before `OpMemoryModel`, `OpExtInst`(12)
    instruction 19 (`exp`), then `x/(1+exp(−x))` via `OpFNegate`(127) + `OpFAdd`(129 with const
    1.0f) + `OpFDiv`(132); 2 args (u16 x ptr + f32 out). Kernel name `bf16_silu`.
- `LevelZero/L0Run.cs` — refactor the shared `TestKernel` into `TestKernelCore` (generic
  `int[] bufferSizes` per arg + `argCount` + `Action<IntPtr[]>`/`Func<IntPtr[],bool>`) with a
  byte-identical-signature `TestKernel` facade; add `ZeCommandListReset` delegate; add a
  **dedicated `RunBf16KernelPhase` after test 5** (inside the try, before the finally) that
  gated-launches `bf16_dot_k16_native`/`_emul` (33 shared buffers: 32×2 B u16 + 4 B f32 out,
  fill via `WriteBf16`, verify reads `bufs[32]`) and `bf16_silu` (persistent module/list +
  `zeCommandListReset` per element, 576 single-element launches) against
  `CpuLeg.Dot` / `CpuLeg.Silu` outputs. Existing tests 1–5 + bisection stay untouched.
- `Program.cs` — add `kernels` mode (CPU + L0 + DX12 gates; skips DX12 gracefully if device
  creation fails), included in `all`.
- `README.md` — **updated in every step's commit** (the probe is a living, run-manually tool —
  its documented state never lags its code): references note, `kernels` build/run line, gate-table
  rows, "verified findings"/limitations updates as they land.
- `docs/SMOLLM-GPU.md` — new "Kernel phase" section with the three-way results (deliverable).
- `docs/TODO.md` — this plan (revised).

CPU leg (inside `CpuLeg`/`KernelGate`, no new kernel code — **nothing reinvented**):
the production Nivara kernels exactly as `NivaraInference` runs them for SmolLM:

- Widen both fixtures once at load via `SafeTensorsLoader.WidenBf16ToF32` (Nivara.Samples, SIMD).
- dot16/dot576 → `LlamaFusedKernels.MatMulTransposedB<float>` with `aRows=1, aCols=K, bCols=1`
  (pure dot through the production GEMV kernel).
- gemv → `LlamaFusedKernels.MatMulTransposedB<float>` (`aRows=1, aCols=576, bCols=1536`, the
  allocation-free BLAS2 GEMV path — same call shape as the fused Llama head).
- silu → production `Activation.Silu(ReverseGradTensor<float>.FromArray(widened))` →
  `GradKernels.Silu`: **sigmoid-then-multiply** (`sigmoid(x) * x`), *not* the `x/(1+exp(−x))` /
  `TensorPrimitives.Exp` formula this section previously described. Values are read back via
  `.Data` + `TryGetSpan` (copied before dispose); no graph nodes are created outside `Grad()`
  scope (inference default).

The probe previously planned hand-written serial CPU kernels (`CpuKernels.cs`) for bit-exact
comparison, then a double-precision oracle (`GoldenReferences.cs`). Both are superseded: the CPU
leg is the *production kernels themselves*, and exactness is judged against them with the
tolerance gate (below), so neither hand-written kernels nor a double golden exist.

## Correctness verification (gates)

- **Oracle:** the **CPU leg (`CpuLeg`) — the production Nivara kernels as `NivaraInference`
  calls them for SmolLM** (dot/GEMV via `LlamaFusedKernels.MatMulTransposedB<float>` in the exact
  fused-head call shape; SiLU via `Activation.Silu` → `GradKernels.Silu`, sigmoid-then-multiply).
  This is the gold target — it depends on Nivara's real reduction order and SIMD kernel, which is
  exactly what a back-end must reproduce within tolerance. No double-precision golden is
  maintained (`GoldenReferences.cs` deleted).
- **Gate (pass/fail) for every kernel, every leg:** `|leg − cpuNivara| ≤ 1e-6 + 1e-5·|cpuNivara|`,
  where `cpuNivara` is the CPU-leg output (f32, promoted to double for the comparison). This
  covers honest f32 accumulation noise (worst-case ~2e-7 absolute for a 576-term dot of these
  magnitudes) with 5–10× margin, while any real bug (wrong index, wrong widen, off-by-one byte
  layout) lands orders of magnitude above it.
- **Diagnostic (reported, not fatal):** per-leg worst `|leg − cpuNivara|` in f32 ULP (computed
  at the reference value's magnitude).
- Existing "bit-exact CPU vs L0 vs DX12 in a single canonical order" and "no-FMA on DX12"
  requirements are **superseded** — the CPU leg is the production SIMD kernels with their own
  reduction order, so exact equality can't and needn't be promised; the tolerance gate against
  the production CPU kernels is the single fixed target.
- Synthetic values ±0.05 → products ≈2.5e-3, sums ≈1e-2·1: no subnormals, no denormal-flush
  risk on any leg.

Perf methodology: warmup (discarded) + median-of-N launches, same fixtures, GFLOPS/GFADD
reporting per leg — mirrors the existing `add_loop` timing. **Widen is a once-only, lossless,
load-time cost and is excluded from per-launch comparisons** on both CPU (widens at load) and GPU
(widens in-register inside the kernel budget).

## Phase-exit / decision criterion

After all gates pass and perf is measured on the same fixtures:

- **Pivot (human decision, after commit 4):** commit 4 proved this driver's IGC
  miscompiles hand-authored SPIR-V FP mul/div (`OpFMul`→`OpFSub`, `OpFDiv`→`OpFMul`,
  mirror-confirmed), so hand-authored bytecode is a dead end for real math. The
  kernel-authoring path for the rest of this phase is the **Intel oneAPI SYCL/DPC++
  toolchain** (`icpx`-produced SPIR-V) — the same stack llama.cpp's SYCL backend uses
  on Arrow Lake Arc iGPUs. The probe learns on the toolchain as it goes; promotion
  decisions (e.g. a `src/Nivara.Gpu`) come *after* this phase, informed by what the
  probe proves.
- **If** a SYCL-backed kernel (dot/GEMV/SiLU, compiled with `icpx`, loaded through
  the probe's proven L0 harness or a small SYCL-runtime shim) passes its gate against
  the production CPU leg **and** its throughput ≥ production-CPU
  `MatMulTransposedB` throughput → record in `docs/SMOLLM-GPU.md` that a
  **SYCL/oneAPI-backed `src/Nivara.Gpu`** is the recommended route (promotion
  decided explicitly later, per the human).
- **If** the SYCL kernels are correct but slower → record the honest verdict
  (correct, not-yet-fast) and the perf gap; DX12 remains the measured comparison leg
  (FL 12_2 / SM 6.8 verified), L0 stays a documented degenerate path (dot-K16/SiLU
  shape only, no indexed memory, FP mul/div miscompiled) either way.
- Exiting the phase *always* requires the `run`/`kernels` modes stay green and the
  gates green; the recommendation text is the phase's written deliverable.

## Verification

- `dotnet build tests/Nivara.GpuProbe` (net11.0 probe; also builds the referenced projects)
  before each probe commit.
- `dotnet run -c Release --project tests/Nivara.GpuProbe -- kernels` — the new three-way gates
  (exit 0 = real gates pass; expected IGC diagnostics reported separately as before).
- Existing modes must stay green: `run` (tests 1–5 + bisection + kernel-phase verdict) and `dx12`.
- Ask before any `dotnet test` / long verification run.
- The oneAPI toolchain is an external install (winget `Intel.OneAPI.BaseToolkit`); its bootstrap
  is part of commit 5 and is a human-informed step.

## Planned commits (one logical change each, local only; README updated in every probe commit)

1. `build: refresh NuGet refs to 11.0.0-rc.1 solution-wide (+ probe references)` — the already
   on-disk solution-wide package refresh (17 csproj: `System.Numerics.Tensors` preview.7→rc.1,
   SourceLink preview.7→rc.1, Microsoft.Extensions.AI 10.9→10.10, Agents.AI 1.19→1.21,
   Microsoft.ML 6.0.0-preview.26160.2→26457.2, Test.Sdk 18.9→18.10, NUnit.Analyzers 4.14→4.15,
   Microsoft.Extensions DI/Hosting/VectorData bumps) **including** the GpuProbe project references
   to `src/Nivara` + `samples/Nivara.Samples` + `System.Numerics.Tensors`. Build-verified.
   README: `References` note (CPU verification reuses existing Nivara kernels; GPU legs stay pure
   P/Invoke).
2. `docs: revise kernel-probe plan (CPU=existing kernels, gates vs production-CPU output, BF16
   host edge)` — this file (current revision).
3. `probe: synthetic SmolLM-shaped BF16 fixtures` — KernelFixtures (+ GoldenReferences, since
   superseded and removed in commit 4); build-verified. README: fixtures note.
4. `probe: L0 BF16 dot K=16 + SiLU kernels — hand-authored-SPIR-V verdict` — CpuLeg
   (production-kernel CPU leg + gate helpers, replaces/deletes GoldenReferences), SpvKernels
   additions (ExtInstImport, Bf16DotK16 native/emul, SiLUBf16, permanent Bf16Binop evidence
   probes), SpvDump, L0Run TestKernelCore split + facade + dedicated RunBf16KernelPhase (gated
   vs CpuLeg) with expected-driver-bug accounting; run `run` CPU-vs-L0 leg. **Verdict:** this
   IGC driver miscompiles OpFMul(131)→OpFSub(130) and OpFDiv(132)→OpFMul(131) in the OpenCL
   kernel model (exact-input probes + Σ(a−b) / x·(1+exp(−x)) mirrors confirm), so hand-authored
   SPIR-V cannot express real math on this driver. README: L0 gate rows + finding #5.
5. `probe: oneAPI/SYCL toolchain sandbox → PROVEN` — human-informed winget install of
   `Intel.OneAPI.BaseToolkit` (2025.1.3.8) + `Microsoft.VisualStudio.2022.BuildTools`
   (MSVC host toolchain is a hard icpx prerequisite). DONE: `Sycl/sycl_runner.cpp`
   (DPC++ `dot16`/`gemv`/`silu`, `sycl::queue` + USM, device-side `<<16` emul-widen),
   `Sycl/build.cmd` (sources VsDevCmd + oneAPI setvars), `Sycl/run.cmd` (sources
   setvars so `sycl8.dll`/`ur_loader.dll` resolve — direct spawn → `STATUS_DLL_NOT_FOUND`),
   `Sycl/SyclLeg.cs` (spawn + gate vs CpuLeg), `sycl` CLI mode. **Decision:** the
   **SYCL-runtime shim** is the commit-5 load path (real UR adapter end-to-end);
   compiler-produced-SPIR-V-via-`zeModuleCreate` is a follow-up option for the
   `kernels` gate mode. **Decisive answer:** *yes* — IGC computes BF16 FP mul/div
   correctly on toolchain-produced bytecode: dot16 0.0 ULP (bit-exact), silu
   576/576, gemv 1536/1536 rows pass the CPU-leg gate (`|diff| = 1.63e-9` on the
   worst row; the 14 336-ULP headline is a near-zero-ref row diagnostic, not a
   failure). README: header verdict + SYCL leg section + gate table + Files.
6. `probe: multi-leg correctness gate harness` — KernelGate with golden-tolerance asserts +
   reporting (CPU + SYCL leg wired first; DX12 joins later). README: `kernels` build/run line +
   gate table. (SYCL's `sycl` mode already gates in-process; this commit generalizes it.)
7. `probe: hand-rolled DX12 compute path` — D3d12Compute.cs (device→PSO→dispatch→fence→readback)
   + HLSL dot16/gemv/silu; build-verified. README: files/results updates.
8. `probe: DX12 kernel gates + perf pass across all legs` — wire DX12 leg into KernelGate;
   `kernels` exit 0; timings for dot16/dot576/gemv/silu across CPU/SYCL/DX12, README results.
9. Cleanup: G2 review → `git rm docs/TODO.md` → `docs: remove TODO.md — plan executed` → offer
   push + PR (human-confirmed).

## Blast radius

- Code: `tests/Nivara.GpuProbe/**` only (probe project, standalone, not in the NUnit suite) plus
  `docs/TODO.md`, `tests/Nivara.GpuProbe/README.md`, `docs/SMOLLM-GPU.md`.
- csproj-only: the solution-wide NuGet refresh in commit 1 touches 17 `.csproj` files across
  `src/`, `samples/`, `tests/` — **package-version bumps only, no source changes**, and required
  to keep the probe's references on one consistent rc.1 line.
- The probe consumes `src/Nivara` and `samples/Nivara.Samples` via project references but makes
  **no changes to them**; `SafeTensorsLoader.WidenBf16ToF32`, `TensorPrimitives`,
  `LlamaFusedKernels.MatMulTransposedB<T>`, `ReverseGradTensor<T>.FromArray` +
  `Activation.Silu<T>` are all public and used read-only.
- **No** changes to `src/Nivara` code, `samples/` code, or the NUnit test projects. The Qwen-bias
  fix is tracked separately as issue #426 and is out of scope.
- Existing probe modes (`run`, `dx12`) must remain green.

## Grounding notes (G1, after this plan is committed)

- D3DCompile / `d3dcompiler_47.dll` inbox availability, SM 5.1 DXBC-in-D3D12 compute validity,
  ID3DBlob vtable layout — via microsoft-learn + Windows docs.
- D3D12 compute pipeline facts: root signatures, `D3D12_RESOURCE_STATES`, descriptor heaps,
  fence/`Signal`/wait, `CopyBufferRegion` — via microsoft-learn before writing D3d12Compute.cs.
- `OpenCL.std` extended-instruction `exp` opcode (expected 19) — verify against the Khronos
  SPIR-V registry (canonical for SPIR-V; not a Microsoft subject).
- **oneAPI/SYCL (commit 5):** `Intel.OneAPI.BaseToolkit` winget package contents + install
  footprint; `icpx -fsycl` device-code-to-SPIR-V flow; whether compiler-produced SPIR-V can be
  loaded by `zeModuleCreate` directly (llama.cpp SYCL backend precedent: oneDNN GEMM + oneMKL,
  Level Zero underneath, targets built-in Arc in Arrow Lake). Via microsoft-learn (Intel® AI
  tools for Microsoft®) + Intel oneAPI docs.
- `LlamaFusedKernels.MatMulTransposedB<T>` semantics (row-major `a`, transposed `b` layout,
  aRows/aCols/bCols contract) — read the implementation before wiring the CPU gemv leg.
- BFloat16 semantics: already authoritative in `docs/BFLOAT16.md`.

## GitHub issues log

- [ ] #426 — `SynthesizeTensors` emits q/k/v biases unconditionally, would corrupt synthetic
  SmolLM weights (found while planning this phase; samples-side fix, out of scope here)