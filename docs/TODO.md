# GPU Kernel Probe Phase: BF16 Dot / GEMV / SiLU — three-way (CPU · L0/SPIR-V · DX12)

Branch: `khurram/gpu-probe` (off `main`). Probe-first, like the previous phase — no change to
`src/Nivara`, `samples/`, or any NUnit test project.

## Problem

`main` now carries the L0/BF16/DX12 availability probe (PR #425). It proved:

- The Arc 140T iGPU runs hand-authored SPIR-V (OpenCL kernel model, SPIR-V 1.0) BF16 compute:
  native `OpConvertBF16ToFINTEL` widen + safe-subset `<<16` widen both exact; 1M-iteration f32
  accumulate exact; **pure `System.Numerics.BFloat16` end-to-end**.
- **Indexed memory access is impossible on this driver** (any access-chain opcode → IGC access
  violation; 17-variant bisection). So a real GEMV (`y[i] = Σⱼ W[i,j]·x[j]`) is not expressible on
  L0 today; the L0 safe subset is: direct loads/stores through kernel-arg pointers, OpPhi loops,
  1-lane atomics, BF16 ops.
- DX12 is available (FL 12_2 / SM 6.8, verified via `D3d12Check.cs`) and targets a DXIL/DXBC
  shader path that bypasses the buggy IGC OpenCL frontend — the viable route to *real* GEMM.

Before committing to a GPU inference backend for SmolLM we must prove the actual compute kernels
correct on both GPU paths, apples-to-apples against the CPU. This phase implements **one or two
kernels SmolLM actually needs**, on **three backends** (CPU reference, L0/SPIR-V hand-rolled,
DX12 hand-rolled), with **correctness gates first, then performance** — all inside GpuProbe.

## Kernel choices (decided with the human: "you pick")

1. **Kernel A — BF16 dot / GEMV** (the workhorse: every SmolLM decode token is dominated by
   linears; 135M params ≈ all in matmuls). Exercises the exact compute model we care about:
   BF16→f32 widen-in-register, f32 accumulate, serial-vs-parallel reduction.
   - **L0 leg:** K=16 fixed-width dot via 33 kernel-arg pointers (`a[0..15]`, `b[0..15]`, `c`),
     unrolled straight-line, direct offset-0 loads only — the proven safe subset. This is L0's
     honest expressible shape (no indexing). Native widen variant + emul-widen variant.
   - **DX12 leg:** full GEMV with real indexing (M×576, SmolLM `q_proj`/`gate_proj`-shaped),
     plus the same K=16 dot for the strict three-way apple.
   - **CPU leg:** serial dot/GEMV (identical summation order) + widen-SIMD variant for the
     "production CPU" perf number.
2. **Kernel B — SiLU** (`x·σ(x)`, SwiGLU FFN activation). Deliberately the other side of the
   envelope: **elementwise** (fully expressible on all three legs — fires on L0 too) and it
   exercises **transcendentals (`exp`)** which neither GPU leg has touched yet (everything so far
   is FAdd/FMul). Validates math-library correctness on both SPIR-V/IGC and DX12.
   - L0: `OpenCL.std` `exp` ext-inst (safe subset + ext-inst import only).
   - DX12/CPU: same formula.

Left out (phase 2+): RMSNorm (reduce → indexed → L0-degenerate), RoPE, attention QK^T/softmax/PV,
LM-head GEMV at full 49152×576 (59 MB fixture; can add a slice later). Phase 2 gates on this
foundation proving out.

## Synthetic data (no samples dependency)

The probe owns a small mirror of the Qwen synthetic generator (`samples/NivaraInference/Qwen.cs`):
deterministic xorshift RNG (seed `0x9E3779B9`), values `(rng/uint.Max - 0.5f) * 0.1f`, narrowed to
`BFloat16` via `BFloat16.CreateChecked`. Shapes from the real SmolLM-135M config on disk
(`samples/data/smollm-135m/config.json`: hidden 576, heads 9, kv 3, intermediate 1536, silu).
Fixtures generated in-memory once per run, byte-identical across all three legs:

- `dot16`: two BF16 vectors len 16 (the strict three-way apple)
- `dot576`: two BF16 vectors len 576 (DX12 vs CPU)
- `gemv`: X ∈ BF16[576], W ∈ BF16[M×576] with M = 4096 (gate_proj-ish; multiple of likely tile
  sizes) — optional `lmhead` slice M = 12288 later
- `silu`: one BF16 vector len 576

The Qwen-bias wrinkle (issue #426) doesn't affect this phase — the probe needs kernel-shaped
fixtures, not full-model tensors.

## Proposed changes (all inside `tests/Nivara.GpuProbe/` + docs)

New files:

- `Kernels/KernelFixtures.cs` — fixture generator (RNG mirror, shapes, BF16 buffers) + shared
  host read/write helpers (write `BFloat16` to native buffer, read f32 back).
- `Kernels/CpuKernels.cs` — CPU references: `Dot16Bf16` / `Dot576Bf16` (serial, f32 accumulate,
  BF16 widen via `(float)` cast — exact), `SiluBf16` (golden computed in **double** for a single
  implementation-independent reference), `GemvBf16` (serial), plus widen-SIMD gemv for the perf
  column.
- `Kernels/KernelGate.cs` — the three-way compare harness: runs each leg over the same fixture,
  asserts and reports, then timings (warmup once + median-of-N, mirroring the `add_loop` timing
  pattern in `L0Run.cs`).

Modified files:

- `LevelZero/SpvKernels.cs` — add:
  - `Bf16DotK16(bool nativeWiden)` — SPIR-V 1.0, 33 args (`ptr<CrossWorkgroup,u16>` ×32 + f32
    out), straight-line: 16 × (widen + FMult), FAdd sum tree in fixed serial order. Native
    (`OpConvertBF16ToFINTEL`, capability 6115) and emul (`OpUConvert` + `<<16`) variants.
  - `SiLUBf16()` — `OpExtInstImport "OpenCL.std"` + `OpExtInst` `exp` (opcode 19, verify against
    Khronos registry), guard with capability declarations; direct load/store through args only.
- `LevelZero/L0Run.cs` — extend the shared `TestKernel` to support arbitrary arg counts/buffers
  (currently hard-limited to a/b/c), add tests 6/7 (or a small dedicated runner) invoking the
  above via the existing `TestKernel` plumbing; outputs written to arg-index 32.
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
  output row, serial K loop, `asfloat(uint<<16)` widen — BF16 stored as `uint16_t` in a
  `ByteAddressBuffer`/`StructuredBuffer<uint>`, matching the raw-pattern layout), `silu` (`exp`).
- `Program.cs` — add `kernels` mode (CPU + L0 + DX12 gates; skips DX12 gracefully if device
  creation fails), included in `all`.
- `README.md` — build/run line for `kernels`, gate-table rows, "verified findings"/limitations
  updates.
- `docs/SMOLLM-GPU.md` — new "Kernel phase" section with the three-way results (deliverable).
- `docs/TODO.md` — this plan.

## Correctness gates (the order that matters)

- **dot16 / dot576**: BFloat16×BFloat16 widened to f32, serial f32 accumulate, single canonical
  order → **bit-exact** equality CPU vs L0 vs DX12. (Synthetic values ±0.05 → products ≈2.5e-3,
  sums ≈1e-2·1: no subnormals, no denormal-flush risk; if a leg still diverges by 1 ULP the gate
  FAILS and we investigate — that's the point.)
- **gemv**: serial order in both CPU and DX12 → bit-exact.
- **silu**: golden in **double** on the host; each leg within relative tolerance 1e-5 (transcendental
  implementations legitimately differ across expf/IGC/DXIL; 1e-5 is loose vs f32 ULP ≈1e-7 but far
  tighter than anything that would mask a real bug).

Perf methodology: warmup (discarded) + median-of-N launches, same fixtures, GFLOPS/GFADD
reporting per leg — mirrors the existing `add_loop` timing.

## Verification

- `dotnet build tests/Nivara.GpuProbe` (net11.0 probe only) before each probe commit.
- `dotnet run -c Release --project tests/Nivara.GpuProbe -- kernels` — the new three-way gates
  (exit 0 = real gates pass; expected IGC diagnostics reported separately as before).
- Existing modes must stay green: `run` (tests 1–5 + bisection) and `dx12`.
- Ask before any `dotnet test` / long verification run.

## Planned commits (one logical change each, local only)

1. `docs: plan GPU kernel-probe phase in TODO.md` — this file.
2. `probe: synthetic SmolLM-shaped BF16 fixtures + CPU reference kernels` — KernelFixtures,
   CpuKernels, fixtures in-memory; build-verified.
3. `probe: L0 BF16 dot K=16 + SiLU kernels` — SpvKernels additions + L0Run TestKernel arg
   generalization + tests 6/7; run `kernels` (CPU-vs-L0 leg).
4. `probe: three-way correctness gate harness` — KernelGate with bit-exact/tolerance asserts +
   reporting (CPU/L0 wired first).
5. `probe: hand-rolled DX12 compute path` — D3d12Compute.cs (device→PSO→dispatch→fence→readback)
   + HLSL dot16/gemv/silu; build-verified.
6. `probe: DX12 kernel gates — three-way parity` — wire DX12 leg into KernelGate; `kernels` exit 0.
7. `probe: perf pass across all legs` — timings for dot16/dot576/gemv/silu, README results.
8. `docs: record kernel-phase results` — probe README + docs/SMOLLM-GPU.md.
9. Cleanup: G2 review → `git rm docs/TODO.md` → `docs: remove TODO.md — plan executed` → offer
   push + PR (human-confirmed).

## Blast radius

- `tests/Nivara.GpuProbe/**` only (probe project, standalone, not in the NUnit suite) plus
  `docs/TODO.md`, `tests/Nivara.GpuProbe/README.md`, `docs/SMOLLM-GPU.md`.
- **No** changes to `src/Nivara`, `samples/`, or the NUnit test projects. The Qwen-bias fix is
  tracked separately as issue #426 and is out of scope.
- Existing probe modes (`run`, `dx12`) must remain green.

## Grounding notes (G1, after this plan is committed)

- D3DCompile / `d3dcompiler_47.dll` inbox availability, SM 5.1 DXBC-in-D3D12 compute validity,
  ID3DBlob vtable layout — via microsoft-learn + Windows docs.
- D3D12 compute pipeline facts: root signatures, `D3D12_RESOURCE_STATES`, descriptor heaps,
  fence/`Signal`/wait, `CopyBufferRegion` — via microsoft-learn before writing D3d12Compute.cs.
- `OpenCL.std` extended-instruction `exp` opcode (expected 19) — verify against the Khronos
  SPIR-V registry (canonical for SPIR-V; not a Microsoft subject).
- BFloat16 semantics: already authoritative in `docs/BFLOAT16.md`.

## GitHub issues log

- [ ] #426 — `SynthesizeTensors` emits q/k/v biases unconditionally, would corrupt synthetic
  SmolLM weights (found while planning this phase; samples-side fix, out of scope here)