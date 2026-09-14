# TODO — GPU probe phase 4a: ILGPU leg (issue #431)

## Problem

`tests/Nivara.GpuProbe` currently gates the three production SmolLM-shaped BF16 kernels
(dot16 · silu · gemv) through five paths (CPU gold · SYCL · DX12 · OV-bf16 · OV-f32).
Issue #431 (phase 4a) adds a **sixth, fully-managed backend**: ILGPU, a pure-C# JIT
kernel runtime that compiles C# kernels to OpenCL C and runs them on the Arc 140T iGPU
via the in-box OpenCL ICD + Intel driver. Deliverables: NuGet pin recorded, kernels
actually gate on the iGPU via a `CLAccelerator` (device printed, **no CPU fallback**),
six-way consolidated table in the probe README, and `docs/ILGPU.md` in the OPENVINO
case-study style (PRO/CON + gotchas keep an ILGPU row). The phase answers the open
empirical question the SYCL leg posed for compiler-produced bytecode, now for
**ILGPU-generated** OpenCL C through the same Intel IGC frontend — measured, not assumed.
Companion phase 4b = ComputeSharp (#432), same contract, own doc.

## Research (grounding, done before this plan — G1)

- **NuGet pin**: `ILGPU 1.5.3` (latest stable, published 2025-07-12; no 1.5.4/2.0) +
  `ILGPU.Algorithms 1.5.3` for `XMath.Exp` (core `IntrinsicMath` has no Exp — verified in
  v1.5.3 sources; `XMath.Exp` → `IntrinsicMath.CPUOnly.Exp` which carries the
  `[MathIntrinsic(MathIntrinsicKind.Exp)]` attribute and compiles to the OpenCL `exp`
  builtin in kernels). Transitive deps: System.Collections.Immutable, System.Memory,
  System.Reflection.Metadata, System.Runtime.CompilerServices.Unsafe. Pure managed —
  **no native SDK install** (Windows ships in-box `OpenCL.dll`; the iGPU's OpenCL comes
  from the Intel graphics driver). No CUDA (Intel-only HW). `ILGPU.Lightning` = no
  (sample lib). `SpawnDev.ILGPU` fork = documented only (native BF16/FP8, out of scope).
- **BF16**: native BF16 kernel types do NOT exist in core 1.5.3 (PR #1221 open/unmerged;
  v2.0 = PR #1577 also open). `Half` can't carry BF16 bit patterns → the issue's
  packed-2-per-uint + in-shader-widen is the **primary** path: `ArrayView<uint>` +
  `Interop.IntAsFloat(uint)` (kernel-safe bit reinterpret, verified in v1.5.3
  `Interop.cs`; `ArrayView<T>.Cast<TOther>()` verified as a `CastView` intrinsic),
  mirroring `D3d12/GemvKernels.PackBf16`/`Widen` byte-for-byte.
- **API shape verified against v1.5.3 sources**: `Context.Create(builder => …
  .Default().Optimize(OptimizationLevel.O2))`, `context.GetCLDevices()`, `CLDevice.Name`
  / `VendorName` / `CreateCLAccelerator(context)`, `accelerator.Allocate1D<T>(long)`,
  `buffer.View` (`ArrayView<T>`), `ref T view[int]` kernel indexer, `buffer.CopyFromCPU`,
  `GetAsArray()`, `LoadAutoGroupedKernel<TIndex, T1, …>` → `Action<AcceleratorStream,
  TIndex, …>`, `stream.Synchronize()`, `MemoryBuffer : AcceleratorObject` (disposable).
- **MS Learn MCP**: no ILGPU content (not Microsoft) — useful only the in-box
  `OpenCL.dll` note; kernel-writing guidance comes from ilgpu.net docs + the fetched
  SimpleKernel sample: high-level loader API (no boxing), implicitly-grouped
  auto-grouped kernels, `OptimizationLevel.O2`, persistent buffers, single stream +
  `Synchronize()` per timed dispatch, default IEEE math (no `MathMode.Fast` — gate).

## Proposed changes (all in `tests/Nivara.GpuProbe` + `docs/`, no `src/Nivara`/`samples`/NUnit)

1. **`Nivara.GpuProbe.csproj`** — first NuGet package refs in the probe:
   ```xml
   <ItemGroup>
     <PackageReference Include="ILGPU" Version="1.5.3" />
     <PackageReference Include="ILGPU.Algorithms" Version="1.5.3" />
   </ItemGroup>
   ```
2. **`Ilgpu/IlgpuKernels.cs`** — the three kernels as ILGPU methods (implicitly-grouped,
   `Index1`-driven), in-shader `Widen(packed, element)` via `Interop.IntAsFloat(uint)`,
   silu via `XMath.Exp`:
   - `Dot16Kernel` — extent 1, serial K=16 f32 accumulate (input = packed [a16|b16]).
   - `SiluKernel` — extent 576, `x / (1f + XMath.Exp(-x))`.
   - `GemvKernel` — extent 1536, one thread per row, serial K=576 with widen on w and x
     (input = packed [w(1536·576)|x(576)]).
3. **`Ilgpu/IlgpuLeg.cs`** — leg runner: O2 context → enumerate `GetCLDevices()`, select
   the Intel GPU by name (assert `Name`/`VendorName` contains Intel / not a CPU
   accelerator — return `null` ⇒ UNBUILT rather than CPU fallback) → print accelerator +
   device → persistent packed `uint` input + `float` output buffers per kernel → load
   kernels → 1 warmup + best-of-25 timed synchronized dispatches (per-kernel µs) →
   read back f32 → `LegResults`. Setup costs (context+accelerator creation, kernel-JIT
   to first sync) timed separately and printed so the README table can stay
   kernel-only, matching the other legs.
4. **`Ilgpu/Availability.cs`** — `ilgpu` mode banner (context, CL devices, chosen
   accelerator), mirroring `OpenVino/Availability`.
5. **`Program.cs`** —
   `"ilgpu" => Ilgpu.Availability.Run() + KernelGate.Run(fixtures, "ILGPU (OpenCL)", Ilgpu.IlgpuLeg.RunLeg)`,
   and the `kernels` gate gains `("ILGPU (OpenCL)", Ilgpu.IlgpuLeg.RunLeg)` → six-way.
6. **`docs/ILGPU.md`** — OPENVINO-style case-study: version pin/install, API shape,
   BF16 path (packed widen primary; `Half` unusable; bf16 PR + SpawnDev fork + v2.0
   roadmap), **measured** IGC/OpenCL verdict for ILGPU-generated OpenCL C, setup-vs-
   steady-state split, kernel-model gotchas, PRO/CON + gotchas matrix row.
7. **Probe `README.md`** — `-- ilgpu` / six-way `kernels` in Build & Run, six-way
   consolidated timing + correctness tables (measured figures), exit-contract update,
   Files section additions, Recommendations row.

## Verification

- `dotnet build tests/Nivara.GpuProbe -c Release` → **0 warnings, 0 errors** (each commit).
- `dotnet run -c Release --project tests/Nivara.GpuProbe -- ilgpu` — banner + gates.
- `dotnet run -c Release --project tests/Nivara.GpuProbe -- kernels` — six-way gate +
  timing table. Exit = failed cells; ILGPU passes keep the documented honest
  baseline (402 OV-bf16-silu + 3 SYCL-unavailable); any new failure is reported honestly.
- Record measured figures + the IGC verdict in the README table and `docs/ILGPU.md`.
  If ILGPU-generated OpenCL C misbehaves (same IGC as the hand-authored SPIR-V leg),
  evidence-probe it per `docs/SPIRV.md` §2 discipline.

## Measured results (Arc 140T, driver 1.15.37858, .NET 11.0 Release)

- **IGC verdict: PASS** — ILGPU-generated OpenCL C compiles and runs correctly;
  device printed = `Intel(R) Graphics`, `CL_DEVICE_TYPE_GPU`, vendor
  `Intel(R) Corporation` (no CPU fallback — asserted and printed).
- Gates (vs production CpuLeg, `|leg−cpu| ≤ 1e-6 + 1e-5·|cpu|`): dot16 **0.0 ULP
  PASS**, silu **576/576 PASS (worst 4.0 ULP)**, gemv **1536/1536 PASS**.
  Failure contribution to `kernels` exit: 0 (exit stays 405).
- Steady state (1 warmup + best-of-25 synchronized dispatches, best of two runs):
  dot16 **11.4–11.7 µs**, silu **6.9 µs** (fastest GPU leg measured in the
  series), gemv **133.4–135.8 µs** (~27× the 3608.7 µs CPU figure in the six-way
  run). Setup split: context+accelerator ~8.8–9.8 ms; per-kernel JIT (load + first
  dispatch) ~1.3–2.7 ms (dot16 up to 3.8 ms in the ilgpu-only run).
- Comparison: gemv trails only OpenVINO bf16's tuned gemm (86.8 µs); beats DX12's
  naive same-shape kernel (246 µs) ~1.8×. silu/dot16 are the fastest GPU rows in
  the consolidated table.

## Planned commits

1. ✓ `docs: plan GPU probe phase 4a (ILGPU leg) in TODO.md`
2. ✓ `probe: pin ILGPU 1.5.3 + ILGPU.Algorithms 1.5.3 in Nivara.GpuProbe` (record snapshot + transitive footprint)
3. ✓ `probe: add ILGPU (OpenCL) leg — kernels, leg runner, availability, six-way gate wiring`
4. (in progress) `docs: add ILGPU case study and six-way probe results` (measured figures, `docs/ILGPU.md`, README)
5. (pending) final build gate + review fixes if any (G2)

## Blast radius

- `tests/Nivara.GpuProbe` only: csproj (2 PackageReferences), `Program.cs` (2 lines),
  new `Ilgpu/` folder (3 files). `Kernels/*` and other legs untouched.
- `docs/ILGPU.md` (new), `docs/TODO.md` (this plan), probe `README.md` (tables/build-run/files).
- No production symbols referenced (ILGPU consumed by the probe only); no NUnit changes.
- NuGet footprint added to the probe project only; solution `Nivara.slnx` build unchanged
  in shape (new package refs restore via the same feed).

## GitHub issues log

- [ ] `#432` — GPU probe phase 4b: ComputeSharp leg (companion, already tracked/cross-linked in #431; out of scope for this phase).
- No new deferred work surfaced during execution — nothing else to log.