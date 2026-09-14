# GPU probe phase 4b: ComputeSharp leg (issue #432)

## Problem

Issue #432 asks for the ComputeSharp leg of the GPU kernel probe: a managed
NuGet library (C# kernels → source-generated HLSL → DXIL via the library's
bundled DXC → D3D12 dispatch) added to the existing multi-leg `KernelGate`,
plus a `docs/COMPUTESHARP.md` case study. Same fixtures, same gold target
(production Nivara CPU kernels), same tolerance gate as phases 1–4a. The
central empirical question: does the source-gen + DXC pipeline produce
correct/fast DXIL for the three SmolLM kernel shapes on the Arc 140T iGPU,
compared against the hand-CS5.1 DX12 leg (`docs/DX12.md`)?

The comment on #432 cross-links phase 4a (ILGPU, issue #431), which is
already merged in this tree. Two issue premises are stale and corrected here:
"first NuGet reference" (ILGPU landed first → ComputeSharp is the second) and
"six-way table" (the `kernels` table is already 6 rows incl. ILGPU → adding
ComputeSharp makes it 7). `GraphicsDevice.GetDefaultDevice()` in the issue is
the 2.x-era name; the pinned 3.2.0 API is `GraphicsDevice.GetDefault()`.

## Proposed changes

### 1. `tests/Nivara.GpuProbe/Nivara.GpuProbe.csproj`

Snapshot-pinned package refs (comment in the ILGPU-leg style; phase 4b,
issue #432; bundled-DXC story):

```xml
<PackageReference Include="ComputeSharp" Version="3.2.0" />
<PackageReference Include="ComputeSharp.Dxc" Version="3.2.0" />
```

`ComputeSharp.Dxc` bundles the native DXC binaries (dxcompiler/dxil) — the
base `ComputeSharp` package only depends on `ComputeSharp.Core`, so the Dxc
package is required for runtime shader compilation. Both MIT, net8.0-targeted
(compatible with the probe's net11.0).

### 2. New `tests/Nivara.GpuProbe/ComputeSharp/`

- **`ComputeSharpKernels.cs`** — the three kernels as ComputeSharp shaders,
  mirroring `D3d12/GemvKernels.cs` exactly: one `ReadOnlyBuffer<uint>` (packed
  concatenated inputs), one `ReadWriteBuffer<float>` output; in-shader BF16
  widen `Widen(packed, element)` via `Hlsl.AsFloat(bits << 16)`, silu via
  `Hlsl.Exp`; per-kernel `[ThreadGroupSize(...)]` matching the DX12 leg
  (dot16 `(1,1,1)`, silu/gemv `(256,1,1)`); bounds-checked padded indexes.
- **`ComputeSharpLeg.cs`** — `RunLeg(KernelFixtures) → LegResults?`:
  try/catch → null (KernelGate UNBUILT row); `GraphicsDevice.GetDefault()`,
  print `Name` + `IsHardwareAccelerated`, assert hardware (WARP ⇒ honest
  null — no CPU fallback); persistent buffers; timed first dispatch (DXC
  compile + PSO, the additive one-time cost) split from 1 warmup + best-of-25
  steady state (ILGPU/OV methodology); readback `CopyTo`; `LegResults`.
- **`Availability.cs`** — `computesharp` mode banner (mirrors
  `Ilgpu/Availability.cs`).

### 3. `tests/Nivara.GpuProbe/Program.cs`

New `"computesharp"` CLI mode; add `("ComputeSharp (DXIL)", …)` to the
`kernels` gate (seven-row table) and to `all`/default chains.

### 4. `docs/COMPUTESHARP.md` (new)

ILGPU.md/OPENVINO.md case-study style: series header; install/version pin;
API shape (3.2.0 shader snippet); source-gen + bundled-DXC story (replaces
d3dcompiler_47; SM 6.x actually produced); BF16 packed-widen path; setup vs
steady-state split; footprint; PRO/CON + gotchas matrix; verdict/gate tables;
decision records. Measured figures filled after the `kernels` run.

### 5. `tests/Nivara.GpuProbe/README.md`

Six GPU paths (was five); summary/back-comparison/gate tables gain the
ComputeSharp row/column; `kernels` narrative becomes seven-way; exit-code
narrative documented in phase-3 style (expected to stay 405 if ComputeSharp
passes 3/3); files list + references.

## Verification steps

- `dotnet build tests/Nivara.GpuProbe -c Release` → 0 warnings/0 errors
  (issue deliverable). Ask before running.
- `dotnet run -c Release --project tests/Nivara.GpuProbe -- kernels` on the
  Arc 140T machine → record measured table rows into README + COMPUTESHARP.md.
  Ask before running.
- No `src/Nivara` / `samples/` / NUnit test changes (probe-only per issue).

## Planned commits

1. `docs: plan GPU probe phase 4b (ComputeSharp leg) in TODO.md`
2. `probe: add ComputeSharp + ComputeSharp.Dxc 3.2.0 package pins to GpuProbe`
3. `probe: add ComputeSharp leg kernels (dot16/silu/gemv, packed-BF16 widen)`
4. `probe: add ComputeSharpLeg runner + availability banner, wire CLI modes`
5. `docs: add COMPUTESHARP.md case study`
6. `docs: update GpuProbe README with ComputeSharp leg rows`

## Blast radius

- `tests/Nivara.GpuProbe/` only — the probe is a standalone run-manually
  console app, **not** part of the NUnit suite. `Program.cs` mode dispatch and
  the `kernels` gate table change, but exit-code contract is additive
  (expected unchanged 405 if the leg passes 3/3).
- New NuGet refs (`ComputeSharp`/`ComputeSharp.Dxc`) affect only the probe
  project; no production dependency, no `src/Nivara` changes.
- Docs: `docs/COMPUTESHARP.md` new; `docs/ILGPU.md` already cross-links #432
  (no change); probe README updated.
- Risk: 3.2.0 API-source-generator compile warnings (must stay 0/0); DXC
  runtime availability on the machine; WARP-vs-hardware device selection
  (asserted, honest null on WARP).

## GitHub issues log

- (none yet — faithfully created during execution if deferred work surfaces)