# GPU Probe Phase 3: OpenVINO leg — four-way (CPU · SYCL/oneAPI · DX12 · OpenVINO)

Branch: `khurram/gpu-probe` (off `main`). Probe-first, like the previous phases — no change to
`src/Nivara`, `samples/`, or any NUnit test project. Issue #428.

## Problem

PR #425 proved L0/BF16/DX12 availability; PR #427 proved the three SmolLM BF16 kernels
(dot16 K=16, silu 576, gemv 1536×576) on the two compiler-backed GPU paths available then —
oneAPI SYCL/DPC++ (icpx → SPIR-V over Level Zero) and hand-rolled pure-P/Invoke DX12 (HLSL
`cs_5_1` DXBC) — both gated PASS vs the production CPU gold, `gemv` ~10–23× faster on GPU.

OpenVINO is the third qualitatively different option: Intel's **first-party inference runtime**,
whose GPU plugin is precompiled and tuned (no compiler needed). The open question is whether its
GPU plugin — historically OpenCL/IGC-frontended, the same frontend that mangled hand-authored
SPIR-V in the L0 leg (access-chain ICE, `OpFMul`→`OpFSub`/`OpFDiv`→`OpFMul` miscompiles) — suffers
the same class of bug on the Arc 140T or is rescued by OV's tuned kernel set. Issue #428 asks for
the fourth leg, a PRO/CON matrix across all four paths, and a rough end-to-end inference estimate.

### Decisions settled with the human before this plan

- **OpenVINO C API leg only** — no ONNX Runtime, no Windows ML, no Foundry Local SDK, no oneMKL.
  Preference: use the Intel GPU directly via a **runtime library** (precompiled DLLs) with **no
  tool/SDK/compiler**, on top of the already-installed Intel GPU driver + OpenCL runtime. NuGet was
  checked: there is **no official Intel OpenVINO runtime NuGet** (only a third-party `OpenVINO.runtime.win`
  repack and the ORT-EP bridge `Intel.ML.OnnxRuntime.OpenVino`, both rejected). Official channel =
  PyPI (`pip install openvino==2026.2.1`, Intel's own recommended version) or WinGet MSIX; pip chosen
  (plain-folder layout, discover-at-runtime like L0/SYCL).
- **oneMKL/DPC++ evaluated and dismissed** — oneMKL GPU = SYCL interfaces only (requires the DPC++
  compiler, which is MSVC-gated and not on NuGet; the same toolchain class we marked unavailable),
  exposes **no GPU C ABI** for P/Invoke, and has **no silu/sigmoid op**. Not a fourth independent
  path; recorded in `docs/OPENVINO.md` §6.
- **SYCL marked unavailable** on this machine (toolchain removed; `sycl_runner.exe` fails
  `STATUS_DLL_NOT_FOUND` 0xC0000135). The four-way table still prints the SYCL row as UNBUILT.
- **BF16: gate both configs** — the OV leg runs twice: `inference_precision=BF16` (as the issue
  specifies) and `=f32` (direct IGC-class proof). Both rows gated and reported. BF16-native rounding
  gaps are reported honestly, not treated as miscode.
- **Branch:** existing `khurram/gpu-probe` (fast-forwarded to `main` @ `4e164b5`).

## Proposed changes (all inside `tests/Nivara.GpuProbe/` + docs)

New files:

- `OpenVino/OpenVinoNative.cs` — pure P/Invoke surface vs `openvino_c.dll` (discovered at runtime;
  no packages): `ov_core_create` (device `"GPU"`), `ov_core_set_property` (`INFERENCE_PRECISION_HINT`,
  `CACHE_DIR`, `ENABLE_MMAD=YES`), `ov_core_read_model` (IR XML + bf16 `.bin`), `ov_core_compile_model`,
  `ov_compiled_model_get_property` (read back `FULL_DEVICE_NAME`, `EXECUTION_DEVICES`,
  `INFERENCE_PRECISION_HINT`), `ov_infer_request_*`, `ov_tensor_create` + `ov_tensor_data`, status-code
  checking (`ov_status_e`). Struct constants read from the pinned release's `openvino.h` shipped in the
  wheel — never from memory (the DX12 lesson); `ValidateLayouts()` asserts `Marshal.SizeOf` == C sizes.
  DLL loading mirrors `LevelZero/L0Native.cs`: locate runtime dir (env `NIVARA_OPENVINO_DIR` >
  `python -c` probe of the `openvino` package dir > documented default), PATH-prepend /
  `LoadLibraryEx` altered search path so `openvino_c.dll` + deps (`openvino.dll`, plugins, TBB)
  resolve; `GetProcAddress`-by-name function pointers.
- `OpenVino/OvIrModels.cs` — dependency-free OpenVINO IR v11 XML + bf16 `.bin` writer for the three
  kernels, with **f32 casts as the model output** (GPU output tensors come back f32 for exact gating):
  - `dot16` — MatMul `A[1,16]×B[16,1]` (bf16) → Convert f32 → `[1]`
  - `silu` — Sigmoid(x) → Multiply (x·σ) over `[576]` (bf16) → Convert f32 → `[576]`
  - `gemv` — W `[1536,576]` bf16 weights + X `[576]` → MatMul → Convert f32 → `[1536]`
- `OpenVino/OpenVinoLeg.cs` — dual-config leg returning **two** `LegResults`:
  **`OV/OpenVINO (bf16)`** and **`OV/OpenVINO (f32)`**. Per config: write IR XML + `.bin` to a temp
  dir (or in-memory via `ov_core_read_model` from string), compile on `"GPU"`, assert the read-back
  `EXECUTION_DEVICES` contains `GPU` (no silent CPU fallback; device is explicitly `"GPU"`, never
  `AUTO`), feed the exact `KernelFixtures` byte-identical BF16 buffers, time compile vs infer
  separately (steady-state best-of-N minus first-call JIT), read f32 outputs back for gating. Returns
  `null` when the runtime or GPU plugin is unavailable (mirrors `SyclLeg` nullable transport).
- `OpenVino/OpenVinoRunner.cs` (small) — shared runtime-dir resolution + `openvino_c.dll` load helper.

Modified files:

- `Program.cs` — add `"openvino"` mode (availability + both configs) and a third leg in the
  `"kernels"` mode (two rows: OV bf16 + OV f32), so the table becomes CPU · SYCL(UNBUILT) · DX12 ·
  OV-BF16 · OV-F32. Include the OV leg in `all`.
- `tests/Nivara.GpuProbe/README.md` — install step (`pip install openvino==2026.2.1`), `openvino`
  mode docs, four/five-way table, verdict paragraph.
- `docs/OPENVINO.md` — **new**, the fourth GPU case-study doc (required deliverable, mirrors
  `docs/SYCL.md` / `docs/DX12.md` structure): §1 toolchain/setup + version pin (incl. the NuGet-Channel
  verdict and oneMKL-dismissed analysis); §2 the IGC-class question on Arc 140T (driver 32.0.101.8826)
  — does OV's tuned GPU plugin suffer the hand-authored SPIR-V bug class; §3 per-kernel gate pattern
  (dot16/silu/gemv vs CPU gold, both precision configs, BF16-native rounding analysis: Xe2 DPAS
  accumulates f32 so dot/gemv stay f32-tier; a BF16-rounded sigmoid intermediate may legitimately sit
  outside the f32 gate — reported honestly); §4 BF16 handling + compile-vs-infer + `CACHE_DIR`; §5
  gotchas; §6 PRO/CON matrix across SPIRV · SYCL · DX12 · OpenVINO (+ oneMKL-dismissed row) + rough
  end-to-end estimate (~270 MFLOP/token decode ⇒ ~33 tok/s @ ~8.9 GFLOP/s measured, labeled an estimate).
- `docs/SPIRV.md` — flip the three `(pending)` OpenVINO refs (lines 10–11, 140, 151) to published.
- `docs/TODO.md` — this plan.

## Correctness gates (unchanged from the probe series)

- Single gate for every leg: `|leg − cpu| ≤ 1e-6 + 1e-5·|cpuNivara|` (`CpuLeg.GateAbs/GateRel`), against
  the **production Nivara kernels** (`LlamaFusedKernels.MatMulTransposedB<float>`, `Activation.Silu`) —
  not a hand-rolled oracle. Byte-identical `KernelFixtures.Generate()` BF16 inputs.
- Exit code = number of failed kernels across all legs (0 = gates pass).

## Verification

- `python -m pip install openvino==2026.2.1` (official Intel wheel; runtime only).
- `dotnet build tests/Nivara.GpuProbe` (net11.0 probe only) before each probe commit.
- `dotnet run -c Release --project tests/Nivara.GpuProbe -- kernels` — five-way gates (exit 0).
- `dotnet run -c Release --project tests/Nivara.GpuProbe -- openvino` — OV-specific availability/readback.
- Existing modes must stay green: `run`, `dx12`, `sycl` (SYCL returns UNBUILT).
- Ask before any `dotnet test` / long verification run.

## Planned commits (one logical change each, local only)

1. `docs: plan GPU probe phase 3 (OpenVINO leg) in TODO.md` — this file.
2. `probe: OpenVINO runtime discovery + openvino_c.dll P/Invoke surface` — OpenVinoNative +
   OpenVinoRunner; build-verified.
3. `probe: OpenVINO IR v11 model writer (dot16/silu/gemv, bf16 weights)` — OvIrModels; build-verified.
4. `probe: OpenVINO leg — dual-config gate (bf16 + f32), GPU device assert` — OpenVinoLeg + Program.cs
   `openvino` mode; build + run `openvino` (GPU plugin availability + readback).
5. `probe: five-way kernels gate (CPU · SYCL · DX12 · OV-bf16 · OV-f32) — OpenVINO PASS` — wire the
   OV leg into `kernels`; `kernels` exit 0.
6. `docs: add GPU case-study doc docs/OPENVINO.md (OpenVINO leg verdict)` — required deliverable.
7. `docs: record OpenVINO leg results + flip SPIRV.md pending refs` — probe README, SPIRV.md, CHANGELOG.
8. Cleanup: G2 review (branch as a whole + vs TODO.md) → `git rm docs/TODO.md` → `docs: remove
   TODO.md — plan executed` → offer push + PR (human-confirmed).

## Blast radius

- `tests/Nivara.GpuProbe/**` only (standalone probe, not in the NUnit suite; dotnet-build + its own
  `--kernels`/`--openvino` modes are the verification) plus `docs/TODO.md`, `docs/OPENVINO.md`,
  `docs/SPIRV.md`, `tests/Nivara.GpuProbe/README.md`, `CHANGELOG.md`.
- **No** changes to `src/Nivara`, `samples/`, or the NUnit test projects.
- Existing probe modes (`run`, `dx12`, `sycl`) must remain green.

## Grounding notes (G1, after this plan is committed)

- OpenVINO C API function/property names, `ov_status_e` values, IR v11 XML element names and BF16
  encoding — via microsoft-learn/docs search + Intel OpenVINO docs (`docs.openvino.ai/2026`) before
  writing `OpenVinoNative.cs` / `OvIrModels.cs`.
- `INFERENCE_PRECISION_HINT`, `EXECUTION_DEVICES`, `FULL_DEVICE_NAME`, `CACHE_DIR`, `ENABLE_MMAD`
  property semantics for the GPU plugin on Xe2 (Arc 140T) — via Intel OpenVINO GPU plugin docs.
- PyPI `openvino==2026.2.1` wheel layout (`openvino\libs\openvino_c.dll` + plugins/build) — verify at
  install time (already confirmed via pip index + docs).
- oneMKL/NuGet findings already verified and recorded at issue #428 planning time (no official Intel
  OpenVINO runtime NuGet; oneMKL GPU is SYCL/DPC++-only).

## GitHub issues log

- (none to date — open items from this phase, if any, get created at discovery time and logged here)