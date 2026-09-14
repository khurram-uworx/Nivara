# Intel OpenVINO GPU Compute from .NET — Lessons from the GpuProbe

> **Series:** [SPIR-V / Level Zero](SPIRV.md) · [SYCL / oneAPI](SYCL.md) · [DX12](DX12.md) · [OpenVINO](OPENVINO.md) · [ILGPU](ILGPU.md) · [ComputeSharp](COMPUTESHARP.md)

This is the fourth GPU-backend case-study doc (after [docs/SPIRV.md](SPIRV.md),
[docs/SYCL.md](SYCL.md), [docs/DX12.md](DX12.md)) and the deliverable for issue #428: the
qualitatively different *first-party* option on the stack. The previous phases
proved three compile/driver paths (hand-authored SPIR-V over Level Zero, the
oneAPI SYCL/DPC++ toolchain over Level Zero, and hand-rolled HLSL `cs_5_1` over
D3D12). OpenVINO asks the open question the plan was built around: **does
Intel's precompiled, tuned GPU plugin — historically OpenCL/IGC-frontended, the
same frontend that mangles hand-authored SPIR-V — survive on the Arc 140T, or
does it suffer the same bug class?** This document records the measured answer.

All findings below were verified on an **Intel Arc 140T** (8086:7DD1, 128 EU,
driver **1.15.37858**) running Windows 10.0.26200 and .NET 11.0 (Release), with
OpenVINO **2026.2.1** (`pip install openvino==2026.2.1`, the official Intel
wheel — `OpenVINO Runtime 2026.2.1-21919-ede283a88e3-releases/2026/2`). The OV
leg of `tests/Nivara.GpuProbe` is **pure P/Invoke against `openvino_c.dll`** —
no packages, no C++/compiler, no subprocess.

---

## 1. Toolchain / setup (the "no toolchain" path)

OpenVINO is Intel's closed inference runtime, shipped as precompiled DLLs on
PyPI. Unlike the previous three legs it needs **no compiler at all** — the GPU
plugin is already tuned by Intel and loaded from the wheel's `libs` folder:

```
pip install openvino==2026.2.1
# wheel layout (what openvino_c.dll pulls in with it):
#   site-packages\openvino\libs\openvino_c.dll   ← the C API entry
#   site-packages\openvino\libs\openvino.dll     ← the runtime core
#   site-packages\openvino\libs\plugins.xml      ← plugin registry (GPU/CPU)
#   ...\plugins\openvino_gpu_plugin.dll etc.     ← compiled, tuned plugins
#   ...\libs\...\tbb.dll etc.                    ← Intel TBB thread pool
```

**Discoverability.** `OpenVinoRunner.cs` mirrors the L0/SYCL pattern of locating
the runtime at runtime, in order: env `NIVARA_OPENVINO_DIR` → `python -c` probe
of the `openvino` package dir → the documented default
`%APPDATA%\Python\Python312\site-packages\openvino\libs`. `openvino_c.dll` is
loaded with `LoadLibraryEx(…, LOAD_WITH_ALTERED_SEARCH_PATH)` from its own
directory so `openvino.dll`, plugins and TBB resolve without polluting `PATH`.
All entry points are `GetProcAddress`-by-name `Cdecl` delegates (`OpenVinoNative.cs`).

**Channel verdict (recorded at issue #428 planning):** there is **no official
Intel OpenVINO runtime NuGet** — only a third-party `OpenVINO.runtime.win`
repack and the ONNX-Runtime EP bridge `Intel.ML.OnnxRuntime.OpenVino`, both
rejected. The official distributions are PyPI (chosen; plain-folder layout
discoverable like L0) and WinGet MSIX.

**oneMKL / DPC++ (dismissed, recorded):** oneMKL GPU = SYCL interfaces only
(requires the DPC++ compiler, which is MSVC-gated and not on NuGet — the same
toolchain class we marked unavailable), exposes **no GPU C ABI** for P/Invoke,
and has **no silu/sigmoid op**. Not a fourth independent path; recorded here
rather than in a dedicated doc.

**API surface.** The pinned wheel ships **no headers**, so structs/enums/
GUIDs were read from the matching release's `openvino.h` on GitHub (tag
`2026.2.1`) — never from memory (the DX12 lesson, §4 of [docs/DX12.md](DX12.md)). The
leg uses the small, stable C subset `ov_core_create` →
`ov_core_set_property`/`ov_core_get_property` → `ov_core_read_model` →
`ov_core_compile_model` → `ov_compiled_model_get_property` →
`ov_compiled_model_create_infer_request` → `ov_infer_request_set_input_tensor`
/`_infer`/`_get_output_tensor_by_index` → `ov_tensor_create`/`ov_tensor_data`,
with `ov_get_last_err_msg()` for the underlying plugin exception text. No C++
objects are ever constructed or laid out from managed code.

## 2. The IGC-class question, answered on the Arc 140T

The L0 phase ([docs/SPIRV.md](SPIRV.md)) proved this driver's IGC OpenCL frontend
**miscompiles hand-authored SPIR-V** — `OpFMul` runs as `OpFSub`, `OpFDiv` as
`OpFMul`, and any access-chain opcode ICEs the compiler. OpenVINO's GPU plugin
is delivered on the same OpenCL/IGC stack, so the first question was whether the
*tuned, precompiled* OV kernel set behaves differently from hand-authored
bytecode — or carries the same bug class.

**Measured: it does not.** The leg gates all three production SmolLM kernels on
the GPU plugin against the production Nivara CPU kernels (§3). The F32-declared
config (direct-IGC-class proof: explicit `INFERENCE_PRECISION_HINT=f32`, no FX
to F16) **passes all three kernels** — dot16 bit-exact, silu 576/576, gemv
1536/1536 — with worst-case `|diff| ≤ 3.7e-9` on silu and `1.6e-9` on gemv. The
plugin runs its own tuned kernels (`NAME`/`Branch`/`MatMul` over DPAS where the
shape qualifies), not user-provided SPIR-V through the buggy frontend. Neither
the `OpFMul`/`OpFDiv` miscompile nor the access-chain ICE surfaces. **OpenVINO
is rescued by its tuned kernel set** — on transparent IR (the topological
`MatMul`/`Sigmoid`/`Multiply` ops below), the driver's optics are clean.

## 3. Per-kernel gate pattern vs the CPU gold

Same single gate as every leg: `|leg − cpuNivara| ≤ 1e-6 + 1e-5·|cpuNivara|`
against the **production Nivara kernels** (`LlamaFusedKernels.MatMulTransposedB`,
`Activation.Silu`) on byte-identical BF16 `KernelFixtures` inputs. The models
are hand-emitted OpenVINO IR v11 with the output tensors cast to F32
(F32 output tensors are read back for exact gating):

| config | kernel | gate | worst (diagnostic) | result |
|---|---|---|---|---|
| `OV (bf16 IR)`, no hint (plugin picks F16) | `dot16` (K=16) | PASS | **0.0 ULP** (bit-exact) | PASS |
| `OV (bf16 IR)`, no hint | `silu` (576) | **402 / 576 fail** | 13 453 ULP @ 139, `\|diff\| = 1.25e-5` @ 0.014 | **honest FAIL** (BF16/F16 elementwise) |
| `OV (bf16 IR)`, no hint | `gemv` (1536×576) | PASS | 14 336 ULP @ 1508 (`\|diff\| = 1.6e-9`, near-zero ref row) | PASS |
| `OV (f32 + hint)` | `dot16` (K=16) | PASS | **0.0 ULP** (bit-exact) | PASS |
| `OV (f32 + hint)` | `silu` (576) | PASS 576/576 | 4.0 ULP | PASS |
| `OV (f32 + hint)` | `gemv` (1536×576) | PASS | 14 336 ULP @ 1508 (`\|diff\| = 1.6e-9`) | PASS |

Same gemv worst-ULP caveat as the SYCL/DX12 legs: after cancellation the
diagnostic row lands near zero where an f32 ULP is ~1.9e-9, so a perfectly good
row reads thousands of ULP while sitting 600× inside the `1e-6` absolute gate.

### The BF16 elementwise finding (reported honestly)

The bf16 config is a **BF16-declared** model (all IR ports + weights BF16,
matching the issue's "BF16" row) compiled on the GPU plugin with **no
`INFERENCE_PRECISION_HINT`** (see §4). The plugin resolves execution to F16 —
read-back and printed hint show `f16` — which is exactly right for the
reductions and exactly wrong for the nonlinearity:

- **dot16 / gemv stay F32-tier.** Products are accumulated by the hardware's
  native model — Xe2 DPAS accumulates BF16 products in F32 — so the reduction
  results gate **bit-exact** (dot16) and far inside the tolerance (gemv
  `1.6e-9`). BF16 wire format + F32 accumulation is exactly the production
  model in [docs/SYCL.md](SYCL.md) §4, this time with **zero host-side widening**.
- **silu carries F16-elementwise rounding.** Sigmoid and the x·σ multiply run
  in F16 on 0.014-magnitude intermediates, so the result lands ~1.9e-5 from the
  F32 CPU gold (13.5k ULP at F16 precision on that row) — legitimately outside
  the tight F32 gate. This is **not a miscode**: the BF16-declared model did
  exactly what it declared, sigmoid/multiply just aren't F32-tier when executed
  in F16. It is reported as the honest gap the issue's BF16 row always implied.
  The f32 config (direct IGC-class proof) passes all three kernels.

**Read: the plugin, precision-declared properly, is correct on both configs; a
BF16-rounded sigmoid is a *precision* finding, not a correctness one.** For a
production BF16 pipeline the clean answer mirrors the CPU path: keep
sigmoid/multiply in F32 (or accept a documented ~1e-4 relative error on
elementwise activations), and keep the F32-accumulated matmuls as the win.

## 4. BF16 handling, compile-vs-infer, and CACHE_DIR

**BF16 weights.** The hand-emitted `.bin` is the raw `BFloat16` byte stream
(`BFloat16.CreateChecked`-narrowed fixtures, exactly as the other legs feed
them) — no widening on the host; the plugin consumes signed 2-byte BF16
literals. The F32 config widens once through the *production* Nivara widen
(`CpuLeg`), the value CPU gold is computed from — never a hand-rolled oracle.

**Precision configs — the two rows, and their sharp edges (verified in Python
before the C# leg):**

- **`OV (bf16 IR)`** — BF16-declared IR, compiled on `"GPU"` with **no
  precision hint**. `INFERENCE_PRECISION_HINT=BF16` is a **CPU-only token** on
  this release; the GPU plugin accepts `{f16, f32, dynamic}` only and rejects
  `BF16`. The plugin's implicit resolution to F16 is what makes the bf16 row
  beat CPU on gemv while staying honest about silu.
- **`OV (f32 + hint)`** — F32-declared IR with **mandatory**
  `INFERENCE_PRECISION_HINT=f32`. Without the hint the plugin *silently* FX-compiles
  to F16 and the gate FAILS (verified in Python validation); with it, every
  kernel passes. This is the direct-IGC-class correctness proof.

**Compile vs infer.** The two are timed separately: the leg reports warm
compile time (`ov_core_compile_model`; 47–127 µs-scale **ms** here — CACHE_DIR
cold) and steady-state infer (1 discarded warmup + best-of-25 timed `infer()`
calls). GPU compile is a first-call cost paid per model; a production
`src/Nivara.Gpu` would compile once and reuse the request for all tokens.

**CACHE_DIR crash class.** `ov_core_set_property(…, "CACHE_DIR")` with a
*reused* cache across configs crashed the process (0xC0000005 — stale
bf16/f32 blob conflict). The leg sidesteps the landmine with a **fresh temp
`CACHE_DIR` per (config, run)** instead of relying on the cache at all. Handle
your own CACHE_DIR: never share one between precision configs.

**Device assert (no silent CPU fallback).** The leg compiles on the explicit
device `"GPU"` — never `AUTO` — and asserts the read-back
`EXECUTION_DEVICES` contains `GPU` (observed `GPU.0`, iGPU) plus the
`FULL_DEVICE_NAME` read-back `Intel(R) Graphics (iGPU)`. A CPU fallback would
violate the gate's whole premise and is impossible here by construction.

## 5. P/Invoke / IR gotchas (what cost real time in this leg)

1. **Every tensor returned or created by the C API is a `new`-allocated wrapper
   you own.** `ov_tensor_create` results and the tensors handed back by
   `_get_output_tensor_by_index`/`_get_input_tensor_by_index` must each be
   released with `ov_tensor_free`; `ov_shape_free` is mandatory after
   `ov_tensor_get_shape`. Forgetting either leaks; double-freeing is a crash.
2. **`ov_get_last_err_msg` is the debug friend.** Status codes collapse to
   `ov_status_e=-1 (general error)`; the *message* holds the real C++
   exception (`src\inference\src\cpp\infer_request.cpp:202: invalid vector
   subscript`). Wire it into every `Check()` — and note that code like
   `get_outputs().at(idx)` means an **empty outputs vector**, which in this
   probe was a generated-IR bug (a dropped `Result` layer — see 3), not a
   runtime problem.
3. **The read-model I/O-count self-check catches IR bugs immediately.** A
   `0`-output model is silently loadable; parsing my first IR gave exactly that
   (`model inputs 1, outputs 0`) because the `ConvertToF32`+`Result` tail layers
   were dropped from the emitted XML. `ov_model_inputs_size`/`ov_model_outputs_size`
   asserted BEFORE the gate made the bug a one-line diagnosis instead of a hunt.
4. **IR v11 shape/port tokens are exact.** Port `precision` tokens are `BF16` /
   `FP32` (the XML token is `FP32`, not `F32`); the `Result` layer declares the
   output, edges are strict `to-layer`/`to-port`/`from-layer`/`from-port`, and
   every layer must be reachable from a `Result` or the model loses that output.
5. **`LOAD_WITH_ALTERED_SEARCH_PATH` + absolute dirs (no `PATH` mutation).**
   `openvino_c.dll`'s own deps (`openvino.dll`, plugins, TBB) are sibling files
   in the wheel's `libs`; altering the search path to the DLL's directory loads
   the whole graph cleanly. Never `Push` onto `PATH` from the probe.
6. **Missing C API symbols (avoid):** `ov_shape_size`,
   `ov_available_devices_size`, `ov_available_devices_get_device`,
   `ov_core_read_model_from_memory` did not resolve on `2026.2.1`; the leg uses
   `ov_core_read_model(xml_path, bin_path)` and `ov_core_get_property("AVAILABLE_DEVICES")`
   instead.
7. **Tensors returned by index are caller-owned; `set_input_tensor` requires
   the model's declared element type.** Feeding the BF16 input tensor into the
   F32-declared model is rejected — input tensors are created per config, not
   shared across the bf16/f32 rows.

## 6. The four-way decision + rough end-to-end estimate

**All measured figures now live in the probe README's consolidated side-by-side
table** ([tests/Nivara.GpuProbe/README.md](../tests/Nivara.GpuProbe/README.md) → "Backend comparison — the numbers,
side by side"), which gathers the SYCL, DX12 and OpenVINO numbers into one view.
For this leg (Release `kernels` run; OV infer = best-of-25 steady-state) gemv
ran **137.9 µs (bf16) / 181.7 µs (f32)** — the ~26–34× CPU win over this run's
4,745 µs CPU measurement; dot16 is 68.5–75.8 µs, silu 51.2–58.9 µs.

Reading: dot16 stays launch-bound everywhere (its only role is the smallest
correctness probe). gemv — the dominating SmolLM decode shape — runs at
**~26–34× the CPU** on the OV plugin; silu is ~1.7–1.9× CPU-competitive even on
the honest elementwise outer loop. Hints observed: `f16` (bf16 row), `f32`
(f32 row), `exec GPU.0`.

### PRO/CON across the four proven paths (= issue #428 matrix)

| path | toolchain | packages | gemv (this machine) | verdict |
|---|---|---|---|---|
| L0 hand-authored SPIR-V | none (hand bytes) | none | n/a | **dead end** — IGC miscompiles `OpFMul`/`OpFDiv`; access-chain ICE ([docs/SPIRV.md](SPIRV.md)) |
| SYCL/oneAPI DPC++ | icpx+VS Build Tools | none (runtime loaded) | ~170–198 µs | **proven**, compiler-backed; footprint = oneAPI SDK, subprocess/`setvars` (toolchain removed on this machine — row UNBUILT) |
| DX12 HLSL `cs_5_1` | none (in-process `d3dcompiler_47`) | none | ~152–525 µs | **proven**, fully managed; per-dispatch overhead highest, no external tooling at all |
| OpenVINO GPU plugin | none | `pip install openvino` | **138–182 µs** | **proven first-party**: no compiler, tuned kernels, byte-exact dot16, gemv ~26–34× CPU; bf16 silu honestly F16-tier |
| oneMKL (dismissed) | DPC++ (gated) | oneMKL | n/a | not a path: SYCL-only ABI, no silu/sigmoid, toolchain-gated |

**Rough end-to-end estimate (labeled an estimate):** SmolLM-135M decode is
dominated by `[1536×576]·[576]` GEMVs (~270 MFLOP/token in the plan's model).
At the measured OV gemv-only rate of **~9.7 GFLOP/s f32 / ~12.8 GFLOP/s bf16**
(1.77 MFLOP per gemv ÷ best infer time), the *gemv-only* floor is ~21–28 ms/token
(≈36–48 tok/s); a realistic whole-model number is lower once attention, norm and
elementwise ops run through the same plugin request (the plan's headline
estimate of **~33 tok/s @ ~8.9 GFLOP/s** sits inside that band). These are
directional — the probe measures kernels, not a full graph — and the honest
label is: first-party, compiler-free, and within striking distance of the
SYCL/DX12 figures from [docs/SYCL.md](SYCL.md) §4 / [docs/DX12.md](DX12.md) §5.

## Verdict

**OpenVINO is the fourth, fully proven backend and the cleanest story of the
probe series: a first-party solver with zero compiler and zero packages.** On
the Arc 140T its tuned GPU plugin does *not* carry the hand-authored SPIR-V bug
class — the F32-declared config passes all three production kernels (dot16
bit-exact, silu 4 ULP, gemv 1.6e-9), and the BF16-declared row keeps the F32-tier
matmul win (~26–34× CPU) while honestly surfacing the F16-elementwise sigmoid
rounding the BF16 row always implied. If Nivara ships a managed GPU backend,
OpenVINO's P/Invoke surface, `pip install` distribution and precompiled plugin
make it the lowest-infrastructure candidate on the table.

## Reproducibility

1. `python -m pip install openvino==2026.2.1`
2. `dotnet build tests/Nivara.GpuProbe -c Release` — 0 warnings, 0 errors.
3. `dotnet run -c Release --project tests/Nivara.GpuProbe -- ov` — runtime
   banner, device readback (`FULL_DEVICE_NAME`), both configs' gates.
4. `dotnet run -c Release --project tests/Nivara.GpuProbe -- kernels` — the
   five-way gate + timing table. Exit contract: number of failed cells —
   **405** on this machine = 402 honest-OV-bf16-silu + 3 SYCL-unavailable
   (baseline); every failing cell is an explicitly-flagged honest one.

Snapshot:

```
OpenVINO: 2026.2.1-21919-ede283a88e3-releases/2026/2  (pip install openvino==2026.2.1)
Driver:   1.15.37858        Device: Intel(R) Graphics (iGPU) — GPU.0
OS:       Windows 10.0.26200   .NET: 11.0 (Release)
```

## Reference links

- [OpenVINO C API (Intel)](https://docs.openvino.ai/2026/api/c_cpp_api/group__ov__c__api.html)
- [OpenVINO GPU plugin / properties (INFERENCE_PRECISION_HINT, EXECUTION_DEVICES, CACHE_DIR)](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html)
- [OpenVINO IR v11 element names / opset1](https://docs.openvino.ai/2026/documentation/openvino-ir-format.html)
- [System.Numerics.Tensors / .NET 11 BFloat16 (Microsoft Learn)](https://learn.microsoft.com/en-us/dotnet/standard/design-guidelines/numeric-guidelines)
- Series: [docs/SPIRV.md](SPIRV.md) · [docs/SYCL.md](SYCL.md) · [docs/DX12.md](DX12.md) · [docs/OPENVINO.md](OPENVINO.md) · [docs/ILGPU.md](ILGPU.md) — the GPU-backend case-study series