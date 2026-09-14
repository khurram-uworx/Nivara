# SYCL / oneAPI GPU Compute on Intel Arc — Lessons from the GpuProbe

> **Series:** [SPIR-V / Level Zero](SPIRV.md) · [SYCL / oneAPI](SYCL.md) · [DX12](DX12.md) · [OpenVINO](OPENVINO.md) · [ILGPU](ILGPU.md)

All findings below were verified on an **Intel Arc 140T** (8086:7DD1, ~2 GB
shared RAM, driver **32.0.101.8826**) running Windows 10.0.26200, .NET 11.0,
oneAPI Base Toolkit **2025.1.3.8**, and VS 2022 Build Tools 17.14 (MSVC
14.44). They apply to Arrow Lake / Lunar Lake integrated Intel GPUs and are
the foundation for the planned SmolLM and Qwen SYCL inference backend.

---

## 1. Toolchain Setup (Windows)

### Prerequisites

| Component | Install | Purpose |
|---|---|---|
| Intel oneAPI Base Toolkit 2025.1 | `winget install Intel.OneAPI.BaseToolkit --exact` | `icpx` (DPC++ compiler), SYCL runtime, Level Zero loader, oneMKL, oneDNN |
| VS 2022 Build Tools (C++ workload) | `winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"` | MSVC host toolchain (`cl.exe`); **icpx requires this on Windows** |

> ⚠️ The oneAPI installer requests admin elevation. Accept the UAC prompt
> when it appears during `winget install`.

### Environment chaining

`icpx` on Windows uses the MSVC ABI and linker. The correct invocation order:

```bat
:: 1. Set up MSVC host tools (cl.exe, LIB, INCLUDE)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64

:: 2. Set up oneAPI (adds icpx, sycl8.dll, ur_loader.dll to PATH)
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

**Order matters:** VsDevCmd first, then setvars. The oneAPI `setvars` adds
compiler and runtime dirs to PATH but does *not* set up MSVC's `LIB`/`INCLUDE`.

### Compile command

```bat
icpx -fsycl -O2 sycl_runner.cpp -o sycl_runner.exe
```

`-fsycl` enables SYCL language support and device compilation. `-O2` is
sufficient for correctness probing; `-O3` or `-Ofast` for production perf.

---

## 2. Known IGC / Driver Bugs (verified on driver 32.0.101.8826)

These bugs were found via hand-authored SPIR-V 1.0 through Level Zero
(`zeModuleCreate` + `zeKernelCreate`). **Bug #5 is the showstopper for
hand-authored kernels; bugs #1-4 are why the probe pivoted to DPC++.**

| # | Symptom | Opcode / construct | Severity |
|---|---|---|---|
| 1 | `IGC: Internal Compiler Error: Access violation` | Any `OpAccessChain` (65), `OpInBoundsAccessChain` (66), `OpPtrAccessChain` (67) | Fatal — no indexed memory access |
| 2 | `"Unsupported indirect pointer argument"` | Generic-pointer kernel args (`ptr<UniformConstant>`) | Fatal — blocks struct-based args |
| 3 | `"variables must have Private FunctionStorage class"` | Kernel-scope `OpVariable` without `Private` storage class | Medium — forces Private on all local vars |
| 4 | Kernel silently drops when `LocalSize ≥ 8` + `OpAtomicIAdd` | Workgroup-level atomics with ≥8 work items | Medium — atomic fallback needed |
| 5 | **`OpFMul`(131) → `OpFSub`(130), `OpFDiv`(132) → `OpFMul`(131)** | FP multiply and divide in the OpenCL kernel model | **Critical — hand-authored SPIR-V cannot do real math** |

### Bug #5 evidence

Exact-input single-op probes:

- `FMul(a,b)` produces `a - b` for every tested f32 pair.
- `FDiv(a,b)` produces `a * b` for every tested f32 pair.
- `FAdd(a,b)` is correct.

Mirror-confirmed via composite kernels:

- dot16 (K=16): `Σ(a × b)` produces `Σ(a − b) = 0.0380020142`.
- silu: `x / (1 + exp(−x))` produces `x × (1 + exp(−x))`.

**Conclusion:** the bug is in IGC's OpenCL/SPIR-V frontend's FP opcode
lowering, not in the hardware's FP units. Toolchain-produced bytecode
(DPC++/icpx → SYCL runtime → Level Zero) is unaffected and executes FP
mul/div correctly. The hardware FP units are fine.

### SPIR-V language ceiling

- Driver reports **SPIR-V max 1.0**. Loading a 1.2 header hangs
  `zeModuleCreate` indefinitely (observed).
- `SPV_INTEL_bfloat16` (capability 6115) is advertised in extensions but
  `OpConvertBF16ToFINTEL` / `OpConvertFToBF16INTEL` are rejected with
  "Invalid opcode" at module load. The `<<16` emul-widen is the reliable
  BF16→f32 path on this driver.
- NPU (`Intel(R) AI Boost`) exposes **no SPIR-V support at all**
  (`spirvVersionSupported = 0`).

---

## 3. Kernel-Authoring Workflow (the canonical pattern)

This is the exact workflow the probe uses to add and verify a GPU kernel —
the pattern to copy for every SmolLM/Qwen kernel that follows. The division
of labor is fixed: **the DPC++ runner owns all device code; the .NET probe
is pure transport + gate** (writes fixtures, spawns the runner, reads
results, gates against the production CPU kernels). No host-side math, no
.NET-side kernel logic — device code lives only in `sycl_runner.cpp`.

### 3.1 The loop, end to end

```
        ┌──────────────────────────────────────────────┐
        │  write kernel in Sycl/sycl_runner.cpp        │
        │  (one Run<Mode> function per kernel)         │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │  Sycl/build.cmd                              │
        │  icpx -fsycl -O2 (VsDevCmd + setvars)        │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │  run.cmd <mode> <in.bin> <out.bin> [dim0] [dim1]                  │
  │  raw u16 BF16 in ──► device widen (<<16) ──► kernel ──► raw f32   │
  └───────────────┬─────────────────────────────────────────┬─────────┘
                  │                                         │
                  ▼                                         ▼
   .NET SyclLeg writes fixtures          runner prints (stdout contract):
   (WriteBf16Concat), spawns runner,     "device: <name> (<driver>)"
   reads out.bin f32s + TIME lines       "  dot16 = ..."        (dot only)
                  │                      "TIME <mode> = <us>"  (per kernel)
                  ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  KernelGate: |leg − cpuNivara| ≤ 1e-6 + 1e-5·|cpuNivara|         │
   │  exit code = failed kernels (0 = all gates pass)                 │
   └──────────────────────────────────────────────────────────────────┘
```

### 3.2 Runner API contract (stable — the .NET side depends on it)

| Piece | Contract |
|---|---|
| argv | `<mode> <in.bin> <out.bin> [dim0] [dim1]`, `mode ∈ {dot16, gemv, silu}` |
| in.bin | raw little-endian `uint16_t` = `System.Numerics.BFloat16` bit patterns. dot16: `a[16]` then `b[16]`; gemv: `W[rows×cols]` row-major then `x[cols]`; silu: `x[n]` |
| out.bin | raw little-endian `float` results. dot16: 1 × f32; gemv: rows × f32; silu: n × f32 |
| exit | `0` success · `1` device/queue/kernel failure · `2` bad args / I/O |
| stdout | `device: <name> (<driver ver>)`, then per-kernel `TIME <mode> = <µs>` (dot16 also prints its value). `SyclLeg.ParseTimeUs` greps `TIME <mode> =` — keep that line parseable |
| stderr | human diagnostics only (echoed by `SyclLeg` on failure) |

### 3.3 Anatomy of a kernel function (gemv is the template)

```cpp
int RunGemv(sycl::queue& q, const std::vector<uint16_t>& input, int rows, int cols, std::vector<float>& out) {
    if (cols <= 0 || rows <= 0)                                // 1. validate args — fail fast
        return Fail("gemv: bad dims", 2);

    auto w = sycl::malloc_shared<uint16_t>(rows * cols, q);    // 2. USM shared = zero-copy on iGPU
    auto x = sycl::malloc_shared<uint16_t>(cols, q);
    auto y = sycl::malloc_shared<float>(rows, q);
    if (!w || !x || !y)
        return Fail("gemv: device allocation failed", 1);

    std::memcpy(w, input.data(), rows * cols * sizeof(uint16_t));  // 3. explicit host→device copy
    std::memcpy(x, input.data() + rows * cols, cols * sizeof(uint16_t));

    int rc = 0;
    try {
        double us = TimedBest([&] {                            // 4. submit + wait, timed best-of-3
            q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> r) {  // one work item per row
                    float acc = 0.0f;
                    for (int k = 0; k < cols; ++k)             // serial K loop = deterministic order
                        acc += WidenBf16(w[r * cols + k]) * WidenBf16(x[k]);
                    y[r] = acc;
                });
            });
            q.wait();                                          // always sync before reading
        });
        out.assign(y, y + rows);                               // 5. read results off device
        std::fprintf(stdout, "TIME gemv = %.1f us\n", us);     //    + the TIME line for the perf harness
    } catch (const sycl::exception& e) {
        rc = Fail(("gemv kernel failed: " + std::string(e.what())).c_str(), 1);
    }

    sycl::free(w, q); sycl::free(x, q); sycl::free(y, q);      // 6. free USM — always
    return rc;
}
```

`main` dispatch by argv `mode` and hands the same `sycl::queue` (created once
via `sycl::queue q(sycl::gpu_selector_v)`) to every `Run<Mode>` — a new queue
per kernel would re-initialize the device and skew the timing rows.

### 3.4 Rules — this is how you write a kernel here

1. **One `Run<Mode>` function per kernel**, dispatched from argv in `main`.
   A kernel owns its buffers; no cross-kernel global state.
2. **BF16 stays raw `u16` on the wire; widen in-register** (`WidenBf16` =
   `uint32(bits) << 16` bitcast to f32). Never host-widen ahead of the kernel —
   that is exactly the production widening cost (and 2× bandwidth) the GPU path
   exists to avoid.
3. **One work item per output row/element, serial K loop inside.** The fixed
   serial order keeps run-to-run determinism; the CPU SIMD kernel has its own
   reduction order, and the tolerance gate (`1e-6 + 1e-5·|cpu|`) covers the
   honest rounding difference.
4. **USM shared for everything** (iGPU = unified memory, zero-copy). Still
   `memcpy` in and read `out` explicitly — it keeps the host/device contract
   obvious and stays portable to discrete GPUs.
5. **`q.wait()` after every submit** before touching results. The probe runs
   one kernel per process; no pipelining, no in-flight reads.
6. **`sycl::free` every USM allocation** — leak-free even in a probe.
7. **Wrap submit+wait in `TimedBest` and print `TIME <mode> = X us`.** The
   first submit pays the JIT compile; best-of-3 yields steady state, and the
   perf harness parses that line (never renumber/reword it without updating
   `SyclLeg.ParseTimeUs`).
8. **Catch `sycl::exception` per kernel**; map errors to exit `1` (device)
   vs `2` (args/I/O), distinct codes the .NET side reports.
9. **`single_task` for dots (K ≤ 256); `parallel_for(range<1>(n))` for
   elementwise/row-parallel.** No `sub_group`/SIMD-lane heroics until a
   measured perf need appears — at SmolLM shapes plain work items saturate
   the EU array.
10. **Simple, deterministic, verifiable first; optimize second.** Gates come
    before speed. Every shipped kernel gets a README gate-table row + a
    [docs/SYCL.md](SYCL.md) lessons entry.

### 3.5 Adding a new kernel, end to end

1. Add `Run<Mode>` + argv dispatch in `Sycl/sycl_runner.cpp` (copy
   `RunGemv` — it is the template).
2. Rebuild: `tests/Nivara.GpuProbe/Sycl/build.cmd`.
3. Standalone smoke test, no .NET involved: write a small `in.bin`, run
   `run.cmd <mode> in.bin out.bin <dim0> [dim1]`, check `out.bin` bytes,
   exit code, and `device:`/`TIME` lines.
4. Wire the transport in `SyclLeg.RunLeg`: fixture write via
   `WriteBf16Concat` → `RunKernel` → f32 readback + `TIME` parse.
5. Gate it: `dotnet build tests/Nivara.GpuProbe -c Release` then
   `dotnet run -c Release --project tests/Nivara.GpuProbe -- kernels`
   (exit 0 = all gates pass; new kernel row appears in the gate table).
6. Update the README (kernel row, Files), add the perf/timing row, and
   commit the probe change.

---

## 4. SYCL Kernel Patterns for BF16 on Intel Arc

### BF16 data representation

BF16 arrives as raw `uint16_t` bit patterns (identical to
`System.Numerics.BFloat16` in .NET). The device-side widen is:

```cpp
inline float WidenBf16(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;  // move bits to f32 upper half
    float f;
    std::memcpy(&f, &u, sizeof(u));
    return f;
}
```

This is lossless (BF16 is the upper 16 bits of f32) and matches the host-side
`SafeTensorsLoader.WidenBf16ToF32` used in the production Nivara kernels.

### USM (Unified Shared Memory)

Use `sycl::malloc_shared` for zero-copy host↔device transfers on integrated
GPUs (where host and device share physical RAM):

```cpp
auto ptr = sycl::malloc_shared<float>(count, q);
// host can read/write ptr directly after q.wait()
sycl::free(ptr, q);
```

Buffers/accessors work but add indirection; USM is simpler for the probe's
file-I/O pattern and matches llama.cpp's SYCL backend.

### Queue creation

```cpp
sycl::queue q(sycl::gpu_selector_v);
```

`gpu_selector_v` selects the Arc 140T via the Level Zero backend. On machines
with both Arc iGPU and NPU, the GPU is picked (NPU has no SPIR-V support).

### Kernel shapes that work

All three production SmolLM shapes run correctly on the Arc 140T via DPC++:

| Kernel | Shape | Dispatch | Notes |
|---|---|---|---|
| dot16 (K=16) | `single_task`, serial K loop | 1 work item | Bit-exact vs CPU serial reduction |
| gemv (1536×576) | `parallel_for(rows)`, serial K loop | 1536 work items | Indexed memory works; CPU SIMD reorder produces different rounding |
| silu (576) | `parallel_for(n)`, elementwise | 576 work items | `sycl::exp` available; 4 ULP worst vs CPU `sigmoid×x` |

### Memory layout

Row-major flat arrays, matching .NET's `BFloat16[]` layout. No padding.
GEMV: `W[rows × cols]` row-major, `x[cols]`, output `y[rows]`.

---

## 5. Runtime Gotchas

### DLL resolution (STATUS_DLL_NOT_FOUND)

When the .NET probe spawns `sycl_runner.exe` directly, the child process
inherits .NET's PATH, which does **not** include the oneAPI runtime DLLs:

- `sycl8.dll` (SYCL runtime) — `C:\Program Files (x86)\Intel\oneAPI\compiler\2025.1\bin\`
- `ur_loader.dll` (Unified Runtime loader) — same dir

Result: `STATUS_DLL_NOT_FOUND` (exit code `0xC0000135 = -1073741515`).

**Fix:** wrap the launch in `run.cmd`:

```bat
@echo off
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
"%~dp0sycl_runner.exe" %*
exit /b %ERRORLEVEL%
```

Then spawn via `cmd /c run.cmd <args>` from .NET. The `setvars` call adds
the runtime dirs to PATH in the child's environment.

> **For production:** avoid the subprocess. P/Invoke directly into the SYCL
> runtime or build a native `.dll` that the .NET host loads via `NativeLibrary`.
> The subprocess path is probe-only overhead (~50ms per kernel launch).

### icpx host compiler

`icpx` uses the MSVC linker under the hood. If `cl.exe` is not on PATH
(VsDevCmd not called), linking fails with missing `LIBCMT.lib` or similar.
Always call VsDevCmd before setvars (see §1).

### Kernel launch overhead

Each `sycl_runner.exe` invocation creates a queue, compiles/loads the kernel
(if not cached), and dispatches. For a single kernel this is ~5–50 ms
dominated by process startup and SYCL runtime initialization, not the kernel
execution itself. For production use, keep the queue alive across multiple
kernel launches (batch decode tokens).

---

## 6. Performance Notes

### CPU vs GPU kernel-only timing (measured, GpuProbe `kernels` mode, Arc 140T)

**Measured figures live in the probe README's consolidated side-by-side table**
([tests/Nivara.GpuProbe/README.md](../tests/Nivara.GpuProbe/README.md) → "Backend
comparison — the numbers, side by side"), which gathers the SYCL, DX12 and OpenVINO numbers into one view; this
doc's range for this leg was dot16 3.4/13–53 µs, silu 33.5/12–34 µs, gemv
197.8/170–198 µs — SYCL is launch-bound at K=16, ~2× CPU on silu, ~23× CPU on
gemv. The **23.1× gemv result is this leg's decisive number**: every decode
token in SmolLM is dominated by [1536×576]·[576] GEMVs, and the GPU runs each
one ~23× faster than the production CPU kernel at these exact shapes.

Measured with a best-of-3 per kernel (post-JIT steady state, submit+wait only,
excluding process/queue setup — the SYCL runner prints `TIME` lines). CPU
times are the production Nivara kernels (`MatMulTransposedB`, `Activation.Silu`)
via `CpuLeg.ComputeLeg`, also warmed once before timing.

### Interpreting the numbers

- **dot16 is launch-bound.** The GPU wins only when the work per kernel is
  large enough to amortize setup. At K=16 nothing does — this kernel exists
  only as the smallest correctness probe.
- **silu and gemv are GPU wins even at SmolLM's small dimensions.** The 23×
  gemv result is the decisive number: every decode token in SmolLM is
  dominated by [1536×576]·[576] GEMVs, and the GPU runs each one ~23× faster
  than the production CPU kernel at these exact shapes.
- **The subprocess overhead (~50 ms/launch) is not part of these numbers.**
  Production will load a native SYCL `.dll` (queue as a long-lived singleton,
  kernels batched per token), eliminating process startup entirely.

### When GPU wins

The GPU advantage is already decisive at SmolLM's dimensions. For Qwen-1.5B
(hidden 14334, intermediate 896) the matmuls are ~25× larger — the ratio
should hold or improve as the EU array saturates. Batching multiple decode
tokens per launch widens the gap further.

### Production recommendation path

1. **Native SYCL library** (`.dll` built with `icpx -fsycl`) loaded via
   `NativeLibrary.Load` from .NET. Keep the SYCL queue as a long-lived
   singleton; batch kernel launches per token.
2. **Alternative:** the oneAPI Level Zero adapter exposes a C API
   (`zeModuleCreate` + `zeKernelCreate`) that can be P/Invoked directly.
   DPC++-compiled modules produce fat binaries that the L0 adapter loads
   transparently. This bypasses `sycl8.dll` but requires managing memory
   allocation and command lists manually.
3. **DX12/HLSL** remains a viable fallback if SYCL runtime footprint is
   unacceptable (e.g., embedded deployment). HLSL `cs_5_1` compiles to DXBC
   through the inbox `d3dcompiler_47.dll`, bypassing IGC entirely.

---

## 7. Lessons for SmolLM / Qwen Kernel Implementation

### Kernels to implement

SmolLM-135M and Qwen-1.5B share the same transformer architecture. The
compute-critical kernels (in order of perf impact):

| Kernel | SmolLM shape | Qwen-1.5B shape | Notes |
|---|---|---|---|
| GEMV (linear) | [576×64] × [64] | [896×14334] × [14334] | Dominates decode; all linear layers |
| SiLU activation | [1536] | [4304] | FFN activation; elementwise |
| RMSNorm | [64] per head | [14334] per head | Per-row normalization; needs row-sum + rsqrt |
| Softmax | [num_heads, head_dim] | [num_heads, head_dim] | Per-row exp + sum; numerically sensitive |
| Attention QK^T | [seq_len × head_dim] × [head_dim] | same | Dot products; K grows with context |
| Batch GEMV (prefill) | [seq_len × 64] × [64] | [seq_len × 14334] × [14334] | Prefill phase; larger matrices |

### BF16 weight loading

Weights are BF16 in the `.safetensors` file. Two strategies:

1. **Load BF16, widen on device** (current probe pattern): keeps file I/O
   small; the `<<16` widen is trivial on GPU. Best for bandwidth-limited
   scenarios (integrated GPU with shared RAM).
2. **Widen to f32 at load time** (what `SafeTensorsLoader.WidenBf16ToF32`
   does on CPU): wastes 2× memory but avoids per-kernel widen overhead.
   Acceptable if RAM budget allows.

Recommendation: **strategy 1 (on-device widen)** for the SYCL path. The
`<<16` is a single instruction per element and the bandwidth savings from
keeping BF16 in memory are significant on integrated GPUs.

### Fused kernels

For SmolLM's small dimensions, consider fusing:
- **GEMV + SiLU** (the FFN first linear + activation) — one kernel launch
  instead of two.
- **RMSNorm + QKV split** — fuse the normalization with the three linear
  projections.

Fusing reduces launch overhead, which dominates at SmolLM's dimensions.

### Numerical gotchas

1. **Softmax overflow:** use the standard `exp(x - max)` trick. Never compute
   `exp(x)` without subtracting the row maximum first.
2. **RMSNorm rsqrt:** compute `1/sqrt(mean + eps)` in f32; avoid `pow(-0.5)`
   which is slow and imprecise on some GPUs.
3. **BF16 accumulation:** Xe2 DPAS accumulates BF16 products in f32 (the
   hardware's native model). For reductions longer than ~256 elements, consider
   staging in f32 to avoid precision loss in the accumulation.
4. **Attention masking:** for causal attention, mask future positions with
   `-inf` before softmax. The `sycl::ext::intel::math::exp` or `sycl::exp`
   handles `-inf → 0` correctly.

### Thread/work-item sizing

- **Elementwise kernels** (SiLU, RMSNorm, masking): 1 work item per element.
  `parallel_for(range<1>(n))`.
- **GEMV:** 1 work item per output row, serial K loop. This is the pattern
  that works on Arc; avoid 2D tiling for now (the probe found indexed memory
  works fine via DPC++ but the GPU's EU count limits parallelism for small N).
- **Dot products:** single task with serial loop for K ≤ 256; `parallel_for`
  with atomic reduction for K > 256.

---

## 8. Build Reproducibility

The exact toolchain versions that produced verified-correct results on the
Arc 140T:

```
icpx:    Intel(R) oneAPI DPC++/C++ Compiler 2025.1.3.8
Driver:  32.0.101.8826
Device:  Intel(R) Graphics (1.15.37858) — Arc 140T (8086:7DD1)
Runtime: SYCL runtime on Level Zero (sycl8.dll)
OS:      Windows 10.0.26200
.NET:    11.0.0
```

If re-testing on a newer driver or different Arc SKU, run:

```bat
dotnet run -c Release --project tests/Nivara.GpuProbe -- kernels
```

This runs the CPU gold leg and the SYCL leg in one pass, reporting per-kernel
gate results. Exit code 0 = all gates pass.

---

## 9. Reference Links

- [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html)
- [Intel DPC++ Programming Guide](https://intel.github.io/llvm-docs/DPCPPIndex.html)
- [Level Zero API Specification](https://spec.oneapi.io/level-zero/latest/core/api.html)
- [llama.cpp SYCL backend](https://github.com/ggerganov/llama.cpp/blob/master/ggml-sycl.cpp) — reference implementation for SYCL on Intel Arc
- [BF16 format (Wikipedia)](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
- [SmolLM-135M architecture](https://huggingface.co/HuggingFaceTB/SmolLM-135M) — 12 layers, 64 hidden, 1536 intermediate, 6 heads
- [Qwen-1.5B architecture](https://huggingface.co/Qwen/Qwen1.5-0.5B) — 24 layers, 14334 hidden, 896 intermediate, 16 heads
- Series: [docs/SPIRV.md](SPIRV.md) · [docs/SYCL.md](SYCL.md) · [docs/DX12.md](DX12.md) · [docs/OPENVINO.md](OPENVINO.md) · [docs/ILGPU.md](ILGPU.md) — the GPU-backend case-study series
