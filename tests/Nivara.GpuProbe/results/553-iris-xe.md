# GpuProbe — Iris Xe same-host eight-leg profile (#553)

Same-host repeated profile of the eight GpuProbe legs for `dot16`, `silu` and
`gemv`, measured on **this** host per issue #553. It is the **Iris Xe half** of the
correction that #552 (Arc 140T) carries the other half of. The `README.md` decision
table, `docs/ILGPU.md`, `docs/ACCELERATION.md` and the per-leg case-study docs are
**not** edited from this file alone — those corrections land jointly, once both hosts
are measured, with the two readings side by side (tracked in #599).

## Host

| | |
|---|---|
| GPU | Intel Iris Xe Graphics |
| PCI id | `8086:9A49` |
| EU count | 96 (Tiger Lake) |
| Host CPU | 11th Gen Intel Core i5-1135G7 @ 2.40 GHz (4C / 8T) |
| Driver | 31.0.101.5333 (2024-02-21) |
| D3D12 feature level / shader model | 12_1 / 6.7 (bogus 6.8 request rejected with `E_INVALIDARG`) |
| OS | Windows 11 Pro, 10.0.26200 |
| .NET | runtime 11.0.0; SDK 11.0.100-rc.1.26425.128 |
| Power | AC only (confirmed at measurement time) |

## Method

- **One `kernels` invocation**, Release build (`net11.0`), all legs in-process.
- Each leg runs its own steady state (1 warmup + best-of-25 synchronized dispatches)
  as before; the whole leg set is then swept **5 times** with a **rotating leg order**
  so no leg is systematically first or last.
- Per cell this file reports the **median and the min–max band** across the 5 sweeps,
  not a single sample. `n=5` for every measured cell.
- Correctness is gated once (round 0) against the production Nivara CPU kernels with
  `|leg − cpu| ≤ 1e-6 + 1e-5·|cpu|`; all runnable legs pass (see below).
- Raw harness stdout: [`553-iris-xe.raw.txt`](553-iris-xe.raw.txt).

## Unavailable on this host

Stated as unavailable, **not** backfilled from the Arc 140T reading:

| leg | reason |
|---|---|
| SYCL/oneAPI | oneAPI not installed (no `icpx`, no `sycl_runner.exe`) |
| OpenVINO bf16 | no `openvino_c.dll` (runtime not installed) |
| OpenVINO f32 | same |

They remain listed as rows in the table below; an absent row and a slow row are
different facts.

## Results — median [min–max] µs per invocation (kernel-only, n=5, AC, Release)

| kernel | CPU (production Nivara) | DX12 (hand-rolled) | ILGPU (OpenCL) | Silk.NET (OpenCL) | ComputeSharp (DXIL) | SYCL/oneAPI | OV bf16 | OV f32 |
|---|---|---|---|---|---|---|---|---|
| `dot16` (K=16) | 7.1 [4.6–11.5] | 219.5 [140.9–249.0] | 78.8 [70.9–106.3] | **71.4 [66.8–77.4]** | 146.1 [128.3–152.0] | unavailable | unavailable | unavailable |
| `silu` (576) | 45.7 [27.7–3233.7] † | 182.7 [121.3–208.7] | 72.9 [70.8–76.5] | **69.2 [66.2–94.3]** | 134.6 [122.4–164.6] | unavailable | unavailable | unavailable |
| `gemv` (1536×576) | 2203.2 [1127.7–11229.6] † | 705.8 [633.5–751.9] | 859.7 [845.5–1212.0] | **486.2 [469.4–583.9]** | 574.0 [536.3–628.2] | unavailable | unavailable | unavailable |

† The CPU leg carries a single outlier in one sweep (silu 3233.7 µs, gemv 11229.6 µs)
that the median resists but the band exposes — the same CPU variance #552 records, and
the reason the CPU column is directional, not spec. The GPU columns are tightly banded.

## Correctness (round 0, vs the production Nivara CPU kernels)

| leg | `dot16` | `silu` (576) | `gemv` (1536×576) |
|---|---|---|---|
| DX12 (hand-rolled) | PASS (0.0 ULP) | PASS (576/576, worst 4.0 ULP) | PASS (1536/1536, worst 12288 ULP) |
| ILGPU (OpenCL) | PASS (0.0 ULP) | PASS (576/576, worst 4.0 ULP) | PASS (1536/1536, worst 12288 ULP) |
| Silk.NET (OpenCL) | PASS (0.0 ULP) | PASS (576/576, worst 4.0 ULP) | PASS (1536/1536, worst 12288 ULP) |
| ComputeSharp (DXIL) | PASS (0.0 ULP) | PASS (576/576, worst 4.0 ULP) | PASS (1536/1536, worst 12288 ULP) |

The 12288-ULP `gemv` diagnostic sits on a near-zero reference row
(`|diff| = 1.4e-9`, far inside the `1e-6` absolute gate), matching the other legs.
`kernels` exit code = 9 = 3 (SYCL) + 3 (OV bf16) + 3 (OV f32); every failure is an
explicitly-flagged unavailable leg, never a miscoded kernel.

## What this host's bands say (scoped to Iris Xe)

- **`gemv`**: the ordering by median is **Silk.NET (486) ≈ ComputeSharp (574) < DX12
  (706) < ILGPU (860)**. Every pair is **disjoint except Silk.NET vs ComputeSharp**,
  whose bands overlap — `[469.4–583.9]` against `[536.3–628.2]`, by ~48 µs — so by the
  rule these issues set that pair is a **tie**, not an ordering. The rest is readable
  *on this host* at these shapes, not asserted as a kernel property until it survives
  the Arc 140T comparison.
- **`silu` and `dot16`**: **Silk.NET and ILGPU are a tie** (bands overlap:
  66.2–94.3 vs 70.8–76.5 on silu; 66.8–77.4 vs 70.9–106.3 on dot16). ComputeSharp
  (DXIL) is disjointly slower than both on the two small kernels. **DX12 and
  ComputeSharp overlap there as well** (dot16 `140.9–249.0` vs `128.3–152.0`; silu
  `121.3–208.7` vs `122.4–164.6`), so DX12's own rank on the small kernels is not
  readable on this host either.
- Silk.NET is the fastest hand-authored leg, and the fastest **or tied-fastest** GPU row
  on both small kernels and on `gemv` on this host.

## Device-dependence vs the Arc 140T (for the joint correction)

Compared against the Arc 140T series figures already in `README.md`, every runnable
leg is **materially slower on the Iris Xe** — `gemv` roughly 4–6× (Silk.NET ≈486 vs
90–100; ILGPU ≈860 vs 133–140; ComputeSharp ≈574 vs 144–151). The Iris Xe is the older
and smaller part (96 EU vs 128, Tiger Lake vs Xe2), so this is the expected direction.
The question the joint correction must answer is the **ordering**, not the absolute
level: whether Silk.NET's `gemv` lead over ILGPU/DX12 and the ILGPU≈Silk.NET tie on the
small kernels **hold on both hosts** or **flip**. The sibling same-host Arc record,
[`552-arc-140t.md`](552-arc-140t.md), answers it pair by pair — that file is the input to
#599, which owns the doc wording.
