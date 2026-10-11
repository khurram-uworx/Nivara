# Plan — #553 GpuProbe: same-host eight-leg profile on Iris Xe

## Problem

`tests/Nivara.GpuProbe/README.md`'s decision table mixes measurements from different
hosts into single ranges and reads orderings off them. #553 is the **Iris Xe** half of
correcting that (companion: #552, Arc 140T). The corrections are shared and must land
**jointly, after both hosts are measured** — the doc edits themselves are the one item
that cannot complete on this branch alone.

The blocking gap: the `kernels` mode currently takes each leg **once** and reports a
single best-of-25 sample per cell (`KernelGate.Run` → `PrintTimingTable`). The issue
requires every leg in one `kernels` invocation, repeated **≥5 times, interleaved**, with
per-cell **median + min–max band** — a statistic the harness cannot currently express.

## Host (captured on this machine, 2026-10-11)

- GPU: Intel Iris Xe Graphics — PCI **8086:9A49**, 96 EU (Tiger Lake, 11th-gen Core i5-1135G7)
- Driver: **31.0.101.5333** (2024-02-21); D3D12 FL **12_1**, highest shader model **6.7**
- OS: Windows 11 Pro, 10.0.26200; .NET runtime 11.0.0; SDK 11.0.100-rc.1
- Power: **AC** (confirmed)
- SYCL: **no oneAPI / no sycl_runner.exe** → unavailable on this host (recorded, not backfilled)
- OpenVINO: **no openvino_c.dll** → both OV rows unavailable on this host

## Proposed changes

### 1. `tests/Nivara.GpuProbe/Kernels/KernelGate.cs` — repeated, interleaved profile
- Sweep the whole leg set `Rounds = 5` times.
- **Rotate** the leg order each round (`Rotate(legs, round)`) so no leg is systematically
  first or last.
- Collect each round's per-leg steady-state sample; print the per-kernel table as
  **median [min–max] (n=…)** per cell, CPU included.
- Keep each leg's internal 1 warmup + best-of-25 as the gauge for a single sample; the
  median/band is the cross-run statistic.
- Gate correctness **once** (round 0) — correctness is deterministic; running the gate
  five times only floods the stdout and the exit code.
- Unavailable cells (SYCL, OV here) print `unavailable on this host` — absent ≠ slow.
- Single-leg overload unchanged (used by `sycl`/`ov`/`ilgpu`/`computesharp`/`silk`/`dx12`).

### 2. Capture the run into a committed results artifact
- New `tests/Nivara.GpuProbe/results/553-iris-xe.md`: host block, method, the full
  median+band table, and the unavailable rows stated as unavailable.
- This is the Iris Xe half of the joint doc correction; it is the input the shared doc
  edits (#552 + #553 landing together) will consume.

### 3. Doc sites — **deferred, recorded**
- `tests/Nivara.GpuProbe/README.md` (table ~L85, Reading ¶, ILGPU verdicts L19/L476/L483/L654),
  `docs/ILGPU.md` (L173/179/210/223), `docs/ACCELERATION.md` §5 item 4/12,
  `OPENVINO.md`/`DX12.md`/`COMPUTESHARP.md`.
- The issue is explicit: land these **once, after both hosts are measured**, readings
  **side by side**, orderings qualified by whether they hold on both hosts. Editing them
  from this host alone would recreate the single-host-as-general defect the issue fixes.
- Action: create a follow-up issue to land the joint doc correction when the Arc 140T
  data (#552) exists. Do **not** leave the doc sites half-updated here.

## Blast radius

- **Test-project only** (`tests/Nivara.GpuProbe`); no `src/Nivara`, no kernel changes —
  both explicitly out of scope.
- `KernelGate.Run` is called by `Program.cs` for the single-leg modes and `kernels`/`all`;
  signatures unchanged. `LegResults`/`CpuLeg` APIs unchanged.
- No NUnit test covers `KernelGate`; verification is the probe building and the modes
  running to the expected exit codes.
- Measurement conditions: Release build, AC power, single box.

## Verification

- `dotnet build -c Release` (0 warnings/errors).
- `kernels` → 5 sweeps, median+band table, exit `6` on this host (402 OV-bf16 silu +
  3 SYCL unavailable + 3 OV-f32 unavailable = 8 cells … confirm against current baseline).
- Single-leg modes (`dx12`, `ilgpu`, `silk`, `computesharp`) still run and exit 0.
- `dotnet test -c Release --filter "Category!=Performance"` — **ask before running**;
  expected untouched (no shared-library change).

## Planned commits

1. `docs: plan GpuProbe Iris Xe same-host profile (#553) in TODO.md`
2. `test(gpuprobe): repeat the kernels sweep with a rotating order; report median + band (#553)`
3. `docs(gpuprobe): record the Iris Xe eight-leg profile (#553)`
4. `docs: remove TODO.md — plan executed` (only after the joint doc correction is
   either landed or tracked in the follow-up issue)

## GitHub issues log

- [x] #599 — land the joint #552/#553 doc correction (both hosts side by side) once both
  hosts are measured. Created while working on #553; blocks the shared doc edits.
