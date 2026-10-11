# Plan — #552 GpuProbe: same-host eight-leg profile on Arc 140T

## Problem

`tests/Nivara.GpuProbe/README.md`'s decision table mixes measurements from **different
hosts** into single ranges and reads orderings off them as if comparable. #552 is the
**Arc 140T** half of correcting that (companion: #553, Iris Xe). The measured figures
currently in the table come from different runs on different days on this host
(`137.9 µs` series-wide vs `87.9–93.4 µs` same-session for OV bf16 `gemv`), so the
table disagrees with a same-session run about which leg wins.

The doc corrections themselves are **not** this branch's job: both issues' bodies and
the owner's execution comment are explicit that they land **once, jointly, after both
hosts are measured** — tracked in #599. Editing them from this host alone recreates the
single-host-presented-as-general defect the two issues exist to fix.

This branch is therefore: run the already-landed repeated-sweep harness on **this**
host, commit the Arc artifact, and stop. No `src/Nivara` change, no kernel change, no
doc-table edit.

## Host (captured on this machine, 2026-10-11 — pre-flight)

| | |
|---|---|
| GPU | Intel Arc 140T iGPU (reported as "Intel(R) Graphics"), PCI `8086:7DD1` (`SUBSYS_513417AA REV_03`) |
| Host CPU | Intel Core Ultra 7 255H (16C) |
| Driver | **32.0.101.8826** (2026-05-29) |
| OS | Windows 11 Pro, 10.0.26300 |
| .NET | SDK 11.0.100-rc.1.26425.128; runtime 11.0.0-rc.1.26425.128 |
| Power | **AC** (confirmed: `Win32_Battery.BatteryStatus = 2`, 99%) |
| OpenVINO | **present** — `openvino` 2026.2.1 at `%APPDATA%\Python\Python312\site-packages\openvino\libs\openvino_c.dll`, which is the harness's own default search path |
| SYCL | **absent** — no oneAPI, no `sycl_runner.exe` → row recorded as unavailable on this host |

Two of the issue's own host facts have **moved** and must be recorded as measured, not
copied: the driver is `32.0.101.8826`, not the issue's `1.15.37858`, and the OS build is
`10.0.26300`, not `10.0.26200`. The issue asks for exactly this ("confirm at measurement
time — record the actual, do not assume this"). EU count and D3D12 feature level are
read from the probe's own `list`/leg output at measurement time.

## Why the harness is carried, not rewritten

`KernelGate.Run` on `khurram/553` (PR #602) already: sweeps the whole leg set `Rounds = 5`
times with a **rotating leg order**, gates correctness once (round 0), and prints per-cell
**median [min–max] (n=…)**. The two hosts' protocols must be identical for the gap between
them to be the device rather than the method, so the Arc run uses the *same* code and the
*same* invocation. Re-implementing it here would break the comparison the two issues exist
to make.

## Proposed changes

### 1. Pre-flight (no code change)
- Confirm host at measurement time: `list` mode (L0 devices/EU count), `Win32_VideoController`
  (driver), battery status (AC), `.NET` versions — recorded in the artifact.
- Confirm OpenVINO resolves: the harness prints the device for the OV legs, or the explicit
  "openvino_c.dll not found" diagnostic. It must **not** silently produce an unavailable OV row.
- Confirm SYCL is absent and is reported as unavailable, not omitted.

### 2. Measure (no code change)
- Release build only (`Debug` compares an unoptimized host against optimized driver code).
- One `kernels` invocation, 5 interleaved sweeps, raw stdout+stderr captured
  (`2>&1`) into `tests/Nivara.GpuProbe/results/552-arc-140t.raw.txt`.
- Capture the process exit code explicitly (`$LASTEXITCODE`), not a pipeline's.
- Expected exit ≈ **405** = 402 (the honest OV-bf16 `silu` precision failure, `402/576`,
  out of scope, its own issue) + 3 (SYCL unavailable on this host). Confirm the actual
  number; a different total is a finding, not a rounding.

### 3. Artifact (only committed work on this branch)
- New `tests/Nivara.GpuProbe/results/552-arc-140t.md`, mirroring `553-iris-xe.md`: host
  block, method, the median+band table, unavailable rows stated as unavailable, the
  correctness table, and a scoped "what this host's bands say" section.
- The device-dependence section states the **open question** (does the Iris Xe ordering
  hold or flip here) and defers the answer to #599 — it must not promote a one-host
  ordering into a general claim.

### 4. Doc sites — **deferred, recorded** (not on this branch)
- `tests/Nivara.GpuProbe/README.md` (decision table ~L85, Reading ¶, ILGPU verdicts
  L19/L476/L483/L654), `docs/ILGPU.md` (L173/179/210/223), `docs/ACCELERATION.md` §5
  item 4 ("OpenVINO bf16 gemm contrast 86.8 µs" — Arc-only; the Iris Xe has no OpenVINO)
  and item 12, and `OPENVINO.md`/`DX12.md`/`COMPUTESHARP.md`.
- These land jointly in **#599** with both hosts' readings side by side. Not touched here.

## Blast radius

- **Test-project only** — `tests/Nivara.GpuProbe/results/` (one new `.md` + one new
  `.raw.txt`). No `src/Nivara`, no kernel, no harness change; `KernelGate.cs` is carried
  unchanged from `khurram/553`.
- No NUnit test covers the probe; it is a run-manually console app. Verification is the
  Release build plus the `kernels` run reaching the expected exit code.
- Merge shape: this branch stacks on `khurram/553`; its PR targets `khurram/553` (PR #602's
  branch), **not** `main`. Landing #552's artifact must not have it read as a `main` PR.

## Verification

- `dotnet build tests/Nivara.GpuProbe/Nivara.GpuProbe.csproj -c Release` — 0 warnings/errors.
- `kernels` → 5 sweeps, median+band table, `n=5` on every available cell; exit code captured
  and explained cell-by-cell.
- Single-leg sanity is not required (the change is data-only), and the full `dotnet test`
  suite is **not** run: no shared-library change. Ask before any long verification.

## Planned commits

1. `docs: plan GpuProbe Arc 140T same-host profile (#552) in TODO.md`
2. `docs(gpuprobe): record the Arc 140T eight-leg profile (#552)` (+ raw harness stdout)
3. `docs: remove TODO.md — plan executed` (only after G2 clears)

## GitHub issues log

- [ ] (pre-existing, referenced) **#599** — land the joint #552/#553 doc correction, both
      hosts side by side. Blocks the acceptance item "every doc site updated".
- [ ] (pre-existing, referenced) **#600** — make the `kernels` sweep count a `--rounds`
      argument; the runbook update rides with #599.
- [ ] (pre-existing, referenced) **#601** — `InferenceCliArgumentTests` 30 s timeout on slow
      hosts (unrelated; found while verifying #553).
- [ ] **NEW — OpenVINO bf16 `silu` precision defect (`402/576`).** Both #552 and #553 bodies
      say "its own issue", but no such issue exists (`gh issue list --search` finds none) and
      `docs/ACCELERATION.md` L148 already calls it "pre-existing and still unfiled". To file
      during execution (confirm the current failure count from this run before filing).
