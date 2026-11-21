# Plan: issue #1384 — opt-in --blas for NivaraIncident

## Problem
`NivaraInference` supports `--blas` (opt-in OpenBLAS GEMM/GEMV). `NivaraIncident` has `--silk`
but no `--blas`, so CPU / Silk / BLAS are not at parity across samples. BLAS wins on larger
square/tall GEMM and most GEMV, loses on small-batch (`batched-8x768`: 0.61x) and is within
noise on `hidden-4096` (1.03x). Leave BLAS opt-in only.

## Proposed changes

### 1. `samples/NivaraIncident/Program.cs`
- Add parse state: `blas`, `blasThreads`/`blasThreadsSet`, `openBlasThreads`/`openBlasThreadsSet` (defaults 1).
- Parse `--blas`, `--threads`, `--blas-threads` with validation (missing/invalid/non-positive → stderr + exit 1).
- Pre-dispatch guards: `--threads`/`--blas-threads` without `--blas` → exit 1.
- In the `streamix` case only, when `blas`: register via `Nivara.Extensions.Blas.NivaraBlas.UseBlas(blasThreads, openBlasThreads)`;
  print status or error+exit 1.
- Update `PrintUsage`.

### 2. `samples/NivaraIncident/README.md`
- Add CLI option rows + BLAS section with probe numbers + example.

## Verification
- Build sample Release.
- CLI reject checks (manual).
- Run streamix --blas.

## Planned commits
1. docs: plan #1384 in TODO.md
2. feat(samples): add opt-in --blas to NivaraIncident CLI
3. docs: document --blas in NivaraIncident README
4. docs: remove TODO.md — plan executed

## Blast radius
- Only sample CLI + sample README. No core changes. No docs pins.

## GitHub issues log
(none yet)
