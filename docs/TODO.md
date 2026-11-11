# Plan: #1119 CLI thread knobs and GEMV probe

Issue: https://github.com/khurram-uworx/Nivara/issues/1119

`WORK.md` (repo root, untracked) holds the agreed CLI contract. This file is the execution plan. Delete both when the plan is concluded.

Reminder: if a step surfaces deferred work, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record it here. Do not hold it in memory.

## Problem

`--threads` is required with `--blas` and is the Nivara row-block count, while `OpenBlasGemmProvider.MatMul` hardcodes `openblas_set_num_threads(1)`. There is no way to try an OpenBLAS pool (`--blas --threads 1 --blas-threads 8`) without a code change.

Part 2's non-GEMM routines were assessed. Only GEMV (`TensorsHelper.MultiplyCore`, `bTransposed && aRows == 1`) is a plausible bind. Dot, norm, axpy, scal, copy, trsm, and LAPACK stay on the BCL. `IGemmProvider` stays GEMM-only (#133 / #444). A sibling `IGemvProvider` is the name if GEMV is bound, and only after a measured median ratio of at least 2×. This session does not run that probe and does not bind.

## Proposed changes

### CLI and provider (this branch)

- `--threads N`: Nivara row-block `Parallel.For` count. Optional. Default 1. Rejected without `--blas`. Missing, zero, or non-integer rejected. The message names the flag.
- `--blas-threads N`: `openblas_set_num_threads`. Optional. Default 1. Same rejects. Not a rename of `--threads`.
- `--blas` alone registers 1 row-block and 1 OpenBLAS thread. It no longer requires `--threads`.
- Both greater than 1 is allowed. Print one line that this is two pools. Do not reject it. Not the default.
- Startup line: `BLAS: enabled (N row-blocks, M OpenBLAS threads)`.
- `UseBlas(int)` stays the row-block count. Add an optional OpenBLAS-thread argument. Do not reinterpret `BlasOptions.ThreadCount`. Add `OpenBlasThreads` (default 1) and fix the comment that calls `ThreadCount` the OpenBLAS pool.
- `OpenBlasGemmProvider` calls `SetNumThreads(openBlasThreads)` outside the `Parallel.For`, instead of the hardcoded 1.
- `IGemmProvider.MatMul` does not grow a thread argument. Native code still runs only after `--blas` registers the provider.

### Probe (this branch, not run here)

- `--cpu-gemv` in `tests/Nivara.PerformanceTests`. No new project. No temp harness: `--cpu-gemm` already owns the OpenBLAS binding; this is a sibling mode, not a GEMM-shape case (M=1 was excluded from that probe).
- `cblas_sgemv` on the probe's own `OpenBlasInterop` only. Not on the Extensions binding, and not dispatched from `MultiplyCore`.
- Shapes: M=1, K in {384, 768, 1024, 4096}, N at that hidden width and one vocab-sized N (30522). Legs: managed `GradKernels.MatMulTransposedB` (the `aRows == 1` path), `sgemv` at 1, 4, and 8 OpenBLAS threads.
- Gate: `maxAbs` against a double-precision dot truth `<= 1e-3`. Timing is 1 warmup and 20 interleaved rounds, median and min printed. No timing assertion.
- Layout: weights are row-major `[N x K]`. `cblas_sgemv(RowMajor, NoTrans, M=N, N=K, lda=K, incX=incY=1, alpha=1, beta=0)`.
- If the DLL is missing, print the reason and exit 2. A numeric failure exits non-zero. Do not run this command in this session.

### Not in this branch

- No `IGemvProvider`. No `MultiplyCore` dispatch change. Half and BFloat16 stay on `NarrowFloatKernels`.
- No dot, norm, axpy, scal, copy, trsm, or LAPACK binding.
- No new sweet-spot numbers. No new `docs/*.md` file. No new `File.cs:NN` or symbol citation (coverage pins stay put).

## Grounding (G1)

Microsoft Learn: `[LibraryImport]` is the .NET 7+ P/Invoke path; it has no `CallingConvention` (use `UnmanagedCallConv` only if the platform default is wrong). The existing binding is x64-only and passes `ref float` plus `int` dimensions (LP64). `cblas_sgemv` follows that declaration. Do not switch the probe to `DllImport` or to `Span` marshalling.

Code-memory / source: the only production GEMV branch is `TensorsHelper.MultiplyCore` when `bTransposed && aRows == 1`. `LlamaDecoderBlock` decode calls it with `aRows == 1`. BERT sentence forwards pass `seqLen` and stay on `IGemmProvider`. `UseBlas(int)` callers are the CLI and `BlasSmokeTests`. Adding an optional second argument is source-compatible because `UseBlas(string?, int)` still requires the path argument, so `UseBlas()` is not ambiguous.

No new decision. The design was cleared in the session that produced `WORK.md`. The bind remains gated on a human-run probe; that is the issue's rule, not an open question.

## Blast radius

- `samples/NivaraInference/Program.cs` — flag parse, help, registration. `InferenceCliArgumentTests` pins the rejects. `--blas` alone now reaches `UseBlas` and then the DLL check, so the argument gate must not happy-path it.
- `NivaraBlas`, `BlasOptions`, `OpenBlasGemmProvider` — registration and `SetNumThreads`. `BlasSmokeTests` (DLL present) checks parity and that a non-default pool is what `openblas_get_num_threads` reports after a matmul, then restores 1. Process-global; the cleanup is required.
- `tests/Nivara.PerformanceTests` — new mode and one `LibraryImport`. No production caller.
- Docs: inference README, `RELEASING.md`, a status note on `docs/BLAS-MKL.md`, performance-tests README. No citation-pin change.

## Verification

- `dotnet build` the touched projects.
- Targeted `InferenceCliArgumentTests` and `BlasSmokeTests`. Not the full suite. Not `--cpu-gemv`.
- `dotnet format Nivara.slnx whitespace --verify-no-changes` before the last code commit.

## Planned commits

1. `docs: plan #1119 CLI knobs and GEMV probe in TODO.md`
2. CLI knobs, provider thread argument, reject tests, README and releasing notes.
3. `--cpu-gemv` probe and its README section. Follow-up issue for the bind, recorded below.
4. Remove this file and `WORK.md` after G2.

## GitHub issues log

- [ ] (pending) GEMV bind — `IGemvProvider` only if `--cpu-gemv` shows a median ratio of at least 2×. Created when the probe lands.
