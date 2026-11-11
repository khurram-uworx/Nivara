# Plan: Issue #1119 Part 1 — BERT-family eval after BLAS

Branch: `khurram/1119`. Base: `main`.

## Problem

Issue #1119 (part 1): the OpenBLAS seam (#1117/#1118) only has GEMM-isolated numbers. Its real
value is the end-to-end number, and #458 established that a large share of the ~6x PyTorch gap is
NOT GEMM. We need to wire `NivaraBlas.UseBlas()` into the BERT-family inference paths and measure
end-to-end throughput on the i5-1135G7 box, then record numbers and decide whether the seam is
worth shipping (acceptance: <2x improvement → revisit).

Part 2 (other BLAS opportunities) is documented as a comment on the issue; this branch implements
Part 1 only.

## Proposed changes

### 1. Env-var gating (no CLI surface change)
Rather than a `--blas` flag (which would touch the pinned `InferenceCliArgumentTests` CLI surface),
enable BLAS via environment variables read in `Main` before any model load:

- `NIVARA_USE_BLAS=1` (or `true`/`yes`) → call `Nivara.Extensions.Blas.NivaraBlas.UseBlas(threadCount)`.
- `NIVARA_BLAS_THREADS=N` (default 1) → thread count passed to `UseBlas`.
- Print `BLAS: enabled (N threads)` / `BLAS: disabled` / `BLAS: requested but DLL not found` to
  stdout. If requested but `UseBlas` returns false, exit 1 (so a missing DLL is never silently
  reported as "managed").

Add `using Nivara.Extensions.Blas;` (transitive via `Nivara.Samples` → `Nivara.Extensions`).

### 2. Benchmark harness additions
None required — existing `minilm`/`distilbert`/`modernbert`/`laya` benchmark modes already time
end-to-end forward passes. The env var switches the provider under them.

### 3. Documentation
- Create `docs/BLAS-BENCH.md` with the measurement template (machine + date + power state + table)
  per issue acceptance, same format as `docs/BLAS-MKL.md` §5.1.
- Update `samples/NivaraInference/README.md`? Only if a BLAS run note is added; keep minimal.

## Verification steps
- Build: `dotnet build Nivara.slnx` (Release).
- Targeted tests: `InferenceCliArgumentTests` (unchanged CLI surface → should still pass).
- Formatting: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- Manual: run the 4 models × 3 configs (managed / BLAS hybrid / BLAS 1t) and record numbers.

## Planned commits
1. `docs: plan issue #1119 part 1 in TODO.md`
2. `feat(inference): env-gated BLAS provider wiring for BERT-family benchmarks`
3. `docs: add BLAS-BENCH.md measurement template`
4. (optional) `docs: record BERT-family BLAS numbers` — only after human runs benchmarks

## Blast radius
- `samples/NivaraInference/Program.cs` — `Main` only (arg-independent early init). No CLI guards
  changed, so `InferenceCliArgumentTests` unaffected.
- `Nivara.Extensions.Blas.NivaraBlas` — already exists; only called.
- `docs/BLAS-BENCH.md` — new file.
- No changes to core `IGemmProvider`/`MultiplyCore`, model layer, or tests.

## GitHub issues log
- (none yet — Part 2 assessment posted as comment on #1119)