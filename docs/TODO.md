# Plan: consolidate the duplicated OpenBlasInterop binding in PerformanceTests (#1193)

## Problem

`tests/Nivara.PerformanceTests/OpenBlasInterop.cs` (~175 lines) is a near-duplicate of
`src/Nivara.Extensions/Blas/OpenBlasInterop.cs` (~281 lines). The probe copy has drifted: it lacks
`ExplicitPath`/`LoadedPath` and the `TryInitialize` hardening landed for #1171. The probe project
already references `Nivara.Extensions` and has `InternalsVisibleTo`, so it can consume the
production binding, eliminating drift.

## Blast radius

- Deleted: `tests/Nivara.PerformanceTests/OpenBlasInterop.cs`
- Edited: `tests/Nivara.PerformanceTests/CpuGemmProbe.cs` (add `using Nivara.Extensions.Blas;`),
  `tests/Nivara.PerformanceTests/CpuGemvProbe.cs` (same).
- Edited: `src/Nivara.Extensions/Blas/OpenBlasInterop.cs` (doc comment: both probes now named).
- Edited (optional, separate commit): `docs/BLAS-MKL.md` §5/§5.1 (stale "P/Invoke stays in that
  project" sentence); `tests/Nivara.PerformanceTests/README.md` (one-line note that legs share the
  shipped binding).
- No csproj/solution changes (default globbing). No pin changes expected (see Verification).
- Consumers in probe: `CpuGemmProbe.cs` (10 call sites) and `CpuGemvProbe.cs` (4). `Program.cs`
  dispatches but does not name the type.

## Proposed changes

1. Delete the probe copy; add `using Nivara.Extensions.Blas;` to both probe files.
2. Update the production binding's doc comment to name both probes.
3. (Optional) Doc honesty: correct `docs/BLAS-MKL.md` §5 item 1 and add a §5.1 bullet; add one
   sentence to the probe README. No new `docs/…md` references or `File.cs::Symbol` citations.

## Verification steps

- `dotnet build tests/Nivara.PerformanceTests/Nivara.PerformanceTests.csproj -c Release`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
  (only if docs changed) — pins must not move.
- Human-run (per probe README, agents do not run): capture `--cpu-gemm` and `--cpu-gemv` output
  before/after; header block, leg list and `gate` columns must be identical; only timing columns
  may move.

## Planned commits

1. `docs: plan #1193 in TODO.md`
2. `refactor(perf): consume Nivara.Extensions.Blas.OpenBlasInterop in probes (#1193)`
3. `docs: correct BLAS-MKL.md §5 now that the binding lives in Extensions (#1193)`
4. `docs: note probes share the shipped OpenBLAS binding in README (#1193)`
5. `docs: remove TODO.md — plan executed` (after G2)

## GitHub issues log

- [x] #1193 — consolidate the duplicated OpenBlasInterop binding in PerformanceTests (this plan)
- (No new follow-ups identified; if any surface during execution, file immediately.)

## Grounding (G1)

- IVT: `src/Nivara.Extensions/Nivara.Extensions.csproj` lines 22-24, unconditional.
- Project ref: `tests/Nivara.PerformanceTests/Nivara.PerformanceTests.csproj:12`.
- Production binding already names the probe in its doc comment (line 7).
- `TryInitialize` missing-DLL text and ILP64 rejection are identical between copies.
- Candidate path resolution (`AppContext.BaseDirectory` 5 levels up) is identical.
- Docs pins: no citation references `tests/Nivara.PerformanceTests/OpenBlasInterop.cs` (verified
  via grep over `docs/`); the file is not a scanned document, so deletion moves no pin.
" 