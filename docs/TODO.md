# Plan: issue #1382 — `--blas` A/B probe in Nivara.SimdProbe

## Problem

Nivara.Extensions ships OpenBLAS-backed GEMM/GEMV providers (`OpenBlasGemmProvider`,
`OpenBlasGemvProvider`, `NivaraBlas`), but there is no standalone, reproducible A/B
probe in `tests/Nivara.SimdProbe` measuring OpenBLAS vs the managed kernel across
representative shapes. The "is it worth wiring?" question has no recorded evidence
in the SimdProbe context.

## Proposed changes

1. `tests/Nivara.SimdProbe/Nivara.SimdProbe.csproj` — add `ProjectReference` to
   `src/Nivara/Nivara.csproj` and `src/Nivara.Extensions/Nivara.Extensions.csproj`.
2. `src/Nivara/Nivara.csproj` and `src/Nivara.Extensions/Nivara.Extensions.csproj` —
   add `InternalsVisibleTo` for `Nivara.SimdProbe` (mirrors `Nivara.PerformanceTests`).
3. `tests/Nivara.SimdProbe/BlasProbe.cs` (new) — `BlasProbe.Run()`:
   - Initialize OpenBLAS via `OpenBlasInterop.TryInitialize`; return exit code 2 if absent.
   - GEMM shapes: square (256³, 512³), tall-skinny (1024×1024×64), batched (8×768×768).
   - GEMV shapes: 768×768, 4096×4096, 768×30522.
   - For each shape: seeded RNG inputs, double-precision truth, managed leg
     (`NivaraBlas.UseManaged()` + `GradKernels.MatMulTransposedB`) and BLAS leg
     (`NivaraBlas.UseBlas(threadCount:1, openBlasThreads:1)` + same call), gate both
     at `maxAbs ≤ 1e-3`, warm up both, then 20 interleaved rounds alternating the
     first-measured route; report median/min ms and `managed/blas` ratio.
4. `tests/Nivara.SimdProbe/Program.cs` — dispatch `blas` before the banner.
5. `tests/Nivara.SimdProbe/README.md` — command, methodology, measured results,
   recorded wire/don't-wire recommendation.

## Verification steps

- `dotnet build tests/Nivara.SimdProbe/Nivara.SimdProbe.csproj -c Release`
- `dotnet run -c Release --project tests/Nivara.SimdProbe -- blas` (requires lib/libopenblas.dll)
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits

1. docs: plan #1382 in TODO.md
2. chore: expose internals to Nivara.SimdProbe
3. feat: add project references to SimdProbe
4. feat: add BlasProbe mode
5. docs: wire `blas` into Program.cs and README

## Blast radius

- Nivara/Nivara.Extensions csproj: assembly attribute only (no runtime behavior change).
- SimdProbe: new mode; no production code touched. No wiring of BLAS into production paths.
- No docs/*.md gated changes except probe README (not gated).

## GitHub issues log

- (none yet)
