# Plan: Explore vectorized A-tile staging in Row4 GEMM kernels (#791)

## Problem
Issue #791 asks whether the A tile can also be staged with vectorized loads in the
Row4 GEMM kernels. #785/#789 vectorized only the B-tile staging (`float4`/`ushort4`
`vload4`); A is still scalar because each thread owns exactly one A element
(`a[outRow * aCols + (k0 + y)]`). Vectorizing A requires a changed thread work
distribution, which can threaten the shared-memory layout and the ascending-K
accumulation order that the byte-identity claim rests on.

## Proposed changes
Prototype A-tile vectorization in a **candidate** kernel (not the production Row4
path) before touching the production macro:

1. Add a candidate OpenCL kernel `gemm_row4_atile_vec` (and matching ILGPU kernel)
   that stages A with `vload4`/`Float4` while keeping the shared-memory layout and
   accumulation order identical to Row4. Distribution: threads with local `y` in
   `[0,4)` each load 4 consecutive K values into slots `x + (y*4 + w)*TILE`;
   threads with `y >= 4` skip A staging. B staging and accumulation unchanged.
2. Gate the candidate against ILGPU with the existing `--silk-kernels` / `--gemm`
   harnesses (byte-identity where required, else `1e-3*(1+|ilgpu|)`).
3. Measure performance (median, not best-of-N) on Laya/DistilBERT/MiniLM shapes.
4. If a measured win and byte-identity hold, promote the A-staging change into
   `GEMM_ROW4_ACCUMULATE` (f32 + BF16 variants) and remove the candidate.
   Otherwise close #791 as a measured null, documenting the result in
   `docs/ILGPU.md` §10.1.

## Verification
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet build` (Release) for `Nivara.Tests`, `Nivara.PerformanceTests`,
  `Nivara.GpuProbe`
- `SilkKernelsCatalogTests` (2/2)
- Gpu namespace tests
- `--silk-kernels` gate (all cells meet criterion)
- `--gemm` gate (all shapes within bound)
- `SilkQwenBf16ParityTests` (BF16 path)

## Planned commits
1. `docs: plan #791 in TODO.md`
2. `feat(silk): add candidate gemm_row4_atile_vec kernel with vectorized A staging`
3. `feat(gpu): mirror candidate kernel in ILGPU GemmKernels`
4. `test(gpu): wire candidate into --silk-kernels / --gemm gates`
5. Either promotion commit or docs measurement note + issue close

## Blast radius
- `src/Nivara.Extensions/Silk/SilkKernels.cs`: new kernel string + catalog entry +
  launch helper. Production Row4 macro untouched unless promotion step runs.
- `samples/Nivara.Samples/Gpu/GemmKernels.cs`: new candidate kernel method.
- `tests/Nivara.PerformanceTests/SilkKernelsGate.cs`, `GemmBenchmark.cs` /
  `GemmLegBenchmark.cs`: candidate row added to gates.
- `tests/Nivara.Tests/Gpu/SharedMemoryAllocationTests.cs`: shared-allocation
  invariant check may need candidate added.
- `docs/ILGPU.md`: measurement note if not viable.

## GitHub issues log
- [ ] #791 — Explore: vectorized A-tile staging in the Row4 GEMM kernels (this plan)
