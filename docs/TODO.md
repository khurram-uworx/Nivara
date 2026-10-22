# Plan: Perf — unroll 16-iter TILE K-loop in Silk rms_norm_lm_head (#795)

## Problem

After #786 (manual unroll of the `GEMM_ROW4_ACCUMULATE[_BF16]` macro inner K-loop), the fused
`rms_norm_lm_head` kernel in `src/Nivara.Extensions/Silk/SilkKernels.cs` still uses its own inline
16-iteration `TILE` loop:

```opencl
for (uint k = 0; k < TILE; ++k)
{
    float av = smem[lid0 + k * TILE];
    uint bBase = TILE + lid1 * BLOCK;
    a0 += av * smem[k + (bBase + 0) * TILE];
    a1 += av * smem[k + (bBase + 1) * TILE];
    a2 += av * smem[k + (bBase + 2) * TILE];
    a3 += av * smem[k + (bBase + 3) * TILE];
}
```

This is the only remaining inline 16-iter TILE accumulation loop in the file. The OpenCL driver
fuses `acc += a * b` by default; `#pragma OPENCL FP_CONTRACT OFF` is load-bearing so that the
f32 sequence matches ILGPU/CPU (mul-then-add, ascending K). Unrolling exposes the ascending-K
mul+add sequence explicitly and removes loop-control overhead, consistent with #786.

## Scope

- Silk (OpenCL) backend only. No ILGPU, CPU, or other backend changes.
- Kernel under change: `rms_norm_lm_head` (OpenCL string in `SilkKernels.cs`).
- No BF16 variant of this kernel exists, so only the f32 form is touched.

## Proposed changes

### Commit 1 — kernel unroll
File: `src/Nivara.Extensions/Silk/SilkKernels.cs` (the `rms_norm_lm_head` kernel).

Replace the loop block with 16 ordered single-statement blocks, ascending K, mirroring #786:

```opencl
{ float av = smem[lid0 + 0 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[0 + (bBase + 0) * TILE]; a1 += av * smem[0 + (bBase + 1) * TILE]; a2 += av * smem[0 + (bBase + 2) * TILE]; a3 += av * smem[0 + (bBase + 3) * TILE]; }
...
{ float av = smem[lid0 + 15 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[15 + (bBase + 0) * TILE]; a1 += av * smem[15 + (bBase + 1) * TILE]; a2 += av * smem[15 + (bBase + 2) * TILE]; a3 += av * smem[15 + (bBase + 3) * TILE]; }
```

Operation order, indexing (`lid0`, `lid1`), and `#pragma OPENCL FP_CONTRACT OFF` are unchanged, so
the byte-identity claim holds.

### Commit 2 (optional, gated on human approval) — perf capture
Add an optional `--json-out <path>` to `SilkKernelsGate.MeasureDecodeShapeAb` writing the tile-side
median (and flat-side median as control) to JSON, mirroring `rmsnorm-before.json` shape, so the
unroll's effect is recorded in the established before/after convention. This is the only perf-harness
touch.

## Verification

1. Build `Nivara.slnx` (Release).
2. Targeted unit tests (agent-run, no long suite):
   - `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkRmsNormGemvTests"`
   - `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkGemvGeometryTests"`
3. Parity: `SilkRmsNormGemvTests` pins bit-identity of `RmsNormLmHead` vs `RmsNorm`+`Gemm`;
   `SilkRmsNormGemvTests.RmsNormGemv_SmolLmCachedHead_MatchesRmsNormThenGemm` covers the head path.
4. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
5. Measurement (human-run, minutes): SilkKernelsGate on main (before) and on branch (after); compare
   tile-side timing; document no-op if within noise.

## Planned commits

1. `docs: plan #795 unroll rms_norm_lm_head TILE loop in TODO.md`
2. `perf(silk): unroll 16-iter TILE K-loop in rms_norm_lm_head`
3. (optional) `test(silk): capture rms_norm_lm_head timing as JSON in SilkKernelsGate`
4. `docs: remove TODO.md — plan executed`

## Blast radius

- `src/Nivara.Extensions/Silk/SilkKernels.cs` — only the `rms_norm_lm_head` kernel body; the
  `Source` string content changes but no C# public surface changes.
- `src/Nivara.Extensions/Silk/SilkLauncher.cs` — unchanged (launch arithmetic identical).
- Callers: `SilkQwenGpuRunner.cs` (3 call sites) and prefill paths — all use the same kernel name,
  no edits needed.
- Tests affected: `SilkRmsNormGemvTests` (parity), `SilkGemvGeometryTests` (source assertions) —
  neither asserts the loop text, so no test edits expected; if the optional JSON commit lands, only
  `SilkKernelsGate.cs`.
- Risk: low. Operation order preserved; byte-identity expected. The only risk is a typo in the
  16-block transcription breaking compilation or parity — caught by unit tests.

## GitHub issues log

- (none yet; capture any deferred work during execution)
