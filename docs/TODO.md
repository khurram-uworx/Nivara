# TODO: Silk fused MLP block GB/s measurement (#1483)

Follow-up to #1473 / #1479. The fused single-token MLP block (`mlp_block` / `mlp_block_bf16`) is a
cached-decode hot path (RMSNorm + gate/up + SwiGLU + down-projection + residual) but has no F32
GB/s measurement; the BF16 one exists only as an optional subsection inside
`MeasureDecodeShapeAbBf16`. This plan defines one canonical bytes-moved model for both variants,
adds F32 measurement, consolidates both variants into a single decode-shape section, and documents
the model in `docs/SILK.md`.

## Problem

- `mlp_block` (F32) has no GB/s number anywhere in the `--silk-kernels` gate or `docs/SILK.md`.
- The BF16 `mlp_block_bf16` GB/s number lives inside `MeasureDecodeShapeAbBf16` as an optional
  subsection, with a bytes formula that is not first-class documented for `mlp_block` itself.
- Issue #1483 asks to agree on and document the formula before publishing numbers so before/after
  comparisons stay valid.

## Kernel bytes actually moved (from source)

`SilkKernels.cs:3778-3827` (`mlp_block`) and `:4466-4515` (`mlp_block_bf16`): single work-group
(256 lanes). Buffers: `x, gamma, Wgate, Wup, Wdown, out, residual`. Intermediate `swiglu` is
`__local float swiglu[MLP_MAX_INTERMEDIATE]` (5120 max, on-chip, not DRAM). `out[j] = acc + residual[j]`
reads `residual`, writes `out` (separate buffers). Norm SSQ pass reads `x` fully; phase 1 reads
`x[k]` again inside the gate/up loops.

## Proposed bytes-moved model (logical, lower-bound, matches #1473 convention)

### F32 `mlp_block`
```
bytes = x(F32) + gamma(F32) + Wgate(F32) + Wup(F32) + Wdown(F32) + out(F32) + residual(F32)
      = hidden*sizeof(float) + hidden*sizeof(float)
        + hidden*intermediate*sizeof(float) + hidden*intermediate*sizeof(float)
        + intermediate*hidden*sizeof(float)
        + hidden*sizeof(float) + hidden*sizeof(float)
      = (4*hidden + 3*hidden*intermediate) * sizeof(float)
GB/s  = bytes / elapsed_seconds / 1e9
```

### BF16 `mlp_block_bf16` (packed weights)
```
bytes = hidden*sizeof(float) + hidden*sizeof(float)
      + 3*hidden*intermediate*sizeof(ushort)
      + hidden*sizeof(float) + hidden*sizeof(float)
GB/s  = bytes / elapsed_seconds / 1e9
```

Assumptions/caveats (documented):
- Logical lower-bound model for comparability, not measured DRAM traffic.
- `x` is read twice internally (SSQ pass + phase-1 compute pass); counted once, matching the
  `rms_norm_gemv` convention (`docs/SILK.md` #1473/#1479). `gamma` counted once.
- `swiglu[intermediate]` is `__local` shared memory; not counted.
- `residual` is read, `out` is written (separate buffers); the issue text's "residual read+write"
  is equivalent in total byte count.
- Single-work-group fused launch (`LinearGroupSize`/256); P1/#1453 caveat applies — GB/s is not a
  promotion criterion on its own.
- No bias term. Formula kept stable across before/after runs.

## Proposed changes

1. `tests/Nivara.PerformanceTests/SilkKernelsGate.cs`
   - Add `MeasureDecodeShapeMlpBlock(silk, buffers)` measuring both F32 and BF16 on the two real
     gate/up shapes `(896,4864)` (qwen) and `(576,1536)` (smollm), with warmups/rounds,
     `Time`+`Median`, `silk.Finish()` inside the timed action, logical bytes formula, `bytes=`
     printed for reproducibility, `[single-WG fused]` tag and P1/#1453 caveat.
   - Remove the BF16 MLP subsection from `MeasureDecodeShapeAbBf16` (lines ~1671-1728) and replace
     with a one-line pointer to `MeasureDecodeShapeMlpBlock`, to avoid duplicate reporting and
     give one canonical spot.
   - Call `MeasureDecodeShapeMlpBlock(silk, buffers)` in the main method alongside the other
     decode-shape measures (after `MeasureDecodeShapeGemvAddF32`).
2. `docs/SILK.md` section 14a
   - Add `### Fused MLP block decode GB/s measurement (#1483)` subsection documenting command,
     shapes, the bytes-moved model for both F32 and BF16, caveats, and comparability rules.
   - Cross-reference the existing BF16 `#1479` measurement (now superseded/consolidated) with a
     note so there is one canonical MLP-block model.

## Verification steps

- Build: `dotnet build tests/Nivara.PerformanceTests/Nivara.PerformanceTests.csproj -c Release`
- Targeted tests: none new (perf gate is human-run/GPU). Confirm `SilkMlpBlockDecompositionTests`
  still green (F32 correctness unaffected): `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkMlpBlockDecompositionTests"` (GPU required).
- Human-run perf gate: `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels`
  (GPU required; user on AC power with iGPU/OpenCL) — confirm F32 and BF16 mlp_block GB/s lines
  appear with bytes model and caveat.
- Docs: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- DocsCoveragePins: check whether any pin (document count, citation count) is affected by the
  SILK.md edits (likely no new citation; document count unchanged).

## Planned commits

1. `docs: plan Silk fused MLP block GB/s measurement (#1483) in TODO.md`
2. `perf: add F32 mlp_block decode GB/s measurement to silk-kernels gate (#1483)`
3. `perf: consolidate mlp_block_bf16 measurement into dedicated section (#1483)`
4. `docs: document fused MLP block bytes-moved model (#1483)`
5. (if needed) `docs: revise DocsCoveragePins for #1483`

## Blast radius

- `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` — edit `MeasureDecodeShapeAbBf16` (remove MLP
  subsection), add `MeasureDecodeShapeMlpBlock`, update main call list. No production code change.
- `docs/SILK.md` — add subsection; no code citation changes (only prose + fenced? no fenced csharp
  blocks added; formulas are plain text), so DocsCoveragePins likely unchanged.
- No change to kernel/launcher/runner code; correctness path (`SilkMlpBlockDecompositionTests`)
  unaffected.

## Grounding notes (G1)

- Kernel source verified: buffers, shared-memory intermediate, single-WG launch, residual read +
  out write. Code-memory MCP used to locate `MlpBlock`/`MlpBlockBf16` symbols.
- Existing 14a convention verified in `docs/SILK.md:1142-1259` and `SilkKernelsGate.cs`.
- microsoft-learn MCP: not needed (no new .NET API; OpenCL kernel is repo-owned). No new .NET
  types involved.

## GitHub issues log

(none created yet)
