# Plan: #786 — unroll GEMM inner K-loop (Silk-only)

## Problem

Silk's GEMM tile kernel accumulates `a0..a3` over a 16-iteration inner loop:

```
for (uint k = 0; k < TILE; ++k) {
    float av = smem[x + k * TILE];
    uint bBase = TILE + y * BLOCK;
    a0 += av * smem[k + (bBase + 0) * TILE];
    a1 += av * smem[k + (bBase + 1) * TILE];
    a2 += av * smem[k + (bBase + 2) * TILE];
    a3 += av * smem[k + (bBase + 3) * TILE];
}
```

The loop is data-dependent only through `smem` (already loaded); unrolling gives the
OpenCL compiler freedom to schedule loads and multiplies. Parity gate must remain
byte-identical: no operation reordering.

## Scope (revised)

- Silk only (`src/Nivara.Extensions/Silk/SilkKernels.cs`).
- No ILGPU/sample edits.
- Unroll every `GEMM_ROW4_ACCUMULATE[_BF16]` macro definition (4 occurrences).
- rms_norm_lm_head inline Tile loop is out of scope for this patch.

## Proposed change

Replace the 8-line inner `TILE` loop in each macro with 16 explicitly unrolled blocks.
Each block uses its own literal `k` in both the `av` load (`smem[x + k*TILE]`) and the
multiply (`smem[k + (bBase + w)*TILE]`), preserving ascending-K accumulation order.

## Blast radius

- `src/Nivara.Extensions/Silk/SilkKernels.cs` — all `GEMM_ROW4_ACCUMULATE` (f32) and
  `GEMM_ROW4_ACCUMULATE_BF16` (3 BF16) macro definitions.
- Downstream: every Silk GEMM kernel that expands a macro (bias/relu/gelu/qkv/add and
  BF16 variants).
- Tests: `SilkKernelsGate` (`--silk-kernels`) byte-identity + timing.
- No type dispatch or API surface change.

## Verification

1. Baseline timing saved by human: `--silk-kernels` output.
2. `dotnet build Nivara.slnx`
3. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
4. Candidate run saved by human; keep only if Silk timings improve and byte-identity holds.

## Planned commits

1. `docs: plan #786 unroll GEMM inner K-loop in TODO.md`
2. `feat(silk): unroll 16-iter GEMM K-loop in row4 macros` (single commit covering all 4 macros)
3. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none yet)
" \