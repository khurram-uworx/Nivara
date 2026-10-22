# Plan: issue #817 — collapse remaining duplicate macros in SilkKernels.cs

## Problem

`src/Nivara.Extensions/Silk/SilkKernels.cs` (`Source` raw string, OpenCL translation unit)
still defines two macros twice:

- `BF16_TO_F32` — defined identically on two consecutive lines.
- `MLP_MAX_INTERMEDIATE` — defined identically in two locations (preceding `mlp_block`
  and `mlp_block_bf16`).

Same drift/maintenance hazard as #804 (duplicate `GEMM_ROW4_ACCUMULATE_BF16`).

## Proposed changes

Edit `src/Nivara.Extensions/Silk/SilkKernels.cs` only:

1. Keep one `#define BF16_TO_F32(x) as_float(((uint)(x)) << 16)` (remove the adjacent duplicate).
2. Keep the first `#define MLP_MAX_INTERMEDIATE 5120` (before `mlp_block`); the preprocessor
   keeps it in scope for `mlp_block_bf16`. Remove the second define only; leave its comment.

No macro text changes; byte-identical macro text retained.

## Blast radius

- Single host-side source string consumed by `SilkRuntime` / OpenCL compile paths.
- No public API or signature changes; no C# symbol changes.
- Only test surfaces: `SilkKernelsCatalogTests`, `--silk-kernels` performance gate.

## Verification

- `grep` counts == 1 for both macros.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Targeted: `SilkKernelsCatalogTests`
- Device G2-ish gate: `--silk-kernels` (human-run, perf class; not polled by agent).

## Planned commits

1. `docs: plan issue #817 in TODO.md`
2. `refactor(silk): collapse duplicate BF16_TO_F32 and MLP_MAX_INTERMEDIATE macro definitions`
3. `docs: remove TODO.md — plan executed` (after G2 reviews)

## GitHub issues log

- #817 — collapse remaining duplicate macros in SilkKernels.cs (this plan).
