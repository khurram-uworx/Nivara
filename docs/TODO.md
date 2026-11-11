# Plan: Issue #1134 — true BF16 weight path in SilkDistilBertTrainer

## Problem
`SilkDistilBertTrainer` only supports F32 and convert-on-load (BFloat16→F32) paths. Need a true BF16 path keeping matrix weights as `ushort` on device (F32 activations/gradients/moments), with BF16 backward GEMM routing and BF16-aware Adam.

## Proposed changes
1. Refactor trainer to common constructor accepting `SilkBertWeights` + flags.
2. Add true-BF16 entry points (constructors + factories) + `ToUshort`.
3. Route forward/backward GEMMs to BF16 variants.
4. Add BF16 in-place Adam kernel + launcher method.
5. Tests: smoke + export F32 compatibility.

## Verification
- Build `Nivara.Extensions`
- Run `SilkDistilBertTrainerTests`, `SilkDistilBertTrainerBf16SmokeTests`, `SilkBf16BackwardProbeTests`, `SilkKernelsCatalogTests` (ask before long runs).
- `dotnet format whitespace --verify-no-changes`.

## Planned commits
1. docs: plan #1134 in TODO.md
2. refactor(silk): common trainer constructor + narrow flags
3. feat(silk): true BF16 entry points for SilkDistilBertTrainer
4. feat(silk): route trainer GEMMs to BF16 variants
5. feat(silk): BF16 in-place Adam kernel + launcher
6. test(silk): true BF16 smoke + export F32
7. docs: remove TODO.md — plan executed

## Blast radius
- `SilkDistilBertTrainer` (partial + new partial), `SilkBertWeights` (unchanged API), `SilkKernels`, `SilkLauncher`.
- Tests under `tests/Nivara.Tests/Gpu`.

## GitHub issues log
- (none yet)
