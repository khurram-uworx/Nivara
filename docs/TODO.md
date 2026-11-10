# Plan: perf(silk) gate BF16-vs-F32 Laya narrow path on a targeted shape (#1110)

Follow-up to #1013. Add a performance-gated comparison (Category Performance, TimingGuards) of
narrow (BF16 matmul-weight) vs F32 Laya on a targeted synthetic shape to lock in the real,
measured benefit. Phase 1 only narrows matmul weights, so the expected win is small and must be
documented with a number rather than claimed.

## Problem

No perf gate exists for the Laya narrow path. Parity tests (`SilkLayaNarrowParityTests`) cover
correctness only. Throughput benefit is currently unmeasured/unclaimed.

## Proposed changes

1. Add `tests/Nivara.Tests/Gpu/SilkLayaNarrowPerfTests.cs`:
   - `[TestFixture]` with `[Category("Gpu")]` and `[Category("Performance")]`.
   - Synthetic targeted shape (Hidden=32, Heads=4, Layers=2, Vocab=64, HeadLayers=2, 64 tokens).
   - Build F32 and BF16 (narrow) tensor dictionaries via helpers mirroring `SilkLayaNarrowParityTests`.
   - Open `SilkLayaSession` for both; warm up; 7 interleaved rounds alternating measurement order.
   - Report median ms and Narrow/F32 ratio via `TestContext.Out.WriteLine`.
   - Assert only finiteness and positive medians (no speedup claim).
   - Skip on no OpenCL GPU (`Assert.Ignore`) and in non-optimized builds (`TimingGuards`).

## Verification steps

- Build: `dotnet build Nivara.slnx`.
- Targeted test on a GPU machine: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkLayaNarrowPerfTests"`.
- Formatting: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits

1. `docs: plan #1110 in TODO.md`
2. `test(gpu): add perf gate for narrow vs F32 Laya`
3. `docs: remove TODO.md - plan executed` (after G2)

## Blast radius

- New test file only; no production code changes.
- Test depends on `SilkLayaSession`, `SilkRuntime`, `ModernBertConfig`, `LayaDecisionHead`, `BFloat16`.
- GPU-gated; skipped in CI.

## GitHub issues log

- (none yet)
