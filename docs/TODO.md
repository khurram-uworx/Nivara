# Plan: Silk decode host/pipeline overhead audit (#1468)

Issue: https://github.com/khurram-uworx/Nivara/issues/1468 (low, performance)
Follow-up to #1453 / PR #1465. Decode profiling reports steady-state warmup and profiled attribution separately.

## Problem

Steady-state decode (`ForwardCachedArgmax`, `ProfilingEnabled == false`) has no explicit `Finish()`, but still has avoidable host/pipeline overhead:
- per-token blocking uploads (token id allocates a new 1-int array every token; pos id upload)
- RoPE table growth by one row per decode token -> per-token Release/Allocate + 2 blocking uploads
- per-token logits buffer Allocate/Release (vocabSize floats)
- argmax partial readback is 2 blocking downloads (small, but unavoidable in current design)
- no `CL_QUEUE_PROFILING_ENABLE` diagnostic path
- queue created with `CommandQueueProperties.None`

## Proposed changes

### A. Steady-state sync/overhead removal (both runners, kept in lockstep)
1. `SilkQwenGpuRunner` / `SilkSmolLMGpuRunner.ForwardCachedArgmax`:
   - Reuse a persistent 1-element host buffer `tokenIdHost` for the token-id upload instead of `new int[] { tokenId }`.
   - Reuse a persistent logits device buffer (`logitsBuf` field) instead of Allocate/Release per token. Ensure capacity >= vocabSize once.
   - Pre-size RoPE tables to `seqCap` on first `EnsureRopeTables` call so decode no longer grows row-by-row (eliminates per-token alloc/upload churn). Change `EnsureRopeTables` to allocate `seqCap` rows when growing (still early-returns when already sufficient).
2. Document the steady-state sync contract (comment block above `ForwardCachedArgmax` in both runners + note in docs/SILK.md) listing each remaining sync point and justification.

### B. Optional queue-profiling diagnostic mode
3. `SilkRuntime`: add constructor parameter `bool enableQueueProfiling = false` (default false, backward compatible). When true, create the command queue with `CommandQueueProperties.ProfilingEnable`. Store and expose `QueueProfilingEnabled`. Add a helper `TryGetEventDurationMs(nint eventHandle, out double)` using `clGetEventProfilingInfo` (Queued/Submit/Start/End). When enabled, `Run` captures the dispatch event into `_lastKernelEvent` (releasing the previous) so callers can read device-side timing after `Finish()`. Default path unchanged (event param stays null).
4. `SilkQwenSession` / `SilkSmolLMSession`: thread `enableQueueProfiling` through constructors and `Open` overloads (default false); expose `QueueProfilingEnabled`.

### C. Measurement / verification
5. Keep steady-state tok/s as primary metric (already the case in Qwen CLI; no change needed to reporting, just re-verify labels).

## Verification steps
- Build `Nivara.Extensions` and `Nivara.Tests`.
- Targeted tests: Silk parity tests (Qwen + SmolLM, bf16), SilkMlpBlockDecompositionTests, SilkDecodeAttentionParityTests, SilkRuntimeLifecycleTests.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- Doc snippet gate: `EveryGatedBlock_CompilesWithoutErrors` (Docs namespace tests) if docs/SILK.md changes.
- Human-run: SmolLM/Qwen synthetic Silk profile (steady-state tok/s before/after). Agent does not run benchmarks.

## Planned commits
1. docs: plan #1468 in TODO.md
2. perf(silk): reuse host token-id buffer + persistent logits buffer in ForwardCachedArgmax (Qwen + SmolLM)
3. perf(silk): pre-size RoPE tables to seqCap to remove per-token growth (Qwen + SmolLM)
4. docs(silk): document steady-state decode sync contract
5. feat(silk): optional queue-profiling diagnostic mode in SilkRuntime (+ sessions)
6. docs: remove TODO.md — plan executed (after G2)

## Blast radius
- `SilkQwenGpuRunner.cs`, `SilkSmolLMGpuRunner.cs` (ForwardCachedArgmax, EnsureRopeTables, fields)
- `SilkRuntime.cs` (constructor, queue creation, Run, new property/helper, new field)
- `SilkQwenSession.cs`, `SilkSmolLMSession.cs` (constructors/Open/property)
- `docs/SILK.md` (documentation)
- Tests: existing parity tests cover correctness; no new test files required, but queue-profiling flag should be exercisable (optional small test).

## GitHub issues log
(none yet)
