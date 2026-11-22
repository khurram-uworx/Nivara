# Plan: #1453 Silk SmolLM/Qwen decode fusion using synthetic-weight profiling

## Problem

Issue: https://github.com/khurram-uworx/Nivara/issues/1453

Synthetic SmolLM CPU/Silk benchmarks (64 prompt / 32 generated, BF16 storage/F32 compute, Intel iGPU) show Silk prefill ~19x faster but single-token decode ~2.3x slower than CPU, so decode dominates total. The decode path is likely dispatch/occupancy-bound for small-model shapes. We need root-cause evidence (dispatch counts/timings), at least one evaluated decode fusion/dispatch reduction behind parity tests, reproducible synthetic-weight harness/docs, and a Qwen 0.5B synthetic baseline + go/no-go fusion plan after #1454.

## Grounding (G1 - before implementation)

- microsoft-learn MCP: query OpenCL dispatch/profiling guidance and .NET Stopwatch/timing patterns relevant to kernel dispatch measurement; confirm no BCL API assumptions.
- code-memory MCP: Silk decode paths and dispatch surfaces.
  - `SilkSmolLMGpuRunner.ForwardCachedArgmax` (`src/Nivara.Extensions/Silk/SilkSmolLMGpuRunner.cs:362-415`) - per token: upload ids/pos, GatherTokEmb, per-layer RmsNormGemv(fused QKV) + 2x Rotary + 2x CopyBuffer + DecodeOproj(fused attention+O proj) + MlpBlock(fused norm+gate+up+swiglu+down+residual), final RmsNormGemv(lmhead) + ArgMaxDevice.
  - `SilkQwenGpuRunner.ForwardCachedArgmax` (`src/Nivara.Extensions/Silk/SilkQwenGpuRunner.cs:353-406`) - parallel shape.
  - `SilkRuntime.Run` (`src/Nivara.Extensions/Silk/SilkRuntime.cs:404-418`) - centralized `clEnqueueNDRangeKernel`, no dispatch counting/timing. `Finish()` at `:424-429`.
  - `SilkLauncher` (`src/Nivara.Extensions/Silk/SilkLauncher.cs`) - typed dispatch; one kernel per launcher call generally.
  - Existing profiling pattern: `SilkModernBertGpuRunner.ProfilingEnabled` + `LastPhaseMs` with Stopwatch+`runtime.Finish()` per phase (`src/Nivara.Extensions/Silk/SilkModernBertGpuRunner.cs:60-61`).
  - Synthetic-weight benchmark pattern: `SmolLM.RunSyntheticBenchmark/RunSyntheticBenchmarkSilk` and Qwen equivalents in `samples/NivaraInference/` (aligned by #1454/#1452).
- Runtime probe: verify `SilkRuntime` can expose dispatch count without changing behavior; verify Finish cost is acceptable only in profiling mode.

## Blast radius

- `src/Nivara.Extensions/Silk/SilkRuntime.cs` - add optional dispatch counting/profiling hooks (read/reset). Low risk: additive, no behavior change when off.
- `src/Nivara.Extensions/Silk/SilkSmolLMGpuRunner.cs` - add decode profiling snapshot/counters; optional per-phase timing behind flag. No change to normal decode path semantics.
- `src/Nivara.Extensions/Silk/SilkQwenGpuRunner.cs` - same profiling surface for comparability.
- `samples/NivaraInference/Program.cs` (SmolLM) and `samples/NivaraInference/Qwen.cs` - add synthetic diagnostic/profile mode reporting kernel/phase timings and dispatch counts; keep existing benchmark format.
- `tests/Nivara.Tests/...` - new parity/profiling smoke tests (CPU/Silk synthetic parity stays green; profile counters sane).
- `docs/SILK.md` / `docs/QWEN.md` - document profiling mode, reproducible commands, baseline/evidence, go/no-go criteria.
- No core `src/Nivara` changes expected. No public API break intended; profiling APIs are additive.

## Proposed changes

### A. Dispatch/profile instrumentation (root-cause evidence)

- Add lightweight kernel-dispatch counting to `SilkRuntime` (e.g. `int KernelDispatchCount`, `void ResetKernelDispatchCount()`; increment in `Run`). Keep always-cheap (int increment) or gated; document semantics (counts kernel enqueues, not copies/uploads).
- Add decode-phase profiling to `SilkSmolLMGpuRunner` and `SilkQwenGpuRunner`:
  - `bool ProfilingEnabled { get; set; }` or equivalent profile mode.
  - `SilkDecodeProfile`/snapshot: total dispatches, per-phase wall ms (embed, qkv norm+proj, rotary, cache copy, attention/oproJ, mlp, final norm+lmhead, argmax, host overhead), token count. Phase timing uses `Stopwatch` + `runtime.Finish()` only when profiling enabled.
  - `ResetDecodeProfile()` / `GetDecodeProfile()`.
- Keep normal path allocation/dispatch behavior unchanged when profiling disabled.

### B. Synthetic diagnostic harness (reproducible)

- Extend SmolLM synthetic benchmark CLI in `samples/NivaraInference/Program.cs` with a profile/diagnostic mode (e.g. `smollm --synthetic-weights profile [--silk]` or `--profile`) that runs fixed workload (warmup + N measured tokens) and prints dispatch count, per-token/per-phase timings, kernels/category summary.
- Extend Qwen synthetic benchmark in `samples/NivaraInference/Qwen.cs` with comparable profile output.
- Ensure output is easy to capture as raw text (human-run benchmarks per AGENTS).

### C. Fusion/dispatch-reduction experiment

- Use profile evidence to pick highest-value candidate. Candidates to evaluate from current decode shape:
  1. **Argmax/lmhead path**: reduce host readback/partial-argmax overhead where possible (dispatch reduction / fewer sync points) while preserving argmax parity.
  2. **Rotary+cache copy fusion opportunity**: evaluate whether rotary/copy dispatches can be reduced/fused (only if parity-safe and evidence-backed).
  3. **Per-layer launch overhead**: assess if small kernel launches dominate; consider fusion only where measurable.
- Implement at least one candidate behind parity tests. Keep existing behavior as default if risky; otherwise replace with equivalent fused path.
- Add/extend synthetic parity tests proving CPU/Silk argmax/logits parity unchanged.

### D. Qwen baseline and go/no-go plan

- #1454 aligned reporting; capture comparable Qwen 0.5B-shape synthetic CPU/Silk baseline (human-run; agent records command and consumes results after confirmation).
- Document Qwen-specific fusion/dispatch-reduction plan and explicit go/no-go criteria based on measured CPU/Silk gap and profile evidence. If shapes/kernels differ materially, record separate experiment.

### E. Docs and evidence

- Update `docs/SILK.md` with profiling mode, dispatch-count semantics, and before/after evidence fields.
- Update `docs/QWEN.md` with Qwen synthetic baseline/evidence section for #1453 if needed.
- Record root-cause evidence and before/after numbers in issue/PR (and docs where durable).

## Verification

- `dotnet build Nivara.slnx -c Release`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Ask before running tests. Targeted first:
  - `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SmolLM|FullyQualifiedName~Qwen|FullyQualifiedName~Silk"`
  - `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"`
  - `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` when docs change
- Human-run benchmarks (do not run/poll without explicit approval):
  - `dotnet run -c Release --project samples/NivaraInference -- smollm --synthetic-weights profile`
  - `dotnet run -c Release --project samples/NivaraInference -- smollm --synthetic-weights --silk profile`
  - `dotnet run -c Release --project samples/NivaraInference -- qwen --synthetic-weights benchmark`
  - `dotnet run -c Release --project samples/NivaraInference -- qwen --synthetic-weights --silk benchmark`

## Planned commits

1. `docs: plan #1453 Silk decode fusion profiling in TODO.md`
2. `feat(silk): add dispatch counting and decode profile instrumentation`
3. `feat(samples): add synthetic decode profile diagnostics for SmolLM/Qwen`
4. `perf(silk): apply highest-value decode fusion/dispatch reduction with parity`
5. `test/docs: cover decode profiling/fusion and record evidence`
6. `docs: remove TODO.md - plan executed` (only after G2 clears)

## GitHub issues log

- [ ] #1453 — Plan Silk SmolLM/Qwen decode fusion using synthetic-weight profiling (this plan)
- [ ] #1457 — Silk: chunked linear+cross-entropy for SmolLM/Qwen training (no full logits) (deferred, training fusion backlog)
- [ ] #1458 — Silk: flash-style attention forward+backward for SmolLM/Qwen training (deferred, training fusion backlog)
- [ ] #1459 — Silk: fuse SwiGLU forward+backward for SmolLM/Qwen (deferred, training fusion backlog)
- [ ] #1460 — Silk: fuse RMSNorm+residual forward+backward for SmolLM/Qwen (deferred, training fusion backlog)
- [ ] #1461 — Silk: fuse RoPE into QKV path for SmolLM/Qwen (deferred, training fusion backlog)
- [ ] #1462 — Silk: fuse LoRA epilogue base(x)+(x*A)*B*scale for SmolLM/Qwen (deferred, training fusion backlog)
- [ ] #1463 — Silk: fuse bias/activation/residual matmul epilogues for SmolLM/Qwen (deferred, training fusion backlog)

## G1 grounding notes

- Cleared by human (2026-05-23): proceed with #1453 as filed (decode fusion profiling/dispatch reduction on Silk). Native-weights training deferred; training-fusion backlog tracked as separate issues. No `release/*` artifacts in this PR.
- CLI shape confirmed: `smollm --synthetic-weights profile [--silk]` and `qwen --synthetic-weights profile [--silk]` as mode `profile`.
- Initial fusion experiment: QK-rotary single-dispatch fusion (1 kernel/layer/token saved) unless profile evidence points to a higher-value candidate. Must be behind parity tests.
- Benchmarks: human granted explicit approval to run synthetic benchmarks; run sequentially and wait for shell to be idle before next command. Capture raw outputs in PR/issue comments or temp files (not `release/*`).
- Training fusion backlog (deferred, create issues): chunked linear+CE (no full logits materialization); flash-style attention fwd+bwd (no full score matrix); SwiGLU fwd+bwd; RMSNorm+residual fwd+bwd; RoPE inside QKV path; LoRA epilogue base(x)+(x*A)*B*scale; bias/activation/residual matmul epilogues.
