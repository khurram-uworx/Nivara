# Plan — Issue #652: Pipeline Silk kernel launches (remove per-kernel clFinish)

## Problem

`SilkRuntime.Run` calls `clFinish` after every kernel dispatch, and `CopyBuffer` does the
same after every device-to-device copy. `SilkSmolLMGpuRunner` issues ~12 kernels per layer ×
30 layers = ~360 launches per decoded token, so a large fraction of per-token time is pure
synchronization overhead, independent of model size.

## Grounding (G1)

- OpenCL 1.2 semantics: the command queue is created with `CommandQueueProperties.None`,
  i.e. in-order. `clEnqueueNDRangeKernel`/`clEnqueueCopyBuffer` enqueue into the in-order
  queue; blocking uploads/downloads still complete all queued work before returning to the
  host. `clSetKernelArg` scalar/buffer arguments are captured at enqueue time.
- `SilkRuntime` already validates every `clSetKernelArg` status individually and every
  dispatch enqueue status; only the finish latency is being removed.
- Blast radius: `SilkRuntime` public API gains `Finish()` and loses the implicit blocking
  contract on `Run`/`CopyBuffer`. All callers already synchronize through blocking
  `Download`, so no runner edits are required; the only timed gate path
  (`SilkKernelsGate`) needs an explicit `Finish()` in its measured lambdas.

## Proposed changes

1. `src/Nivara.Extensions/Silk/SilkRuntime.cs`
   - `Run`: drop the trailing `cl.Finish(queue)`; keep enqueue status checking.
   - `CopyBuffer`: drop the trailing `cl.Finish(queue)`; keep enqueue status checking.
   - Add public `Finish()` that calls `cl.Finish(queue)` and throws on non-zero status.
   - Keep `Download` blocking; its blocking read still synchronizes callers.
2. `docs/SILK.md`: document that launches are now non-blocking and synchronization is via
   blocking Upload/Download/Copy-enqueue-order or `Finish()`.
3. `tests/Nivara.PerformanceTests/SilkKernelsGate.cs`: append `silk.Finish()` to timed
   Silk launch lambdas (RunGemm and direct `silk.Run(...)`), so measured Silk time still
   includes kernel execution rather than enqueue-only dispatch.

## Verification steps

- Build: `dotnet build Nivara.slnx`
- Silk kernel gate (Release): `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels`
  (32/32 argmax parity must hold; per-kernel timings re-baselined.)
- Docs snippet gate: `EveryGatedBlock_CompilesWithoutErrors`.
- Report SmolLM greedy-decode timings across Python (PyTorch), Nivara CPU, and Silk GPU.

## Planned commits

1. `docs: plan #652 (pipeline Silk launches) in TODO.md`
2. `perf(silk): make Run/CopyBuffer non-blocking; add Finish()`
3. `docs: document non-blocking Silk launch semantics`
4. `test(silk): synchronize timed Silk gate launches with Finish()`
5. `docs: remove TODO.md — plan executed` (only after G2 review)

## GitHub issues log

- (none discovered yet)
   - Deferred concerns, if any, will be captured at discovery time.
