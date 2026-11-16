# Plan: #1234 — AV in SilkMemObject release during NivaraChat cleanup

## Problem

`NivaraChat --qwen … --silk` (tool loop) runs to completion but crashes with
`0xC0000005` (`Silk.NET.OpenCL.CL.ReleaseMemObject`) during cleanup. Reproduced
on `main` (F32) and `khurram/1222` (BF16), so not caused by #1222.

Root cause: the chat tool paths wrap the user-supplied `IChatClient` in a
`FunctionInvokingChatClient`. The outer `ExecuteSilk` also holds the
`SilkQwenChatClient` in a `using`, whose `Dispose` calls `SilkQwenSession.Dispose`.
If the wrapper (or the outer `using`) disposes the inner client, the session is
disposed twice; the second pass re-releases the same OpenCL handles, possibly
after `SilkRuntime` has torn down the context. `NivaraInference` avoids this
because it never wraps the session in a `FunctionInvokingChatClient`.

## Proposed changes

1. `src/Nivara.Extensions/Silk/SilkQwenGpuRunner.cs` — add `bool disposed;` guard;
   make `Dispose()` a no-op on the second call.
2. `src/Nivara.Extensions/Silk/SilkQwenSession.cs` — add `bool disposed;` guard;
   make `Dispose()` a no-op on the second call.
3. `src/Nivara.Extensions/Silk/SilkRuntime.cs` — make `Release(nint)` a no-op when
   `disposed` is true (in addition to the existing `buffer == 0` guard).

## Blast radius

- `SilkQwenGpuRunner` (both F32 and BF16 constructors share one `Dispose`).
- `SilkQwenSession` (both constructors share one `Dispose`).
- `SilkRuntime.Release` (used by all Silk runners).
- No public API signature changes; behavior change is idempotent disposal only.

## Verification

- Build `Nivara.Extensions` (Release).
- Run `SilkQwenToolsWeatherLoopTests` (Gpu) where GPU available.
- Repro command exits 0.

## Planned commits

1. docs: plan #1234 in TODO.md
2. fix: make Silk Qwen disposal idempotent and teardown-safe
3. docs: remove TODO.md — plan executed

## GitHub issues log

(none created during this plan)
