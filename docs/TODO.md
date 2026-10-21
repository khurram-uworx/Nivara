# Plan: Track Silk Nivara type-safe route performance gap versus Ollaya (#708)

## Problem
The Silk-backed in-process TypeSafe route takes ~1216 ms per assessment versus Ollaya's
~653.5 ms. `NivaraTypeSafeClient.SystemOneAsync` runs six independent question forwards
serially, with all CPU prompt work inside the device lock, and exposes no phase timing
to identify the dominant cost. There is also no shared immutable work beyond state
tokenization.

## Proposed changes

### Commit 1 — Phase timing + encode/head separation
- Add `LastEncodeMilliseconds` / `LastHeadMilliseconds` to `ILayaGpuSession` (GpuSeam.cs),
  implement in `SilkLayaSession` and `IlgpuLayaSession`.
- Add `SystemOnePhaseTimings` record and `LastPhaseTimings` property to
  `NivaraTypeSafeClient`, measuring state text, state tokenization, question mapping,
  prompt build, forward (encode + head), decode, answer mapping, allocations, total.
- Update `TypeSafe.Benchmark` to accumulate and print a phase breakdown.

### Commit 2 — Share immutable state/question work
- Hoist `ToLayaQuestion` and `BuildSequence` for all six questions outside the `gate`
  lock; keep only `session.Forward` + decode + answer mapping locked.
- Pre-size `answers` and derived dictionaries.

### Commit 3 — Docs/coverage pins (if required)
- Update `DocsCoveragePins` only if doc/citation counts change.

## Verification steps
- `dotnet build Nivara.slnx` (Release).
- Targeted: `NivaraTypeSafeClientTests`, `InferenceCliArgumentTests`.
- `typesafe nivara verify` (all AGREE) — human-run.
- `laya --silk compare` (PASS) — human-run.
- `typesafe nivara benchmark` median < 1216 ms — human-run.

## Planned commits
1. `feat(inference): add phase timing to NivaraTypeSafeClient.SystemOneAsync`
2. `perf(inference): share immutable state/question work across TypeSafe questions`
3. `docs: remove TODO.md — plan executed`

## Blast radius
- `src/Nivara.Extensions/AI/GpuSeam.cs` — interface addition; both session impls.
- `src/Nivara.Extensions/Silk/SilkLayaSession.cs` — timing.
- `samples/Nivara.Samples/Gpu/LayaGpuSession.cs` — timing.
- `samples/NivaraInference/NivaraTypeSafeClient.cs` — phase timings + hoisting.
- `samples/NivaraInference/TypeSafe.cs` — benchmark phase breakdown.

## GitHub issues log
- (none yet; deferred work captured during execution)
