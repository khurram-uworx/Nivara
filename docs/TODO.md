# Plan: Promote SilkDistilBertTrainer primitives (issue #1304)

## Problem

`SilkDistilBertTrainer` keeps `Backward`, `AdamUpdateAll`, and `ExportGradients`
`internal` while `SilkLayaHeadTrainer` exposes the equivalent primitive quartet
publicly. Callers therefore cannot write the same
`Forward → Backward → AdamUpdateAll → ExportCheckpoint` loop against both
trainers without `InternalsVisibleTo`.

## Proposed changes

Visibility-only change in `src/Nivara.Extensions/Silk/SilkDistilBertTrainer.cs`:

- `Backward(int batch, int seqLen, ReadOnlySpan<float> dLogitsSpan)` → `public`
  (manual-loop primitive; caller supplies the batch-averaged softmax-CE gradient).
- `Backward(int batch, int seqLen)` → `public` (device-buffer-populated overload used
  by `Train`; promoted for consistency).
- `AdamUpdateAll(float learningRate, float weightDecay)` → `public`.
- `ExportGradients()` → `public`.

Also tidy the two XML doc phrases that still say "internal"/"Exposed internally …".

No behavioral change. `Forward`, `Train`, `ExportCheckpoint`, `CreateTrueBf16` are
already public.

## Blast radius

- Source: single file, four members of one partial class.
- Downstream: `SilkDistilBertTrainerTests`, `SilkBertBf16*ParityTests`,
  `SilkDistilBertTrainerBf16*Tests`, `SilkDistilBertTrainingProfileProbe`
  (uses `InternalsVisibleTo`) — all unaffected functionally.
- Public API surface widens; no removals.

## Verification steps

1. `dotnet build Nivara.slnx`
2. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
3. Targeted Gpu tests:
   - `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkDistilBertTrainer"`
   - `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkBertBf16"`
4. Build performance probe: `dotnet build tests/Nivara.PerformanceTests/Nivara.PerformanceTests.csproj -c Release`

## Planned commits

1. `docs: plan issue #1304 in TODO.md`
2. `feat(silk): promote SilkDistilBertTrainer.Backward/AdamUpdateAll/ExportGradients to public`
3. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none created yet)
