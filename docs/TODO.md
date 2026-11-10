# Plan: reuse partial/frozen training policy for future Laya trainer (#1107)

Follow-up to #1012 / PR #1101 and #1106.

## Problem

`SilkDistilBertTrainer.TrainableLayerCount` is the canonical state for partial/frozen
training, but the policy that derives trainable flags from it is duplicated inline in two
places:

- `SilkDistilBertTrainer.TrainableLayerCount` setter (layer/embedding/head marking)
- both `SilkBertWeights` constructors (f32 and BF16/FP16 narrow)

It is also hard-wired to the BERT layer layout, so a future Silk Laya/ModernBERT trainer
would have to copy-paste it, risking drift. No Silk Laya trainer exists yet; this plan
extracts the policy into a reusable, model-agnostic helper so that trainer can call it
instead of reinventing a second freeze policy.

## Proposed changes

1. Add `src/Nivara.Extensions/Silk/SilkPartialTrainingPolicy.cs` (internal, static) with:
   - `Normalize(requested, numLayers, trainable)` → clamped count (`trainable == false` → 0; `requested < 0` → `numLayers`).
   - `EmbeddingsTrainable(normalized, numLayers, trainable)`.
   - `LayerTrainable(layerIndex, numLayers, normalized, trainable)`.
   - `HeadTrainable(trainable)`.
2. Refactor both `SilkBertWeights` constructors to call the helper instead of inline
   `firstTrainable` / `trainableLayerCount == config.NumHiddenLayers` logic.
3. Refactor `SilkDistilBertTrainer.TrainableLayerCount.set` to delegate layer/embedding/head
   marking to the helper; keep backward-skip and profiling logic unchanged.
4. Add non-GPU NUnit tests pinning clamping and selection.
5. Update `docs/SILK.md` partial/frozen section to name the helper as canonical reuse point.

## Blast radius

- `src/Nivara.Extensions/Silk/SilkPartialTrainingPolicy.cs` (new)
- `src/Nivara.Extensions/Silk/SilkBertWeights.cs` (both constructors)
- `src/Nivara.Extensions/Silk/SilkDistilBertTrainer.cs` (setter)
- `tests/Nivara.Tests/...` new test file
- `docs/SILK.md`

No public API change; helper is `internal` (reachable via existing `InternalsVisibleTo`).

## Verification

- `dotnet build src/Nivara.Extensions/Nivara.Extensions.csproj -c Release`
- targeted non-GPU tests for the helper (no OpenCL device required)
- existing GPU parity tests unchanged
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits

1. docs: plan #1107 in TODO.md
2. feat(silk): add reusable partial/frozen training policy helper
3. refactor(silk): use policy helper in SilkBertWeights
4. refactor(silk): use policy helper in SilkDistilBertTrainer
5. test(silk): pin partial/frozen training policy
6. docs(silk): name policy helper as canonical reuse point
7. docs: remove TODO.md — plan executed

## GitHub issues log

- #1107 — feat(silk): reuse partial/frozen training policy in future Laya trainer (this issue)
