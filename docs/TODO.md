# #1238 DistilBert: ForwardBatched backward produces no parameter gradients

## Problem

`DistilBertSequenceClassificationTests.ForwardBatched_Backward_ProducesParameterGradients` fails with
"Parameter 'Weight' received no gradient through the batched attention path." Root cause is not batched-
specific and not in the attention kernel.

Two related defects:

1. **Same-dtype `LoadStateDict` drops `RequiresGrad`.** `Linear.LoadStateDict` (same-dtype branch,
   `Linear.cs:208`) and `Embedding.LoadStateDict` (same-dtype branch, `Embedding.cs:309`) call
   `Module<TModel>.CloneTensor(source)` with no `requiresGrad`, so the parameter adopts the source's
   flag. `StateDictLoader.LoadLinearWidened` / `LoadEmbedWidened` build sources via
   `ReverseGradTensor<TWeight>.FromMatrix(...)` whose `requiresGrad` defaults to `false`. Result: every
   weight loaded through those helpers becomes inference-only, so `ShouldTrack()`/`ShouldTrackGrad`
   return false, no `OpNode` is attached, and no gradient accumulates. This is the same class of bug
   already fixed for the mixed-target bind sites in 1.5.2 (commit `ca183e1f`); the same-dtype path was
   left un-fixed.
2. **`LayerNorm.Forward` gates on `input.RequiresGrad` alone.** Even if its own affine params require
   grad, a constant input makes it take the `ForwardInference` path, so its params never train. With
   defect 1 fixed, the embedding LayerNorm's input becomes tracking and this masks; the gate is still
   a latent hole.

## Blast radius

- `src/Nivara/AutoDiff/Nn/Linear.cs` — `LoadStateDict` same-dtype `Weight` branch.
- `src/Nivara/AutoDiff/Nn/Embedding.cs` — `LoadStateDict` same-dtype `Weight` branch.
- `src/Nivara/AutoDiff/Nn/LayerNorm.cs` — `Forward` tracking gate.
- Tests: `LinearLoadPathTests`, `EmbeddingLoadPathTests`, `LayerNormTests`,
  `DistilBertSequenceClassificationTests` (end-to-end).
- Consumers: DistilBert loader (`samples/Nivara.Samples/DistilBertModel.cs`), ResNet18/MobileNetV2
  inference loaders (`samples/NivaraInference`) — these load with `requiresGrad:false` sources but run
  outside `Grad()`, so `GradientUtils.IsGradEnabled` keeps them inference-only; no perf regression.
- Doc pins: no new `docs/*.md` files or `File.cs:NN` citations; `CHANGELOG.md` is at repo root
  (not scanned by the doc gate).

## Proposed changes

### Commit 1 — preserve `requiresGrad` on same-dtype load

```csharp
// Linear.cs:208, Embedding.cs:309
var cloned = Module<TModel>.CloneTensor(source, weight?.Tensor.RequiresGrad ?? true);
```

Mirrors the mixed-dtype branch directly below each site, `Module.LoadStateDict`, and the `?? true`
deferred default already used by `StateDictLoader`'s bind sites.

### Commit 2 — LayerNorm tracks when only its params require grad

Replace `if (!input.RequiresGrad)` gate with:

```csharp
var shouldTrack = GradientUtils.IsGradEnabled &&
    (input.RequiresGrad || (weight?.Tensor.RequiresGrad ?? false) || (bias?.Tensor.RequiresGrad ?? false));
```

Keep the `ForwardInference` fast path when `!shouldTrack`; guard `AccumulateGradient(input, …)` in the
gradFn with `if (input.RequiresGrad)` (matching `Linear`), since `AccumulateGradient` sets `.Grad`
unconditionally.

### Commit 3 — docs

- `docs/AUTODIFF.md` load-path bullet: loading preserves the target's `requiresGrad`, defaulting to
  `true` when deferred.
- `CHANGELOG.md` `[1.5.2] → Changed`: same-dtype bind sites no longer silently drop `RequiresGrad`;
  `LayerNorm` tracks when only its affine params require grad.

## Verification

- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~DistilBertSequenceClassificationTests"`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~LinearLoadPathTests|FullyQualifiedName~EmbeddingLoadPathTests|FullyQualifiedName~LayerNormTests"`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.AutoDiff"`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Targeted sweep of AI/model fixtures (Laya, ModernBert, DistilBert, BertEncoder).

## Planned commits

1. `fix(AutoDiff): preserve parameter requiresGrad in same-dtype LoadStateDict`
2. `fix(AutoDiff): track LayerNorm gradients when only its parameters require grad`
3. `docs: record #1238 fix (same-dtype requiresGrad, LayerNorm gate)`

## GitHub issues log

- [ ] #1238 — DistilBert: ForwardBatched backward produces no parameter gradients (pre-existing) (working on this)
- [ ] #1240 — AutoDiff: RMSNorm/BatchNorm/TransformerBlock share LayerNorm's input-only tracking gate (created while working on #1238)
- [ ] #1241 — AutoDiff/StateDictLoader: silently skip missing keys on partial checkpoint load (created while working on #1238)
