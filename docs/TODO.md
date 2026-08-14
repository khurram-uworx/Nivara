# Plan: XML documentation across the AutoDiff public API

Tracking issue: [#197 Missing XML documentation across AutoDiff public API](https://github.com/khurram-uworx/Nivara/issues/197)

## Problem

The entire AutoDiff public API surface (~51 files, ~350 public declarations) lacks XML
doc comments. IntelliSense shows no tooltips for the primary ML training surface
(modules, losses, optimizers, training loops, serialization, initializers). The core
columnar library is well-documented by comparison (e.g. NivaraColumn.cs: 469 XML doc
lines for 55 public members).

## Proposed changes

Add `/// <summary>` (plus `<param>`, `<returns>`, `<typeparam>`, `<exception>`,
`<see cref>` where applicable) to every public type and member in the files below.
Style follows the already-documented reference files: `TypeValidator.cs`,
`TypeConverter.cs`, `GradientUtils.cs`, `ForwardGradOperations.cs`,
`ReverseGradOperations.cs`, `Im2Col.cs`, `Parameter.cs`, `LayerNormKernel.cs`.

- No `<inheritdoc/>` (unused in repo).
- Doc text must match actual code, not the (slightly stale) review doc — e.g. current
  `Module<T>` has no `Forward(input1, input2)` / `NamedParameters()`.
- Accuracy source: `docs/AUTODIFF.md` (canonical reference) + CHANGELOG PyTorch-parity notes.

Scope by area (one commit per area):

| # | Area | Files |
|---|---|---|
| 1 | Nn core | `Module.cs`, `Linear.cs`, `Sequential.cs`, `Dropout.cs`, `Activation.cs`, `IMultipleInputModule.cs` |
| 2 | Convs/pooling | `Conv1d.cs`, `Conv2d.cs` (incl. `ConvTranspose2d`), `DepthwiseSeparableConv2d.cs`, `MaxPool2d.cs`, `AdaptiveAvgPool2d.cs` |
| 3 | Norms | `BatchNorm.cs`, `BatchNormKernel.cs` (internal — doc per LayerNormKernel precedent), `LayerNorm.cs` |
| 4 | Attention | `MultiheadAttention.cs`, `TransformerBlock.cs` (incl. `NormType`) |
| 5 | Embeddings/tokenizer | `Embedding.cs`, `SparseEmbedding.cs` (gap-fill), `TextTokenizer.cs` |
| 6 | Generative | `VAE.cs`, `ConvVAE.cs`, `Sampler.cs`, `ElboLossType.cs` |
| 7 | Losses | `Loss.cs`, `Reduction.cs`, `MSELoss.cs`, `L1Loss.cs`, `BCELoss.cs`, `BCEWithLogitsLoss.cs`, `CrossEntropyLoss.cs` |
| 8 | Initializers | `IInitializer.cs` + 7 implementations |
| 9 | Optimizers | `Optimizer.cs` (gap-fill), `SGD.cs`, `Adam.cs`, `AdamW.cs` (gap-fill) |
| 10 | Training | `TrainingLoop.cs`, `DataParallelTrainer.cs`, `DataLoader.cs`, `TensorDataset.cs`, `Batch.cs`, `DataParallelResult.cs` |
| 11 | Serialization | `ModelSerializer.cs`, `Checkpoint.cs` |
| 12 | Operations & tensors | `ReverseGradOperations.cs`, `ForwardGradOperations.cs`, `ReverseGradTensor.cs`, `ForwardGradTensor.cs` |

Reference files with partial gaps (documented too — human confirmed 2026-08-14):
`ReverseGradOperations.cs` (32: core ops + class), `Parameter.cs` (10),
`ReverseGradTensor.cs` (5: operators), `ForwardGradTensor.cs` (5: operators),
`ForwardGradOperations.cs` (1: Gelu).

Fully documented already (no change): `Im2Col.cs`, `LayerNormKernel.cs`,
`GradKernels.cs`, `AttentionKernels.cs`, `GradientUtils.cs`, `TypeValidator.cs`,
`TypeConverter.cs`, `GradTensor.cs`, `AutoGradExceptions.cs`,
`NivaraAutoGradExtensions.cs`, `AutoDiffDiagnostics.cs`, `GraphInfo.cs`.
Internal, out of scope: `ComputationGraph.cs`, `OpNode.cs`, `RMSNormKernel.cs`,
`ModuleHelpers.cs`.

## Blast radius

- Pure documentation change: no behavior, signature, or semantics change; compiles identically.
- Files touched: ~51 under `src/Nivara/AutoDiff/` (NN modules, losses, optimizers,
  training, serialization, initializers) plus `docs/REVIEW-2026-08-12.md` and
  `CHANGELOG.md` (bookkeeping).
- Downstream: nothing depends on doc comments at runtime. No public API shape changes,
  so no caller/test impact. Existing tests remain the consistency guardrail.
- Verification: a temporary doc-coverage script (outside the repo) flags public
  declarations lacking an immediately-preceding `///` block; target is 0 flags. `dotnet
  build Nivara.slnx` must stay clean. `dotnet test` only with human confirmation.

## Verification steps

1. Gate: `dotnet build src/Nivara/Nivara.csproj -p:GenerateDocumentationFile=true` reports
   **0 CS1591** anywhere under `src/Nivara/AutoDiff/` (baseline: 412 unique members).
2. `dotnet build Nivara.slnx` succeeds (no warnings/errors introduced).
3. `dotnet test` — ASK HUMAN BEFORE RUNNING (AGENTS.md rule).
4. Review doc-count deltas per file (documented files should now have `///` counts
   roughly matching public-member counts).

## Planned commits

1. `docs: plan XML docs for AutoDiff public API (#197) in TODO.md`
2. `docs: XML doc comments for Nn core modules (#197)`
3. `docs: XML doc comments for conv/pooling modules (#197)`
4. `docs: XML doc comments for normalization modules (#197)`
5. `docs: XML doc comments for attention/transformer modules (#197)`
6. `docs: XML doc comments for embedding/tokenizer modules (#197)`
7. `docs: XML doc comments for generative modules (#197)`
8. `docs: XML doc comments for loss functions (#197)`
9. `docs: XML doc comments for initializers (#197)`
10. `docs: XML doc comments for optimizers (#197)`
11. `docs: XML doc comments for training API (#197)`
12. `docs: XML doc comments for serialization API (#197)`
13. `docs: XML doc comments for operations and tensor classes (#197)`
14. `docs: mark REVIEW-2026-08-12 finding #1 resolved; CHANGELOG (#197)` +
    `git rm docs/TODO.md` (plan executed)

## GitHub issues log

- No issues created yet. As work executes, any deferred work/concern found outside this
  plan must be captured immediately via `gh issue create --repo khurram-uworx/Nivara`
  and recorded here — do not rely on memory (compaction can lose it).
