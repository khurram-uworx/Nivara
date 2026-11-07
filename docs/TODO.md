# Plan: perf(samples): use SilkBertEncoderRunner for dev eval in RunTrainSilk (issue #964)

## Problem
`RunTrainSilk` exports the trained weights to the CPU model (`model.LoadWeights(exported)`)
every epoch to run dev evaluation. That's 66M params × 2 conversions (device→HF→CPU) per
epoch on top of the already-slow training loop. The inference runner `SilkBertEncoderRunner`
already exists and runs on the device.

## Goal
Use `SilkBertEncoderRunner` for dev-set evaluation instead of exporting to the CPU model
each epoch. Speeds up training and exercises the inference path during training.

## Proposed changes

### 1. `src/Nivara.Extensions/Silk/SilkDistilBertTrainer.cs`
- Store the `BertConfig` passed to the constructor (`readonly BertConfig config;`).
- Add public `CreateInferenceRunner()` returning a `SilkBertEncoderRunner` that shares
  the trainer's live device weights (no copy).

### 2. `src/Nivara.Extensions/Silk/SilkBertEncoderRunner.cs`
- Add `readonly bool ownsWeights;`.
- Add internal constructor
  `SilkBertEncoderRunner(SilkRuntime runtime, BertConfig config, SilkBertWeights weights, bool ownsWeights)`
  that performs the existing workspace allocation but borrows the supplied weights.
- Public tensor constructor chains into the internal one (creates its own weights, ownsWeights=true).
- `Dispose` releases workspace buffers always; disposes weights only when `ownsWeights`.

### 3. `samples/NivaraFineTuning/Program.cs` (`RunTrainSilk`)
- Create runner once: `using var runner = trainer.CreateInferenceRunner();`.
- Replace per-epoch `ExportCheckpoint` + `LoadWeights` + `Evaluate` with device eval via
  `EvaluateSilk(runner, ...)`.
- Add `EvaluateSilk` helper (mirrors `RunEvalSilk`).
- Keep final save using `ExportCheckpoint` + `ModelSerializer.Save`.

## Blast radius
- SilkDistilBertTrainer: new public method + stored field; backward compatible.
- SilkBertEncoderRunner: new internal constructor + field; public API unchanged.
- Sample only consumer affected.
- Tests: SilkDistilBertTrainerTests, SilkBertEncoderRunner usage unaffected (public API stable).

## Verification
- `dotnet build Nivara.slnx`
- Targeted Gpu tests: `SilkDistilBertTrainerTests`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits
1. docs: plan #964 in TODO.md
2. feat(silk): shared-weight inference runner support
3. refactor(samples): device-side dev eval in RunTrainSilk
4. docs: remove TODO.md — plan executed

## GitHub issues log
- None yet.

## Decisions for human (G1)
- Sharing live device weights requires the runner to not dispose them (trainer owns).
  Confirmed by design above.
- Dev eval seqLen/batchSize clamping follows existing RunEvalSilk conventions.