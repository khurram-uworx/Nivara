# TODO: Issue #1085 — Samples: remaining hand-counted accuracy

## Problem

Follow-up to #1065. Three sample sites still hand-count classification accuracy:

1. `samples/NivaraChat/Training/SentimentTrainer.cs` line ~60 loop (3-class).
2. `samples/NivaraChat/Training/ValidatorTrainer.cs` line ~60 loop (2-class).
3. `samples/NivaraFineTuning/Program.cs` three spots (lines ~467, 614, 670) — predictions per batch, accuracy accumulated alongside per-batch CE.

Goal: migrate Sentiment/Validator like AgentsValidatorTrainer; migrate FineTuning by collecting predicted/actual labels across batches then deriving accuracy from `Nivara.Metrics.ConfusionMatrixFrame`. Console output and numbers unchanged.

## Proposed changes

### Step 1: SentimentTrainer (single commit)
- Add `using Nivara.Metrics;`.
- Replace `int correct` loop with `predictedLabels`/`actualLabels` arrays; call `NivaraMetrics.ConfusionMatrixFrame`; sum `"Correct"` into `long correct`.
- Console line unchanged.

### Step 2: ValidatorTrainer (single commit)
- Same pattern.

### Step 3: NivaraFineTuning.Program.cs (single commit)
- Add `using Nivara;` and `using Nivara.Metrics;`.
- In `RunEvalSilk`, `EvaluateSilk`, `Evaluate`: remove `int correct` accumulation; append predicted + actual to `List<int>` buffers; after loop build frame when non-empty; derive `correct` and `acc` unchanged. Keep loss accumulation.

## Blast radius

- Sample projects only: `NivaraChat`, `NivaraFineTuning`. No core/test changes.
- No public API change; only demonstration code.
- `NivaraColumn<int>.Create(int[])` and `NivaraMetrics.ConfusionMatrixFrame` already validated by #1065.

## Verification

- `dotnet build samples/NivaraChat/NivaraChat.csproj -c Release`
- `dotnet build samples/NivaraFineTuning/NivaraFineTuning.csproj -c Release`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Smoke-run affected sample entry points to confirm unchanged accuracy strings (human-run if needed).

## Planned commits

1. docs: plan issue 1085 in TODO.md
2. samples: use Nivara.Metrics in SentimentTrainer
3. samples: use Nivara.Metrics in ValidatorTrainer
4. samples: use Nivara.Metrics in NivaraFineTuning eval paths
5. docs: remove TODO.md — plan executed

## GitHub issues log

- (none created yet)
