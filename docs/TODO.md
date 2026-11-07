# Plan: fix(samples): consistent --max-len clamping in RunTrainSilk (issue #965)

## Problem
`RunTrainSilk` clamps `maxBatch` and `maxSeqLen` to device caps (8 / 128) and warns, but:

1. Tokenization and the `allTokens`/`allMask` buffers still use `cli.MaxLen`, so a `--max-len 256`
   run tokenizes 256-length sequences and allocates 256-wide buffers while the trainer caps at 128.
2. `seqLen` is computed as `Math.Min(cli.MaxLen, maxSeqLen)` (redundant and only accidentally correct).
3. Dev eval (`Evaluate`) is called with `cli.MaxLen`, not the clamped value.

Result: training runs at 128 while tokenization/dev-eval run at 256 — a silent shape/layout mismatch.

## Goal
Use the clamped `maxSeqLen` consistently for tokenization, the flattened training buffers, `seqLen`,
and dev eval, with the existing single clamping warning.

## Proposed changes

### `samples/NivaraFineTuning/Program.cs` (`RunTrainSilk`)
Replace every `cli.MaxLen` reference inside `RunTrainSilk` with `maxSeqLen`:
- `Sst2Dataset.Tokenize(tokenizer, trainData, maxSeqLen)` and its `seqLen={maxSeqLen}` print.
- `Sst2Dataset.Tokenize(tokenizer, dataset.Dev, maxSeqLen)`.
- `int seqLen = maxSeqLen;` (drop the `Math.Min`).
- `int[] allTokens = new int[sampleCount * maxSeqLen];`
- `float[] allMask = new float[sampleCount * maxSeqLen];`
- copy loop `for (int j = 0; j < maxSeqLen; j++)` with `i * maxSeqLen + j`.
- `Evaluate(model, lossFn, tokenizer, dataset.Dev, maxSeqLen, cli.BatchSize)`.

The existing warning block stays as-is (single warning).

## Blast radius
- Only `RunTrainSilk` in the sample. No library/API changes.
- No tests directly reference this sample method.

## Verification
- `dotnet build samples/NivaraFineTuning/NivaraFineTuning.csproj -c Release`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Manual smoke if data/OpenCL available: `--mode train --silk --max-len 256 --max-examples 8 --epochs 1`.

## Planned commits
1. docs: plan #965 in TODO.md
2. fix(samples): consistent --max-len clamping in RunTrainSilk
3. docs: remove TODO.md — plan executed

## GitHub issues log
- None yet.

## Decisions for human (G1)
- Clamping (not early error) is chosen to match existing behavior and acceptance criteria.