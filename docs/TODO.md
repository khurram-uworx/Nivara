# Plan: Issue #1362 — TinyShakespeare Silk OOM guard

## Problem
`SilkTinyShakespeareTrainer` stores `maxBatch`/`maxSeqLen` but `Train(...)` never validates
`batchSize * blockSize <= maxBatch * maxSeqLen`. An oversized batch silently overruns the fixed
activation buffers (sized to `maxSeqLen` per sequence). `Forward`/`Backward` already reject
`seqLen > maxSeqLen` but with an empty message. The CLI has no `--max-batch`/`--max-seq`
overrides and help text doesn't document them.

## Proposed changes

### 1. Trainer guards (`src/Nivara.Extensions/Silk/SilkTinyShakespeareTrainer.cs`)
- Constructor: keep `<= 0` checks with clear messages.
- `Forward`/`Backward`: keep `seqLen <= 0 || seqLen > maxSeqLen` throw; add clear message;
  add direct-caller product guard `seqLen > maxBatch * maxSeqLen`.
- `Train`: add `batchSize <= 0` / `blockSize <= 0` throws; add headline guard
  `(long)batchSize * blockSize > (long)maxBatch * maxSeqLen`.

### 2. CLI overrides (`samples/NivaraChat/Transformer/TransformerMode.cs`)
- `TransformerOptions`: `MaxBatch = 0`, `MaxSeq = 0`.
- Parse `--max-batch`, `--max-seq`.
- `TrainSilk`: compute caps (default to batch/block) and pass to trainer.
- `PrintHelp`: document both flags.

### 3. Docs (optional): `docs/SILK.md` §13g note.

## Verification
- Build `Nivara.Extensions` and `NivaraChat`.
- Run `SilkTinyShakespeareTrainerTests` (GPU, may be skipped without device).
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits
1. docs: plan #1362 in TODO.md
2. fix: enforce maxBatch/maxSeqLen OOM guard in SilkTinyShakespeareTrainer
3. feat: add --max-batch/--max-seq overrides + help text in TransformerMode
4. docs: remove TODO.md — plan executed

## Blast radius
- `SilkTinyShakespeareTrainer`: constructor, Forward, Backward, Train.
- `TransformerMode`: ParseArgs, TransformerOptions, TrainSilk, PrintHelp.
- Tests: `SilkTinyShakespeareTrainerTests` (GPU). No CPU-side tests directly.

## GitHub issues log
- (none yet)
