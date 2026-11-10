# Plan: issue #1114 — feat(silk): let DistilBERT trainer ingest BF16 checkpoints (convert on load)

## Problem
`SilkDistilBertTrainer` only accepts `Dictionary<string, (float[] Data, int[] Shape)>`. A BF16
checkpoint cannot be loaded into the only existing BERT-family training path without an external
F32 conversion, so the memory saving of a BF16 checkpoint is lost. (Follow-up to #1013/#1012;
scoped subset of #1103.)

## Proposed changes
1. Change `SilkDistilBertTrainer` class declaration to `partial`.
2. Add `SilkDistilBertTrainer.Bf16.cs` with a `ToFloat` helper (BFloat16 → float) and two
   constructors (trainableLayerCount + freezeEncoder convenience) delegating to the F32 path.
3. Add `SilkDistilBertTrainer.Fp16.cs` with the same shape using `Half` → float.
4. No device-side narrow storage, no new backward kernels (those belong to #1103).

## Blast radius
- `src/Nivara.Extensions/Silk/SilkDistilBertTrainer.cs` (class keyword only).
- New files in same directory.
- Downstream: samples/tests that construct the trainer with float tensors unaffected.
- DocsCoveragePins: no doc files changed; no pin updates expected.

## Verification
- `dotnet build Nivara.slnx` (Extensions).
- Targeted Gpu tests if OpenCL device available.

## Planned commits
1. docs: plan #1114 in TODO.md
2. refactor(silk): mark SilkDistilBertTrainer partial
3. feat(silk): ingest BF16 checkpoints in DistilBERT trainer (convert on load)
4. feat(silk): ingest FP16 checkpoints in DistilBERT trainer (convert on load)
5. docs: remove TODO.md — plan executed

## GitHub issues log
- (none yet)
