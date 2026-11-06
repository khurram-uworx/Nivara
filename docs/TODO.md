# Plan: docs update Silk story for NivaraFineTuning (#954)

## Problem

After #952 (Silk inference) and #953/#956 (Silk training) landed, the documentation still
describes NivaraFineTuning training as CPU-only in `docs/SILK.md` §13d, the sample README
has no `--silk` usage examples, and the `--silk-kernels` gate notes do not reflect that the
trainer's backward/Adam kernels are parity-tested separately.

## Proposed changes

1. `docs/SILK.md` §13d: add a `NivaraFineTuning` bullet (inference + optional training
   coverage); remove the stale "AutoDiff training is CPU-only" line.
2. `docs/SILK.md`: extend the `--silk-kernels` gate bullet to note the trainer's
   backward/Adam kernels are exercised by `SilkDistilBertTrainerTests` (gradient parity),
   not byte-identity.
3. `samples/NivaraFineTuning/README.md`: add a "Silk (OpenCL GPU)" usage subsection with
   `--silk` examples for train/eval/predict.

No source or test changes. No new citations or fenced C# blocks, so doc coverage pins and the
snippet gate are unaffected.

## Verification

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`

## Planned commits

1. `docs: plan #954 in TODO.md`
2. `docs: update Silk story for NivaraFineTuning (#954)`

## Blast radius

Docs only. No public API, no tests beyond doc coverage (unchanged counts).

## GitHub issues log

(none created)
