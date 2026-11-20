# Plan: TinyShakespeare Silk save/load round-trip + loss parity tests (#1360)

## Problem

`SilkTinyShakespeareTrainer` has `ExportCheckpoint()` and `BatchedTransformer.LoadFromSilkWeights(...)`,
but no test exercises the full save/load round-trip parity: device train → export → import → CPU
`Forward` logits match device forward within tolerance (argmax exact). The parity gate also
compares gradients/weights but not the CE loss value.

## Proposed changes

- Add a test method `SaveLoadRoundTrip_LogitsAndLossMatchCpu` to
  `tests/Nivara.Tests/Gpu/SilkTinyShakespeareTrainerTests.cs` that:
  - computes CPU `CrossEntropyLoss` and Silk host-computed CE loss on identical initial weights
    and asserts they agree within 1e-3;
  - performs one explicit device step (Forward/Backward/AdamUpdateAll);
  - exports via `ExportCheckpoint()`, imports into a fresh `BatchedTransformer<float>`;
  - asserts CPU `Forward` logits match post-step Silk `Forward` logits within 1e-3 and argmax
    exact per position.
- Add two small static helpers (`ArgMax`, `CrossEntropy`) to the fixture.
- No production code changes.

## Verification steps

- `dotnet build tests/Nivara.Tests/Nivara.Tests.csproj -c Release`
- Targeted test run (requires OpenCL device; otherwise ignored):
  `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkTinyShakespeareTrainerTests"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits

1. docs: plan #1360 in TODO.md
2. test: add Silk TinyShakespeare save/load round-trip + loss parity test
3. docs: remove TODO.md — plan executed

## Blast radius

- Only `tests/Nivara.Tests/Gpu/SilkTinyShakespeareTrainerTests.cs` touched.
- No public API or storage changes; no docs pins affected.

## GitHub issues log

- (none yet)
