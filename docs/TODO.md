# Plan: NivaraChat CPU-vs-Silk parity test for TinyShakespeare (#1314)

## Problem

Issue #1314 (parent #1270, depends on #1313): other Silk runners have
CPU-vs-GPU parity coverage, but the TinyShakespeare port does not. Need a
test fixture comparing CPU `BatchedTransformer` logits/top-1 against the Silk
runner on a small randomly initialized model.

## Proposed changes

Add a single test fixture (no product-code changes):

- File: `tests/Nivara.Tests/Gpu/SilkTinyShakespeareParityTests.cs`
- Category: `Gpu` (device-guarded skip when no OpenCL device).
- Build `BatchedTransformer<float>(32, 16, 1, 2, 8)` (random init).
- Export weights via `ToSilkWeights()`; open `SilkTinyShakespeareSession` with
  matching `TinyShakespeareConfig` (eps 1e-5).
- Test 1 `ForwardPrefill_...`: fixed prompt `[1,2,3,4]`, assert top-1
  equality and logits `maxAbs <= 1e-3 * (1 + max|cpu|)`.
- Test 2 `ForwardCached_...`: assert prefill top-1 agrees, then 3 cached
  steps, asserting top-1 equality and logits closeness per step.

## Verification steps

- `dotnet build Nivara.slnx` (Release)
- Targeted test: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkTinyShakespeareParityTests"` on a machine with OpenCL.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits

1. `docs: plan #1314 in TODO.md`
2. `test: add CPU-vs-Silk parity test for TinyShakespeare`
3. `docs: remove TODO.md — plan executed` (after G2 review)

## Blast radius

- New test file only; no changes to src or samples.
- Test references `NivaraChat.Transformer.BatchedTransformer`, `Nivara.Silk.*`,
  already exposed to `Nivara.Tests` via InternalsVisibleTo / project references.
- No DocsCoveragePins changes expected (no docs/*.md added).

## GitHub issues log

- (none yet)
