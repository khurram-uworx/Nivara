# Plan: Issue #1223 — Qwen parity fixtures (qwen_tool_*_py.bin)

## Problem

`QwenNativeCheckpointTests` and `QwenInstructParityTests` reference PyTorch-generated
fixtures that are not present in the repo / checkpoint snapshot. The reference generator
exists at `samples/NivaraInference/Python/qwen_tool_reference.py` (and
`qwen_plain_reference.py` for the plain-prompt fixtures), but outputs are not committed.
Fixture-dependent Qwen tests skip on every machine, including CI.

## Proposed changes

1. **Ship the fixtures** — add `.gitignore` exceptions for the 10 small fixture files
   (6 tool + 4 plain) under `samples/data/qwen2.5-0.5b-instruct/`. `.bin` files remain
   LFS-tracked via the existing `.gitattributes`. Commit the generated files.
2. **Regeneration docs** — document exact commands + expected file list + byte sizes in
   `docs/QWEN.md`.
3. **Native parity checks** — add two tests to `QwenNativeCheckpointTests` mirroring
   `QwenInstructParityTests.Model_GreedyToolLoop_MatchesTorchGeneratedIds` and
   `Model_GreedyFinalAnswer_SemanticParityAndFinalLogitsWithinTolerance`, but against the
   native `LlamaForCausalLM<float,BFloat16>` model.
4. **DocsCoveragePins** — revise if docs/citation counts move.

## Blast radius

- Files: `.gitignore`, `docs/QWEN.md`, `tests/Nivara.Tests/Inference/QwenNativeCheckpointTests.cs`,
  `tests/Nivara.Tests/Docs/DocsCoveragePins.cs` (maybe).
- New untracked data files become tracked (10 fixtures, ~1.2 MB).
- Tests: QwenInstructParityTests (already passing), QwenNativeCheckpointTests (3 existing + 2 new).

## Planned commits

1. `docs: plan 1223 in TODO.md`
2. `chore: un-ignore Qwen parity fixtures`
3. `test: add native Qwen fixture parity checks`
4. `docs: QWEN.md regeneration procedure + pinned fixture sizes`
5. `test: adjust DocsCoveragePins if needed`

## GitHub issues log

- (none yet)

## Verification

- `dotnet test ... --filter QwenInstructParityTests` (green)
- `dotnet test ... --filter QwenNativeCheckpointTests` (new tests green)
- `dotnet test ... --filter "FullyQualifiedName~Nivara.Tests.Docs"` (green)
- snippet gate `EveryGatedBlock_CompilesWithoutErrors` (green)
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` (clean)
