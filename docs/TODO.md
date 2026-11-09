# Plan for issue #1051: simplify NivaraInference sample CLI

## Problem
`samples/NivaraInference` mixes visitor-facing inference with evaluation/benchmark/parity tooling.
Goal: remove evaluation-only functionality from the visitor CLI surface while preserving it in an
appropriate harness (`tests/Nivara.PerformanceTests` / `tests/Nivara.SimdProbe` when needed).

## Additional requested scope
- Fully remove `--synthetic-weights` from the sample CLI.
- Move `compare` and `compare_diag` out of the sample CLI and expose them through
  `tests/Nivara.PerformanceTests`.

## Blast radius
- `samples/NivaraInference/Program.cs` (CLI dispatch, modes, guards, output)
- `samples/NivaraInference/README.md`
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`
- `tests/Nivara.PerformanceTests/Program.cs` and new probe files
- Docs: `docs/QWEN.md`, `docs/ACCELERATION.md`, `docs/HALF-F16.md` as needed

## Planned commits
1. `docs: plan #1051 work in TODO.md`
2. `refactor(inference): split help into inference and evaluation sections`
3. `refactor(inference): remove synthetic weights from visitor CLI`
4. `refactor(inference): move compare and compare_diag out of visitor CLI`
5. `test(perf): expose relocated compare commands in performance harness`
6. `docs: reorganize NivaraInference README quick start`
7. `docs: update affected docs for relocated tooling`
8. `test(inference): update CLI argument gate for reduced surface`
9. `docs: remove TODO.md - plan executed`

## GitHub issues log
- #1097 — follow-up: expose removed compare/parity commands in performance harness (created while simplifying sample CLI)
