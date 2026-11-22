# TODO — #1441 ft-eval --silk dispatch guard test

## Problem

Issue #1441 (`NivaraFineTuning: guard ft-eval --silk silent CPU fallback`, closed) was resolved by #1439: `RunFtEval` dispatches `--silk` to `RunFtEvalSilk` (`samples/NivaraFineTuning/Program.cs:1462-1465`), and `RunFtEvalSilk` fails clearly with no CPU fallback when OpenCL is unavailable. Report/markdown behavior is covered by `LoraSftRunReportTests` and the generation seam by `WorklogSftGenerationTests`.

Residual gap: no targeted automated test pins the CLI dispatch guard itself — i.e., that `RunFtEval` with `Silk=true` never silently takes the CPU eval path. The #1441 acceptance criteria included "CLI/report behavior covered by targeted tests"; report is covered, CLI dispatch is not explicitly pinned.

## Proposed changes

Add a targeted test in `tests/Nivara.Tests/FineTuning/WorklogSftGenerationTests.cs` (or a small new FineTuning test file if cleaner) that proves `RunFtEval` dispatches to the Silk path when `cli.Silk` is true.

Planned approach (grounded, environment-independent, no real model/OpenCL required):

- Create a temp directory named like `qwen-dispatch-test` with a minimal `config.json` and a minimal `test.jsonl` (one valid worklog row or empty-as-needed), plus explicit `OutDir` in temp.
- Build a `CliArgs` with `Mode="ft-eval"`, `Silk=true`, `ModelDir=tempDir`, `TestData=testJsonl`, `OutDir=tempDir`, `Template="qwen"`, `MaxExamples=1`.
- Call `NivaraFineTuning.Program.RunFtEval(cli)` (internal, exposed via `InternalsVisibleTo`).
- `RunFtEval` resolves modelDir/testData, determines qwen, and should dispatch to `RunFtEvalSilk` before loading CPU model. `RunFtEvalSilk` will fail while trying to load tokenizer/tensors/session in the Silk path (no real tokenizer/safetensors in temp dir). That failure is proof of Silk-path dispatch, not CPU fallback.
- Assert: the call throws (expected from Silk-path loading) OR returns non-zero with Silk-specific error; and no `ft-eval-report.json` / `ft-eval-report.md` is written with `Backend="cpu"`. Also capture/assert console does not contain the CPU header `=== Nivara LoRA SFT Eval ===` if feasible.
- Keep the test focused and readable; avoid brittle exact exception-message assertions beyond key phrases.

Alternative considered: call `RunFtEvalSilk` directly. Rejected as primary pin because the issue was about `RunFtEval`/CLI dispatch; direct `RunFtEvalSilk` call would not pin the guard in `RunFtEval`. May add as secondary assertion if cheap.

## Blast radius

- **Files changed**: `tests/Nivara.Tests/FineTuning/WorklogSftGenerationTests.cs` or a new `tests/Nivara.Tests/FineTuning/FtEvalCliDispatchTests.cs` (preferred if it keeps concerns separate), possibly `docs/TODO.md` (deleted at end).
- **Production code**: none expected. `Program.RunFtEval` / `RunFtEvalSilk` already `internal static`; `InternalsVisibleTo` already present.
- **Downstream impact**: test-only; no public API/behavior change.
- **DocsCoveragePins**: no docs/*.md files added/removed; no citations added/removed. Expect no pin revision needed; verify during G2.
- **Risk**: low. Test must be environment-independent (no OpenCL, no real model files).

## Verification steps

- [ ] `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` (before PR/push)
- [ ] Targeted test: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~FtEval|FullyQualifiedName~WorklogSftGenerationTests"` (ask/confirm before running - user gave explicit go-ahead to test during this task)
- [ ] Confirm no `Backend="cpu"` report written by the dispatch test
- [ ] G2 review: branch as a whole + branch against this TODO
- [ ] DocsCoveragePins revision check (expect none needed)

## Planned commits

1. `docs: plan #1441 dispatch guard test in TODO.md`
2. `test: pin ft-eval --silk dispatch away from CPU path`
3. `docs: remove TODO.md — plan executed` (only after G2 clears)

## GitHub issues log

- [ ] No new issues expected. #1441/#1439 already closed; this is a follow-up hardening test.

## Reminder

As each task executes, if you find deferred work or a concern, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number in the log — don't rely on memory or wait until the end of the plan, as compaction during execution can lose important items.
