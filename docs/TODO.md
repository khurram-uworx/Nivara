# Plan: #1439 — NivaraFineTuning Silk path for causal-LM ft-eval

## Problem

`NivaraFineTuning --mode ft-eval` is CPU-only for causal-LM LoRA adapters (`samples/NivaraFineTuning/Program.cs:RunFtEval`). Passing `--silk` is silently ignored (#1441/G2). After #1429, Silk sessions/runners can load adapter bundles, but ft-eval cannot evaluate on Silk/OpenCL.

Issue: https://github.com/khurram-uworx/Nivara/issues/1439 (milestone 1.5.4, target base branch khurram/1438).

## Grounding / G1

- Grounded with codebase navigation (code-memory MCP symbol relations for `RunFtEval`, grep/read) and local docs (`docs/plan/LORA.md` D2a/G2 #1441).
- Microsoft-learn MCP not required for API semantics here; this is repo-local Silk/session/CLI behavior. If implementation surfaces external API uncertainty, escalate before assuming.
- No silent CPU fallback for `--silk` paths (AGENTS.md / LORA guardrail).

## Blast radius

- **Changed:** `samples/NivaraFineTuning/Program.cs` (RunFtEval dispatch, new Silk eval path, report fields), `samples/NivaraFineTuning/WorklogSftGeneration.cs` (Silk/cached greedy generation helper or reusable generator seam), possibly `samples/NivaraFineTuning/LoraSftRunReport.cs` only if report shape needs fields (prefer no schema change).
- **Touched docs:** `docs/plan/LORA.md` (mark #1439/#1441 status when done), `samples/NivaraFineTuning/LORA-TUNING.md` if CLI behavior documented; `DocsCoveragePins` if citation/doc counts change.
- **Tests:** `tests/Nivara.Tests/FineTuning/LoraSftRunReportTests.cs` and/or new targeted FineTuning tests; GPU-gated smoke only if feasible/appropriate.
- **Downstream callers:** CLI `ft-eval` behavior; `WorklogSftGeneration.Evaluate` existing CPU callers (`Program.cs`, `WorklogLoraSftTrainer.cs`) must remain behavior-compatible.

## Proposed changes

1. **CLI dispatch/guard** — `RunFtEval` checks `cli.Silk`; dispatch to `RunFtEvalSilk`. No silent ignore, no CPU fallback. Clear OpenCL unavailable error.
2. **Silk eval path** — load Qwen/SmolLM config/tokenizer/tensors, create `SilkQwenSession`/`SilkSmolLMSession`, run base eval; if adapter provided, load adapter bundle via session `LoadAdapterBundle` (or fresh session for base/adapted separation), run adapted eval. Report `Backend="silk"`, `Device=<device>`, same metrics/report shape.
3. **Generation seam** — reuse `WorklogSftEvaluator.Evaluate(examples, predict)`. Add Silk greedy generator using `ForwardPrefillArgmax`/`ForwardCachedArgmax`/`ResetCache` (cached greedy). Keep CPU `GenerateGreedy`/`Evaluate(LlamaForCausalLM,...)` intact for existing callers.
4. **Report parity** — JSON/Markdown includes backend/device, base/adapted metrics, deltas, eval split/examples, adapter hash/format. Parity status remains honest/not release-certified unless evidence supports otherwise.
5. **Docs/pins** — update `docs/plan/LORA.md` for #1439/#1441; review DocsCoveragePins after branch review if doc/citation counts changed.

## Verification

- [ ] `dotnet build Nivara.slnx -c Release` (or targeted projects) passes.
- [ ] Targeted tests for report/CLI behavior pass (ask before running).
- [ ] GPU-gated Silk ft-eval smoke only where appropriate; missing OpenCL should fail clearly.
- [ ] `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` before PR.
- [ ] Docs snippet gate / `FullyQualifiedName~Nivara.Tests.Docs` if docs change (ask before running).

## Planned commits

1. `docs: plan #1439 Silk ft-eval path in TODO.md`
2. `feat: add Silk generation seam for worklog ft-eval`
3. `feat: add Silk ft-eval CLI path and report parity`
4. `test: cover Silk ft-eval report and CLI guard behavior`
5. `docs: update LORA plan and docs coverage pins for #1439` (if needed)
6. `docs: remove TODO.md — plan executed` (after G2 clears)

## GitHub issues log

- [ ] #1439 — NivaraFineTuning: add Silk path for causal-LM ft-eval (this work)
- [ ] #1441 — ft-eval --silk silently ignored (resolved by #1439 guard/path; confirm closure note in PR)
- [ ] #1437 — Generation base-vs-adapted eval in Silk ft-lora (related, not in scope unless discovered necessary)
- [ ] #1438 — D2a parity certification (base branch context; not changed here)

## Notes

- As each task executes, if deferred work or concerns are found, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record it here.
- Do not push/create PR without explicit user confirmation. (User has given explicit confirmation for this task.)
- Do not run full `dotnet test` suite unprompted.
