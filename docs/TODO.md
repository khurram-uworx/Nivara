# LoRA SLM CLI and report for SLM LoRA showcase (#1420)

Status: **active implementation on `khurram/1420`**.

## Problem

Issue #1420 asks to expose the generic Llama-family causal-LM LoRA SFT showcase in
`NivaraFineTuning` with a clear, honest report and Silk/CPU parity. The current code
already ships `ft-lora`/`ft-eval`/`ft-generate` modes, a CPU reference trainer, a real
Silk trainer, a shared adapter bundle, and a JSON report. Remaining gaps versus the
issue's acceptance criteria:

1. CLI surface does not include `--max-memory` (memory budget option named in issue).
2. Report is JSON-only; issue comment asks for structured JSON **and** markdown artifact.
3. Report lacks explicit base/adapters/trainable param breakdown, memory-budget field,
   loss-curve summary, and structured eval metrics (base vs adapted before/after).
4. `ft-lora` does not integrate eval/before-after into its output.
5. SmolLM verification command path is not documented in README/help.

## Decisions (to be cleared at Gate G1)

- D1 — `--max-memory`: treat as an **advisory budget reported in the artifact**, not a hard
  runtime cap, because the LoRA trainers are per-example loops with no bounded buffer
  manager in scope. Hard enforcement would require wiring `StreamingBufferManager` into
  the sample and is out of scope for #1420. Report the requested value and whether it was
  honored (advisory). Flag this for human confirmation.
- D2 — Markdown companion: emit `ft-lora-report.md` / `ft-eval-report.md` next to JSON,
  rendered from the same `LoraSftRunReport` data. No new data model.
- D3 — Loss curve: extend `WorklogLoraSftResult` with `PerEpochValLoss` list (and train loss)
  collected during training. CPU path populates it; Silk path records per-epoch val loss.
- D4 — Eval integration in ft-lora: optional `--eval` flag (default true) runs base + adapted
  eval on val split using existing `EvaluateWorklogSft` helper; reports before/after in both
  console and report artifact. Reuses existing code path.
- D5 — SmolLM: document a verification command in README and help; no code change needed
  beyond ensuring `TrainSmolLMFloatBf16` path is exercised. Add targeted test for path
  selection if feasible.
- D6 — Parity wording stays honest: `IsParityReady` remains `false`; report parity status
  must never imply certification.

## Proposed changes (blast radius)

### A. CLI (`samples/NivaraFineTuning/Program.cs`)
- Add `MaxMemory` field to `CliArgs` (default 0 = "not specified"; interpret 0 as
  advisory default `1024 * 1024 * 1024`? or report as null). Decision: default 0 →
  report `null`/`"unspecified"` to avoid false claim. Parse `--max-memory` (bytes).
- Add `--eval` bool flag (default true) for ft-lora eval integration.
- Wire `MaxMemory` and `Eval` into `RunFtLora` / `RunFtLoraSilk` / `RunFtEval`.
- Update `PrintHelp` with new options and SmolLM verification example.

### B. Report model (`samples/NivaraFineTuning/LoraSftRunReport.cs`)
Additive fields:
- `AdapterParams` (int) — trainable adapter parameter count (alias clarity).
- `TotalParams` (int) — base + adapter.
- `MemoryBudgetBytes` (long?) — requested budget (null if unspecified).
- `MemoryBudgetMode` (string) — "advisory" (or "unspecified").
- `LossCurve` (string?) — JSON-serialized per-epoch train/val losses, or structured list.
  Use `List<LoraSftEpochLoss>` nested record for clarity.
- `BaseMetrics`/`AdaptedMetrics` already exist; ensure populated by ft-lora when eval runs.
- `EvalRan` (bool), `EvalSplit` (string), `EvalExamples` (int).

### C. Trainer result (`samples/NivaraFineTuning/WorklogLoraSftTrainer.cs`)
- Extend `WorklogLoraSftResult` with `List<float> TrainLossPerEpoch`, `List<float> ValLossPerEpoch`.
- Populate during `Train` loop (already computes trainAvg/valLoss per epoch).
- Silk path: collect per-epoch val loss in `RunFtLoraSilk`.

### D. Eval integration
- In `RunFtLora` CPU path: if `cli.Eval`, run base + adapted eval using existing
  `EvaluateWorklogSft` on val split and populate report metrics + deltas.
- In `RunFtLoraSilk`: same where model/adapter load path exists; if not feasible,
  report `EvalRan=false` with reason rather than false claim.

### E. Markdown report
- Add `LoraSftRunReport.SaveMarkdown(path)` (or a static helper) rendering a
  human-readable summary from the same fields.
- Call from both ft-lora and ft-eval paths.

### F. Docs
- README LoRA section: add SmolLM verification command; document `--max-memory`,
  `--eval`, markdown artifact.
- `docs/plan/LORA.md`: update #1420 status (optional, low risk).

### G. Tests (targeted)
- Unit tests for `LoraSftRunReport` serialization (new fields, markdown).
- CLI parse tests for `--max-memory`/`--eval`.
- Path selection test (Qwen vs SmolLM) if model-independent.

## Verification steps

- Build: `dotnet build samples/NivaraFineTuning/NivaraFineTuning.csproj -c Release`.
- Targeted tests: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~WorklogLoraSftTrainerTests|FullyQualifiedName~Nivara.Tests.FineTuning"`.
- Doc snippet gate if docs under `docs/` touched: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`.
- Human-run smoke (requires model data + optional OpenCL):
  - Qwen CPU ft-lora with `--max-memory 1073741824 --eval`.
  - Qwen Silk ft-lora on available device.
  - SmolLM verification ft-lora.
  - ft-eval before/after.

## Planned commit list

1. `docs: plan #1420 in TODO.md`
2. `feat(ft): add --max-memory and --eval CLI options`
3. `feat(report): extend LoraSftRunReport with params/memory/loss/eval fields + markdown`
4. `feat(trainer): capture per-epoch loss curve in WorklogLoraSftResult`
5. `feat(ft-lora): integrate eval before/after + markdown report (CPU)`
6. `feat(ft-lora): integrate eval + markdown report (Silk)`
7. `docs: SmolLM verification command path + CLI option docs`
8. `test: report/CLI/loss-curve coverage for #1420`
9. `docs: remove TODO.md — plan executed` (at G2)

## Blast radius

- `samples/NivaraFineTuning/Program.cs` — CLI parse, ft-lora CPU/Silk, ft-eval, help.
- `samples/NivaraFineTuning/LoraSftRunReport.cs` — report model + serialization.
- `samples/NivaraFineTuning/WorklogLoraSftTrainer.cs` — result record + Train loop.
- `samples/NivaraFineTuning/README.md` — docs.
- `docs/plan/LORA.md` — status (optional).
- Tests under `tests/Nivara.Tests/FineTuning/` — new coverage.
- No core library changes; no public API changes outside sample.

## GitHub issues log

- [ ] (none yet) — create during execution if deferred work is found.

## Notes

- No false iGPU claim: parity status stays "not release-certified" until evidence supports flip.
- Memory budget is advisory only; do not claim enforcement.
- Do not run full test suite unprompted; ask before full suite.
