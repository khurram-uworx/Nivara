# TODO: Issue #1417 — LoRA generic Llama-family CPU SFT trainer

Branch: `khurram/1417` (off `khurram/1416`)
Issue: https://github.com/khurram-uworx/Nivara/issues/1417
PR target: `khurram/1416`

## Problem

Issue #1417 asks for a generic Llama-family LoRA SFT trainer (CPU reference) for the behavioral worklog showcase. A scaffold landed in `84223292` (`WorklogLoraSftTrainer`, `WorklogSftDataset`, `ft-lora`/`ft-eval`/`ft-generate` CLI, `WorklogSftEvaluator`). The scaffold covers the happy path but several acceptance criteria are unmet or unverified:

- Trainer core still takes a `bool qwen` flag and calls `LlamaLoader.LoadModel(..., qwen: qwen)` — a Qwen/SmolLM split bleeds into the trainer signature.
- No early stopping/patience option.
- No gradient accumulation / batch option (per-example steps only).
- No memory guard defaults/validation.
- Structured worklog metrics (field accuracy, valid structure, convention/clarification) are computed only in `ft-eval`, not surfaced from the trainer.
- Adapter save/load is index-based with no target/rank/shape/model metadata validation.
- No tests pin base-weight immutability, adapter updates, determinism, or overfit sanity.
- No `WorklogSftEvaluator` unit tests.
- `NivaraFineTuning` README has no LoRA SFT section.

## Blast radius

- `samples/NivaraFineTuning/WorklogLoraSftTrainer.cs` — trainer core, snapshot/restore, save/load.
- `samples/NivaraFineTuning/Program.cs` — CLI wiring, ft-lora/ft-eval/ft-generate, adapter loading, eval metrics.
- `samples/NivaraFineTuning/WorklogSftDataset.cs` — already covered by #1416; consume only.
- `samples/NivaraFineTuning/WorklogSftEvaluator.cs` — metrics; add unit tests.
- `samples/NivaraFineTuning/README.md` — add LoRA SFT section.
- `tests/Nivara.Tests/FineTuning/` — new trainer + evaluator tests.
- `tests/Nivara.Tests/Nivara.Tests.csproj` — already references NivaraFineTuning.
- Docs/pins: no new `docs/*.md` expected; if citations change, revise DocsCoveragePins.

Downstream callers: `ft-lora`/`ft-eval` CLI and (future) #1418 Silk path. Keep public API compatible or update call sites in same commit.

## Proposed changes

1. **Generic trainer API** — remove `bool qwen` from `Train<TModel,TWeight>`; load via `LlamaLoader.LoadModel<TModel,TWeight>(modelDir, requiresGrad:false)` (auto-detects Qwen from config). Keep `options.Template` for chat rendering only. Update `TrainQwenFloatBf16`/`TrainSmolLMFloatBf16` convenience wrappers to call generic `Train`. Drop qwen console labeling or derive from model/config.
2. **Early stopping / patience** — add `Patience` (default 0 = disabled) and `MinDelta` to `WorklogLoraSftOptions` + CLI `--patience`, `--min-delta`. Stop training when no val improvement for `Patience` epochs.
3. **Gradient accumulation** — add `GradientAccumulationSteps` (default 1) + CLI `--grad-accum`. Accumulate gradients across N examples before `optimizer.Step()`/`ZeroGrad()`. Keep per-example forward for memory.
4. **Memory guard** — add `MaxSeqLenGuard` validation (reject > 2048 or > model max), document small defaults; validate `maxLen` against model `MaxPositionEmbeddings`. CLI defaults remain small (`maxLen=256`, `maxExamples=0`).
5. **Structured metrics in result** — extend `WorklogLoraSftResult` with `TrainLoss`, `ValLoss` series? Instead, expose final `ValMetrics` (loss + WorklogEvalMetrics on val set) and `TrainableParamCount`/`BaseParamCount` (already present). Keep result compact: add `BestEpochValMetrics`? Decision: add `InitialValLoss` (present), `BestValLoss` (present). Add optional `EvalMetrics` field computed on val set after best restore. Keep minimal.
6. **Harden adapter contract** — include `targets`, `rank`, `alpha`, `template`, `baseModelId`, per-adapter `projection` name, and shapes in saved JSON. On load, validate format, count, shapes, and (if present) targets/rank/alpha. Fail loudly on mismatch.
7. **Tests (fast, no external weights)**:
   - `WorklogSftEvaluatorTests`: parse single/multi/malformed calls; clarification detection; perfect prediction metrics; multi-ticket exactness; convention all-day/half-day; clarification scoring.
   - `WorklogLoraSftTrainerTests` or a tiny-model fixture: base weights unchanged after training; adapter params change; deterministic given seed (same loss/adapter snapshot); overfit sanity on tiny set (loss decreases). Use a tiny in-memory LlamaForCausalLM with random weights and synthetic tokenized examples (no model files).
8. **README** — add LoRA SFT section documenting CLI, defaults, metrics, adapter format.
9. **Verification** — build, targeted tests, smoke run on SmolLM if weights available (ask before long runs).

## Planned commits

1. `docs: plan issue 1417 LoRA generic CPU SFT trainer`
2. `feat(ft): make Llama SFT trainer generic over model loader (drop qwen flag from core)`
3. `feat(ft): add early stopping, patience, and gradient accumulation to SFT trainer`
4. `feat(ft): harden adapter save/load contract with metadata validation`
5. `test(ft): add WorklogSftEvaluator unit tests`
6. `test(ft): add trainer invariants (base unchanged, adapter updates, determinism, overfit)`
7. `docs(ft): document LoRA SFT CLI, defaults, and adapter format`
8. `docs: remove TODO.md — plan executed`

## Verification

- `dotnet build samples/NivaraFineTuning/NivaraFineTuning.csproj -c Release`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~WorklogSft"` (ask before running)
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` before push
- Smoke: SmolLM/Qwen if weights available (ask before long runs; use `--max-examples` small).

## GitHub issues log

- [ ] #1417 — LoRA: generic Llama-family CPU SFT trainer (this branch)
- [ ] #1416 — LoRA: behavioral worklog SFT dataset and deterministic generator (dependency, already on base)
- [ ] (add discoveries here immediately during execution)

## Grounding / decisions

- Grounded in codebase: `WorklogLoraSftTrainer.cs`, `LlamaForCausalLM.cs`, `LoRALinear.cs`, `LlamaLoader` (auto-detect qwen via `IsQwenModel` using ropeTheta/vocabSize), `CrossEntropyLoss.Forward(..., ignoreIndex)`, `AdamW`, `GradientUtils.Grad`.
- Decision: keep `options.Template` for chat rendering; derive model loading from config auto-detection.
- Decision: early stopping optional (patience=0 disables); gradient accumulation default 1 (no change to existing behavior).
- Decision: adapter save format stays `nivara-lora-sft-adapters-v1` but gains metadata; load validates when metadata present, tolerates older files (best-effort).
- Decision: tests use tiny in-memory LlamaForCausalLM (random weights) to avoid external model dependency; overfit sanity asserts loss decreases (not exact convergence).
- Red flag to watch: `LlamaLoader.LoadModel` auto-detection must correctly classify SmolLM vs Qwen; verify with config heuristics.
- No new .NET/BCL tensor APIs introduced; microsoft-learn grounding not strictly required for this internal sample/test work, but G1 still applies to any new API usage.
