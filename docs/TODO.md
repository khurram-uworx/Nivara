# TODO: SFT trainer structured worklog eval metrics (#1425)

> As each task executes, if you find deferred work or a concern outside this plan, create a tracked issue immediately via `gh issue create --repo khurram-uworx/Nivara` and record its number in the GitHub issues log below. Don't rely on memory or wait until the plan finishes — compaction can lose it.

## Problem

Issue [#1425](https://github.com/khurram-uworx/Nivara/issues/1425) (milestone 1.5.4): `WorklogLoraSftTrainer` surfaces only loss (`InitialValLoss`, `BestValLoss`). Field/structure/convention/clarification accuracy are computed only in the `ft-eval` CLI path via `WorklogSftEvaluator`. Training reports need these metrics without requiring a separate `ft-eval` run.

## Grounding / constraints

- Reuse `WorklogSftEvaluator.Evaluate(IReadOnlyList<WorklogSftExample>, Func<WorklogSftExample,string>)` (`samples/NivaraFineTuning/WorklogSftEvaluator.cs:101`).
- Greedy generation on val set after best-checkpoint restore (issue note).
- Optional/gated to avoid doubling training time on large runs (issue note).
- Base noted as `khurram/1417`; 1417 is closed and its trainer code is present on current mainline. Implement on `khurram/1425`.

## Proposed changes

### 1. Shared greedy generation helper (extract from Program.cs)

`Program.cs` has private helpers hardcoded to `LlamaForCausalLM<float,float>`:
- `EvaluateWorklogSft` (`Program.cs:1427`)
- `GenerateGreedy` (`Program.cs:1466`)
- `RenderEvalPrompt` (`Program.cs:1445`)
- `GetStopTokenIds` (`Program.cs:1498`)

Extract to a reusable location in `samples/NivaraFineTuning` (e.g. `WorklogSftGeneration.cs` or static methods on `WorklogSftEvaluator`/new helper class). Make generation generic over `TModel,TWeight : IFloatingPointIeee754` so trainer can use it. Argmax should compare by converting logits to float.

Sketch:

```csharp
public static class WorklogSftGeneration
{
    public static WorklogEvalMetrics Evaluate<TModel, TWeight>(
        LlamaForCausalLM<TModel, TWeight> model,
        Gpt2BpeTokenizer tokenizer,
        IReadOnlyList<WorklogSftExample> examples,
        string template,
        int maxNewTokens,
        int[] stopIds)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>;

    public static int[] GenerateGreedy<TModel, TWeight>(...); // same semantics as Program.GenerateGreedy
    public static int[] GetStopTokenIds(Gpt2BpeTokenizer tokenizer, string template);
    public static string RenderEvalPrompt(IReadOnlyList<WorklogMessage> messages, string template);
}
```

Update `Program.cs` to call shared helper; keep behavior identical.

### 2. Trainer options gate

Add to `WorklogLoraSftOptions` (`WorklogLoraSftTrainer.cs:15`):

```csharp
public bool ComputeWorklogEvalMetrics { get; init; } = false;
public int EvalMaxNewTokens { get; init; } = 128;
public int MaxEvalExamples { get; init; } = 0; // 0 = all val examples
```

Default off. Gated because greedy generation is expensive.

### 3. Trainer result carries structured metrics

Extend `WorklogLoraSftResult` (`WorklogLoraSftTrainer.cs:39`):

```csharp
public sealed record WorklogLoraSftResult(
    ... existing params ...,
    WorklogEvalMetrics? BestValMetrics = null,
    int EvalExamples = 0,
    long EvalElapsedMs = 0);
```

Add defaulted tail parameters to reduce source breakage; audit call sites.

### 4. Trainer computes metrics after best-checkpoint restore

In `WorklogLoraSftTrainer.Train` (`:201-214`), after `RestoreAdapters` and before save/return:

- If `options.ComputeWorklogEvalMetrics` is false, skip.
- Need original val `WorklogSftExample`s and `tokenizer`. Options:
  - retain originals when loaded from paths (`:103-113` currently discards them);
  - accept optional `IReadOnlyList<WorklogSftExample>? valExamples = null` parameter (for pretokenized/test callers).
- Need generation settings: template (`options.Template`), `EvalMaxNewTokens`, stop IDs from tokenizer.
- Cap eval examples by `MaxEvalExamples` if > 0.
- `model.Eval()`, no `GradientUtils.Grad()`.
- `metrics = WorklogSftGeneration.Evaluate(model, tokenizer, evalExamples, options.Template, options.EvalMaxNewTokens, stopIds);`
- Log `metrics.ToReport("Best-checkpoint val eval")`.
- Return metrics in result.

### 5. CLI/report plumbing

- `ft-lora` CLI: when `cli.Eval` is enabled today it already computes base+adapted metrics outside trainer by reloading float/float model. To satisfy #1425 without doubling work, pass `ComputeWorklogEvalMetrics`/eval options into trainer and use `result.BestValMetrics` for adapted metrics where possible; keep base-vs-adapted delta reporting. Be careful not to change semantics unexpectedly.
- `LoraSftRunReport`: currently stores metrics as formatted strings. Add structured fields if needed (default null) so machine-readable reports carry them; keep existing string fields for compatibility. Update `ToMarkdown` to include structured summary when present.
- `ft-eval` path: reuse shared generation helper; behavior unchanged.

## Decisions / red flags (G1 - resolve before implementation)

1. **Generic generation vs float-only.** Existing `GenerateGreedy` is `float,float`. Trainer is generic `TModel,TWeight`. For `Train<float,BFloat16>` (Qwen/SmolLM headline paths), can we generate generically by converting logits to float for argmax? This is feasible but must be confirmed. Alternative: restrict eval-metrics to `float,float` or add a specialized overload.
2. **CLI double-eval avoidance.** `ft-lora` default `--eval` currently computes base+adapted metrics post-training by reloading model (`Program.cs:995-1033`). If trainer computes adapted best-checkpoint metrics, running both wastes time. Need decision: (a) trainer computes adapted metrics only, CLI still computes base and reuses adapted from result; (b) keep CLI eval as-is and also surface trainer result; (c) refactor CLI eval to use trainer result. Prefer (a) with clear reporting.
3. **Original val examples availability.** Pretokenized/test callers pass `trainTok,valTok` but no original `WorklogSftExample`s. Evaluator needs originals. Plan adds optional `valExamples` param; when missing and option enabled, either throw clear error or skip with warning. Need decision on strictness.
4. **Report schema.** Add structured numeric fields to `LoraSftRunReport` or keep only string `BaseMetrics/AdaptedMetrics`? Strings already work; structured fields are nicer for tooling. Backward-compatible additive fields preferred.

## Verification steps

- [ ] `dotnet build Nivara.slnx -c Release` (ask before long-running commands as needed; build is fine)
- [ ] Targeted tests: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~WorklogLoraSftTrainerTests|FullyQualifiedName~WorklogSftEvaluatorTests|FullyQualifiedName~LoraSftRunReportTests|FullyQualifiedName~WorklogSftDatasetTests"`
- [ ] Confirm `docs/TODO.md` excluded from DocsCoveragePins (it is per AGENTS.md). No doc pin changes expected unless docs/*.md added.
- [ ] Ask before running broader suites.

## Planned commits

1. `docs: plan SFT trainer worklog eval metrics in TODO.md` (this file)
2. `Extract generic worklog SFT generation helpers` (shared helper + Program.cs uses it, behavior-preserving)
3. `Surface worklog eval metrics from SFT trainer` (options, result, trainer eval after restore, tests)
4. `Plumb worklog eval metrics through ft-lora report/CLI` (report fields, CLI wiring, tests)
5. `docs: remove TODO.md — plan executed` (only after G2 reviews clear)

## Blast radius

- `samples/NivaraFineTuning/WorklogLoraSftTrainer.cs` — options/result/train signature/behavior; downstream: `Program.cs`, trainer tests.
- `samples/NivaraFineTuning/WorklogSftEvaluator.cs` or new helper file — shared generation; downstream: `Program.cs` ft-lora/ft-eval, trainer.
- `samples/NivaraFineTuning/Program.cs` — CLI eval wiring; private helper extraction.
- `samples/NivaraFineTuning/LoraSftRunReport.cs` — additive report fields/markdown.
- Tests: `WorklogLoraSftTrainerTests`, `LoraSftRunReportTests`, possibly new generation helper tests.
- No `src/Nivara` / `src/Nivara.Extensions` production library changes expected; this is samples + sample tests.

## GitHub issues log

- [ ] #1425 — SFT trainer: surface structured worklog eval metrics (field/structure/convention/clarification) (this plan)
