# TODO: #1437 — Add generation base-vs-adapted eval to Silk ft-lora

> Tracking issue: https://github.com/khurram-uworx/Nivara/issues/1437
> Branch: `khurram/1437`
> Milestone: 1.5.4

## Reminder

As each task executes, if you find deferred work or a concern outside this plan, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number in the GitHub issues log below. Do not rely on memory or wait until the end of the plan.

## Problem

`NivaraFineTuning --mode ft-lora --silk` currently reports loss/val-loss but does not run integrated base-vs-adapted worklog generation eval. CPU `ft-lora` does this when `cli.Eval` is enabled (post-training: load val rows, evaluate base model, load adapter/evaluate adapted, compute deltas, emit report fields). `ft-eval --silk` already has generation eval parity via `SilkQwenSession`/`SilkSmolLMSession` + `WorklogSftGeneration.GenerateGreedyCached` + `WorklogSftGeneration.Evaluate(...)`. The Silk ft-lora path lacks equivalent generation-eval content.

## Acceptance criteria (from issue)

- `--mode ft-lora --silk --eval` produces base-vs-adapted worklog eval metrics comparable in shape to CPU path.
- JSON/markdown reports include eval metrics/deltas for Silk runs when eval is enabled.
- No silent CPU fallback for the Silk eval path.
- Targeted tests cover report schema/behavior; GPU-gated smoke where appropriate.

## Grounding / facts

- CPU ft-lora eval: `samples/NivaraFineTuning/Program.cs::RunFtLora` (993-1048) computes base metrics on fresh base model, adapted on adapter-loaded model, uses `WorklogSftGeneration.Evaluate`, builds deltas, fills `LoraSftRunReport` eval fields. Uses val data (`EvalSplit="val"`), caps eval examples at `Max(1, cli.MaxExamples/4)` when `MaxExamples>0`.
- Silk ft-lora: `Program.cs::RunFtLoraSilk` (1105+) trains with `SilkLlamaLoRATrainer`, saves best adapter, writes report with no eval fields. `cli.Eval` is parsed/defaulted (default enabled) but not honored for generation eval in Silk path.
- Silk ft-eval parity already exists: `Program.cs::RunFtEvalSilk`/`RunFtEvalSilkWithSession` (1450-1617) opens `SilkQwenSession`/`SilkSmolLMSession`, uses `GenerateGreedyCached`, evaluates base then loads adapter and evaluates adapted, writes `LoraSftRunReport` with `Backend="silk"`, `EvalRan=true`, `EvalSplit="test"`, metrics/deltas.
- Eval primitives are backend-agnostic: `WorklogSftGeneration.Evaluate(tokenizer, examples, template, maxNewTokens, generateGreedy, stopIds)` and `GenerateGreedyCached(...)` in `samples/NivaraFineTuning/WorklogSftGeneration.cs`.
- Report schema already supports eval: `LoraSftRunReport` has `EvalRan`, `EvalSplit`, `EvalExamples`, `BaseMetrics`, `AdaptedMetrics`, `EvalDeltas`, `BaseWorklogMetrics`, `AdaptedWorklogMetrics`, `EvalElapsedMs`.
- `SilkQwenSession` / `SilkSmolLMSession` (`src/Nivara.Extensions/Silk/`) expose `Open`, `DeviceName`, `ForwardPrefillArgmax`, `ForwardCachedArgmax`, `ResetCache`, `LoadAdapterBundle`, `Dispose`; each owns its own `SilkRuntime`.
- `SilkLlamaLoRATrainer.Dispose()` releases owned buffers but does not dispose the externally supplied runtime. Trainer ends at last epoch weights; best adapter is saved to disk by path.

## Proposed changes

### 1) Extract reusable Silk worklog generation eval helper

Refactor `RunFtEvalSilkWithSession` so the core eval computation can be reused by Silk ft-lora. Keep ft-eval behavior/report output unchanged.

Sketch:

```csharp
static (WorklogEvalMetrics Base, WorklogEvalMetrics? Adapted, string? Deltas) EvaluateSilkWorklogGeneration(
    CliArgs cli,
    Gpt2BpeTokenizer tokenizer,
    IReadOnlyList<WorklogSftExample> evalRows,
    int[] stopIds,
    int maxSequenceLength,
    Func<int[], int> prefillArgmax,
    Func<int, int, int> cachedArgmax,
    Action resetCache,
    Action<string> loadAdapterBundle,
    string? adapterPath)
{
    int[] Generate(int[] promptIds) => WorklogSftGeneration.GenerateGreedyCached(
        prefillArgmax, cachedArgmax, resetCache, promptIds, cli.MaxNewTokens, stopIds, maxSequenceLength);

    var baseMetrics = WorklogSftGeneration.Evaluate(tokenizer, evalRows, cli.Template, cli.MaxNewTokens, Generate, stopIds);
    Console.WriteLine(baseMetrics.ToReport("Base"));

    WorklogEvalMetrics? adaptedMetrics = null;
    if (!string.IsNullOrWhiteSpace(adapterPath))
    {
        Console.WriteLine($"Loading adapter bundle: {adapterPath}");
        loadAdapterBundle(adapterPath);
        adaptedMetrics = WorklogSftGeneration.Evaluate(tokenizer, evalRows, cli.Template, cli.MaxNewTokens, Generate, stopIds);
        Console.WriteLine(adaptedMetrics.ToReport("LoRA-adapted"));
    }

    string? deltas = adaptedMetrics is null ? null : FormatWorklogEvalDeltas(baseMetrics, adaptedMetrics);
    if (deltas is not null) Console.WriteLine(deltas);
    return (baseMetrics, adaptedMetrics, deltas);
}

static string FormatWorklogEvalDeltas(WorklogEvalMetrics baseMetrics, WorklogEvalMetrics adaptedMetrics) =>
    $"valid_structure: {adaptedMetrics.ValidStructureRate - baseMetrics.ValidStructureRate:+0.0%;-0.0%;0%}\n" +
    $"ticket:          {adaptedMetrics.TicketAccuracy - baseMetrics.TicketAccuracy:+0.0%;-0.0%;0%}\n" +
    $"hours:           {adaptedMetrics.HoursAccuracy - baseMetrics.HoursAccuracy:+0.0%;-0.0%;0%}\n" +
    $"date:            {adaptedMetrics.DateAccuracy - baseMetrics.DateAccuracy:+0.0%;-0.0%;0%}\n" +
    $"convention:      {adaptedMetrics.ConventionAccuracy - baseMetrics.ConventionAccuracy:+0.0%;-0.0%;0%}";
```

`RunFtEvalSilkWithSession` will call this helper and keep its existing report-save behavior.

### 2) Honor `cli.Eval` in `RunFtLoraSilk`

After training and best-adapter save, if `cli.Eval` is enabled, run Silk generation eval:

- Use val rows for eval (`EvalSplit="val"`), same cap rule as CPU ft-lora: `Max(1, cli.MaxExamples/4)` when `MaxExamples>0`.
- Dispose trainer, then dispose the training runtime, then open a fresh eval session (`SilkQwenSession.Open` / `SilkSmolLMSession.Open`) with the already-loaded config/tensors. No CPU fallback.
- Base eval = fresh session (no adapter). Adapted eval = load best `adapterPath`.
- Measure eval elapsed time (`Stopwatch`).
- Populate `LoraSftRunReport`: `EvalRan`, `EvalSplit="val"`, `EvalExamples`, `BaseMetrics`, `AdaptedMetrics`, `EvalDeltas`, `BaseWorklogMetrics`, `AdaptedWorklogMetrics`, `EvalElapsedMs`. Keep backend/device explicit.

Sketch (inside `RunFtLoraSilk`, before building report):

```csharp
WorklogEvalMetrics? baseMetrics = null;
WorklogEvalMetrics? adaptedMetrics = null;
string? evalDeltas = null;
int evalExamples = 0;
long evalElapsedMs = 0;
if (cli.Eval)
{
    if (string.IsNullOrWhiteSpace(adapterPath) || !File.Exists(adapterPath))
        throw new InvalidOperationException("Silk ft-lora eval requires a saved best adapter; no adapter was produced.");
    var evalRows = valRows;
    if (cli.MaxExamples > 0 && evalRows.Count > cli.MaxExamples)
        evalRows = evalRows.Take(Math.Max(1, cli.MaxExamples / 4)).ToList();
    evalExamples = evalRows.Count;
    if (evalExamples == 0)
        throw new InvalidOperationException("Silk ft-lora eval requires at least one validation example.");

    trainer.Dispose();
    runtime.Dispose();
    trainerDisposed = true; runtimeDisposed = true; // restructure using blocks

    var swEval = Stopwatch.StartNew();
    // open SilkQwenSession/SilkSmolLMSession, evaluate, dispose
    (baseMetrics, adaptedMetrics, evalDeltas) = EvaluateSilkWorklogGenerationOnSession(...);
    swEval.Stop();
    evalElapsedMs = swEval.ElapsedMilliseconds;
}
```

Implementation detail: restructure the `using (runtime)` / `using var trainer` scopes so trainer+runtime can be disposed before opening the eval session, without double-dispose hazards. Session creates its own runtime.

### 3) Failure policy (decision/red flag)

Issue states: "No silent CPU fallback; if Silk eval is not possible, fail clearly or mark unsupported explicitly (but parity is the goal)".

Planned policy: **fail clearly** when `cli.Eval` is enabled and Silk eval cannot be completed (session open failure, missing best adapter, no eval rows, generation/eval exception). Do **not** fall back to CPU. Rationale: silent/partial success is easy to misread; parity is the goal and the issue allows clear failure. This is stricter than CPU ft-lora's "Eval integration skipped" catch, but CPU path is not Silk-fallback; I will preserve CPU behavior as-is.

**G1 decision needed:** confirm this failure policy is acceptable (fail clearly) vs. marking unsupported in report while returning success. I recommend fail clearly.

## Verification

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Targeted tests (no full suite without asking): `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.FineTuning"`
- GPU-gated smoke (if OpenCL available): targeted filter for new smoke test; otherwise test is `[Category("Gpu")]` + Assert.Ignore.
- Review branch diff for DocsCoveragePins impact: no `docs/*.md` added/removed expected; no citations added/removed; ADR not added. If anything doc-related changes, run Docs gate and revise pins as required.

## Blast radius

- `samples/NivaraFineTuning/Program.cs` — `RunFtLoraSilk`, `RunFtEvalSilkWithSession` refactor, new helper(s). Affects ft-lora Silk and ft-eval Silk code paths. ft-eval Silk behavior must remain unchanged.
- `tests/Nivara.Tests/FineTuning/LoraSftRunReportTests.cs` — add Silk eval report schema/markdown coverage.
- Possibly new GPU-gated smoke test under `tests/Nivara.Tests/FineTuning/` or appropriate GPU test location; must be `[Category("Gpu")]` and skipped when no OpenCL.
- No core library/API changes. No `Nivara.Extensions` public API changes expected. No docs pins expected.

## Planned commits

1. `docs: plan #1437 Silk ft-lora generation eval in TODO.md`
2. `refactor: extract reusable Silk worklog generation eval helper` (if separated; may be combined with impl if small)
3. `feat: add generation base-vs-adapted eval to Silk ft-lora`
4. `test: cover Silk ft-lora eval report schema and behavior`
5. `docs: remove TODO.md — plan executed` (only after G2 clears)

## GitHub issues log

- [ ] None yet.
