# TODO: Align Qwen synthetic CPU/Silk benchmark reporting (#1454)

> Plan for issue #1454. As each task executes, if you find deferred work or a concern, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number in the GitHub issues log below — don't rely on memory or wait until the end of the plan, as compaction during execution can lose important items.

## Problem

SmolLM synthetic benchmarks report consistent fields on CPU and Silk (warmup + 3 runs + median, same 64-prompt/32-generated workload, prefill/decode/total `tok/s`). Qwen synthetic weights/config already exist (`QwenSyntheticConfig`, real Qwen2.5-0.5B shape; `Qwen.SynthesizeTensors<T>`), but benchmark reporting is inconsistent:

| Needed | SmolLM now | Qwen now |
|---|---|---|
| Synthetic Qwen 0.5B shape | n/a | Yes (`QwenSyntheticConfig`) |
| CPU warmup + 3 runs + median, prefill/decode/total `tok/s` | Yes | Partial (cached vs full, 3-run median, but reports `ms/token`; decode tok/s not primary) |
| Silk warmup + 3 runs + median, prefill/decode/total `tok/s` | Yes | No (single warmup + single timed run, `ms/token`, no prefill/decode split) |
| Same CPU/Silk workload fields | Yes | Not quite (`MaxNewTokensSynthetic` is 24, Silk warmup is 8 tokens) |

A consistent Qwen-shape synthetic baseline is needed before deciding whether Silk decode fusion/dispatch reduction is warranted for Qwen.

## Blast radius

- `samples/NivaraInference/Qwen.cs`
  - `MaxNewTokensSynthetic` const (24 -> 32) — used only by synthetic benchmark paths.
  - `GenerateSilk` (public, returns `List<int>`) — keep signature; factor into timed helper for benchmark reporting. Callers: `RunToolsSilk`, `RunPlainSilk`, `RunDistillSilk`, `RunBenchmarkSilk`, synthetic benchmark.
  - `RunSyntheticBenchmarkSilk` — reporting/workload alignment.
  - `RunSyntheticBenchmark<TModel,TWeight>` — workload/label alignment; delegates to `RunDecodeBenchmark`.
  - `RunDecodeBenchmark` / `TimeGeneration` / `TimeResult` — shared by real-weight `RunBenchmark` and `RunPlainBenchmark`; format change propagates to those CPU benchmark surfaces. Add decode tok/s while preserving cached-vs-full and speedup signal.
- `samples/NivaraInference/Program.cs` — CLI dispatch already routes `qwen --synthetic-weights benchmark [--silk]`; no dispatch change expected.
- `tests/Nivara.Tests/...` — `InferenceCliArgumentTests` only if CLI surface changes (not expected). No core `src/Nivara` behavior change.
- Docs — check for stale Qwen synthetic benchmark output examples; update only if needed.

## Proposed changes

### A. Silk synthetic benchmark reporting (`Qwen.cs`)

Add timed Silk generation helper and align `RunSyntheticBenchmarkSilk` to SmolLM format:

```csharp
public static (List<int> Ids, long PrefillMs, long DecodeMs) GenerateSilkTimed(...)
{
    session.ResetCache();
    var prefillSw = Stopwatch.StartNew();
    int next = useKvCache ? session.ForwardPrefillArgmax(promptIds.ToArray()) : session.ForwardLastArgmax(promptIds.ToArray());
    prefillSw.Stop();
    ...
    var decodeSw = Stopwatch.StartNew();
    ... decode loop ...
    decodeSw.Stop();
    return (gen, prefillSw.ElapsedMilliseconds, decodeSw.ElapsedMilliseconds);
}

public static List<int> GenerateSilk(...) => GenerateSilkTimed(...).Ids;
```

`RunSyntheticBenchmarkSilk`: 64 prompt, 32 generated (`MaxNewTokensSynthetic`), warmup + 3 timed runs + median. Report prefill ms, decode ms, decode tok/s, total tok/s. Precision label clear (`BFloat16 storage / F32 compute` for BF16/native; `F32` for F32).

### B. CPU synthetic benchmark reporting (`Qwen.cs`)

- Change `MaxNewTokensSynthetic` from 24 to 32 for workload consistency with SmolLM/issue.
- Update `RunDecodeBenchmark`/`TimeResult`/`TimeGeneration` to report decode `tok/s` consistently (not only `ms/token`).
- Keep cached-vs-full comparison and speedup signal; add consistency rather than losing information.
- Primary comparable fields: prefill ms, decode ms/tok/s, total tok/s.

```csharp
// TimeResult gains DecodeTokS or report computes it
double decodeTokS = r.DecodeMsTok > 0 ? 1000.0 / r.DecodeMsTok : 0;
// report: prefill {ms} ms, decode {ms/token} ms/token ({tok/s:F2} tok/s), total {tok/s:F2} tok/s
```

### C. CLI/parity surface

- Confirm `qwen --synthetic-weights benchmark [--silk]` dispatch unchanged.
- Update/extend `InferenceCliArgumentTests` only if CLI surface changes.
- No core `src/Nivara` behavior change expected.

### D. Baseline capture

- Run CPU and Silk synthetic Qwen benchmarks and record raw outputs (human-run unless explicitly approved; record machine/device/runtime).
- Post results to issue / PR for fusion planning.

## Verification

- [ ] `dotnet build Nivara.slnx -c Release`
- [ ] `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- [ ] Targeted tests for touched inference/sample areas (ask before long tests; user gave go-ahead for targeted tests in this session).
- [ ] `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"` (targeted)
- [ ] CPU/Silk synthetic Qwen benchmark raw outputs captured with consistent fields (human-run unless explicitly approved).

## Planned commits

1. `docs: plan Qwen synthetic benchmark reporting alignment (#1454)`
2. `Align Qwen Silk synthetic benchmark reporting with SmolLM format`
3. `Align Qwen CPU synthetic benchmark decode tok/s reporting`
4. `docs: remove TODO.md — plan executed` (after G2 reviews clear)

## GitHub issues log

- [ ] #1454 — Align Qwen synthetic CPU/Silk benchmark reporting for fusion planning (this plan)

## G1 grounding notes (to complete before implementation)

- Ground API/behavior claims via microsoft-learn MCP and code-memory navigation.
- Confirm `Stopwatch`/timing patterns are appropriate (BCL). No new .NET preview APIs expected.
- Flag decisions/red flags to human before implementation.
