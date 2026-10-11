# Plan: #580 — reject the last four inert `NivaraInference` options

- **Branch:** `khurram/580` (off `khurram/577`, which carries the #571 guards + seedable init)
- **PR target:** `khurram/577` (stacked PR, per the human's instruction)
- **Issue:** https://github.com/khurram-uworx/Nivara/issues/580

## Problem

#571 rejected `--teacher-examples` / `--seed` / `--force` outside `qwen distill`, closing the
silent-acceptance failure mode for the three flags `Qwen.RunDistill` consumes. Four more
`NivaraInference` options are still parsed for every model but read only on a narrower path, so the
operator cannot tell from the run that the option had no effect:

- `--plain` — consumed only by the `qwen` case (`Program.cs`: the `--synthetic-weights` branch and
  `RunPlain` / `RunPlainBenchmark` at 448-469). `minilm --plain` is accepted and does nothing.
- `--text` — consumed only by the qwen prompt paths (`Qwen.RunPlain` at 469, `Qwen.RunTools` at 470).
- `--no-kv-cache` — consumed only by `Qwen.RunTools(..., useKvCache: !noKvCache, ...)` at 470.
- `--simd-widen` — flips the global `NivaraPrimitives.UseWidenSimd` at 325, but only a bf16/fp16 run
  reads it; an f32 run is unchanged.

The issue's proposed shape was "reject unless the model/mode that consumes it is selected, naming
the flag and the requirement, before any disk touch."

## Grounding corrections to the issue's proposed shape

Two details in the issue body do not match the dispatch switch, confirmed by reading it:

1. **The tools path is `qwen` *and* `qwen tools`, not "bare qwen" only.** `case "qwen"` has no
   `tools` branch — `mode == "tools"` falls through to the final `return Qwen.RunTools(...)` exactly
   as bare `mode == ""` does. So a guard that required `mode.Length == 0` would reject the README's
   canonical `qwen tools --no-kv-cache`. The consumed condition is instead
   `isQwen && !syntheticWeights && mode is not ("distill" or "benchmark")`.
2. **`--simd-widen` is not SmolLM-only.** `TensorsHelper.DotFor<T>` (line 167) routes every
   `BFloat16`/`Half` dot through `WidenPrimitives.Dot`, which reads the global set at 325, so the
   flag is effective for `distilbert` / `distilbert_sst` / `laya` / `modernbert` at bf16/fp16 —
   `docs/BFLOAT16.md:353` documents `distilbert_sst --precision bf16` "with and without
   `--simd-widen`". Rejecting unless `modelType == "smollm"` would regress a documented lever. The
   precise axis is the **precision**; the human chose precision-based rejection.

Also confirmed: `--text`'s parser branch only consumes its value when `i + 1 < args.Length`, so a
trailing `--text` with no value falls through to the positional-mode branch and is read as a mode —
`minilm --text` silently runs `minilm` with mode `--text`. The guard cannot be sound unless that
branch is fixed the way `--teacher-examples` / `--seq` already are.

## Proposed changes

### 1. Parser — `samples/NivaraInference/Program.cs`

Add presence tracking beside `text` (mirroring #571's `seedSet`: the `""` default hides an explicit
value):

```csharp
string text = "";
bool textSet = false;
```

Make the `--text` branch consume its value the way the other value flags do, so a missing value can
never be read as a mode name (and so presence is detectable):

```csharp
else if (args[i] == "--text")
{
    if (i + 1 >= args.Length)
    {
        Console.Error.WriteLine("--text needs a value.");
        return 1;
    }
    text = args[i + 1];
    textSet = true;
    i++;
}
```

### 2. Guards — new block after the #571 guards (after line 254), before `modelDir` at 256

```csharp
// #580: the last options parsed for every model but consumed on one path. Same reasoning as
// #571: a flag scoped to one model's mode is a typo elsewhere, and a typo should cost a second
// rather than a 1.5 GB safetensors read.
if (plain && !isQwen)
{
    Console.Error.WriteLine("--plain is a 'qwen' option only: no other model has a plain-prompt surface.");
    return 1;
}

// --text and --no-kv-cache ride the qwen prompt paths: RunPlain / RunTools are reached for
// 'qwen', 'qwen tools' and any non-benchmark/distill mode; RunTools is the only KV-cache consumer.
bool qwenPromptPath = isQwen && !syntheticWeights && mode is not ("distill" or "benchmark");
string got = $"{modelType}{(mode.Length > 0 ? " " + mode : "")}";

if (textSet && !qwenPromptPath)
{
    Console.Error.WriteLine($"--text is a 'qwen' / 'qwen tools' prompt option only; got '{got}'.");
    return 1;
}

if (noKvCache && (!qwenPromptPath || plain))
{
    Console.Error.WriteLine($"--no-kv-cache is the 'qwen tools' path only; got '{got}'.");
    return 1;
}

if (simdWiden && precision == "f32")
{
    Console.Error.WriteLine("--simd-widen only changes a bf16/fp16 run (it enables the widen-compute-narrow SIMD kernels for BFloat16/Half); an F32 run is unchanged. Add --precision bf16|fp16 or drop the flag.");
    return 1;
}
```

The existing `--plain` mode guard at 452-456 stays unchanged (the issue asked it to). `--precision`
stays model-wide and is not guarded.

### 3. Help text

- In the `Qwen options:` block, after the #571 negative line (182):
  `  (--plain / --text / --no-kv-cache are 'qwen' options only; --no-kv-cache is the 'qwen tools' path only.)`
- In the `SIMD (narrow-float models)` block, after 193:
  `  (--simd-widen is a no-op for --precision f32; only a bf16/fp16 run changes.)`

### 4. Changelog

`CHANGELOG.md` `[Unreleased]` → `### Fixed`: a `NivaraInference` entry (#580) naming the four
guards, the explicit-presence tracker, the `--text` missing-value fix, the help negatives, the
`--simd-widen` precision axis (and why it is not SmolLM-only), and "Sample CLI only — no
`src/Nivara` change."

## Blast radius

- **Single file changes behaviour:** `samples/NivaraInference/Program.cs` (`Main` argument parsing
  + dispatch pre-checks + help text). `Program.Main` has 0 in-repo relationship records (sample
  process entry point), so the blast radius is direct `dotnet run` invocations only.
- **No `src/Nivara` change; no public API.** `CHANGELOG.md` doc-only.
- **Behaviour change for previously-inert input:** the four rejections plus `--text`-without-value
  now exits `1`; valid invocations (`qwen`, `qwen tools`, `qwen --plain`, `qwen --text "…"`,
  `qwen --no-kv-cache`, `qwen tools --no-kv-cache`, `distilbert --precision bf16 --simd-widen`,
  etc.) are unchanged.
- **No test coverage impact:** the sample exe is not reachable from `Nivara.Tests`; there is no
  automated gate, exactly as #568/#571 recorded. Verification is by Release build + a manual
  exit-code matrix.

## Verification

- **Build:** `dotnet build samples/NivaraInference -c Release` (expect 0 warnings).
- **Reject matrix (expect exit 1 + the new message):** `minilm --plain`; `smollm --plain`;
  `minilm --text "hi"`; `qwen benchmark --text "hi"`; `qwen --synthetic-weights --text "hi"`;
  `qwen distill --text "hi"`; `minilm --no-kv-cache`; `qwen benchmark --no-kv-cache`;
  `qwen --plain --no-kv-cache`; `minilm --simd-widen`; `smollm --simd-widen`;
  `qwen --simd-widen`; `minilm --text` (missing value → "--text needs a value").
- **Pass-the-guard matrix (reach "Model file not found", i.e. the guard let them through):**
  `qwen`; `qwen tools`; `qwen tools --no-kv-cache`; `qwen --no-kv-cache`; `qwen --plain`;
  `qwen --text "hi"`; `qwen distill --seed 7`; `smollm --precision bf16 --simd-widen`;
  `distilbert --precision bf16 --simd-widen`.
- **Help:** `-h` renders both new negative lines.
- **Full suite:** ask the human first; always `--filter "Category!=Performance"`. No test changes
  are planned, so the suite is regression-only.

## Planned commits

1. `docs: plan #580 inert-flag guards in TODO.md`
2. `docs: record #580 G1 grounding in TODO.md` (G1 gate)
3. `fix(samples): reject Qwen-only --plain/--text/--no-kv-cache and F32 --simd-widen (#580)`
4. `docs: remove TODO.md — plan executed` (G2 gate, only after both reviews clear)

## G1 grounding (2026-10-11)

- **Exit-code convention:** [Main() return values](https://learn.microsoft.com/dotnet/csharp/fundamentals/program-structure/main-command-line#main-return-values)
  and [Environment.ExitCode](https://learn.microsoft.com/dotnet/api/system.environment.exitcode)
  both state "a nonzero value means there's an error" / "0 indicates success", so
  `Console.Error.WriteLine(...)` + `return 1` is the idiomatic reject for a bad option — consistent
  with the #474/#568/#571 guards this mirrors.
- **Blast radius re-confirmed via code-memory SQL:** `NivaraInference.Program.Main(string[] args)`
  (`Program.cs:21-475`) has **0** relationship records targeting it — it is the sample process entry
  point with no in-repo callers, so the guards affect only direct `dotnet run` invocations. The four
  consuming members (`Qwen.RunTools` `Qwen.cs:411`, `RunBenchmark` `541`, `RunPlainBenchmark` `560`,
  `RunPlain` `574`, `RunDistill` `708`) are reached only from `case "qwen"`.
- **Dispatch re-read for the tools path:** `mode == "tools"` has no branch in `case "qwen"` and falls
  through to `return Qwen.RunTools(...)`, exactly as bare `mode == ""` does. The guard condition is
  therefore `isQwen && !syntheticWeights && mode is not ("distill" or "benchmark")`, not
  `mode.Length == 0` as the issue's "bare qwen" wording implied — otherwise `qwen tools
  --no-kv-cache` (the README's canonical tools invocation) would be wrongly rejected.
- **`--simd-widen` is not SmolLM-only:** `TensorsHelper.DotFor<T>` (`TensorsHelper.cs:167`) routes
  every `BFloat16`/`Half` dot through `WidenPrimitives.Dot` → `ShouldWiden` → the global set at
  `Program.cs:325`; `KernelSelector.DetermineKernelType<T>` (`KernelSelector.cs:29`) does the same
  for the column path. So the flag is effective for `distilbert` / `distilbert_sst` / `laya` /
  `modernbert` at bf16/fp16, corroborated by `docs/BFLOAT16.md:353` (`distilbert_sst --precision
  bf16` argmax 8/8 "with and without `--simd-widen`"). The precision axis is the correct scope.
- **`--text` missing value is a soundness prerequisite:** the current branch only consumes its value
  when `i + 1 < args.Length`, so a trailing `--text` falls through to the positional-mode branch and
  becomes a mode — `minilm --text` silently runs `minilm` with mode `--text`. The guard can only see
  `--text` if the branch is fixed the way `--teacher-examples` / `--seq` already are ("the flag's
  value can never fall through to the positional-mode branch"). Treated as part of this change, not
  a separate issue; it is the same flag's handling.
- **Decisions surfaced and resolved with the human before implementation:** (1) `--simd-widen`
  scope — precision-based (reject on f32) rather than the issue's SmolLM-only wording, because the
  latter regresses a documented narrow-float lever; (2) `--text` / `--no-kv-cache` strictness —
  inert-free (reject whenever the consuming path is not selected). Both answered during planning.
- **No new red flags surfaced by grounding** beyond the two corrections above (already folded into
  the plan) and the two follow-ups filed below.

## GitHub issues log

- [x] #586 — `--precision bf16|fp16` is silently ignored by the vision models (mobilenet_v2/resnet18)
      (created while working on #580)
- [x] #587 — an unrecognised qwen mode is silently ignored and runs the tools path
      (created while working on #580)

> As each task executes, any deferred work or concern is filed immediately via
> `gh issue create --repo khurram-uworx/Nivara` and recorded here — not held in memory, because
> compaction during execution can lose it.
