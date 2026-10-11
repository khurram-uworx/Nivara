# Plan: close the NivaraInference CLI silent-acceptance whirlpool (#586, #587, #589, #591)

Branch: `khurram/issues` (off `khurram/583`).

## Problem

`Program.Main` in `samples/NivaraInference` re-derives the CLI's supported surface in a series of
scattered `if` guards that are separate from the dispatch `switch`. The two drift, so each guard
that closes one hole opens the next. Measured on the built sample:

- `mobilenet_v2 --precision bf16` loads BF16 weights and runs **F32** (exit 0) — #586.
- `qwen garbag` falls through to `Qwen.RunTools` — #587.
- `mobilenet_v2 --precision bf16 --simd-widen` passes the (precision-based) `--simd-widen` guard and
  changes nothing (exit 0) — #591.
- Adjacent, same class: `mobilenet_v2 garbag` prints `Image not found` and exits **0**;
  `minilm`/`distilbert`/`distilbert_sst`/`modernbert`/`smollm` silently run the default path for an
  unknown mode; `laya compare --precision bf16` and `modernbert compare --precision bf16` run the F32
  compare and drop the precision; the `--synthetic-weights` scope, `qwen` fp16, `--gpu`+bf16/fp16 and
  `qwen` bf16-mode rejects all run **after** the model-file check.

#589 asks for a gate over "the remaining rejects". Because its own guidance is that new cases must
be machine-independent (weights may be absent), every reject must be observable before the first
disk touch — which forces the late guards to move up.

## Root cause

Knowledge the dispatch switch already owns ("which models read a narrow runner", "which flags a path
consumes", "which modes the GPU path supports") is duplicated in guards that do not share a source.
The fix: one small capability block, one pre-disk validation pass, dispatch reads the same predicates.

## Proposed changes

### 1. Capability block (new, in `samples/NivaraInference/Program.cs`)

```csharp
static readonly Dictionary<string, string[]> SupportedModes = new()
{
    ["mobilenet_v2"]   = ["", "benchmark", "compare", "compare_diag"],
    ["resnet18"]       = ["", "benchmark", "compare", "compare_diag"],
    ["minilm"]         = ["", "benchmark", "compare", "similarity"],
    ["distilbert"]     = ["", "benchmark", "compare"],
    ["distilbert_sst"] = ["", "benchmark", "compare", "predict"],
    ["laya"]           = ["", "benchmark", "compare"],
    ["modernbert"]     = ["", "benchmark", "compare", "compare_diag"],
    ["smollm"]         = ["", "benchmark", "ab", "generate"],
    ["qwen"]           = ["", "tools", "distill", "benchmark"],
};

static readonly Dictionary<string, string[]> GpuModes = new()
{
    ["minilm"]         = ["", "benchmark", "compare"],
    ["distilbert"]     = ["", "benchmark", "compare"],
    ["distilbert_sst"] = ["", "benchmark", "compare"],
    ["laya"]           = ["", "benchmark", "compare"],
    ["modernbert"]     = ["", "benchmark", "compare"],
};

// The dispatch selects a BFloat16/Half runner for these (model, mode) pairs.
static bool ConsumesNarrow(string model, string mode, bool synthetic) => model switch
{
    "minilm" or "distilbert" or "distilbert_sst" or "smollm" => true,
    "laya" or "modernbert" => mode is not ("compare" or "compare_diag"),
    "qwen" => mode == "benchmark" || synthetic,
    _ => false, // mobilenet_v2 / resnet18 have no narrow-float path
};

// --simd-widen changes the run only where a narrow runner reads the toggle and does not set it
// itself: smollm/qwen auto-enable it for narrow, so the flag is inert there.
static bool ConsumesWiden(string model, string mode)
    => model is "minilm" or "distilbert" or "distilbert_sst" or "laya" or "modernbert"
       && ConsumesNarrow(model, mode, synthetic: false);

static bool IsVisionModel(string model) => model is "mobilenet_v2" or "resnet18";
static bool IsSupportedMode(string model, string mode)
    => SupportedModes[model].Contains(mode) || (IsVisionModel(model) && mode.Length > 0 && File.Exists(mode));
```

### 2. One pre-disk validation pass

All `return 1` guards move above the `modelPath` / `File.Exists` check at the former line ~301.
Order (existing messages preserved verbatim):

1. `--seq`/`--warmup`/`--iters` scope (existing).
2. **new**: unknown model — `Unknown model type: {model}` (moved up from the switch default).
3. `--gpu` model scope (existing).
4. `--teacher-examples`/`--seed`/`--force` scope (existing).
5. `--plain` non-qwen (existing) and **new**: `--plain` with a qwen mode other than `benchmark`.
6. `--text` / `--no-kv-cache` scope (existing).
7. **new**: mode allow-list (`Unsupported mode '{mode}' for '{model}'; supported: …`; vision accepts
   an existing image path; qwen names `tools/distill/benchmark or none`).
8. **new**: `--gpu` + unsupported mode (`--gpu does not support mode '{mode}' for '{model}' …`).
9. `qwen` fp16 (moved down, message unchanged).
10. `--gpu` + bf16/fp16 (moved down, message unchanged).
11. `--synthetic-weights` scope (moved down: qwen benchmark only; non-qwen; qwen non-benchmark).
12. **new**: `--precision bf16|fp16` where the selected run is not narrow-capable — `!ConsumesNarrow`.
    Covers #586 (vision), `laya`/`modernbert` `compare`/`compare_diag`, and the qwen bf16 non-benchmark
    case (keeping its specific message).
13. **replaces** the old precision-only `--simd-widen` guard: reject unless
    `precision != "f32" && ConsumesWiden(model, mode)` — #591 (strict; excludes smollm/qwen/vision/`ab`).

### 3. Dispatch switch loses the now-dead duplicates

Delete, from the switch: the minilm `--gpu similarity`, distilbert `--gpu predict/compare_diag`,
distilbert_sst `--gpu predict`, laya `--gpu compare_diag` + arbitrary-gpu-mode + CPU `compare_diag`,
modernbert `--gpu compare_diag`, qwen synthetic-mode and qwen bf16-mode rejects (all superseded by
the pre-disk pass). The qwen case collapses to: synthetic → benchmark → distill → plain → tools.

## GitHub issues log

- [ ] #586 — vision models ignore `--precision bf16|fp16` (assigned to this branch)
- [ ] #587 — unrecognised `qwen` mode silently runs the tools path (assigned to this branch)
- [ ] #589 — extend the CLI argument gate to the remaining rejects (assigned to this branch)
- [ ] #591 — `--simd-widen` silent no-op on the vision models (assigned to this branch)

## Blast radius

- `samples/NivaraInference/Program.cs` — `Main` and new helpers; sample CLI only, no `src/Nivara`
  change. The sample builds inside `Nivara.slnx`; `NivaraInference.dll` is referenced by
  `tests/Nivara.Tests` (the gate spawns it).
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs` — extended.
- `samples/NivaraInference/README.md`, `CHANGELOG.md`, `docs/SMOLLM.md` — flag wording.
- Downstream: none in `src/Nivara`. The only coupling is the gate process launching the DLL.

## Verification

1. `dotnet build Nivara.slnx -c Release` (sample Release is required; Debug changes perf only here).
2. Targeted: `dotnet test -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"`.
3. Re-run the four issue reproductions; each expects exit 1 with the new message. A real image path
   and a valid narrow run still reach their runners.
4. Full suite: **ask the human first**; if run, carry `--filter "Category!=Performance"`.

## Planned commits

1. `fix(samples): reject inert and unrecognised NivaraInference options pre-disk (#586 #587 #591)`
2. `test(inference): extend the CLI argument gate to the full reject surface (#589)`
3. `docs: document the NivaraInference guard surface and the strict --simd-widen scope`

## Notes / accepted trade-offs

- `--simd-widen` is now rejected for `smollm`/`qwen` (their narrow runs auto-enable widen) and for
  `smollm ab` (which toggles it itself). This corrects the `docs/SMOLLM.md` claim that the flag is an
  A/B lever there; the `ab` mode remains the A/B.
- `runimageinference`'s `"Image not found"`/exit-0 path becomes unreachable from the CLI because the
  mode allow-list requires the path to exist; the method is left as a defensive no-op.
