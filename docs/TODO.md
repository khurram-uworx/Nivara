# Plan: Qwen `distill` teacher pass on the Silk (OpenCL) backend — issue #386

## Status

- Branch: `386`, cut from `main` at `27480b19`.
- Worktree: `C:\Users\khurram.aziz\.local\share\opencode\worktree\d3844b\386-silk-distill`
  (kept off `khurram/346` so that branch's uncommitted Quantile/storage work is neither
  disturbed nor compiled into any measurement).
- `samples/data/qwen2.5-0.5b-instruct` is a junction to the main checkout's checkpoint; the
  model directory stays gitignored and uncopied.

## Problem

Issue #386 asks for a full `qwen distill` teacher-labelling cycle. The filed cost model
(~178–206 s per sentence, ~1–1.5 h per cycle) was measured on the **full re-forward** path,
which `distill` does not use:

| Path (CPU, F32, 206-token tools prompt, 19 generated tokens) | Median total | Prefill | Decode |
|---|---:|---:|---:|
| KV-cached — the path `RunTeacherClassify` uses | 11,142 ms | 8,027 ms | 164.0 ms/token |
| Full re-forward — the path behind the issue's estimate | 206,475 ms | 9,957 ms | 10,342.9 ms/token |

The same session measured the Silk/OpenCL path (Iris Xe) against CPU on the teacher-shaped
call: 11,500 ms vs 11,142 ms — a tie. Silk wins only on the short plain prompt
(3,750 ms vs 4,452 ms, 1.19×). Numbers and caveats are posted on
[#386](https://github.com/khurram-uworx/Nivara/issues/386#issuecomment-6289830955).

So this change is **not** a performance fix. It is feature parity: `qwen` already exposes
`tools`, plain and `benchmark` on the hand-authored Silk kernel stack, and `distill` is the
one Qwen surface that hard-rejects `--silk`. Users on a machine where the iGPU is the
stronger device (the documented Ultra 7 255H row is ~4× faster than this Iris Xe on the same
shape) should not have to drop the flag to label their data.

## Proposed changes

### 1. Seam the teacher classifier (`samples/NivaraInference/Qwen.cs`)

Split `RunTeacherClassify` so prompt construction and label parsing are shared and only the
generation call is backend-specific. `LoadOrRunTeacher` then takes the classifier as a
`Func<string, string>` instead of `(model, config, tokenizer)`:

```csharp
static int[] BuildClassifyTurnIds(Gpt2BpeTokenizer tokenizer, string sentence)
{
    string toolJson = QwenChatTemplate.ToolJson(
        ClassifyToolName, ClassifyToolDesc,
        [("text", "string", null, true), ("label", "string", null, true)]);
    string systemMessage = QwenChatTemplate.BuildToolsSystemMessage(toolJson);
    string prompt = QwenChatTemplate.RenderFirstTurn(
        systemMessage, $"Classify the sentiment of this movie review: \"{sentence}\"");
    return tokenizer.Encode(prompt).ToArray();
}

static string ParseClassification(string text) { /* existing parse + fallback, unchanged */ }
```

`LoadOrRunTeacher(cachePath, sentences, limit, force, classify)` keeps the cache, the
resume logic, the per-sentence stopwatch and the `({seconds}s)` print exactly as they are —
so the printed time measures whichever backend is under test.

### 2. Extract the shared distill body

`RunDistill` currently loads the model itself. Split it:

- `RunDistill(tensors, modelDir, teacherExamples, force, seed)` — CPU entry point. Keeps the
  `LoadModel` call (the CPU teacher needs it), builds the CPU classifier closure, delegates.
- `RunDistillCore(config, tokenizer, modelDir, teacherExamples, force, seed, classify, deviceLine)`
  — everything after loading: cache/resume, labels, hashed features, student training inside
  `GradientUtils.Grad()`, linear baseline, DistilBERT SST-2 eval table, accuracy table.
- `RunDistillSilk(tensors, modelDir, teacherExamples, force, seed)` — Silk entry point:
  config + tokenizer only, one `SilkQwenSession` reused for all 18 calls, classifier closure
  over `GenerateSilk`.

The device line is a parameter because it is the only part of the output that differs:

```
Device: GPU (Silk)  Precision: F32 (BF16-upcast)  KV cache: on
```

`RunDistillSilk` deliberately does **not** call `LoadModel`: `LlamaLoader.Load<float, float>`
materializes a second ~1.9 GB host copy that a GPU teacher never reads. Session shape mirrors
`RunToolsSilk` (build stopwatch, `Device: {session.DeviceName}`, config line).

Reuse across calls is safe: `GenerateSilk` begins with `session.ResetCache()`.

### 3. CLI dispatch (`samples/NivaraInference/Program.cs`)

- `SilkModes["qwen"]` → `["", "benchmark", "tools", "distill"]`.
- In the `useSilk` branch of `case "qwen"`, route `mode == "distill"` to `RunDistillSilk`
  before the plain/tools fallbacks.
- Usage text under `--silk` gains `distill`.
- `--precision bf16` stays rejected by the existing central Silk guard (Silk is F32-only).

### 4. Tests (`tests/Nivara.Tests/`)

Change → test coupling, per AGENTS.md:

- `InferenceCliArgumentTests`: add `["qwen", "--silk", "distill"]` to `AcceptedQwenCases`
  and delete `SilkQwen_DistillMode_Rejected` (the behaviour it pins is now gone).
- `Silk_RejectedBeforeModelFileLookup` currently uses `qwen --silk distill` as its vehicle;
  it must move to a still-rejected Silk mode (`qwen --silk compare`) so it keeps testing the
  guard-ordering property it exists for.
- Add `SilkQwen_DistillPrecision_Rejected` for `qwen --silk distill --precision bf16`,
  naming the precision guard. A rejection that fires on the mode allow-list would not pin
  the precision path.
- `SilkQwenParityTests`: add a cached-decode parity case (prefill, then one
  `ForwardCached` step) on synthetic weights. The existing test covers prefill only; the
  teacher loop is prefill + N cached steps, so the step that was unmeasured is the one the
  feature depends on. Assert argmax and logits within the existing `1e-3` relative bound.

### 5. Docs

- `docs/QWEN.md` — the Silk sentence ("tools, plain, benchmark") gains `distill`.
- `docs/SILK.md` — the Qwen mode list gains `distill`, and the note about which modes stay
  rejected drops `distill` if it names it.
- `samples/NivaraInference/README.md` — add the `qwen --silk distill` command next to the
  CPU one.

## Blast radius

| Change | Affected | Depends on it | Covering tests |
|---|---|---|---|
| `LoadOrRunTeacher` signature | `RunDistill` only | — | distill CLI tests, `QwenDistillStudentParityTests` |
| `RunDistill` split | `Program.cs:850` | CLI distill | `InferenceCliArgumentTests` |
| `RunDistillSilk` (new) | `Program.cs` dispatch | CLI | new Accept/precision cases |
| `SilkModes["qwen"]` | all `qwen --silk` invocations | CLI usage text | `Silk*` guard tests |
| Parity test | test suite | — | itself |

No `src/` change: every API used (`SilkQwenSession`, `GenerateSilk`, `ToSilkConfig`) already
exists and is unchanged. Core library behaviour is untouched, so the public surface of
`Nivara` / `Nivara.Extensions` does not move.

Memory note for the docs: a Silk distill run holds ~3.5 GB of device buffers (805 MB KV cache
for `maxPos = 32768` + ~2.7 GB of weights) on top of the host tensor dictionary. It also does
**not** need the CPU model, which item 2 preserves.

## Verification

1. `dotnet build Nivara.slnx -c Release` — clean.
2. Targeted: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests|FullyQualifiedName~SilkQwen"`.
3. Docs namespace (docs changed): `--filter "FullyQualifiedName~Nivara.Tests.Docs"`.
4. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
5. Manual parity on the real checkpoint: back up `qwen_distill_labels.json`, run
   `qwen --silk distill --teacher-examples 12 --force`, then restore and run the CPU form;
   diff the two label maps. Identical labels are the gate; a divergence is a finding, not a
   tolerance to widen.
6. Perf: the two runs above also produce per-sentence timings (the CLI prints them), giving
   the CPU-vs-Silk teacher-pass comparison on the real path. Record both on #386.

Runs 5–6 are ~12 minutes of teacher time and are the only long-running steps.

## Planned commits

1. `docs: plan Silk distill backend in TODO.md`
2. `refactor(qwen): seam the distill teacher classifier behind a generate delegate`
3. `feat(qwen): run the distill teacher pass on the Silk backend`
4. `test(qwen): pin --silk distill acceptance, precision guard and cached-decode parity`
5. `docs(qwen,silk): document the Silk distill surface`
6. `docs: remove TODO.md — plan executed` (only after G2 clears)

## GitHub issues log

- [ ] #386 — Qwen distillation full cycle: this branch adds the Silk teacher path and its
  measurements; the full cycle itself stays open.
