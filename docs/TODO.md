# Plan: Issue #341 — BFloat16 demo completeness: `--precision` flag, fp16 symmetry, conditional precision test

## Problem

Pr: #340 added BFloat16 transformer inference to `NivaraInference` and fixed a real token-ID
corruption bug in `Embedding<T>` (narrow-precision dtypes like BFloat16/Half can't represent
vocabularies ~30k, so token IDs must stay `int`). The `bf16` run mode works end-to-end and
`distilbert_sst` is verified 8/8 vs the F32 reference.

Issue #341 tracks deferred follow-ups (per our scoping conversation, after investigating):

1. **Check in the F32 reference fixtures** for base DistilBERT + MiniLM so `bf16` modes can show a
   quantitative cosine diff. — **Declined by design**: `git check-ignore` shows
   `samples/data/distilbert/` and `samples/data/minilm/` are **entirely gitignored**
   (`.gitignore:352-355`) because they hold the ~268 MB weights. Their generated `.bin` fixtures can
   never be tracked. The README already states these are "not checked into the repo." Decision: do
   **not** check them in; instead document how a curious user generates them locally, and rely on
   graceful degradation ("fixture not found; skipping diff").
2. **(Optional stretch)** end-to-end `DistilBertForSequenceClassification<BFloat16>` inference test
   guarding the fix. — Subsumed by task 3 below, but made **conditional on the weight file existing**
   so it never breaks default/CI runs (user-requested).
3. **Generalize the precision flag** to `--precision f32|bf16|fp16` and add an `fp16`/`Half` mode
   for symmetry with `bf16`. — **Agreed** to implement.

## Proposed changes

### Task A — `--precision` flag + `Half`/fp16 symmetry (`samples/NivaraInference/Program.cs`)

Today the precision mode is positional (`bf16` as `args[1]`). Generalize to a `--precision
f32|bf16|fp16` argument while keeping `bf16` as a backward-compatible alias:

- Parse `--precision` (or a bare `bf16`/`fp16`/`half` positional) into a precision selector.
- Load weights at the chosen element dtype:
  - `f32` → `SafeTensorsLoader.Read(modelPath)` (already the default `float`).
  - `bf16` → `SafeTensorsLoader.Read<BFloat16>(modelPath)`.
  - `fp16` → `SafeTensorsLoader.Read<Half>(modelPath)` (already supported by the loader:
    `DtypeToArray<Half>` converts on-disk F32 via `ConvertF32<Half>` → `T.CreateChecked`, same
    mechanism as BF16; F16/BF16 on-disk dtypes also widen losslessly).
- Route `minilm`, `distilbert`, `distilbert_sst` to a precision-parameterized run:
  - Add `Run*Half`/`fp16` methods parallel to the existing `Run*BFloat16` methods for all three
    text models (`RunMiniLMHalf`, `RunDistilBertHalf`, `RunDistilBertSstHalf`), using `<Half>`:
    `MiniLMDistilled<Half>.LoadWeights<Half, Half>`, `BertEncoder<Half>` +
    `DistilBertLoader.LoadEncoderWeights<Half, Half>`, and
    `DistilBertForSequenceClassification<Half>.LoadWeights<Half>`.
  - `Half` path uses the exact-int token array (`Array.ConvertAll(tokenIds, x => (int)x)`) and a
    `Half` mask tensor — same as the BF16 path (Half is exact only to 2048; vocab ~30k).
- Keep the F32 modes and `bf16` alias working so the documented quick-start is unchanged.
- Report chosen precision + halved weight memory in the printed summary.

### Task B — `Half` (fp16) helpers in `DistilBertSst.cs`

- `LoadHalf(...)` → `DistilBertForSequenceClassification<Half>` from the F32 on-disk weights via
  `LoadWeights<Half>` (parallel to `LoadBFloat16`).
- `PredictLogitsHalf(model, tokenizer, text, maxLen)` — exact-int `Forward(intIds, mask, 1, len)`
  with a `Half` mask (parallel to `PredictLogitsBFloat16`).
- `SaveHalfCompareOutput(...)` — writes logits+softmax probs for the 8 SST-2 compare sentences
  (parallel to `SaveBFloat16CompareOutput`).

### Task C — conditional precision-inference regression test (`tests/Nivara.Tests/AutoDiff/DistilBertPrecisionInferenceTests.cs`)

Runs **only if the ~268 MB weight file exists** (never breaks default/CI), matching the established
`Assert.Ignore` pattern in `AutoDiff/PerfTests.cs`:

- Resolve `samples/data/distilbert_sst/{model.safetensors,config.json,vocab.txt}` relative to
  `TestContext.CurrentContext.TestDirectory` (same `..\..\..\..\..` walk as `PerfTests`).
- `Assert.Ignore("... not found; skipping ...")` if any is missing; wrap `SafeTensorsLoader.Read`
  in try/catch that `Assert.Ignore`s on load failure (mirror `PerfTests.DistilBert_Inference_Latency`).
- Two tests, building the F32 model + a narrow model and asserting **argmax agreement 8/8** over
  `DistilBertSst.CompareSentences`:
  - `DistilBertSst_HalfInference_PreservesFloatArgmax` — `<Half>` via the exact-int path.
  - `DistilBertSst_BFloat16Inference_PreservesFloatArgmax` — `<BFloat16>` via the exact-int path.
- Compare Nivara-F32 vs Nivara-narrow (NOT vs the gitignored Python fixture) so the test runs from
  just the weight file, avoids tracked-fixture issues, and still locks in the correctness property
  the bf16/fp16 demo must show (narrow precision preserves every prediction).

### Task D — README (`samples/NivaraInference/README.md`)

- **Reference-fixture generation** note (the "corner case" for curious users) for:
  - base `distilbert` → `python samples/NivaraInference/Python/distilbert_compare.py`
    → `samples/data/distilbert/last_hidden_state_py.bin`
  - `minilm` → `python samples/NivaraInference/Python/minilm_compare.py`
    → `samples/data/compare_minilm_embeddings_py.bin`
  - `distilbert_sst` → `python samples/NivaraInference/Python/distilbert_sst_compare.py`
    → `samples/data/compare_distilbert_sst_py.bin`
  - Clarify these are gitignored, generated locally on demand, and the demo degrades gracefully
    without them.
- Update Quick start / Usage / BFloat16 sections for `--precision` + `fp16`/`half` usage; note the
  exact-int token-ID correctness applies to Half too.

## Verification

- `dotnet build Nivara.slnx` (after each task; low cost, run without asking).
- Manual (Release, weights present locally under `samples/data/`):
  - `--precision fp16` / `half` runs for `minilm`, `distilbert`, `distilbert_sst` produce sensible
    output; SST-2 argmax preserved.
  - `bf16` alias and F32 default still work.
  - Compare modes degrade gracefully when fixtures absent.
- Ask before `dotnet test` (NUnit suite): targeted run of `DistilBertPrecisionInferenceTests` +
  existing `PerfTests` weight-dependent tests to confirm the conditional-skip pattern; then full
  suite before conclusion (human-gated).

## Planned commits

1. `feat(samples): validate --precision flag parsing + fp16/half aliases in NivaraInference`
   (Task A arg parsing + routing + Task B Half helpers).
2. `feat(samples): add fp16 run modes for MiniLM/DistilBERT/SST-2` (Task A run methods + Program.cs
   wiring).
3. `test: add conditional F32-vs-half/bf16 argmax inference tests for DistilBERT SST-2` (Task C).
4. `docs: document --precision/fp16 and reference-fixture generation in NivaraInference README`
   (Task D).

## Blast radius

- **Sample only — no engine source changes.** Files touched:
  - `samples/NivaraInference/Program.cs` — arg parsing + new `Run*Half` methods (additive; bf16
    aliases preserved).
  - `samples/NivaraInference/DistilBertSst.cs` — additive Half helpers.
  - `tests/Nivara.Tests/AutoDiff/DistilBertPrecisionInferenceTests.cs` — new test file.
  - `samples/NivaraInference/README.md` — docs.
- Test project already references `Nivara.Samples` (verified in `.csproj`), so no new references.
- Downstream callers: none outside the sample. The `bf16` invocations documented in the README keep
  working (back-compat alias). Existing `PerfTests` weight-dependent tests are unaffected.
- Tests covering the touched surface: `EmbeddingBFloat16Tests` (token-ID fix), `BFloat16Tests`,
  `PerfTests.MiniLm_Inference_Latency` / `DistilBert_Inference_Latency` (weight-skip pattern to
  mirror), and the new `DistilBertPrecisionInferenceTests`.

## GitHub issues log

- None yet. If Task A/C surfaces deferred work (e.g. `--precision` flag not applied to vision
  models, a `--precision` form that should also cover `benchmark`/`compare`), file a tracked issue
  here immediately (`gh issue create --repo khurram-uworx/Nivara`) rather than holding it in memory.
