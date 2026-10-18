# TODO — Parameterize the decision-model and add `typesafe benchmark`

Branch: `khurram/typesafe` (continuing the landed work — the demo move, Ollaya dispatch and
`--ollama` support are already on it and pushed as PR #675).

## Problem

1. The `typesafe` demo hardcodes one model per server — Ollaya `laya:en`, Ollama `tev1:0.8b`.
   Both servers hold models the demo cannot reach, so a user with a different decision model
   installed has no way to run it.
2. There is no way to measure a decision model's latency through this CLI, so comparing decision
   models or the two backend servers (Ollaya vs Ollama) means eyeballing the assessment output.

## Decisions confirmed with the human

- Model selection is a **single optional `--model <name>` scoped to `typesafe`** (not per-server
  flags, not encoded into the URL). Defaults stay Ollaya `laya:en` / Ollama `tev1:0.8b`.
- Benchmark = **"warmup + median"**, the repo's existing protocol.
- Ollaya is started by the human (`ollaya serve`), not by this work.
- **Restrict to the told models.** Both servers currently hold exactly one decision model each —
  Ollaya `laya:en`, Ollama `tev1:0.8b` — and not every model on a server is a decision model
  (Ollama's `beam:cloud` answers `400 beam is not a decision model`). So examples, docs and
  verification use only those two; `--model` stays a general override because the server is the
  authority on what is a decision model.

## Grounding (verified before implementation)

- **SDK puts the model per request.** `TypeSafeClient.cs:72` resolves
  `Model = options?.Model ?? request.Model ?? Options.DefaultModel`. The demo calls the
  `SystemOneAsync(state, questions)` extension (`TypeSafeClientExtensions.cs:7`), which builds a
  request with `State`/`Questions` and no model — so the model is read from `DefaultModel` at
  construction. Setting `DefaultModel` is therefore the one place to parameterize; no SDK change.
- **Benchmark protocol already exists in this sample.** `Laya.cs:198` and `ModernBert.cs:197`
  both print `{warmup} warmup + {iterations} timed passes, median reported.` with defaults
  `ModernBert.cs:53 DefaultWarmup = 1` and 3 timed passes, then `Summarize` → `(median, min)`.
  `typesafe benchmark` mirrors that: **1 warmup + 3 timed passes per item, median (and min)
  reported**, measured around the `SystemOneAsync` round trip — i.e. client + transport + server
  inference, which is exactly what "compare the backends" means.
- **`--warmup` / `--iters` are pinned to modernbert.** `Program.cs:358` rejects them for any other
  model, and `InferenceCliArgumentTests.SeqWarmupIters_OnNonModernBertModel_Rejected` pins the
  message *"are options of 'modernbert benchmark' only"* for `minilm benchmark`. So
  `typesafe benchmark` uses the fixed protocol and does not accept them — extending the scope is
  tracked as deferred work (#676). Note `laya benchmark` / `modernbert benchmark` are **different
  paths**: they time in-process forward passes over weights, whereas `typesafe benchmark` times an
  HTTP round trip to a server, so the two are not comparable numbers and must be labelled as such.
  `--seq` stays modernbert-only (a shape parameter is meaningless for a fixed question set).
- **No `--model` flag exists** in `Program.cs` (verified) — nothing collides.
- **Both servers verified live**: `GET :11434/v1/models` → 9 models, of which only `tev1:0.8b` is a
  decision model (`beam:cloud` answers `400 beam is not a decision model`);
  `GET :11435/v1/models` → `laya:en`, the only model Ollaya holds.

## Planned commits

1. **`--model <name>` (typesafe-only).**
   - Parse in `Main` with the value consumed even when nonsense (same pattern as `--text`, so it
     cannot fall through to the positional-mode branch and be read as a mode).
   - Scope guard: `--model is a 'typesafe' option only; got '{got}'.`
   - `TypeSafe.Run(bool useOllama, string? ollamaUrl, string? model)` → resolve
     `model ?? (useOllama ? OllamaModel : OllayaModel)` and feed `TypeSafeClientOptions.DefaultModel`.
   - Banner prints the effective server, URL and model (already does; now driven by the override).
   - Usage: document `--model <name>` under the TypeSafe block.
   - Gate tests: `minilm --model laya:en` → scope reject; `typesafe --model` → value missing.
2. **`typesafe benchmark`.**
   - Add `"benchmark"` to `SupportedModes["typesafe"]`.
   - New benchmark runner in `TypeSafe.cs`: 1 warmup + 3 timed passes over each of the six news
     items, per-item median/min, plus an overall median and items/s; exit 1 if every request failed
     (same server hint as the assessment path).
   - Usage: document the mode.
   - Gate tests: `typesafe benchmark --warmup 1` → rejected by the existing modernbert-only scope
     (pins that the fixed protocol is not silently swappable); coverage doc line updated.
3. **Docs** — `samples/NivaraInference/README.md` (quick-start lines for `--model` and
   `typesafe benchmark`, and the section) plus a sentence in `docs/LAYA.md`'s ecosystem overview.
   Prose/bash only → no snippet-gate impact, but `DocReferenceTests`' pinned document-reference
   count moves if a `docs/…md` link is added (re-pin deliberately).

## Verification steps

- `dotnet build Nivara.slnx -c Release` after each code commit.
- Targeted: `dotnet test -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"`.
  Known pre-existing environmental failures: the two Qwen checkpoint cases (#673).
- Live, on both servers (happy path is deliberately ungated — CI has no servers), restricted to
  the told decision models: `typesafe`, `typesafe --model laya:en`, `typesafe benchmark`,
  `typesafe --ollama`, `typesafe --ollama --model tev1:0.8b`, `typesafe --ollama benchmark`.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` + the documentation
  snippet gate before concluding. Full suite only with the human's go-ahead.

## Blast radius

- `samples/NivaraInference/Program.cs` — one parse branch, one scope guard, one dictionary entry,
  usage lines. All existing guards and pinned messages untouched.
- `samples/NivaraInference/TypeSafe.cs` — `Run` signature, model resolution, new benchmark runner.
  Sole caller is the one dispatch site in `Program.cs`.
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs` — new reject cases + a coverage
  line. Existing cases unaffected (`typesafe` was already a key; `--model` is new).
- docs only otherwise. **No `src/` changes; no public library contract; CHANGELOG not required.**

## GitHub issues log

- [ ] #NNN — deferred: allow `--warmup`/`--iters` on `typesafe benchmark` (they are pinned to
  modernbert by #474's gate message), so users can raise the sample count for stable medians
  against a slow server. Create with `gh issue create --repo khurram-uworx/Nivara` while working.
