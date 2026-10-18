# TODO — Parameterize the decision-model and add `typesafe benchmark`

Branch: `khurram/typesafe` (continuing the landed work — the demo move and Ollaya dispatch are
already on it and pushed as PR #675).

## Problem

1. The `typesafe` demo hardcodes one model per server — Ollaya `laya:en`, Ollama `tev1:0.8b`.
   Both servers hold models the demo cannot reach, so a user with a different decision model
   installed has no way to run it.
2. There is no way to measure a decision model's latency through this CLI, so comparing decision
   models or the two backend engines (Ollaya vs Ollama) means eyeballing the assessment output.

## Decisions confirmed with the human

- Model selection is a **single optional `--model <name>` scoped to `typesafe`** (not per-server
  flags, not encoded into the URL). Defaults stay Ollaya `laya:en` / Ollama `tev1:0.8b`.
- Benchmark = **"warmup + median"**, the repo's existing protocol.
- Ollaya is started by the human (`ollaya serve`), not by this work.
- **Restrict to the told models.** Both engines currently hold exactly one decision model each —
  Ollaya `laya:en`, Ollama `tev1:0.8b` — and not every model on an engine is a decision model
  (Ollama's `beam:cloud` answers `400 beam is not a decision model`). So examples, docs and
  verification use only those two; `--model` stays a general override because the engine is the
  authority on what is a decision model.
- **CLI shape reworked to `typesafe [engine] [mode] [--url <url>] [--model <name>]`**, replacing
  phase 1's `--ollama [url]`. The human's reasoning: a third engine later should be a row in a
  table, not another boolean flag. Three things forced this over keeping `--ollama`:
  (a) its *optional* value is a non-flag token that competes with the mode, which really did
  misparse `typesafe --ollama benchmark` as a URL and reject a well-formed command — a
  required-value `--url` has no such token; (b) one boolean flag per engine does not scale to a
  third provider; (c) `--url` says what it does where `--ollama <url>` did not. **No alias** for
  `--ollama`: this is the first implementation and it never merged, so the flag simply does not
  exist. The human also settled the terminology: *engine* (not server/provider) in prose.
- **No `--warmup`/`--iters` on `typesafe benchmark`** — see #676; `laya benchmark` is explicitly
  *not* part of that issue, since it is an in-process path rather than a round trip.

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
3. **Docs — one new document, not two patched ones** (the human's call, after the content landed in
   both the sample README and `docs/LAYA.md`): **`docs/DECISION-MODELS.md`** holds the ecosystem
   (what a decision model is, the TypeSafe route, the two engines, "not every model is a decision
   model"), the demo's command line, and `typesafe benchmark`'s protocol **plus the measured
   numbers from PR #675**. `samples/NivaraInference/README.md` keeps a table, five commands and a
   link; `docs/LAYA.md`'s ecosystem section shrinks from ~20 lines to a pointer, so neither is
   polluted. Re-pinned `DocReferenceTests`: `ExpectedFileCount` 709 → 710 (+ the new file) and
   `ExpectedDocumentReferenceCount` 360 → 363 (+1 README link, −1 README's old `docs/LAYA.md` link,
   +1 in `docs/LAYA.md`, +2 `docs/LAYA.md` mentions in the new doc — the extractor only counts
   `docs/…md` occurrences, so the parts sum to +3).

## Commits (status)

1. `3deb0fce` **`--model <name>` (typesafe-only)** — value consumed even when nonsense (the `--text`
   pattern), scope guard, resolved into `TypeSafeClientOptions.DefaultModel` (the SDK's per-request
   resolution at `TypeSafeClient.cs:72` makes this the one parameterization point; no SDK change),
   usage text, two gate cases. **Live-verified on both engines.**
2. **CLI rework to `typesafe [engine] [mode] [--url <url>] [--model <name>]`** — `--ollama` removed
   (no alias; it never merged), engine as a positional over a table in `TypeSafe.cs`, `--url` as a
   required value, `--url`/`--model` scope guards, engine-positional + `--url` gate cases. Verified:
   build clean, 4 targeted cases pass.
3. **`typesafe benchmark`** — `SupportedModes["typesafe"]` gains `"benchmark"`, runner in
   `TypeSafe.cs` (1 warmup + 3 timed per item, per-item median/min/max, overall median + items/s,
   exit 1 when every request failed), shared question set/news items so the timed request is the
   demo's request, usage text, `--warmup`/`--iters` rejection cases. Verified: build clean,
   6 targeted cases pass.
4. **Docs** — `docs/DECISION-MODELS.md` new; README and `docs/LAYA.md` slimmed to pointers; pins
   re-set as above; this plan updated.

## Verification steps

- `dotnet build … -c Release` after each code commit ✅ (0 warnings, 0 errors) — **Release only**,
  per the Debug-build rule for any timing-bearing measurement.
- Targeted gate: `dotnet test -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"` →
  **44 passed / 2 failed**, the two pre-existing environmental Qwen checkpoint cases (#673);
  total 44 → 46 with the new cases ✅.
- Live, on **both engines** (happy path deliberately ungated — CI has no engines), restricted to the
  told decision models — all exit 0 ✅: `typesafe`, `typesafe ollama`,
  `typesafe --url http://localhost:11435 --model laya:en`, `typesafe benchmark`,
  `typesafe ollama benchmark`.
- Reject paths, all exit 1 with their pinned message ✅: `typesafe bogus`, `typesafe ollama bogus`,
  `typesafe ollama ollaya`, `minilm ollama`, `minilm --url …`, `typesafe --url` (missing),
  `typesafe --url not-a-url`, `typesafe --model` (missing/blank), `minilm --model …`,
  `typesafe benchmark --warmup 1`.
- Benchmark results posted to PR #675 and copied into `docs/DECISION-MODELS.md`; **two runs
  recorded** because absolute latency drifted ~30% run to run while the ordering held
  (Ollama 3.4× then 2.8× Ollaya). The doc says to re-run both rather than trust the absolutes.
- Remaining: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`, the
  documentation snippet gate, and `DocReferenceTests`. Full suite only with the human's go-ahead.

## Blast radius

- `samples/NivaraInference/Program.cs` — one parse branch (`--url`), one engine-positional branch,
  two scope guards, dispatch, usage lines. The pinned `--warmup`/`--iters`, `--ilgpu`,
  `--precision` and mode-allow-list messages are untouched.
- `samples/NivaraInference/TypeSafe.cs` — engine table + `Resolve`/`CreateClient`/`BuildQuestions`
  helpers, `Run`, `Benchmark`. Sole caller is the one dispatch site in `Program.cs`.
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs` — `Ollama_Beyond…` replaced by
  `Url_BeyondItsScopeOrWithoutAValue_Rejected`; new `EnginePositional_…` and `Model_…`;
  `typesafe benchmark` cases inside `SeqWarmupIters_OnNonModernBertModel_Rejected`; `benchmark` is
  no longer asserted as an unsupported mode for `typesafe` (it is one now).
- docs: `docs/DECISION-MODELS.md` (new), `docs/LAYA.md`, `samples/NivaraInference/README.md`.
  **No `src/` changes; no public library contract; CHANGELOG not required.**

## GitHub issues log

- [x] #676 — created: allow `--warmup`/`--iters` on `typesafe benchmark` (pinned to modernbert by
  #474's gate message) so users can raise the sample count against a slow engine. Scoped to
  `typesafe` only — `laya benchmark` is an in-process path and deliberately out of scope.
