# PLAN — `NivaraTypeSafeClient`: an in-process TypeSafe engine for Laya

Status: **proposed, not started**. Scope: one workflow, executed under `iterative-work`
(feature branch, one commit per logical change unit, G1 decisions raised before implementation,
two-gate review at G2).

Goal: add `NivaraTypeSafeClient`, an in-process `ITypeSafeClient` that answers a TypeSafe System One
question set by running the Laya decision head on Nivara's Silk (OpenCL) backend, and expose it as a
third engine (`nivara`) in the `typesafe` demo.

Correctness is checked against **Ollaya** (which serves Laya as `laya:en`). Ollama has no Laya model,
so it cannot serve as the reference for this engine.

---

## 1. Non-goals (this cut)

- CPU / ILGPU backends, or any backend-selection flag for the `nivara` engine. Silk (F32) only.
- Promoting the client into `Nivara.Extensions` — a separate follow-up issue (step 7).
- Serving HTTP / a loopback server. The client is in-process.
- Batched multi-state requests, multi-device concurrency, streaming answers.
- Narrow precision (`--precision bf16|fp16`) — the Silk Laya stack is F32 throughout.

## 2. Settled decisions

**D1 — in-process `ITypeSafeClient`, not a loopback server.** The SDK's typed surface is reused
directly: `QuestionSet`/handles and the `SystemOneAsync(string, QuestionSet)` extension on
`ITypeSafeClient`. The client implements `SystemOneAsync(SystemOneRequest, …)` and returns a
`SystemOneResponse`; `TypeSafeClientExtensions` builds the `QuestionSetResult`. `QuestionSetResult`
has an `internal` constructor — never construct it.

**D2 — `Confidence` is Laya's calibrated `AnswerConfidence`, not the entropy `Confidence`.** Laya's
docs state only `AnswerConfidence` is calibrated and safe to gate on. `ScoreAnswer.Confidence` gets
the same quantity. `NoulAnswer` has no confidence property, so it goes into `AdditionalData`.

**D3 — JSON state and JSON instructions are supported.** Serialized as the Python reference does —
`json.dumps(…, ensure_ascii=False, separators=(", ", ": "))` — because that string is what the
server-side stringification produces and what the prompt was validated against.

**D4 — the engine's model name is the exact Hugging Face repo id: `convaiinnovations/laya`.** The
local checkpoint is `samples/data/laya`. Ollaya serves the same weights under its own tag
(`laya:en`); the two engines are compared on answers only, never on the model name, so the two
spellings never need reconciling.

**D5 — Silk is implied, not flagged.** `--ilgpu` and `--silk` stay rejected for `typesafe` (existing
guards), and `typesafe nivara` runs the Silk backend without a flag. A backend selector is a later
change, deliberately deferred.

**D6 — verification is against Ollaya.** Ollama has no Laya model; Ollaya serves `laya:en`, the same
Laya weights the `nivara` engine loads from disk, so the parity check runs `nivara` in-process
against Ollaya over HTTP.

**D7 — the client lives in the sample app** (`samples/NivaraInference/NivaraTypeSafeClient.cs`).
`Nivara.Tests` already project-references `NivaraInference`, so the mapping and content logic is
unit-testable without a checkpoint or a GPU. Promotion into `Nivara.Extensions` is follow-up step 7.

**D8 — forwards are serialized.** The Silk session owns device buffers and is not re-entrant, so the
client takes a lock per request. `SystemOneAsync` returns a completed `Task` (the work is a
synchronous device forward); SDK-side parallelism (`SystemOneManyAsync`) is therefore safe but
sequential. Batched concurrency is a later feature, not a silent corruption now.

## 3. Files

New:

- `samples/NivaraInference/NivaraTypeSafeClient.cs` — the client, the model resource, the content
  serializer, and the Laya↔TypeSafe mapping.
- `tests/Nivara.Tests/Inference/NivaraTypeSafeClientTests.cs` — mapping + content tests (no model).
- `docs/plan/TYPESAFE-LAYA.md` — this file (see the pin arithmetic in §10; it is inside the docs gate's scan scope).

Changed:

- `samples/NivaraInference/TypeSafe.cs` — `Engine` gains a kind (HTTP vs in-process), the `nivara`
  row, a client factory that returns `ITypeSafeClient`, the `verify` mode, header/description text.
- `samples/NivaraInference/Laya.cs` — `AgentConfig`, `LoadAgentConfig`, `LoadEncoderConfig`,
  `LoadTokenizer` become `internal` so the client reuses the one authoritative parse (no second copy).
- `samples/NivaraInference/Program.cs` — typesafe guard additions (nivara `--url`, nivara `--model`,
  verify engine guard), the `verify` dispatch, and the help text.
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs` — new reject cases (§9) and the
  `verify` mode in the allow-list comment.
- `docs/DECISION-MODELS.md` — the engines table gains the `nivara` row; the command line gains
  `verify`; the benchmark section explains why the in-process row is not comparable.
- `docs/LAYA.md` — points at the demo as the in-process consumer of the head.
- `samples/NivaraInference/README.md` — a `typesafe nivara` section (not in the docs gate's scope).

Reused unchanged: `Nivara.Silk.SilkLayaSession.Open`, `ILayaGpuSession.Forward`,
`LayaPromptBuilder.BuildSequence`, `LayaCalibration.Decode`, `SafeTensorsLoader.Read` (F32),
`Gpt2BpeTokenizer.LoadFromTokenizerJson`, `ModernBertConfig.FromJson`.

## 4. Question mapping (TypeSafe → Laya)

| TypeSafe | Laya |
| --- | --- |
| `Question.Id` | `LayaQuestion.Id` |
| `Instructions` (text) | `Instructions` verbatim |
| `Instructions` (JSON) | `ContentToText` (D3 serializer) |
| `Instruction == null` | `string.Empty` (the field is required and the builder indexes into it) |
| `ChoiceQuestion.Criteria` (`label → content?`) | `ChoiceOptions` = `(label, description?)`, in the dictionary's declaration order |
| `NoulQuestion.Criteria.Yes` / `.No` | `NoulTrueCriterion` / `NoulFalseCriterion`; labels left `null` so Laya's defaults render. A null side keeps its default rather than borrowing the other side's text. |
- `ScoreQuestion.Criteria` (ordered list) → `ScoreCriteria`, one entry per level, `null` preserved.

`LayaQuestion.Id` is the question's key in `SystemOneRequest.Questions`, so answers key back to it.

## 5. Content (state / instructions) → string

One helper, `ContentToText(TypeSafeContent?)`:

- text content is returned verbatim;
- JSON content is serialized in a single pass over the element tree, preserving member order, with
  `", "` between items and `": "` between a name and its value, and non-ASCII characters emitted raw
  (not `\uXXXX`).

Known divergences, documented rather than discovered later:

- Number literals are preserved as written (`JsonElement.GetRawText()`), which matches
  `json.dumps` output for canonical literals and *preserves* a non-canonical one (`1e16`, `1.50`)
  where Python would normalise it;
- exponent formatting differs for some magnitudes (`1E+17` here, `1e+17` in Python).

Both are only reachable for states a caller hand-wrote with a non-canonical literal. The parity cases
that matter (the shape `json.dumps` produces) are pinned by tests.

`[MASK]` handling stays where it already is: `LayaPromptBuilder` substitutes it for instructions and
options, and the client substitutes it in the state before pre-tokenizing with
`tokenizer.Encode(stateText.Replace("[MASK]", " ", StringComparison.Ordinal))` (the same call the
inference path makes).

## 6. Answer mapping

| Laya | TypeSafe |
| --- | --- |
| `LayaDecision.Choice` | `ChoiceAnswer.Choice` — the option **key**, not the rendered `key: description` |
| `LayaDecision.Probabilities` | `ChoiceAnswer.Probabilities` — keyed by option key, same order |
| `LayaDecision.Score` | `ScoreAnswer.Score` — expectation over option indices, 0-based; matches the SDK's "probability-weighted level" |
| option index → criterion | `ScoreAnswer.Legend` — index → level content |
| — | `ScoreAnswer.Probabilities` — index → probability |
| `LayaDecision.NoulProbability` | `NoulAnswer.Probability` — p(true) |
| `LayaDecision.AnswerConfidence` | `ChoiceAnswer.Confidence` / `ScoreAnswer.Confidence`; for `NoulAnswer` (no confidence property) it goes in `AdditionalData` |

`Answer.AdditionalData` carries the diagnostics a gated consumer may want, under stable keys:
`answer_confidence`, `confidence` (the entropy reading, explicitly *not* the calibrated one),
`temperature`, and `act_probability`. The act head reads ~1.0 on the shipped checkpoint and is not a
signal — it is carried for completeness, and the key is documented as such.

`Usage`: `InputTokens` = the sum of prompt token counts across the request's questions (each question
is its own forward over the same state), `OutputTokens` = 0. This is a local convention, stated as
such in the doc comment, not a figure the engine reported.

## 7. CLI surface

Engine table gains one row: `nivara`, no URL, model `convaiinnovations/laya`. Modes: default,
`benchmark`, and `verify` (new).

```text
NivaraInference typesafe [engine] [mode] [--url <url>] [--model <name>]
```

- `typesafe nivara` — the same news items and question set as the HTTP path, answered in-process, so
  the two outputs line up side by side. Prints the Silk device, the load time, and a note that the
  first request includes kernel compilation.
- `typesafe nivara benchmark` — one sample is a whole `SystemOneAsync` call (six encoder forwards on
  the device, no transport). The output says explicitly that it is comparable neither with the HTTP
  rows nor with `laya benchmark` (one forward).
- `typesafe nivara verify` — runs the same items through the in-process client and through an HTTP
  engine (default Ollaya, `--url` to point elsewhere) and reports per-item, per-question deltas.

Guards, all firing before the first disk touch (that is what makes them testable on a runner with no
checkpoint):

- `--url` with the `nivara` engine → rejected: the engine is in-process. (`--url` *is* meaningful in
  `verify`, where it names the HTTP side to compare against.)
- `--model` with the `nivara` engine in `verify` → rejected: each side uses its own default. In the
  default/benchmark modes `--model` is accepted only when it names the one supported checkpoint; any
  other name is rejected with a message naming `convaiinnovations/laya`.
- `verify` with any engine other than `nivara` → rejected, and the message names `verify` (so the
  reject pins the mode in the allow-list rather than being caught by "Unsupported mode").
- `--ilgpu` / `--silk` / `--precision bf16|fp16` with `typesafe` → unchanged; the existing guards
  already reject them and the help text explains that the `nivara` engine is F32 on Silk.

Missing checkpoint: the client factory throws `FileNotFoundException` naming
`samples/data/laya/model.safetensors`; the demo prints `Model file not found: <path>` and returns 1,
matching the message shape the other models use.

## 8. Steps (each is one commit)

0. Bump the docs-gate pins this file moves (§10) — or, if the plan is treated as transient, delete
   this file at G2 and revert them then. Never leave the tree with a red docs gate.
1. `NivaraTypeSafeClient.cs`: the content serializer, the question/answer mapping, and the model
   resource, with the Laya×TypeSafe glue behind a small seam so the mapping is testable without
   weights. Unit tests for the serializer and the mapping (step 5's file, written here).
2. Silk wiring: the factory that loads agent config, encoder config, tokenizer and F32 safetensors
   from `samples/data/laya` and opens `SilkLayaSession`; the request loop (state tokenized once, one
   forward per question, `LayaCalibration.Decode`), the lock, `IDisposable`, cancellation checks.
   `Laya.cs` helpers become `internal`.
3. CLI: the engine row, the client factory returning `ITypeSafeClient`, header text for a URL-less
   engine, and the guard additions in `Program.cs`.
4. `verify`: the two-engine comparison loop, its report and its exit code.
5. Tests: `InferenceCliArgumentTests` reject cases plus the new fixture.
6. Docs: `docs/DECISION-MODELS.md`, `docs/LAYA.md`, `samples/NivaraInference/README.md`.
7. File the follow-up issue (promote the client into `Nivara.Extensions`; add a CPU/backend
   selector). `gh issue create --repo khurram-uworx/Nivara --body-file <temp>` — body from a file.

## 9. Tests

Unit (no model, no device):

- `ContentToText`: text passthrough; nested object/array; `", "` and `": "` separators; non-ASCII raw;
  integral vs decimal numbers; empty object/array; null; a JSON `null` state.
- question mapping: choice keys and order, score level order, noul criteria and the partially
  specified case, null instructions, null content.
- answer mapping: choice = key, probabilities keyed by key/index, score = expectation, noul =
  p(true), confidence = `AnswerConfidence` (and explicitly *not* the entropy reading), the
  `AdditionalData` keys.

CLI (`InferenceCliArgumentTests` — every case must be rejected before any load, so it runs on a
machine with no checkpoint):

- `typesafe ollama verify` → the verify-engine guard (this is what pins `verify` in the allow-list).
- `typesafe nivara --url http://localhost:11435` → the in-process guard.
- `typesafe nivara --model tev1:0.8b` → the checkpoint guard, message naming
  `convaiinnovations/laya`.
- `typesafe nivara verify --model laya` → the verify `--model` guard.
- `typesafe nivara bogus` → `Unsupported mode 'bogus'`, proving `nivara` parses as an engine.

## 10. Verification

Local, in this order:

```console
dotnet build Nivara.slnx
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Inference"
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"
dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore
```

The docs gate is in scope because `docs/DECISION-MODELS.md` and `docs/LAYA.md` change — that means the
whole `Nivara.Tests.Docs` namespace, not one fixture. The pin arithmetic for this change:

- `DocCitationTests.ExpectedDocumentCount` 52 → 53 (this file joins the markdown scan);
- `DocReferenceTests.ExpectedFileCount` 711 + this file **+ one per new `.cs` file**
  (`NivaraTypeSafeClient.cs`, `NivaraTypeSafeClientTests.cs`);
- `DocReferenceTests.ExpectedDocumentReferenceCount` 364 + one per `docs/…md` reference in this file
  and in the two documents it edits;
- the citation counts (`ExpectedCitationCount` 113, `ExpectedCodeCitationCount` 126) do **not** move
  unless a `File.cs:NN` citation is added — this plan deliberately cites none.

Pin the first three from the gate's own output rather than by hand: it prints the observed count, and
hand arithmetic on a file that is still being edited is how a "count" stops matching what it counts.

This file must contain no ```` ```csharp ```` fence: a snippet-bearing document that is in neither
`GatedDocuments` nor `UngatedDocuments` fails the coverage assertion, and this is not a reader-facing
API document. Command blocks above are fenced `console`, which the extractor ignores.

Runtime, needs the Laya checkpoint and an OpenCL device:

```console
dotnet run --project samples/NivaraInference -c Release -- typesafe nivara
```

Parity, needs Ollaya running (human-run — it needs a live engine and a GPU):

```console
dotnet run --project samples/NivaraInference -c Release -- typesafe nivara verify
```

`verify` compares, per item and per question:

- **hard gate** — the typed decision agrees: choice key equal, noul on the same side of 0.5, score on
  the same integer level;
- **reported** — deltas on per-option probability, `AnswerConfidence`, and score, against a tolerance
  of 1e-3 to start with, re-pinned from the observed distribution on the first real Ollaya run. If
  Ollaya serves the checkpoint at a narrower precision the distribution will say so, and widening the
  bound then needs that reason recorded rather than a quiet edit.

## 11. Open questions for the human

1. **`verify` in this cut, or a follow-up?** It needs a live Ollaya and a device, so it cannot be
   CI-verified; the alternative is to land `nivara` first and add the comparison next.
2. **`--model` on the `nivara` engine.** Rejecting anything but `convaiinnovations/laya` is the
   chosen rule (a model name that silently loads a different checkpoint would be a false claim). Say
   if you would rather it accept a directory name under `samples/data/` instead.
3. **The plan's home.** This file is in the docs gate's scan scope and moves three of its pins; the
   repo's designated transient plan file is the excluded one, excluded by name. Keep `docs/plan/TYPESAFE-LAYA.md`
   and move the pins, or move this to the excluded path and leave them alone?

## 12. Definition of done

- `typesafe nivara` answers the demo's six items with the same question set the HTTP engines answer.
- `Confidence` equals Laya's `AnswerConfidence` on every choice and score answer.
- A JSON state round-trips through `ContentToText` with `json.dumps`-shaped separators.
- `typesafe nivara verify` reports agreement with Ollaya on every typed decision, or names the
  disagreements item by item.
- The `Nivara.Tests.Inference` and `Nivara.Tests.Docs` fixtures pass in Release, formatting is clean,
  and this file's pins are moved or its exclusion is recorded.
- The follow-up issue for promoting the client into `Nivara.Extensions` exists.
