# Plan — `NivaraTypeSafeClient`: in-process TypeSafe engine for Laya

Derived from `docs/plan/TYPESAFE-LAYA.md` (kept in tree; pins already moved).

## Problem

Nivara can answer the TypeSafe `System One` question set only over HTTP (Ollaya/Ollama). The Silk/OpenCL
Laya decision head already runs in-process, but there is no `ITypeSafeClient` over it, so the demo's
`typesafe` path cannot exercise Nivara's own backend and cannot be parity-checked against Ollaya except
by external processes.

## Proposed changes

1. `samples/NivaraInference/NivaraTypeSafeClient.cs` — `ITypeSafeClient` + `IDisposable`:
   - content serializer `ContentToText(TypeSafeContent?)` (D3 separators, raw non-ASCII, numbers via `GetRawText`);
   - question mapping (TypeSafe -> `LayaQuestion`) and answer mapping (`LayaDecision` -> `ChoiceAnswer`/`ScoreAnswer`/`NoulAnswer`),
     with `Confidence = AnswerConfidence` (not the entropy reading) and `AdditionalData` carrying
     `answer_confidence`, `confidence`, `temperature`, `act_probability`;
   - Silk model resource (agent/encoder config, tokenizer, F32 safetensors, `SilkLayaSession`), per-request lock,
     mask-aware state tokenisation, one forward per question, `LayaCalibration.Decode`;
   - `Usage.InputTokens` = sum of prompt token counts, `OutputTokens = 0` (local convention, documented).
2. `Laya.cs` — `AgentConfig`, `LoadAgentConfig`, `LoadEncoderConfig`, `LoadTokenizer` become `internal`.
3. `TypeSafe.cs` — engine gains an `InProcess` flag, `nivara` row, client factory returning `ITypeSafeClient`,
   URL-less header/description text, benchmark support.
4. `Program.cs` — `typesafe` gains `verify`; guards:
   - `--url` with `nivara` rejected
   - `--model` with `nivara` + `verify` rejected
   - `--model` with `nivara` otherwise accepted only when exactly `convaiinnovations/laya`
   - `verify` with any engine other than `nivara` rejected (message names `verify`)
   - missing checkpoint prints `Model file not found: <path>` and returns 1
5. `verify` mode — compare in-process vs HTTP engine (default Ollaya), per item per question:
   hard gate on typed decision, reported deltas on probabilities/AnswerConfidence/score at 1e-3.
6. `tests/Nivara.Tests/Inference/NivaraTypeSafeClientTests.cs` — serializer + mapping golden (no weights).
7. `InferenceCliArgumentTests.cs` — new reject cases pinning the guards.
8. Docs: `docs/DECISION-MODELS.md`, `docs/LAYA.md`, `samples/NivaraInference/README.md`.

Model name is the exact HF repo id `convaiinnovations/laya`; checkpoint root `samples/data/laya`.

## Blast radius

- New sample-owned types consumed by `Nivara.Tests` (already project-references `NivaraInference`).
- `Laya.cs` visibility changes (public -> internal) are source-compatible within the repo.
- CLI dispatch table gains `verify` mode for `typesafe`; tests pin the expanded surface.
- Docs gate pins already updated for this file; new `.cs` additions require re-running the gate to
  refresh `ExpectedFileCount`.

## Planned commits

0. Branch setup + this plan (`docs: plan typesafe-laya in TODO.md`).
1. Mapping seam + `ContentToText` + unit tests.
2. Silk wiring + internal Laya helpers + model factory.
3. CLI engine row + client factory + header/description text.
4. `verify` comparison loop + report + exit code.
5. CLI guards + tests.
6. Docs updates.
7. Follow-up issue (promote client into Nivara.Extensions + backend selector).

## Verification

- `dotnet build Nivara.slnx`
- targeted inference tests, then full docs-gate namespace (human confirms)
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- runtime (checkpoint + OpenCL): `typesafe nivara`, `typesafe nivara verify`

## GitHub issues log

- [x] #700 � promote `NivaraTypeSafeClient` into `Nivara.Extensions` + backend selector (filed during execution)
