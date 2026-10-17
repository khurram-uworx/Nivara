# TODO — Move the TypeSafe demo to NivaraInference and add Ollama decision-model support

Branch: `khurram/typesafe` (already on it; confirmed with the human).

## Problem

The TypeSafeAI news-assessment demo lives in `samples/NivaraChat` (`Modes/TypesafeMode.cs`,
wired as `--typesafe`), but it exercises **inference** against a decision-model server — it
belongs in `samples/NivaraInference` next to the other inference demos. It also talks only to
Ollaya (`http://localhost:11435`, model `laya:en`); Ollama 0.35+ now serves the same TypeSafe
`/v1/systemone` API with its own decision models, and the demo should reach it too.

## Grounding (verified before implementation)

- **The TypeSafeAI client's wire shape**: SDK source at `E:\github\TypeSafeAI.Net` —
  `TypeSafeClient.cs:44` resolves `_systemOneUri` via `HttpPipeline.Resolve("/v1/systemone")`,
  i.e. `new Uri(BaseUrl, "/v1/systemone")`. Since the path is absolute, it replaces any base
  path: `http://localhost:11434` + `/v1/systemone` = `http://localhost:11434/v1/systemone`.
- **Ollama serves the same endpoint** (ollama.com/blog, 2026-09-29, "Ollama now supports
  Jev-style decision models"): Ollama 0.35+ exposes `/v1/systemone` — `state` + named
  questions, one request. Decision models available: `nimble` (9B, Bespoke Labs), `tev1` (4B,
  Together AI), `tev1:0.8b` (0.8B). The human chose **`tev1:0.8b`** as the Ollama model.
- So the same `TypeSafeClient` works against both servers; only `BaseUrl` and `DefaultModel`
  change. No SDK changes needed.
- **NivaraChat's `--ollama` convention** (`samples/NivaraChat/Program.cs:25`): bare flag →
  default URL `http://localhost:11434`; an optional non-flag value overrides the URL. The new
  switch mirrors these semantics.
- **NivaraInference CLI contract**: model-dispatch via `SupportedModes`; every guard rejects
  with exit 1 + stderr message **before the first disk touch**; the surface is pinned by
  `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs` (spawns the built DLL per arg
  set). `typesafe` has no checkpoint, so its dispatch must run before the model-file lookup.
- Tensors/.NET-API grounding (microsoft-learn MCP) is **not applicable**: this work touches
  sample CLI wiring and an external HTTP API, no tensor or BCL-surface changes.

## Decisions confirmed with the human

- Dispatch spelling: **positional `typesafe`** (`NivaraInference typesafe [--ollama [url]]`),
  matching this CLI's bare-token model convention — not NivaraChat's `--typesafe` style.
- **4 commits**: move · wire Ollaya dispatch · add Ollama · docs.
- Ollama model: `tev1:0.8b`. Ollaya stays the default server (it emerged first; `laya:en`).

## Planned commits

1. **Move the TypeSafe demo NivaraChat → NivaraInference.** Mostly staged already: the rename
   with namespace/class rename, `TypeSafeAI` package moved from `NivaraChat.csproj` to
   `NivaraInference.csproj`, `--typesafe` case + usage line removed from NivaraChat's
   `Program.cs`. Clean the noise: revert `samples/Nivara.Samples/Nivara.Samples.csproj`
   entirely (its only change vs HEAD is a UTF-8 BOM; `.editorconfig` says `charset = utf-8`,
   no BOM — and the earlier `<Folder Include="TypeSafe\" />` artifact must not land), and
   strip the BOM from `NivaraChat.csproj` keeping its package removal.
   Verify: `dotnet build Nivara.slnx`.
2. **Wire `typesafe` into the NivaraInference CLI (Ollaya parity).**
   - `SupportedModes["typesafe"] = [""]` (no sub-modes).
   - Dispatch after the last scope guard (`--simd-widen`), **before** the model-file lookup —
     the demo is an HTTP client, there is no `samples/data/typesafe/model.safetensors`.
   - Fix `TypeSafe.cs`: banner "NivaraInference" (not "NivaraChat"), `Run()` → `Task<int>`
     (exit 1 if every news item failed, with a "is the server running" hint), drop unused
     usings added by the move.
   - Usage text: `typesafe` in the model list + a modes line.
   - Gate tests in `InferenceCliArgumentTests`: `typesafe benchmark` → "Unsupported mode
     'benchmark' for 'typesafe'"; `typesafe --ilgpu` → "--ilgpu is not supported for
     'typesafe'"; `typesafe --precision bf16` → the F32-only reject; update the class
     doc-comment's coverage list. The happy path is deliberately **not** gated — it needs a
     live server, which CI does not have.
   Verify: build + targeted `InferenceCliArgumentTests` (Release).
3. **Introduce Ollama decision-model support.**
   - `--ollama [url]` parsed in `Main` with NivaraChat's exact semantics (bare → default
     `http://localhost:11434`; optional value overrides); scoped reject for non-`typesafe`
     models: "--ollama is a 'typesafe' option only".
   - Per-server constants in `TypeSafe.cs`: Ollaya `http://localhost:11435` / `laya:en`;
     Ollama `http://localhost:11434` / `tev1:0.8b`.
   - Banner prints which server + model it is talking to.
   - Gate test: `minilm --ollama` → the scope reject; usage text for the flag.
   Verify: build + targeted `InferenceCliArgumentTests` (Release).
4. **Docs.**
   - `samples/NivaraInference/README.md`: quick-start lines (`typesafe`, `typesafe --ollama`)
     + a short section: the two servers, their models, `--ollama [url]`, the shared TypeSafe
     `/v1/systemone` wire shape, link to the Ollama decision-model announcement.
   - `docs/LAYA.md`: short overview section before "## Model overview" — TypeSafe AI SDK /
     System One API, Ollaya emerging first (laya:en), Ollama 0.35+ adopting the same endpoint
     (`nimble`, `tev1`, `tev1:0.8b`), the "OpenAI Chat Completions of decision models"
     framing. Prose only — no csharp fenced blocks, so no snippet-gate impact.
   Verify: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` and the
   documentation snippet gate (`EveryGatedBlock_CompilesWithoutErrors`), plus the targeted
   CLI gate again.

## Verification summary (pre-PR, per AGENTS.md)

- `dotnet build Nivara.slnx` after each code commit.
- Targeted: `dotnet test -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"`.
- Before concluding: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
  + snippet gate. Full suite only with the human's explicit go-ahead.

## Blast radius

- `samples/NivaraChat/` — loses the demo, its package ref, and the `--typesafe` entry. No
  other NivaraChat mode references `TypesafeMode`; verified by repo-wide search.
- `samples/NivaraInference/` — gains `TypeSafe.cs` + the package; `Program.cs` gains one
  dictionary entry, one arg-parsing branch, one dispatch branch, usage lines. All guards
  untouched; `typesafe` flows through the existing model-scoped guards correctly
  (`--precision bf16` → F32-only reject via `ConsumesNarrow`'s `_ => false`).
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs` — new reject cases only;
  existing cases unaffected (`typesafe` is a new key, existing models unchanged).
- No `src/` changes at all. No public library contract touched; CHANGELOG not required.

## GitHub issues log

- [ ] #673 — `InferenceCliArgumentTests` fails on dev machines with the Qwen checkpoint
  present (it assumes the checkpoint is absent; only true on CI). Discovered while running
  the fixture during commit 2. Pre-existing, not caused by this branch.
- [ ] #674 — `DocReferenceTests` tolerated-mentions gate fails whenever `docs/TODO.md` exists,
  i.e. on every `iterative-work` branch while its plan is live (the test asserts via
  `DocumentFaults`, which needs the target to be absent). Discovered while running the Docs
  gate during commit 4. Pre-existing workflow/gate conflict; resolves for this branch when
  TODO.md is deleted at G2.
