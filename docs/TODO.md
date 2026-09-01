# TODO: Reorganize samples/NivaraChat (split Program.cs + group root files)

Reminder: as each task executes, if you find deferred work or a concern outside this plan,
create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record
its number in the GitHub issues log below. Don't rely on memory.

## Problem

`samples/NivaraChat` has two compounding organizational problems:

1. **`Program.cs` is 1262 lines.** Every mode runner (`--train`, `--workflow`, `--agents`,
   `--interactive`, `--handoff`, `--tools`, `--critic`, `--intent`, `--online-learning`,
   `--embed`, `--rag`, `--rag-agent`, `--intent-train`) is a top-level local function
   capturing the outer option variables (`modelsDir`, `ollamaUrl`, `modelName`, ...), plus
   four model loaders, shared run/print helpers (`RunSingleShot`, `RunLoop`,
   `PrintAgentResults`), `GetRepoRoot`, the interactive menu, and `PrintUsage`.
2. **The root folder holds 25 flat-`NivaraChat` files** (13 executors, 6 text-model/chat
   classes, 4 helpers, 1 tools file, 1 interface) while the rest of the project already
   uses sub-namespaced folders (`Data` → `NivaraChat.Data`, `Training` →
   `NivaraChat.Training`, `Transformer` → `NivaraChat.Transformer`, `SmolLM` →
   `NivaraChat.SmolLM`). The root files break that convention.

## Decided direction (confirmed by human)

- **Scope:** extract all modes into per-mode static classes AND group the root files into
  sub-namespaced folders (full restructure).
- **Namespaces:** introduce sub-namespaces (`NivaraChat.Modes`, `NivaraChat.Executors`,
  `NivaraChat.Models`, `NivaraChat.Helpers`, `NivaraChat.Tools`).
- **Behavior:** purely structural + minor tidy-ups. No change to any mode's output or CLI
  contract.
- **`Nivara.Samples` is NOT touched.** Only `samples/NivaraChat/` + its README.
- **`TreatWarningsAsErrors=true`** (Directory.Build.props) → build must stay at 0 warnings
  (clean usings, code style).

## Ground truth (analyzed from source)

- Program.cs modes and their shared helpers (line numbers on `main`):
  - `RunTraining` (166), `RunIntentTraining` (190) — reference top-level `args.Length` for
    CLI-vs-menu messaging → extraction needs an explicit `fromCli` param.
  - `RunIntentMode` (202, uses `GetRepoRoot`, `MiniLMEmbeddingGenerator`,
    `DocumentChunker`, `NivaraToolFunctions`, intent executors, `CommunityToolkit` vector store, `ExtractIntent` 289).
  - `RunOnlineLearning` (353, uses `FeedbackCollector`, `LoadIntentModel`).
  - `RunWorkflow` (419), `RunAgents` (519, uses `NivaraChatClient`/`AsAIAgent`,
    `RunSingleShot`/`RunLoop`), `RunHandoff` (592), `RunTools` (687, uses
    `NivaraToolFunctions`), `RunCritic` (771, uses `CriticExecutor`,
    `WriterCriticLoop`).
  - Shared `RunSingleShot` (827), `RunLoop` (834), `PrintAgentResults` (850), model
    loaders `LoadValidatorModel` (894), `LoadSentimentModel` (904), `LoadEntityModel`
    (913), `LoadIntentModel` (922).
  - `PrintUsage` (931), `RunEmbeddingSearch` (961, `MiniLMEmbeddingGenerator`,
    `TensorPrimitives`), `RunRagPipeline` (1037, nested `RunQuery` 1095),
    `RunRagAgentPipeline` (1137, nested `RunQuery` 1224).
  - Menu: `RunInteractiveMenu` (98), `ShowMainMenu` (134), `AskOllama` (148),
    `GetRepoRoot` (161).
- The nested `BuildWorkflow()` and `RunQuery(...)` local functions capture method locals —
  they can stay as local functions *inside* the extracted static methods (grounded: legal
  per official C# docs).
- Root type cross-references that create new `using` edges after the namespace split:
  - `ModelInferenceHelper` (→ Helpers) used by executors, text models, FeedbackCollector,
    NivaraToolFunctions.
  - `DocumentChunk`/`DocumentChunker` (→ Helpers) used by FactualExecutor + RAG modes.
  - `ITextModel` (→ Models) used only within Models.
  - `NivaraToolFunctions` (→ Tools) used by ToolsMode/IntentMode.
- **No references to any root `NivaraChat` type exist outside `samples/NivaraChat/`**
  (verified across `.cs`, `.md`, `.slnx`, `.csproj`). Blast radius fully contained.

## Target layout

```
samples/NivaraChat/
├── Program.cs                      # thin dispatcher: consts, arg parse, switch → XxxMode.Run, menu, usage
├── Modes/            (ns NivaraChat.Modes)
│   ├── ModeContext.cs              # record: ModelsDir, OllamaUrl, ModelName, UseOllama, SingleShotText, ConfidenceThreshold, DocsDir, TopK
│   ├── ModeHelpers.cs              # GetRepoRoot, model loaders (Sentiment/Entity/Validator/Intent), RunSingleShot, RunLoop, PrintAgentResults
│   ├── TrainingMode.cs             # --train (RunTrain), --intent-train (RunIntentTrain)
│   ├── WorkflowMode.cs             # --workflow
│   ├── AgentsMode.cs               # --agents / --interactive
│   ├── HandoffMode.cs              # --handoff
│   ├── ToolsMode.cs                # --tools
│   ├── CriticMode.cs               # --critic
│   ├── IntentMode.cs               # --intent (+ ExtractIntent)
│   ├── OnlineLearningMode.cs       # --online-learning
│   ├── EmbeddingMode.cs            # --embed
│   ├── RagMode.cs                  # --rag (+ local RunQuery)
│   └── RagAgentMode.cs             # --rag-agent (+ local RunQuery)
├── Executors/       (ns NivaraChat.Executors)
│   ├── TextRouter.cs, SentimentExecutor.cs, EntityExtractor.cs, ValidatorExecutor.cs,
│   ├── LlmExecutor.cs, ConfidenceRouter.cs, CriticExecutor.cs, IntentClassifier.cs,
│   ├── ChitchatExecutor.cs, CommandExecutor.cs, EscalationExecutor.cs, QuestionExecutor.cs, FactualExecutor.cs
├── Models/          (ns NivaraChat.Models)
│   ├── ITextModel.cs, SentimentTextModel.cs, EntityTextModel.cs, ValidatorTextModel.cs,
│   ├── PassthroughTextModel.cs, NivaraChatClient.cs
├── Helpers/         (ns NivaraChat.Helpers)
│   ├── ModelInferenceHelper.cs, DocumentChunk.cs, FeedbackCollector.cs, WriterCriticLoop.cs
├── Tools/           (ns NivaraChat.Tools)
│   └── NivaraToolFunctions.cs
├── Data/            (unchanged, ns NivaraChat.Data)
├── Training/        (unchanged, ns NivaraChat.Training)
├── Transformer/     (unchanged, ns NivaraChat.Transformer)
├── SmolLM/          (unchanged, ns NivaraChat.SmolLM)
└── README.md                       # update architecture tree
```

## Proposed changes

### 1. `Modes/ModeContext.cs` + `Modes/ModeHelpers.cs` (new files)
- `public sealed record ModeContext(string ModelsDir, string OllamaUrl, string ModelName,
  bool UseOllama, string? SingleShotText, float ConfidenceThreshold, string? DocsDir,
  int TopK)`.
- `internal static class ModeHelpers`:
  - `GetRepoRoot()` (from Program.cs 161).
  - `LoadSentimentModel`, `LoadEntityModel`, `LoadValidatorModel(bool useAgentsFormat)`,
    `LoadIntentModel` (from Program.cs 894–929; note `LoadValidatorModel` has an optional
    param — keep it).
  - `RunSingleShot(Workflow, string)`, `RunLoop(Func<Workflow>)`, `PrintAgentResults(Run)`
    (from Program.cs 827–892).
- `AskOllama()` + the default consts stay in Program.cs (menu-only; modes receive values
  via `ModeContext`).

### 2. `Modes/XxxMode.cs` — one static class per mode (extraction)
- Each is a `public static class XxxMode` with `public static async Task Run(ModeContext ctx)`
  (or `void`/`Task` as needed); body moved verbatim from the Program.cs local function with:
  - `modelsDir` → `ctx.ModelsDir`, `ollamaUrl` → `ctx.OllamaUrl`, `modelName` →
    `ctx.ModelName`, `useOllama` → `ctx.UseOllama`, `singleShotText` →
    `ctx.SingleShotText`, `threshold` → `ctx.ConfidenceThreshold`, `docsDir` →
    `ctx.DocsDir`, `topK` → `ctx.TopK`.
  - Calls to shared local functions → `ModeHelpers.X`.
  - `GetRepoRoot()` → `ModeHelpers.GetRepoRoot()`.
  - Capturing local functions (`Workflow BuildWorkflow()`, RAG `async Task RunQuery(...)`)
    stay as local functions inside the static method (they capture that method's locals —
    legal; no static modifier needed).
  - `RunTraining`/`RunIntentTraining`: `args.Length > 0` → `bool fromCli` parameter.
  - RAG modes: the duplicated event-print loop between `--rag` and `--rag-agent` is NOT
    extracted (they print different things — rag prints retrieved chunks + LLM response,
    rag-agent prints agent events already via `PrintAgentResults`-style code); verify no
    over-reach in tidy-up.
- Mode class needs (usings per mode):
  - Common: `Microsoft.Agents.AI`, `Microsoft.Agents.AI.Workflows`,
    `Microsoft.Extensions.AI`, `OllamaSharp`, `Nivara.AutoDiff.Nn`,
    `Nivara.AutoDiff.Serialization`, `Nivara.Samples`, plus the new sub-namespaces used
    (`NivaraChat.Executors`, `NivaraChat.Models`, `NivaraChat.Helpers`,
    `NivaraChat.Tools`), and `CommunityToolkit.VectorData.InMemory` for intent/rag.
  - Keep only what each mode actually uses (build gate enforces cleanliness).

### 3. Group root files → sub-namespaces (`git mv`)
- Move the 25 root `.cs` files per layout using `git mv` (preserves history).
- Update `namespace NivaraChat;` → `namespace NivaraChat.Executors;` (etc.) in each file.
- Add the new `using` edges (verified above): executors (FactualExecutor →
  `using NivaraChat.Helpers;`), all text models + NivaraChatClient →
  `using NivaraChat.Helpers;` (ModelInferenceHelper), NivaraToolFunctions →
  `using NivaraChat.Helpers;`.
- No other behavior/type changes.

### 4. Rewrite `Program.cs` as a thin dispatcher
- Keep: default consts, `args` parse loop, mode switch (14 cases incl. `--tinyshakespeare`
  and `--smollm` → unchanged `TransformerMode`/`SmollmMode` calls), `RunInteractiveMenu`,
  `ShowMainMenu`, `AskOllama`, `PrintUsage`.
- Replace mode bodies with `TrainingMode.RunTrain(ctx, fromCli: args.Length > 0)` /
  `await XxxMode.Run(ctx)` calls; build a `ModeContext` once from parsed options.
- Interactive menu calls the same mode classes with a fresh `ModeContext`.
- Update `PrintUsage` to mention all modes (unchanged text apart from `--smollm` line
  already present on main).

### 5. README
- Update `## Architecture` file tree to `Modes/`, `Executors/`, `Models/`, `Helpers/`,
  `Tools/`. Mode names/outputs unchanged → no behavior docs need edits.

### 6. Minor tidy-ups (safe, behavior-neutral)
- Remove the commented-out dead line in `RunAgents`
  (`//new NivaraChatClient(new PassthroughTextModel(ollamaClient)).AsAIAgent(...)`).
- Preserve the repo's private-field naming (`camelCase`, no `_` prefix) in new code only;
  do NOT reformat moved code (build enforces 0 warnings).
- Intent mode: `DocumentChunk` (Helpers) + `MiniLMEmbeddingGenerator` (Nivara.Samples)
  usings must be present; no other changes.

## Verification (ask before long runs per AGENTS.md)

1. `dotnet build Nivara.slnx -c Release` → 0 warnings, 0 errors (after each commit).
2. `dotnet run --project samples/NivaraChat -- --smollm plain --text "The capital of France is"`
   → streams the known reply; proves dispatcher + SmolLM intact.
3. `dotnet run --project samples/NivaraChat -- --workflow`
   → prints "Models not found. Run with --train first." (fast path proves WorkflowMode wiring).
4. `dotnet run --project samples/NivaraChat -- --tinyshakespeare --help` → help renders.
5. `dotnet run --project samples/NivaraChat -- --help`-less modes: `--embed` requires
   MiniLM data which exists at `samples/data/minilm`? — check; if present, `--embed` smoke
   is optional (not required), else skip.
6. `dotnet run --project samples/NivaraChat` (no args) → menu renders, "1"/"q" work.

## Blast radius

- **All changes confined to `samples/NivaraChat/`** (+ README). No core, no `Nivara.Samples`,
  no other sample/test projects.
- Public types in the moved files (`NivaraToolFunctions`, `DocumentChunk`,
  `DocumentChunker`) are only consumed inside this exe — no external callers (verified).
- Files touched: 25 root files (namespace + using), ~13 new Modes files, Program.cs rewrite,
  README. No test project covers NivaraChat; verification = build + smoke runs.
- Downstream consumers of `samples/NivaraChat` binary output: none (it's an `Exe`).

## Planned commits (one logical unit, build before each commit)

1. `docs: plan NivaraChat reorganization in TODO.md` — this file.
2. `refactor: add Modes/ModeContext + Modes/ModeHelpers with shared mode helpers` — new files.
3. `refactor: extract Program.cs mode runners into Modes/XxxMode classes` — new mode files +
   Program.cs switch now calls them; intermediate build must still pass with the mode classes
   present and Program.cs bodies removed.
4. `refactor: move root files into Executors/ Models/ Helpers/ Tools/ sub-namespaces` —
   `git mv` + namespace/using updates.
5. `refactor: slim Program.cs to a thin dispatcher` — final Program.cs (menu/usage/switch)
   + remove any leftover helpers; build → commit.
6. `docs: update NivaraChat README architecture tree` — README commit.

## GitHub issues log

- (none yet — purely structural; any deferred concern found during execution will be captured
  here as a tracked issue before `docs/TODO.md` is deleted.)