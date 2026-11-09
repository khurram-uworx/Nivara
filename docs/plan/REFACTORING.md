# Refactoring plan: stop `Nivara.Tests` referencing CLI projects

## Motivation

`tests/Nivara.Tests/Nivara.Tests.csproj` currently references three executable sample projects:

- `samples/NivaraChat/NivaraChat.csproj`
- `samples/NivaraClassifier/NivaraClassifier.csproj`
- `samples/NivaraInference/NivaraInference.csproj`

Those references exist only because reusable logic is hosted in CLI projects:

- `NivaraChat`: `Helpers/*`, `Modes.ModeContext`, `Qwen.QwenChatTemplate`, `Qwen.QwenToolCallParser`, `Shared.TokenizePrefixCache`
- `NivaraClassifier`: `BenchStats`
- `NivaraInference`: `Qwen.GenerateSilk`, `Qwen.SynthesizeTensors`, `Qwen.RunToolsSilk`, and type-safe client helpers

The ideal shape is that `Nivara.Tests` references only `Nivara.Samples`, with CLI projects acting as thin orchestration layers over sample-owned reusable code.

## Principle

Prefer transferring compile ownership rather than physically moving files in the first pass. Keep source paths and namespaces stable initially to avoid citation churn in `docs/`, then do namespace/path cleanup later if desired.

## Dependency inventory

### `NivaraChat` helper upward dependencies

Only these helper files depend on `NivaraChat` namespaces:

- `FeedbackCollector.cs` -> `NivaraChat.Training` (apparently unused)
- `MultiViewIndex.cs` -> `NivaraChat.Modes` (`ModeContext`)
- `MultiViewIndexCache.cs` -> `NivaraChat.Modes` (`ModeContext`)
- `RagIndexCache.cs` -> `NivaraChat.Modes` (`ModeContext`)
- `RagSearch.cs` -> `NivaraChat.Modes` (`ModeContext`)
- `WriterCriticLoop.cs` -> `NivaraChat.Executors` (`CriticExecutor`)

`ModeContext` must be promoted before the four `ModeContext`-dependent helpers. `WriterCriticLoop` should remain in `NivaraChat` because it is not used by tests and pulls in executor/agent dependencies.

## Waves

### Wave 0 — Prune upward dependencies

- Promote/copy `NivaraChat.Modes.ModeContext` into `Nivara.Samples` (pure option record).
- Remove the unused `using NivaraChat.Training;` from `FeedbackCollector.cs`.
- Keep `WriterCriticLoop.cs` in `NivaraChat`.

After this, remaining helper files only depend on `Nivara`, `Nivara.Samples`, `Nivara.AI`, and framework/BCL types.

### Wave 1 — Transfer `NivaraChat.Helpers` to `Nivara.Samples`

- Include all `NivaraChat.Helpers` files except `WriterCriticLoop.cs` in `Nivara.Samples`.
- Exclude the same files from `NivaraChat` compilation to avoid duplicate definitions.
- Keep namespaces as `NivaraChat.Helpers`.
- `NivaraChat` consumes the helpers through its existing `Nivara.Samples` reference.

### Wave 2 — Transfer `BenchStats` from `NivaraClassifier`

- Include `samples/NivaraClassifier/BenchStats.cs` in `Nivara.Samples`.
- Preserve namespace `NivaraClassifier`.
- Update `NivaraClassifier` to use the sample assembly copy.
- Process-spawning tests (`ClassifierCliArgumentTests`) remain unchanged.

### Wave 3 — Promote test-needed inference helpers

Do not promote the whole 67K `Qwen.cs`. Add a small sidecar in `Nivara.Samples` containing:

- synthetic tensor creation currently exposed as `NivaraInference.Qwen.SynthesizeTensors`
- Silk generation currently exposed as `NivaraInference.Qwen.GenerateSilk`
- Silk tools loop currently exposed as `NivaraInference.Qwen.RunToolsSilk`

Refactor `NivaraInference.Qwen` methods to call the promoted helpers.

For `NivaraTypeSafeClientTests`, inspect and promote only the reusable type-safe client and minimal loading helpers; leave CLI orchestration (`TypeSafe.cs`, `DistilBertSst.cs`, etc.) in the inference project.

### Wave 4 — Promote remaining Qwen utilities

Include or move:

- `NivaraChat/Qwen/QwenChatTemplate.cs`
- `NivaraChat/Qwen/QwenToolCallParser.cs`
- `NivaraChat/Shared/TokenizePrefixCache.cs`
- `NivaraChat/Shared/TokenSampling.cs` only if required by promoted types

Preserve existing namespaces in this pass.

### Wave 5 — Remove CLI project references from tests

Remove from `Nivara.Tests.csproj`:

- `samples/NivaraChat/NivaraChat.csproj`
- `samples/NivaraClassifier/NivaraClassifier.csproj`
- `samples/NivaraInference/NivaraInference.csproj`

Keep process-spawning tests, which locate built DLLs by path and do not require compile-time type references.

Verify no remaining source references to `NivaraChat.`, `NivaraClassifier.`, or `NivaraInference.` after the moves.

### Wave 6 — Validation

- Build `Nivara.slnx`.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- Run docs gates (`EveryGatedBlock_CompilesWithoutErrors`, `DocsCoverageTests`).
- Run targeted tests: `FullyQualifiedName~Samples`, `FullyQualifiedName~Qwen`, `FullyQualifiedName~Inference`.

Namespace and physical-path cleanup, if desired, should be a separate follow-up.

## Risks

1. Circular dependency: `MultiViewIndex`, `MultiViewIndexCache`, `RagIndexCache`, and `RagSearch` use `ModeContext`, so Wave 0 must happen first.
2. Dependency weight: promoted files may require `Microsoft.Extensions.AI`, `OllamaSharp`, `VectorData`, or `System.Numerics.Tensors`. `Nivara.Samples` already references most; add missing packages there as needed, never in the opposite direction.
3. Summary counts and citations: keeping paths/namespaces stable in Waves 1–4 avoids large docs coverage and citation edits in the first pass.
4. Scope discipline: only promote functionality with existing tests; leave app-only orchestration in CLI projects.
