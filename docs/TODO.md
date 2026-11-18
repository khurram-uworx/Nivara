# Plan: Issue #1272 — NivaraChat adaptive eval Laya Silk parity

## Problem

`RagAdaptiveEvalMode` runs embeddings + index parity under `--silk`, but the parity report is ambiguous about which components were compared. The report states "Scope: MiniLM embedding generation only. ModernBERT feature extraction, decisions and head training are CPU in both runs," while the mode actually compares Laya encoder features (`ExtractFeatures` hidden values) between CPU and Silk engines. A drifted Laya path would not be clearly caught, and the report's scope line contradicts reality.

## Proposed changes

1. `AdaptiveSilkParity`: add Laya comparison methods (`CompareLaya`, `CompareLayaScores`, `CompareLayaFeatures`, `CompareScores`, `CompareFeatures`, `CompareHidden`, `CompareDecisionHead`) plus `Compare(float[], float[])` / `Compare(ReadOnlySpan<float>, ...)` overloads, all delegating to a shared `CompareFloats` returning `SilkParityReport`. Tolerance default stays `1e-5`.

2. `RagAdaptiveEvalMode`: replace hand-rolled Laya feature loop with `AdaptiveSilkParity.CompareLaya(...)`; feed `layaMaxAbs`/`layaWithin` from the report.

3. `AdaptiveReportSilk`: add init property `ComponentsCompared` (nullable) plus computed aliases (`ComparedComponents`, `Components`, `Compared`, `EmbeddingsCompared`, `LayaCompared`, `ComparedEmbeddings`, `ComparedLaya`) defaulting from `Available`/`LayaAvailable`.

4. `AdaptiveReport.Markdown`: add `- Components compared: {components}` line in Silk scope; replace contradictory scope line with accurate note.

5. `AdaptiveReport.MetricsJson`: include `componentsCompared` in silk object.

## Verification

- `dotnet build Nivara.slnx`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Adaptive"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits

1. docs: plan #1272 in TODO.md
2. feat: add Laya comparison to AdaptiveSilkParity
3. refactor: use AdaptiveSilkParity for Laya parity in RagAdaptiveEvalMode
4. feat: state compared components in adaptive report
5. docs: remove TODO.md — plan executed

## GitHub issues log

- (none yet)
