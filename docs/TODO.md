# Plan — #983: adaptive retrieval/decision loop (`--rag-adaptive-eval`)

Child D of #916. Wire frozen-encoder + Laya decision layer + online-updating head into an
end-to-end adaptive retrieval/decision loop, with evaluation and honest Silk scope.

## Scope

In scope: `samples/NivaraChat`, `tests/Nivara.Tests/Samples`, `docs/RAG.md`, `samples/NivaraChat/README.md`, `CHANGELOG.md`.
Zero changes under `src/`.

Out of scope (issue non-goals): ONNX export, ANN/vector DB, encoder fine-tuning, Silk weight steps.

## Problem

Children B (#981) and C (#982) are merged, but the loop is not wired:
- `RagAdaptiveMode` retrieves → decides → answers; never trains, evaluates, persists, or reports.
- `--rag-adaptive` uses `RagIndex` whose chunk ids are `{FileName}-{ordinal}`; judge labels and `AdaptiveTrainingData` require `{relativePath}#{ordinal}` from `MultiViewIndex` → labels never resolve.
- `LayaRetrievalDecisionHead.Evaluate` constructs `new LayaDecisionEngine(modelDir)` per call → re-reads the 842 MB checkpoint per query.
- `AdaptiveTrainingExample.Features` stores the full `[seqLen, 1024]` hidden state (~2 MB/example at max_len 512).

## Proposed changes

### New files

- `samples/NivaraChat/Helpers/AdaptiveEvaluation.cs` — `AdaptiveObservation`, `AdaptiveCalibrationBin`, `AdaptiveCalibration`; `AdaptiveEvaluation.BinIndex`, `Calibrate`, `Decisions` (reuses `AdaptiveChoiceMetrics`).
- `samples/NivaraChat/Helpers/AdaptiveDecisionScoring.cs` — `AdaptiveCandidate`; `Rerank`, `Filter`.
- `samples/NivaraChat/Helpers/AdaptiveFeedback.cs` — `AdaptiveBuildLimits`, `AdaptiveFeedbackBuild`; `Build` (truncating, deterministic), `AllowsAutoLabels`.
- `samples/NivaraChat/Helpers/AdaptiveReport.cs` — `AdaptiveReportInput`, `Markdown`, `MetricsJson`, `ResolvePath`, `MetricsPath`, `Write`.
- `samples/NivaraChat/Helpers/AdaptiveSilkParity.cs` — `SilkParityReport`, `Compare`.
- `samples/NivaraChat/Modes/RagAdaptiveEvalMode.cs` — orchestration.

### Edits

- `Program.cs` — `--rag-adaptive-eval` case; flags `--max-queries`, `--max-examples`, `--head-weights`, `--head-epochs`, `--head-lr`, `--update-batch`, `--resume`, `--no-persist`; usage lines.
- `Modes/ModeContext.cs` — 8 new trailing parameters (head defaults: 5 epochs, 5e-4 lr).
- `Helpers/LayaRetrievalDecisionHead.cs` — `Evaluate(engine, ...)`, `Decide(engine, MultiViewChunk)`, `BuildQuestion`, `DecideFromFeatures`; keep convenience overload documented as per-call.
- `Helpers/AdaptiveTraining.cs` — `Observe(head, examples, calibration)`; `Evaluate` reimplemented over it.
- `Helpers/AdaptiveFeedbackCollector.cs` — `Flush()` → `TrainAndPersist(true)`; new `FlushRemaining()`.
- `Helpers/AdaptiveTrainingData.cs` — extract `CreateExample`; keep judge-only guard.
- `Helpers/MultiViewIndex.cs` — `public bool FromCache { get; }`.
- `Modes/RagAdaptiveMode.cs` — switch to `MultiViewIndex` + one engine per session.
- `README.md`, `docs/RAG.md`, `CHANGELOG.md` — docs.

### Tests

- `Samples/AdaptiveEvaluationTests.cs`, `AdaptiveDecisionScoringTests.cs`, `AdaptiveFeedbackTests.cs`, `AdaptiveReportTests.cs`, `AdaptiveSilkParityTests.cs`.
- `Samples/AdaptiveFeedbackCollectorTests.cs` — add `FlushRemaining_TrainsBelowThreshold`, `FlushRemaining_EmptyBuffer_ReturnsFalse`.
- Existing fixtures untouched.

## Metric definitions (report verbatim)

Candidate set = index top-K (within-run comparison; report states non-comparability with `--rag-learn`).

Retrieval (per query, mean, K): `fixed`, `adaptive-rerank(before/after)`, `adaptive-filter(before/after)`.
P@K/R@K/MRR/NDCG@K. Unlabelled retrieved chunks count as not relevant (report prints count).
Invariant: `rerank` preserves membership ⇒ P@K/R@K unchanged (stated as a check and pinned by test).

Decision (graded pairs in top-K, before/after): choice accuracy; accept precision/recall/F1 at `--threshold`;
calibration on `AnswerConfidence` vs (choice correct): ECE (10 bins), Brier, reliability table.
Confidence = `AnswerConfidence(SoftmaxAtTemperature(logits, 2, Resolve(Choice,2).Scale), 2)`.

Timings: index build (cpu/silk + FromCache), test extraction forwards, test decisions, train extraction forwards,
train decisions, head updates (flushes × epochs), persistence (saves + bytes), evaluation forwards, total.

Silk: max-abs parity, mismatched rows, tolerance 1e-5; CPU vs Silk build times; decisions/training CPU in both.

## Verification

```
dotnet build Nivara.slnx
dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Samples"
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"
```
Docs pins unchanged (60/116/19/53); no new `File.cs::Symbol` citations; no new `docs/*.md`.

## GitHub issues log

- (none created yet)
