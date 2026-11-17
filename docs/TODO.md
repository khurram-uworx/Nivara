# Plan: Issue #1268 — Wire Silk Laya into NivaraChat

## Problem

NivaraChat's adaptive RAG path uses `LayaRetrievalDecisionHead.LayaDecisionEngine`, which hardcodes
`ModernBertEncoder<float,float>` and `LayaDecisionHead<float,float>` (CPU-only float). `--silk` does not
affect Laya. Core already ships `SilkLayaSession`, `SilkLayaHeadGpuRunner` (F32/BF16/FP16),
`SilkLayaHeadTrainer`, and the `ILayaGpuSession`/`OpenLayaSession*` seam, exercised only by tests.

## Decision (taken with human)

Wire (not trim). F32 only initially; narrow precision (bf16/fp16) native-weights training deferred to
next release. No silent mixed-backend reporting (G12).

## Proposed changes

### Phase 0 — Truthful reporting (docs/help)
- Update `--silk` help line + README so Laya status is honest at each stage.

### Phase 1 — Backend-aware `LayaDecisionEngine`
File: `samples/NivaraChat/Helpers/LayaRetrievalDecisionHead.cs`

Add constructor selector:
```
LayaDecisionEngine(string modelDir,
    bool loadAdaptiveWeights = true,
    string? adaptiveWeightsPath = null,
    bool useSilk = false)
```
- CPU path unchanged.
- Silk path: tensors = SafeTensorsLoader.Read<float>(path); open
  `ILayaGpuSession session = GpuSeamFactories.Laya(config, tensors, agent.HeadLayers)`.
- ExtractFeatures: Silk uses `session.Encoder.Forward(ids, length)` -> hidden span.
- Decide: Silk uses `session.Forward(...)` -> `LayaHeadOutput`; `calibration.Decode`.
- Head/DetachHead: still load CPU `LayaDecisionHead<float,float>` from same tensors (adaptive weights)
  so AdaptiveTraining/FeedbackCollector/persistence keep working (training stays CPU AutoDiff in
  this PR; device training is Phase 3).

### Phase 2 — Wire into RagAdaptive modes
Files: `samples/NivaraChat/Modes/RagAdaptiveMode.cs`, `samples/NivaraChat/Modes/RagAdaptiveEvalMode.cs`
- Pass `useSilk: ctx.UseSilk` to engine constructor.
- RagAdaptiveEvalMode: add Laya parity (hidden/logits max-abs) to the Silk report so coverage is
  explicit (G12), not just embeddings.

### Phase 3 — Training on device (deferred follow-up)
Device training via `SilkLayaHeadTrainer` is a larger change (gradient derivation for CE loss);
tracked as a follow-up issue. This PR keeps training on CPU AutoDiff and reports "training: CPU"
honestly in the parity report.

### Phase 4 — Tests
- Unit: backend selector + smoke (OpenCL-guarded), mirroring `SilkLayaHeadTrainerSmokeTests`.
- No changes to existing Adaptive* CPU tests.

### Phase 5 — Docs/help + pins
- Finalize help/README; run snippet gate; adjust DocsCoveragePins if needed.

## Verification steps
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- snippet gate `EveryGatedBlock_CompilesWithoutErrors`
- targeted tests: `--filter "FullyQualifiedName~Adaptive|FullyQualifiedName~Laya|FullyQualifiedName~SilkLaya"`
- manual smoke of `--rag-adaptive --silk` / `--rag-adaptive-eval --silk` where OpenCL available

## Planned commits
1. docs: plan #1268 in TODO.md
2. feat(sample): backend selector in LayaDecisionEngine (Phase 1)
3. feat(sample): wire Silk Laya into RagAdaptive modes (Phase 2)
4. test(sample): Laya backend selector + smoke (Phase 4)
5. docs: truthful --silk reporting incl. Laya (Phase 0/5)
6. docs: remove TODO.md — plan executed

## Blast radius
- `LayaRetrievalDecisionHead.LayaDecisionEngine` (constructor, ExtractFeatures, Decide, Dispose, Head).
- Call sites: RagAdaptiveMode, RagAdaptiveEvalMode, LoadTrainingContext.
- Tests: AdaptiveTrainingTests (unchanged, CPU path), AdaptiveFeedbackCollectorTests.
- Report shape: AdaptiveReportSilk / new Laya parity field.

## GitHub issues log
- (none yet; create follow-ups during execution)
