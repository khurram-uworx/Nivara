# Plan: Issue #1276 — Per-component backend reporting (Silk as first-class)

Branch: khurram/1276 (off khurram/1274). Reporting only; no behavior change.

## Problem

A single run can mix backends (e.g. `--qwen --silk` with `tools-nivara` uses Silk for Qwen
but CPU for the domain models and Laya when adaptive). Console prints one "Backend: Silk/CPU"
line, implying uniform coverage. Parity/perf conclusions unreliable.

## Proposed changes

1. Add `ComponentBackend` + `BackendMatrix` records in `AdaptiveReport.cs`.
2. Add `Backends` (defaulted) to `AdaptiveReportInput`; render "## Backends" in Markdown and
   `backends` in MetricsJson.
3. Add `BackendMatrixPrinter` helper for console matrix.
4. Wire into modes that mix backends: ToolsMode, AgentsMode, WorkflowMode, IntentMode, CriticMode,
   HandoffMode, OnlineLearningMode, RagAdaptiveMode, RagAdaptiveEvalMode, EmbeddingMode, plus
   Qwen/SmolLM/Transformer LLM row.
5. RagAdaptiveEvalMode: populate `AdaptiveReportInput.Backends` from actual resolved backends.

## Verification

- `dotnet build Nivara.slnx`
- targeted: `AdaptiveReportTests`, snippet gate
- format whitespace check

## Planned commits

1. docs: plan #1276 in TODO.md
2. feat: add BackendMatrix records + AdaptiveReport rendering
3. feat: add BackendMatrixPrinter console helper
4. feat: wire backend matrix into modes
5. feat: populate AdaptiveReport backends in rag-adaptive-eval
6. test: cover backend matrix rendering
7. docs: remove TODO.md

## Blast radius

- AdaptiveReport.cs (records + render) — AdaptiveReportTests
- New helper file
- Modes/*.cs console output only
- No public API changes to core Nivara

## GitHub issues log

- [ ] #1276 — mixed backends reporting (this issue)
- [ ] follow-up: Qwen Silk tools-nivara omits entitySilk wiring (entity stays CPU)