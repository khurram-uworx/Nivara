# Plan: issue #1062 — settle `PerClassAccuracyFrame` name before 1.6.0

## Problem

`NivaraMetrics.PerClassAccuracyFrame` returns `Class`, `Tp`, `Fp`, `Fn`, `Precision`,
`Recall`, `F1` — no `Accuracy` column. Per-class accuracy is `Tp / (Tp + Fn)`, exactly
`Recall`; global accuracy is `sum(diagonal) / validPairs`, a scalar from the matrix frame.
The name is misleading. Rename before 1.6.0 while free.

## Decision

Option 1 (recommended): rename to `PerClassMetricsFrame`. Also document the global-accuracy
identity (option 3 fallback paragraph). Do not add a redundant `Accuracy` column.

## Proposed changes

- `src/Nivara/Metrics/NivaraMetrics.cs`: rename method; diagnostic operation
  `"Metrics.PerClassAccuracy"` -> `"Metrics.PerClassMetrics"`.
- `src/Nivara/Metrics/ClassificationMetrics.cs`: comment wording.
- `tests/...`: rename test class/file and test methods; update calls and diagnostic assertion.
- `docs/METRICS.md`: method name in table/snippet, diagnostic row, add global-accuracy
  identity paragraph.
- `docs/plan/AISTACK-ROADMAP.md`, `AGENTS.md`: method name.

## Blast radius

Public API method rename (pre-1.6.0, no callers outside repo found). Tests: 3 fixtures.
Docs pins unaffected (no `NivaraMetrics.cs::PerClass*` citation exists).

## Verification

- `dotnet build Nivara.slnx`
- `dotnet test ... --filter "FullyQualifiedName~Metrics"` (39 passed)
- `dotnet test ... --filter "FullyQualifiedName~Nivara.Tests.Docs"` (66 passed)

## Planned commits

1. docs: plan #1062 in TODO.md
2. rename PerClassAccuracyFrame to PerClassMetricsFrame

## GitHub issues log

- (none created)
