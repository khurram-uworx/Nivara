# Plan: Decide the home of ranked-retrieval metrics (issue #1068)

## Problem

`samples/NivaraChat` carries a ranked-retrieval evaluation surface (`RagEvalMetrics`,
`RetrievalEvaluation`, `RagEvalJudge`, `RagEvalLabels`, `Modes/RagEvalMode.cs`) that
returns no `NivaraFrame`. It sits awkwardly against the `Nivara.Metrics` boundary in
`docs/METRICS.md`: per-query nDCG/MRR could be frame-producing metrics, while
pipeline-level evaluation belongs to the platform (ML.NET has ranking evaluators).

## Decision

Leave the metrics in `samples/NivaraChat`. Ranked-retrieval metrics operate on ordered
per-query id lists, not on a predicted/actual column pair, and turning them into
frame-producing metrics would require a new `string[]` retrieved-ids convention plus
tests — a real design job, not a port. Defer until a second consumer wants it.

This is recorded as documentation only. No code moves.

## Proposed changes

1. `docs/METRICS.md` — add a "Ranked retrieval and pipeline evaluation" section
   stating that `PrecisionAtK`, `RecallAtK`, `Mrr`, `NdcgAtK` and the sign-test
   p-value are demonstrated in `samples/NivaraChat` and are not part of this module,
   with a pointer to the sample files. Use the issue's exact boundary sentence.
2. `samples/NivaraChat/README.md` — add the same one-line boundary note in the
   "RAG evaluation (`--rag-eval`)" section.
3. (Optional, same commit if wanted) one-line boundary notes in `docs/RAG.md`
   (section 8, "The metrics") and `samples/NivaraChat/RAG.md` (evaluation
   section), written without markdown links so the document-reference pin stays
   stable. Default plan: land the two required edits first; the RAG-note additions
   are deferred unless requested.

Follow-up: a design issue may be opened later if a second consumer wants
frame-producing ranked-retrieval metrics. No code changes in this issue.

## Verification steps

- `dotnet build Nivara.slnx`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Docs namespace tests (snippet gate + coverage pins):
  `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
- Targeted sample tests:
  `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~RagEvalMetricsTests"`
- Confirm no `DocsCoveragePins` change is needed (no citation or `docs/...md`
  reference is added — the sentence uses backticks only).

## Planned commits

1. `docs: plan issue 1068 in TODO.md`
2. `docs(metrics): state ranked-retrieval boundary` — edit `docs/METRICS.md`
3. `docs(samples): note ranked-retrieval metrics are sample-local` — edit
   `samples/NivaraChat/README.md`
4. `docs: remove TODO.md — plan executed` (after G2 review)

## Blast radius

Documentation only. Affected symbols/locations:

- `docs/METRICS.md` — new prose section only; no citation added, so
  `DocCitationExtractor` / `DocReferenceExtractor` counts are unchanged.
- `samples/NivaraChat/README.md` — new prose line; README is in the scan scope,
  but no `docs/...md` reference is added, so the document-reference pin is
  unchanged.
- No source files, no tests, no pins. Risk: documentation-integrity gates
  (snippet gate, coverage pins). Both remain green because no fenced code is
  added and no counts move.

Symbols involved (from code-memory):

- `RagEvalMetrics` — `samples/NivaraChat/Helpers/RagEvalMetrics.cs` (class)
- `RagEvalMetricsTests` — `tests/Nivara.Tests/Samples/RagEvalMetricsTests.cs`
- `NivaraMetrics` — `src/Nivara/Metrics/NivaraMetrics.cs` (class)

Related methods in `RagEvalMetrics`: `PrecisionAtK`, `RecallAtK`, `Mrr`,
`NdcgAtK`, `SignTestPValue` (plus `WinLossTie`, `CohensD`, etc.).

## GitHub issues log

- None created yet.
