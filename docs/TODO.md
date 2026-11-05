# RAG LLM-as-Judge Evaluation Plan (#943 follow-up)

## Problem

Issue #943 asks whether multi-view retrieval improves over single-view retrieval. We need an eval harness inside `samples/NivaraChat` that builds both modes, retrieves top-k, grades relevance with an Ollama LLM judge, and reports metrics, diagnostics, and cost. No core `src/` changes.

## Branch / base

- Branch: `khurram/943`
- Plan source: `docs/plans/RAG-LLM-EVAL.md`
- Sample-local only.

## Proposed changes

1. Wire `--rag-eval` mode in `samples/NivaraChat/Program.cs`.
2. Add helpers:
   - `samples/NivaraChat/Helpers/RagEvalMetrics.cs` — pure metric functions.
   - `samples/NivaraChat/Helpers/EvalGroundTruth.cs` — TSV loader plus deterministic auto-derived heading/title pairs.
   - `samples/NivaraChat/Helpers/RagEvalLabels.cs` — LLM judge label cache.
   - `samples/NivaraChat/Helpers/RagEvalJudge.cs` — Ollama judge client/prompts/batching.
3. Add `samples/NivaraChat/Modes/RagEvalMode.cs` orchestration.
4. Generate `samples/data/nivarachat/eval/queries.tsv`.
5. Add unit tests in `tests/Nivara.Tests/Samples/RagEvalMetricsTests.cs`.
6. Document mode in `samples/NivaraChat/README.md`.

## Verification

- `dotnet build Nivara.slnx` / sample build.
- Targeted NUnit tests for new helpers.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- Docs/snippet gates if docs change.

## Planned commits

1. `docs: plan #943 rag eval in TODO.md`
2. `feat(NivaraChat): add rag eval metrics and ground-truth helpers`
3. `feat(NivaraChat): add llm judge label cache`
4. `feat(NivaraChat): add rag-eval mode orchestration`
5. `feat(NivaraChat): wire --rag-eval cli`
6. `test(NivaraChat): add rag eval metric tests`
7. `docs(NivaraChat): document --rag-eval mode`

## Blast radius

- Sample project only; no `src/` changes.
- `Program.cs` switch and `ModeContext` reuse.
- README changes may require `DocsCoveragePins` updates.
- New eval data files under `samples/data/nivarachat/eval/`.

## GitHub issues log

- (none yet)
