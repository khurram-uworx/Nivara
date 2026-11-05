# RAG Evaluation Harness Plan (#943)

## Goal

Evaluate whether multi-view retrieval improves over single-view retrieval, as a follow-up to #914. The harness is sample-local only; no core `src/` changes.

## Scope

- Add a `--rag-eval` mode to `samples/NivaraChat`.
- Build both single-view and multi-view indexes on the same corpus.
- Read a version-controlled ground-truth TSV.
- Retrieve top-k from each mode and ablation.
- Compute IR metrics and print paired comparisons, diagnostics, and cost.

## Ground truth

File: `samples/data/nivarachat/eval/queries.tsv`

Format:

```
query<TAB>chunk_id_1,chunk_id_2,...<TAB>notes
```

Optional graded relevance: `chunk_id:grade` (grade defaults to 1).

Sources of pairs:

1. Hand-curated cross-document queries (at least 25).
2. Auto-derived heading-as-query pairs.
3. Auto-derived title-as-query pairs.

Chunk IDs must use `MultiViewChunk.ChunkId` (`{relativeSourcePath}#{index}`).

## Chunk-ID alignment

The single-view and multi-view chunkers differ. The decisive comparison is therefore performed within one chunk space: multi-view `fused` versus multi-view `Content-only`. The legacy single-view pipeline is reported as a secondary baseline with a documented content-overlap mapping and mapping coverage.

## Metrics

- Precision@k
- Recall@k
- MRR
- nDCG@k
- Mean +/- std
- Win/loss/tie counts
- Sign-test p-value
- Effect size

## Ablations

- `fused`: default weights.
- `content-only`: `content=1,title=0,section=0,code=0`.
- Leave one view out: zero each weight in turn.

## Diagnostics

- `BestSpace` distribution.
- Agreement buckets: Agree, Mild, Strong, InsufficientViews.
- Divergence by view.
- Named categories derived from per-view ranks.

## Cost

- Index build time.
- Unique text count.
- Index size on disk.
- Query latency per mode.
- Relative cost factor.

## Implementation steps

1. Add `RagEvalMetrics` helper.
2. Add `EvalGroundTruth` loader and generator.
3. Add `RagEvalMode`.
4. Wire `--rag-eval` in `Program.cs`.
5. Generate `queries.tsv`.
6. Add unit tests for metrics and parsing.
7. Document mode in README.
8. Run formatting and docs gates.

## Tests

- `tests/Nivara.Tests/Samples/RagEvalMetricsTests.cs`
- CLI argument coverage if added.
- Docs coverage pins updated for README changes.

## Risks

- Chunker mismatch prevents exact single-view pairing.
- MiniLM availability for heavy eval runs.
- Docs gate count changes when adding documentation.
