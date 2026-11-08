# RAG LLM-as-Judge Evaluation Plan (#943 follow-up)

## Goal

Evaluate whether multi-view retrieval improves over single-view retrieval using an
Ollama LLM as the relevance judge. The harness runs inside `samples/NivaraChat`
(sample-local only; no core `src/` changes) and requires `--ollama` so a large
cloud model (for example GPT-OSS or another Ollama cloud model) can act as judge.

## Motivation

Issue #943 asks for measurement, not just demonstration. Instead of relying only
on manually curated ground truth, this plan uses an LLM judge to grade the chunks
retrieved by each mode, while keeping the original issue's structure:

- Claim A: fused top-k surfaces more relevant chunks than single-view.
- Claim B: multi-view names which representation matched.
- Claim C: cost is acceptable for any gain.

A and B stay independent.

## High-level approach

1. Chunk the corpus once with `MultiViewChunker` so all modes share one chunk space and one set of `ChunkId` values (`{relativeSourcePath}#{index}`).
2. Embed four views per chunk (Content, Title, Section, Code) as `--rag-multiview` already does, including deduplication.
3. Treat `fused` (default weights) versus `content-only` (`content=1,title=0,section=0,code=0`) as the decisive comparison. Issue #943 identifies this as critical because single-view `--rag` embeds the whole chunk, approximately the Content view alone.
4. For each query, retrieve top-k from `fused`, `content-only`, and leave-one-view-out configurations.
5. Use the `--ollama` model as judge on a 0/1/2 scale.
6. Aggregate multiple runs and report paired statistics, diagnostics, and cost.

## CLI surface

New mode: `--rag-eval`

Shared flags:

- `--ollama <url>` (required), `--model <name>`
- `--docs-dir <path>`, `--top-k <n>`, `--chunk-size <n>`, `--silk`
- `--acl-group <name>`, `--rebuild`, `--no-cache`, `--batch-size <n>`
- `--weights <spec>`, `--report <path>`, `--explore`

New eval flags:

- `--queries <path>` (default `samples/data/nivarachat/eval/queries.tsv`)
- `--runs <n>` (default 3)
- `--temperature <f>` (default 0.2)
- `--grade-all` (grade every visible chunk for true recall; slower)
- `--verbose` (print per-query chunk movement)
- `--offline` (use cached labels only; no judge calls)

The judge model is the `--model` supplied with `--ollama`, so users can point at Ollama cloud models such as GPT-OSS.

## Ground truth and LLM grading

### Query file

Version-controlled text file: `samples/data/nivarachat/eval/queries.tsv`

Format:

    query<TAB>chunk_id_1,chunk_id_2,...<TAB>notes

Graded relevance can be encoded as `chunk_id:grade`.

Query sources:

1. At least 25 hand-curated cross-document queries.
2. Auto-derived heading-as-query pairs.
3. Auto-derived title-as-query pairs.

### LLM-as-judge labels

- Prompt returns JSON array of grades 0/1/2.
- Batch roughly 8 passages per call.
- Run each query `--runs` times at `--temperature`.
- Aggregate by majority vote per chunk; store mean and variance.
- `--grade-all` grades every visible chunk (true recall); otherwise grade retrieved plus sampled non-retrieved chunks and report relative recall.

### Cache

Labels cached to `samples/data/nivarachat/eval/relevance.tsv` with columns:

    query<TAB>chunk_id<TAB>grade<TAB>runs<TAB>stddev

`--offline` uses cache without judge calls.

## Chunk-id alignment

`RagChunker` and `MultiViewChunker` differ, so exact cross-chunker pairing is avoided.
Primary comparison is within one chunk space: multi-view `fused` versus multi-view `content-only`. Legacy `--rag` is a secondary baseline with documented overlap mapping.

## Metrics

Per query/config: Precision@k, Recall@k, MRR, nDCG@k (graded).

Aggregate: mean/std, paired win/loss/tie, sign-test p-value, Cohen's d.

Configurations: fused, content-only, and leave-one-view-out (zero each of Content, Title, Section, Code).

## Diagnostics

- `BestSpace` distribution.
- Agreement buckets: Agree, Mild, Strong, InsufficientViews.
- Divergence by view.
- Named categories: hidden-gem, misleading-title, context-match, implementation-evidence.
- Per-query chunk movement.

## Cost

- Index build time.
- Unique view text count.
- Index size on disk.
- Average query latency.
- Relative query cost factor.

## Implementation steps

1. Wire `--rag-eval` into `Program.cs`.
2. Add `RagEvalJudge.cs`, `RagEvalMetrics.cs`, `RagEvalLabels.cs`, and `EvalGroundTruth.cs` (TSV loader plus deterministic auto-derived heading/title pair generator).
3. Add `RagEvalMode.cs` orchestrating build, label, retrieval, scoring, output.
4. Reuse `MultiViewIndex.BuildAsync` and `MultiViewScoring.Rank`.
5. Generate initial `queries.tsv` (hand-curated plus auto-derived pairs).
6. Add offline cached-label path.
7. Document mode in README.
8. Add unit tests for metrics/parsing in `tests/Nivara.Tests/Samples/RagEvalMetricsTests.cs`.
9. Run formatting and docs/snippet gates.

## Acceptance criteria mapping

- Ground-truth file with at least 20 hand-curated cross-doc queries plus auto-derived pairs.
- Metrics printed as paired comparison.
- Ablation results reported.
- Diagnostics reported.
- Cost reported.
- No core Nivara changes.
- Docs/whitespace/snippet gates remain green.

## Risks and notes

- LLM judge bias; aggregate runs and keep temperature low.
- Cloud model availability/latency; cache labels aggressively.
- Chunker mismatch handled by content-only decisive comparison.
- Deterministic fallback: metric helpers are pure and unit-tested; `--offline` runs from cached labels only.
- MiniLM availability affects runtime.