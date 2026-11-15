# RAG Implementation Guide — Differentiable Retrieval with Nivara

This guide walks through building a **differentiable retrieval** system on top of Nivara:
a multi-view RAG pipeline whose late-fusion weights are learned with AutoDiff, compared
against a fixed baseline on held-out queries, with reproducible baselines and a coverage
gate that refuses stale labels. The reference implementation lives in
`samples/NivaraChat` (`--rag-multiview`, `--rag-eval`, `--rag-learn`).

The guide is organized as one implementation flow, but the sample itself is layered: each
mode builds on the previous one. Read top-to-bottom, or jump to the layer you care about.
Each section names the Nivara APIs and the sample files that use them, so you can read it
top-to-bottom and then jump into the code.

## 0. Progressive tour (RAG 101 → Adaptive Intelligence)

NivaraChat ships one corpus (`docs/`) and five retrieval modes that build on each other.
Run them in order and watch the system grow from "search" to "self-learning":

| Step | Mode | What it adds | Issue |
|---|---|---|---|
| 1 | `--rag` | Basic RAG: chunk docs, MiniLM embeddings, top-K retrieval, optional LLM answer | #912 |
| 2 | `--rag-multiview` | Four views per chunk (Content/Title/Section/Code) + late fusion + disagreement report | #914 |
| 3 | `--rag-eval` | LLM-as-judge labels (`relevance.tsv`) + paired metrics | #912/#914 |
| 4 | `--rag-learn` | Learn the fusion weights + optional query gate via AutoDiff, vs fixed baseline | #915 |
| 5 | `--rag-adaptive` | Laya decision layer accepts/rejects retrieved chunks before answering | #916/#981 |
| 6 | `--rag-adaptive-eval` | End-to-end loop: retrieve → decide → learn → persist → evaluate → report | #916/#983 |

Commands (run from the repo root, `docs/` is the default corpus):

```
# 1. Basic RAG (LLM optional)
dotnet run --project samples/NivaraChat -- --rag --text "What is Nivara?"

# 2. Multi-view retrieval + disagreement report (no LLM)
dotnet run --project samples/NivaraChat -- --rag-multiview --text "What is Nivara?"

# 3. Judge labels (needs an LLM; use --ollama)
dotnet run --project samples/NivaraChat -- --rag-eval --regenerate judge --ollama --force

# 4. Differentiable retrieval: learn fusion weights, compare on held-out queries
dotnet run --project samples/NivaraChat -- --rag-learn
dotnet run --project samples/NivaraChat -- --rag-learn --labels auto --epochs 2   # offline smoke

# 5. Adaptive RAG: Laya decision layer filters retrieved chunks
dotnet run --project samples/NivaraChat -- --rag-adaptive --text "What is Nivara?"

# 6. Adaptive eval: retrieve -> decide -> learn -> persist -> evaluate -> report
dotnet run --project samples/NivaraChat -- --rag-adaptive-eval --labels judge --no-persist
dotnet run --project samples/NivaraChat -- --rag-adaptive-eval --labels judge --silk
```

What to notice at each step:

- **Step 1 → 2**: retrieval stops being "one embedding per chunk"; the disagreement
  report shows which view won and where views diverge. The four views are still fused by
  hand-picked weights `(0.5, 0.2, 0.2, 0.1)`.
- **Step 2 → 4**: `--rag-learn` prints `Init parity ≈ 0` (the learned model is numerically
  the fixed baseline at step 0) and then a held-out before/after table. A non-significant
  result is a valid result and is reported as such.
- **Step 4 → 5**: `--rag-adaptive` adds a typed decision (accept / fallback / abstain)
  with calibrated confidence per chunk, before any answer is generated.
- **Step 5 → 6**: `--rag-adaptive-eval` trains a small head on frozen ModernBERT
  features via AutoDiff, persists it with `ModelSerializer`, and re-scores the held-out
  split; the report shows decision accuracy and calibration before vs after, plus CPU vs
  Silk parity. Feedback is simulated from judge labels and is stated as such.

The rest of this guide (Sections 1-11) is the deep dive into step 4 and beyond. Sections
2-9 cover the differentiable retrieval machinery; Section 10 covers the adaptive loop.

## 1. Overview

The system has three stages:

1. **Indexing** — chunk the corpus, embed each chunk in several "views" (Content, Title,
   Section, Code), cache to Parquet.
2. **Retrieval / scoring** — for a query, compute per-view cosine similarities and fuse
   them with late-fusion weights.
3. **Learning** — make the fusion weights (and an optional per-query gate) trainable via
   AutoDiff, train on labeled queries, evaluate on held-out queries.

Two non-negotiable design rules:

- **Candidate generation stays outside the gradient path.** The trainable part only
  reranks a fixed candidate pool. This makes the comparison a reranking comparison, not
  three different retrieval systems.
- **The fixed baseline and the learned model share arithmetic.** The learned model is
  initialized so that at step 0 it is numerically identical to the fixed baseline
  (`max |learned - fixed| ≈ 1e-7`). Every later delta is attributable to learning, not
  to a different fusion formula.

## 2. Corpus and chunking

The corpus is every `docs/**/*.md` plus the sample README. Plan documents under
`docs/plan/` and `docs/plans/` are excluded (they are working notes, not reader-facing
docs).

Chunking is section-scoped with a configurable max size (`--chunk-size`, default 500).
Each chunk carries `ChunkId = "{relativePath}#{ordinal}"` — a positional identity. The
ordinal is on purpose: judge labels key on `path#ordinal`, so any docs edit that renumbers
chunks invalidates labels, and the coverage gate (Section 6) catches it.

Relevant code: `samples/NivaraChat/Helpers/MultiViewChunk.cs`,
`MultiViewChunker.ChunkMarkdown`.

## 3. Multi-view indexing

Each chunk is embedded **four times** in one 384-d MiniLM space, once per view
(Content, Title, Section, Code). Views are deduplicated before embedding because titles
and sections repeat heavily, then expanded back to per-chunk rows.

The index is cached to Parquet keyed by a content hash + chunker version + chunk size +
backend. `MultiViewIndex.DiscoverCorpusFiles` produces a canonical, path-sorted file list
(README inserted in sort order) so the hash is order-independent and rename-aware.

Relevant code: `samples/NivaraChat/Helpers/MultiViewIndex.cs`,
`MultiViewIndexCache.cs`, `RagIndexCache.ComputeContentHash(mdFiles, repoRoot)`.

Nivara APIs used: `NivaraFrame`, `NivaraColumn<T>`, `TensorPrimitives` (via MiniLM),
Parquet I/O (`ToParquet` / `LoadParquet`).

## 4. Retrieval and scoring

For a query, `MultiViewScoring.ScoreAll` embeds the query and computes all 4 view
cosines for every chunk (read-side, no gradients). Each index row gets
`(score, present)` per view.

Late fusion per row:

```
fused = ((S · M) @ w) / (M @ w)
```

where `S` is the `[rows, 4]` cosine scores, `M` the presence mask, and `w` the 4 fusion
weights. The denominator is never zero: softmax weights are strictly positive and every
pooled row has at least one scored view.

`MultiViewScoring.Fuse` is the reference scorer; the learned path calls the same
arithmetic via `MaskedFusion.Apply`, so the number a reader sees in `--rag-multiview` is
the number the model optimizes.

Relevant code: `samples/NivaraChat/Helpers/MultiViewScoring.cs`,
`RetrievalTrainer.MaskedFusion`.

## 5. Making it differentiable (AutoDiff)

The four weights become parameters:

- `w = softmax(theta)`, `theta ∈ R^4`. Initialized to `ln(0.5, 0.2, 0.2, 0.1)` so
  `softmax(theta)` returns the defaults exactly — the parity property.
- `FusionModel : Module<float>` holds `logits : Parameter<float>` of shape `[4, 1]`.
- `GatedFusionModel` adds a per-query gate: `weight : Parameter` of shape `[4, features]`
  (zero) plus `bias : Parameter` of shape `[4, 1]` (initialized to `ln(defaults)`),
  so the gate is input-invariant at step 0.

Losses (listwise by default):

- Listwise softmax cross-entropy against a `2^grade` target, normalized per query.
- Pairwise logistic via `BCEWithLogitsLoss` through a `+/-1` selector matrix.

Training uses `GradientUtils.Grad()` to scope reverse-mode AD, Adam for updates, and
`StateDict()` / `LoadStateDict()` for best-checkpoint restore. Validation nDCG drives
early stopping with `Patience`.

Nivara APIs used: `Module<T>`, `Parameter<T>`, `ReverseGradTensor`, `GradNode`,
`ReverseGradOperations` (`LogSoftmax`, `MatMul`, `Divide`, `Sum`), `Linear<T>`,
`Loss<T>` / `BCEWithLogitsLoss`, `SGD`/`Adam`/`AdamW`, `TrainingLoop`,
`Module<T>.StateDict` / `LoadStateDict`.

Relevant code: `samples/NivaraChat/Helpers/RetrievalTrainer.cs`.

## 6. Training data and labels

A training example is a **whole query's candidate pool**, not a row. One AutoDiff graph
and one Adam step per query.

Pipeline per query:

1. `MultiViewScoring.ScoreAll` computes all 4 view cosines for all chunks.
2. `CandidatePool.Select` builds the pool = union of each view's own top-K rows plus
   every row graded above zero, returned ascending. Selection is deliberately
   **weight-agnostic** so a chunk only the Code or Title view ranks highly can still be
   promoted.
3. Rows with zero present views are dropped and counted (never divided by zero).
4. `TrainingSetBuilder.Create` flattens to `[rows, 4]` score + presence blocks, normalizes
   the `2^grade` loss target (uniform fallback when all zero).

Labels (three sources, all projecting to `query -> row -> grade`):

- **Judge** — `relevance.tsv` from `--rag-eval`, graded 0/1/2 per chunk. Headline source.
- **Auto** — heading/title-derived, leakage diagnostic only (Title/Section score their own
  positive near 1.0).
- **Fixture** — deprecated alias for judge (single file now).

Coverage gate: refuse to train when fewer than 50% of label rows resolve against the
current corpus (`RetrievalLabels.MinimumCoverage = 0.5`, `Accepts`). A stale label file
silently shrinks the relevant set and lowers every ranker; the gate makes it loud
(`REFUSED:` marker, exit code 2).

Split: 80/10/10 by an FNV-1a hash of the query string with the seed mixed in
(`QuerySplit`), because `string.GetHashCode()` is randomized per process.

Relevant code: `samples/NivaraChat/Helpers/RetrievalTrainingData.cs`,
`RetrievalLabels.cs`.

## 7. Reproducibility and baselines (916-A)

The same corpus + seed must produce the same chunk ids, split and baseline metrics. Three
things pin that:

1. **Chunk identity** — `(SourcePath, ordinal, ChunkerVersion, ChunkSize)`.
2. **Corpus identity** — `CorpusIdentity` (content hash, file count, chunk count, chunker
   version, chunk size, docs dir, backend, dimension), captured after the index resolves
   on both cache-hit and rebuild paths. One identity line is printed per run.
3. **Split** — FNV-1a over UTF-8 bytes with the seed mixed in.

Content hash: `RagIndexCache.ComputeContentHash(mdFiles, repoRoot)` orders files by
repo-relative path and hashes `relativePath \0 fileBytes \0` per file — order-independent
and rename-aware.

Baseline artifact: `RetrievalBaseline` (JSON, indented, no timestamps) records schema
version, corpus identity, label kind/description/source path, matched/total/coverage/
is-complete, seed, k, candidates, split counts, and the two fixed baselines
(`fixed(default)`, `fixed(content-only)`) over **both** scopes — all resolved usable label
queries (wide, stable starting point) and the held-out test split (the scope the learned
comparisons use). Default path:
`samples/data/nivarachat/eval/baseline-{kind}-s{seed}-k{k}.json` (gitignored),
overridable with `--baseline <path>`.

Full-set fixed metrics are not leakage: the fixed weights never see labels. Learned models
are scored on the test split only.

Regeneration: `--regenerate <queries|judge|all>` rebuilds label artifacts deterministically
(sorted rows, LF, no BOM, single trailing newline), with a provenance header on
`relevance.tsv` and a manifest for `all`. Refuses to overwrite without `--force`.

Relevant code: `samples/NivaraChat/Helpers/RetrievalEvaluation.cs`,
`MultiViewIndex.cs`, `RagIndexCache.cs`.

## 8. Evaluation

The mode prints a paired, held-out comparison. Neither side trains on the test split;
validation nDCG drives checkpoint selection, and the test split scores once at the end.

Metrics: Precision@K, Recall@K, MRR, nDCG@K (same definition as `--rag-eval`), plus a
paired sign test and Cohen's d on per-query nDCG. `Init parity ≈ 0` is the step-0 sanity
check; a large value means a bug, not a result. A non-significant result is a valid
result and is reported as such.

Files written: `--report <path>` (markdown) and the shared multi-view Parquet cache
(`--rebuild` to force).

## 9. How to run

```
dotnet run --project samples/NivaraChat -- --rag-learn
```

Defaults: judge labels when the cache exists, 200 epochs, lr 0.05, weight decay 0.01,
patience 20, `stats` gate, `listwise` loss, candidate top-10, seed 42, K 10.

Key flags: `--labels judge|auto`, `--epochs`, `--lr`, `--weight-decay`, `--patience`,
`--gate stats|embed|none`, `--loss listwise|pairwise`, `--candidates`, `--seed`,
`--train-queries`, `--k`, `--docs-dir`, `--silk`, `--report`, `--baseline`,
`--regenerate`, `--force`.

Offline smoke: `--rag-learn --labels auto --epochs 2` (after generating queries).
Reproducibility check: rerun with the same `--seed` and expect weights to agree to 1e-7.

## 10. Adaptive decision loop (#983)

`--rag-adaptive-eval` runs the full loop: retrieve (multi-view top-K) -> understand (MiniLM
views + frozen ModernBERT features) -> decide (Laya choice head) -> learn (online AutoDiff
updates on train-split simulated feedback) -> persisted update (`laya_head.json`, unless
`--no-persist`) -> evaluate on the held-out split -> report.

- Candidate set is the index's top-K per query; the table is a within-run reranking comparison.
  `adaptive-rerank` preserves top-K membership, so its P@K/R@K are invariant by construction;
  only `adaptive-filter` can move them.
- Feedback is simulated from judge labels and is stated as such.
- Auto labels (`--labels auto`) are a declared leakage diagnostic.
- Calibration: ECE (10 equal-width bins) and Brier on `AnswerConfidence` vs choice correctness,
  before and after.
- Timings split feature extraction (encoder) from head updates (AutoDiff) and persistence;
  index build times are recorded for both CPU and Silk (when `--silk`).
- Silk scope: MiniLM embedding generation and parity only. Decisions and head training are CPU
  in both backends. Hand-written Silk weight steps are a follow-up.
- Report + metrics written to `samples/data/nivarachat/adaptive/` (gitignored):
  rag-adaptive-report.md, `metrics-{kind}-s{seed}-k{k}.json`. `laya_head.json` is ~147 MB;
  use `--no-persist` or `--head-weights <temp>` for read-only runs.

Run:

```
dotnet run --project samples/NivaraChat -- --rag-adaptive-eval --labels judge
dotnet run --project samples/NivaraChat -- --rag-adaptive-eval --labels judge --silk
dotnet run --project samples/NivaraChat -- --rag-adaptive-eval --labels judge --no-persist --max-queries 5
```

Key flags: `--labels`, `--seed`, `--k`, `--max-queries`, `--max-examples`, `--head-epochs`,
`--head-lr`, `--update-batch`, `--resume`, `--head-weights`, `--no-persist`, `--report`,
`--silk`, `--docs-dir`, `--threshold`.

## 11. References

- `samples/NivaraChat/Helpers/MultiViewChunk.cs`, `MultiViewChunker` — chunking.
- `samples/NivaraChat/Helpers/MultiViewIndex.cs`, `MultiViewIndexCache.cs` — indexing,
  `CorpusIdentity`, `DiscoverCorpusFiles`.
- `samples/NivaraChat/Helpers/MultiViewScoring.cs` — scoring and fusion.
- `samples/NivaraChat/Helpers/RetrievalTrainer.cs` — differentiable fusion, losses,
  training loop.
- `samples/NivaraChat/Helpers/RetrievalTrainingData.cs`, `RetrievalLabels.cs` — data,
  labels, split.
- `samples/NivaraChat/Helpers/RetrievalEvaluation.cs` — metrics, baseline.
- `samples/NivaraChat/Helpers/RagIndexCache.cs`, `RagEvalLabels.cs`,
  `EvalGroundTruth.cs` — corpus hash, judge labels, auto labels.
- `samples/NivaraChat/Modes/RagLearnMode.cs`, `RagEvalMode.cs` — mode runners.
- `samples/NivaraChat/Program.cs`, `Modes/ModeContext.cs` — CLI plumbing.

Nivara core APIs exercised: `NivaraFrame`, `NivaraColumn<T>`, `TensorPrimitives`,
`Module<T>`, `Parameter<T>`, `ReverseGradTensor`, `ReverseGradOperations`,
`Linear<T>`, `Loss<T>` / `BCEWithLogitsLoss`, `Adam`/`AdamW`, Parquet I/O,
`NivaraTensorExtensions`.

