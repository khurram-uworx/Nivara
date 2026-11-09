# NivaraChat RAG - Design Notes

Consolidated design notes for the retrieval work in samples/NivaraChat.

- Part 1: #915 Differentiable Retrieval and Learning-to-Rank (--rag-learn).
- Part 2: #916-A / #980 Reproducible evaluation baseline over docs/.

The two efforts are complementary: 915 is the learning machinery, 916-A is the reproducibility and baseline discipline that makes 915 numbers trustworthy. See Section 7 for how 916-A supports 915.

---

# #915 — Differentiable Retrieval and Learning-to-Rank (`--rag-learn`)

## 1. Overview of the ask

`--rag-multiview` (#914) landed four independently scored embedding views of a chunk
(Content, Title, Section, Code) with a single hand-picked late-fusion weight vector
`(0.5, 0.2, 0.2, 0.1)`. Those weights are a design choice in a file, not a result, so
the flagship claim — *"Nivara is a differentiable DataFrame: data to query, to numerical
computation, to loss, to gradients, to optimization"* — was demonstrated only by samples
(AutoDiff, MicroGpt) that never touch the retrieval stack.

Issue #915 closes that gap with a new `samples/NivaraChat` mode that

1. learns the four fusion weights,
2. learns an optional query-gated mixture of the four cosine scorers,
3. compares both against the fixed #914 baseline on queries neither model saw,
4. prints and writes the comparison (and never overclaims a win).

Scope is **sample-only** — no `src/Nivara` changes. Candidate generation stays outside the
gradient path; the trainable part only reranks retrieved candidates. Non-goals, explicitly
out of scope: metric learning, a learned similarity kernel, an ANN index, PDF extraction,
and Spearman/Kendall. CPU-only by default; with `--silk` the fusion training (forward, gradient, Adam) runs on the
OpenCL device via `SilkFusionTrainer` (see section 5.1 of the README and `docs/SILK.md`).

## 2. Executive description of the design (the model)

The data is still the #914 multi-view dataset: one row per chunk, four cosine scores in one
384-d space. The change is that the four weights become parameters and each becomes a
softplus/softmax of a learnable logit.

- `w = softmax(theta)`, `theta` in R^4, so every weight stays strictly positive and the
  vector sums to one.
- `fused = ((S . M) @ w) / (M @ w)`, with `S` the cosine scores of shape `[rows, 4]`
  and `M` the presence mask. That is the tensor form of the per-row renormalization the
  sample already computes, so the fused number a reader sees in `--rag-multiview` is the
  same number the model optimizes — nothing is re-derived.
- The denominator cannot be zero. Softmax is strictly positive and every pooled row has at
  least one scored view, so `ReverseGradOperations.Divide`'s zero-divisor pre-check cannot
  fire.

**The parity property (load-bearing).** `FusionWeights.Default` sums to exactly `1.0`, so
initializing `theta = ln(0.5, 0.2, 0.2, 0.1)` makes `softmax(theta)` return the defaults
exactly and the learned model is numerically the #914 baseline at step 0 (max
`|learned - fixed| ≈ 1e-7`). A gate initialized at `V = 0`, `b = ln(defaults)` is
likewise invariant to its input. So every later delta is attributable to learning and not
to a different fusion formula — and the mode prints that parity number so a drifted path
cannot pass silently.

**Query-gated mixture of experts.** The four cosine scorers are the experts; the gate
produces their weights per query. The default feature vector is each view's mean and max
cosine over the candidate pool (missing views excluded from the mean): 8 features into 4
logits, 36 parameters — the size that can survive roughly a hundred training queries.
`--gate embed` trades it for a `Linear<float>(384, 4)` over the raw MiniLM embedding
(1,540 parameters) for comparison; `--gate none` skips the gated model entirely.

**Loss.** Listwise softmax cross-entropy against a `2^grade` target, normalized per
query, so grade 2 outranks 1 outranks 0 and one query builds one graph (no row-gather
op is needed). `--loss pairwise` is the gather-free logistic variant through a `+/-1`
selector matrix. `BCEWithLogitsLoss` handles the per-pair term.

**Labels.** Auto labels (heading/title derived) are trivially learnable by the Title and
Section views, so they are a leakage diagnostic only and never the headline. The headline
is the LLM judge cache written by `--rag-eval`; the offline fallback is a fixture keyed
by **document path** rather than `path#chunk` id, because chunk ordinals move whenever a
document is edited and that is exactly how the judge cache went stale after the `docs/SILK.md`
change.

### Training data — what one example actually is

Important: a training example is a **whole query's candidate pool**, not a row. The loss is
listwise over the pool, and there is one AutoDiff graph + one Adam step per query.

**Ingredients (three files):**

| Source | What | Scale |
|---|---|---|
| Corpus | every `docs/**/*.md` + the sample README, section-scoped into chunks, each chunk embedded 4 times in 384-d MiniLM space | ~2,389 chunks |
| Index | `samples/data/nivarachat/rag_multiview_cache/rag_multiview_*.parquet` built once on corpus hash; MiniLM embeddings (CPU or `--silk`) | ~3,386 unique view texts after dedup |
| Labels | one of the three sources below, all projecting onto the same `query -> row -> grade` map | judge: ~453 queries, ~21k rows; fixture: 40 queries; auto: 500 queries |

Label sources:

- **Judge** (`samples/data/nivarachat/eval/relevance.tsv`, from `--rag-eval`) — keys are
  `path#chunk` ids, grades 0/1/2. The headline source.
- **Judge** (`relevance.tsv`, generated by `--rag-eval --ollama`, human-editable) — graded 0/1/2 per chunk. The separate fixture file is retired; `--labels fixture` is a deprecated alias.
  to every chunk of that document at load time; survives re-chunking and docs edits.
- **Auto** (`samples/NivaraChat/queries.tsv`, opt-in) — heading/title-derived, self-labeling;
  prints a leakage warning and is never the headline.

The mode **refuses to train when fewer than 50 % of label rows resolve** against the current
corpus, because a stale label file shrinks the relevant set and lowers every ranker
silently.

**How a single query becomes a training row (`TrainingSetBuilder.Create`):**

1. `MultiViewScoring.ScoreAll` embeds the query and computes **all 4 view cosines for all
   2,389 chunks** (read-side scoring, no gradients), so each index row has
   `(score, present)` for Content/Title/Section/Code.
2. `CandidatePool.Select` builds the pool = **union of each view's own top-K rows plus every
   row graded above zero**, returned in ascending row order. The selection is deliberately
   **weight-agnostic** — a chunk only the Code or Title view ranks highly can still be promoted.
   Drawing the pool from the default fusion's top-K would let the baseline choose its competitors.
3. Rows with zero present views are **dropped and counted** (dropping rather than dividing by
   zero; `ReverseGradOperations.Divide` throws on a zero divisor).
4. The outcome (one `TrainingQuery`):
   ```
   query.Scores     float[rowCount * 4], row-major, 0f where a view is absent
   query.Present    bool  [rowCount * 4]                      // the mask
   query.LossTarget  float[rowCount], 2^grade normalized per query; uniform fallback when all zero
   query.QueryVector                      float[384]          // used by --gate embed
   query.Candidates    RankedCandidate[]   (Row, ChunkId, Grade)
   ```
   Pool sizes are 30-100 rows depending on `--candidates` (default 10 per view) and label
   overlap.

**What one epoch does:** shuffle usable queries; for each query, one forward + backward + Adam
step (gradient scoped through `GradientUtils.Grad()`); then a validation nDCG@K with patience
and a best-checkpoint `StateDict()`/`LoadStateDict`. Concretely: judge run has 451 usable
queries, split 341 / 60 / 50, and 50 held-out test queries; fixture run 40 queries, 34 / 4 / 2.

**Knobs that move the data distribution:**

| Flag | Effect on the data |
|---|---|
| `--candidates N` | per-view top-K fed into the pool |
| `--labels judge \| fixture \| auto` | which grader — judge grades, doc-path fixture, auto pseudo-queries |
| `--train-queries N` | cap the training split |
| `--seed N` | hash-re-split (the split is process-independent by design) |
| `--train-queries 20` | end-to-end smoke on the judge cache |

## 3. Design — how it is implemented

### Modules (`samples/NivaraChat/Helpers/RetrievalTrainer.cs`)

- `IFusionRanker` — the shared forward contract: `FeatureCount`, `TrainableParameters`,
  `FeatureVector`, `ForwardMasked` (returns a `[rows, 1]` score tensor), `WeightsFor`.
  One interface so the model, its metric and its report never disagree on arithmetic.
- `FusionModel : Module<float>` — `logits : Parameter<float>` of shape `[4, 1]`,
  initialized via `ln(default weights)`. `ForwardMasked` builds the score block from the
  query, applies the learned softmax, and calls `MaskedFusion.Apply`. A global `Forward`
  survives for the module contract (all views present).
- `GatedFusionModel` — `weight : Parameter` of shape `[4, features]` (zero), plus
  `bias : Parameter` of shape `[4, 1]` initialized to `ln(default weights)`. `Forward` is
  the gate, `ForwardMasked` runs the gate on the query features then fuses.
- `MaskedFusion` — the central arithmetic `((S . M) @ w) / (M @ w)`, building the mask
  tensor from the flattened `bool[]`. Also holds `Weights` (max-stable softmax for
  reporting) and `LogWeights`.
- `ListwiseLoss` — `-sum(target * LogSoftmax(fused, dim:0))` reshaped as `0 - sum`.
- `PairwiseLoss` — `A @ fused` then `BCEWithLogitsLoss` against ones.
- `TrainingOptions`, `EpochReport`, `TrainingReport`, and `RetrievalTrainer.Train` — the
  Adam loop with gradient scoping via `GradientUtils.Grad()`, early stopping on
  validation nDCG with `Patience`, best-checkpoint restore through `StateDict()` /
  `LoadStateDict`, and per-group weight decay.

### Training data (`samples/NivaraChat/Helpers/RetrievalTrainingData.cs`)

- `TrainingQuery` — query text, embedding, candidates, the flattened `[rows, 4]` score
  and presence blocks, the normalized `2^grade` target, the count of dropped rows and the
  label source.
- `CandidatePool.Select` — union of each view's top-K ranking (ties broken by ascending
  row index) plus every row graded above zero, returned sorted ascending. Deliberately
  weight-agnostic.
- `TrainingSetBuilder.Create` — pools, drops any row with zero present views (counting
  it rather than allowing a zero denominator), normalizes the loss target with a uniform
  fallback when everything is grade 0.
- `RetrievalLabels` — three sources: judge (`RagEvalLabels.Load`, keyed by chunk id),
  fixture (`relevance-fixture.tsv`, keyed by document path), auto (`EvalGroundTruth.Load`,
  heading-derived). `Resolve` maps target ids onto the built chunks, so a document path
  grades every chunk of that document and a chunk id grades every matching row.
  `Coverage` and `IsComplete` make a stale file fail loudly.
- `QuerySplit` — 80/10/10 by an FNV-1a hash of the query string with the seed mixed in,
  because `string.GetHashCode()` is randomized per process.

### Evaluation (`samples/NivaraChat/Helpers/RetrievalEvaluation.cs`)

- `FixedScores` calls `MultiViewScoring.Fuse` (the reference scorer) rather than a
  second tensor implementation — the fixed baseline is the sample's own arithmetic.
- `Scores` reads the model's fused scores through `MaskedFusion.ToArray`, since
  `ReverseGradTensor.AsSpan()` is internal to `Nivara`.
- `Rank` breaks ties by candidate index (which is ascending row order, since the pool is
  built sorted).
- `Measure`, `Evaluate`, `EvaluateFixed`, `MeanNdcg` and `Compare` (paired comparison on
  per-query nDCG: sign test and Cohen's d).

### The mode (`samples/NivaraChat/Modes/RagLearnMode.cs`)

Loads MiniLM, builds the multi-view index, resolves labels (refusing to train below 50%
coverage), builds and splits the training set, trains both models, and prints the table,
the parity line, the learned weights, the paired comparisons, and a significance note when
the held-out split is too small. Writes the markdown report.
- `ModeContext` gained `Labels`, `Epochs`, `LearningRate`, `WeightDecay`, `Patience`,
  `Gate`, `Loss`, `Candidates`, `Seed`, `TrainQueries`, `EvalK`; `Program.cs` parses
  them as `--labels`, `--epochs`, `--lr`, `--weight-decay`, `--patience`, `--gate`,
  `--loss`, `--candidates`, `--seed`, `--train-queries`, `--k`, dispatches
  `--rag-learn` and prints usage.

## 4. Features — how to use, what to tweak, what happens with what

### Basic usage

```
dotnet run --project samples/NivaraChat -- --rag-learn
```

Defaults: judge labels when the cache exists and `relevance-fixture.tsv` otherwise, 200
epochs, lr 0.05, weight decay 0.01, patience 20, `stats` gate, `listwise` loss, candidate
top-10, seed 42, all training queries, K 10.

### Options

| Flag | Default | What it changes |
|------|---------|-----------------|
| `--labels judge\|fixture\|auto` | judge / fixture | Label source; `auto` is opt-in and prints a leakage warning |
| `--epochs <n>` | 200 | Training epochs |
| `--lr <f>` | 0.05 | Adam learning rate |
| `--weight-decay <f>` | 0.01 | Per-group weight decay |
| `--patience <n>` | 20 | Early stopping on validation nDCG |
| `--gate stats\|embed\|none` | stats | Gate feature source; `none` skips the gated model |
| `--loss listwise\|pairwise` | listwise | Ranking loss |
| `--candidates <n>` | 10 | Rows each view contributes to the pool |
| `--seed <n>` | 42 | Split and shuffle seed |
| `--train-queries <n>` | all | Cap the training split |
| `--k <n>` | 10 | Metric cut-off |
| `--docs-dir` | `docs/` (recursive) | Corpus to index |
| `--silk` | off | MiniLM embeddings (device) plus, for `--rag-learn`, device-native fusion training (forward, gradient, Adam) via `SilkRetrievalTrainer` on listwise loss; pairwise falls back to CPU || `--report <path>` | `.../learn/rag-learn-report.md` | Markdown report location |
| `--acl-group`, `--rebuild`, `--no-cache`, `--chunk-size` | | As in `--rag-multiview` |

### What happens with what

- `_train` with too few labels (fewer than half the rows resolve) refuses to run; enrich the
  judge cache or use `--labels fixture`.
- `Init parity ≈ 0` confirms both fusion paths agree; a large value means a bug, not a result.
- `learned-global vs fixed(default)` wins/losses are a reranking comparison, not a retrieval
  one: the pool is identical for all rankers.
- `best epoch / val nDCG` and the early-stop line: with `Patience = 20` one bad-less run
  stops early, restoring the best validation checkpoint.
- `Learned weights` shows how the global model moved; the gated line averages gates across
  test queries, since they vary per query by construction.
- With 40 fixture queries the test split is 2 queries and the sign test cannot reach p < 0.05,
  so that note is printed; the judge cache (452 queries) gives a 50-query test set.
- The fixture (`relevance-fixture.tsv`) keys grades by document path so they expand to every
  chunk of a document, and survives re-chunking and docs edits.

### What to tweak

- To push the model hard: raise epochs and lr, lower patience, use `--gate embed`; expect the
  384-to-4 gate to need more data (it degrades results on ~100 queries by design).
- `--loss pairwise` on small pools with a single relevant item per query is mostly a tie-break
  exercise; listwise is the recommended headline.
- Raise `--candidates` to grow the pool per query; it grows training time roughly per row.
- Switch label source to compare judge vs. fixture vs. auto and read the deltas as a diagnostic
  on the label agreement itself, not on the model.
- Use `--train-queries 20` for a quick end-to-end smoke (the full run embeds the whole corpus
  and takes ~2 min on CPU; the multi-view cache makes subsequent runs fast).
- To force a clean rebuild: `--rebuild`; for offline environments: keep `--labels fixture`
  (or `--labels judge` with a cached `relevance.tsv`).

## 5. How to evaluate

The mode prints a paired, held-out comparison. Neither side is trained on the test split; only the
validation nDCG drives checkpoint selection, and the test split scores once at the end. The 40-query
fixture and the 453-query judge cache are the same code path, only the label source changes.

### What is printed

- `Resolved: M/N rows (...) K queries against C chunks` -- coverage, i.e. how many label rows
  actually matched a chunk this corpus produces. Below 50% the mode refuses to train rather than
  silently scoring a shrunken relevant set.
- The metric table: `P@k R@k MRR nDCG@k Queries` for `fixed(default)`, `fixed(content-only)`,
  `learned-global` and (unless `--gate none`) `learned-gated(...)`.
- `Init parity: max |learned - fixed|` -- the step-0 sanity check (target approximately 1e-7).
- `learned-*: best epoch E/N val nDCG ... train loss ... (S steps, Q queries)` -- plus
  `early stop` when patience fired.
- `Learned weights: fixed/learned/gated` -- how the global weights moved and the mean gate across
  test queries.
- The paired `wins=`/`losses=`/`ties=` line plus the sign-test p and Cohen's d against
  `fixed(default)`, on per-query nDCG. `IsSignificant` requires a strictly positive delta and
  p < 0.05.
- A note when the held-out split is too small to reach significance (fewer than six decisive
  queries).
- `Report written to <path>`.

### The metrics

- `Precision@K` -- fraction of the top-K retained as relevant (grade above zero). Indicates how
  clean the top of the list is.
- `Recall@K` -- fraction of the pool's relevant rows that the top-K recovers. Bounded below by
  the candidate-pool size and the per-view setting.
- `MRR` -- reciprocal rank of the first relevant row; sensitive to the single best hit.
- `nDCG@K` -- `DCG@K / iDCG@K` using the graded `2^grade` relevance. Carries the same
  definition as in `--rag-eval`, so the two samples' numbers are comparable.
- Per-query `Ndcg` is what `Compare` reduces to a sign test: win/loss/tie over the same query
  set, then a two-sided p = 2 * Binom(0.5, non-ties) and Cohen's d on the differences.

### How to read a report

- `fixed(default)` vs `fixed(content-only)`: the #914 contribution, restated on the same pool. It
  sets the bar a learned model must beat.
- `learned-global` vs `fixed(default)`: the headline. A good outcome is a positive dNDCG with
  enough non-ties that p < 0.05 and a coherent weight shift.
- `learned-gated(stats/embed)` vs `fixed(default)`: expected to match or slightly underperform
  `learned-global` on roughly 100 queries. Treat as evidence, not as an improvement.
- A non-significant result is a valid result. Both the README and the report say so, because with
  a handful of held-out queries the sign test cannot discriminate either way.
- `Init parity ~= 0`: without this line a "gain" is uninterpretable because it could come from
  changing the fusion formula. With it, the comparison is the baseline plus learning.

### Files written

- `--report <path>` (default `samples/data/nivarachat/learn/rag-learn-report.md`, gitignored):
  label source and coverage line, split counts, candidate-pool description, loss and
  hyperparameter lines, parity line, metric table, paired comparisons, learned weights and a
  reading note.
- `samples/data/nivarachat/rag_multiview_cache/rag_multiview_*.parquet`: the shared multi-view
  index, so a second run does not re-embed about 3,000 texts. `--rebuild` forces it.

### Reproducing

- Full headline (judge, default seed): `--rag-learn --labels judge --gate stats --loss listwise`
  (needs the `relevance.tsv` cache from `--rag-eval`; re-running the judge can take a while).
- Offline smoke (fixture): `--rag-learn --labels fixture --epochs 2 --gate embed --loss pairwise`.
- Determinism: rerun with the same `--seed` and expect the weights to agree to 1e-7.
- To confirm the hold-out story is not artefactual, compare `--gate none` with `--gate stats` and
  read the significance column honestly: most runs on roughly 100 queries are inconclusive, and
  that is the result, not a bug.

## 6. Recommended follow-ups and next actions

Nothing in the numbered items below is required before merging: #915 ships as a sample that runs
end to end. They are the shortest list that takes it from "demonstrates the science" to "reusable
by someone else."

1. **Silk training (GPU) — DONE.** `--silk` now trains the fusion loop and gate on the OpenCL
   device through `SilkFusionTrainer` (`samples/NivaraChat/Helpers/SilkRetrievalTrainer.cs`),
   reusing the `fusion_forward`, `fusion_wgrad` and `adam_update` kernels. The pairwise loss
   still falls back to CPU; listwise is the stable default path. The 36-parameter stats gate
   stays per query cold-path — the same measurement bar from Part 1 still applies.
2. **A low-rank gate that learns cleanly.** The 1,540-parameter embed gate underperforms the
   global fusion at ~100 queries. A low-rank design (`384 -> 8 -> 4`) plus better joint
   training is the next experiment, and is the follow-up reserved by the Silk section.
3. **More queries.** The judge cache is the headline label source and it is the thing holding back
   significance. A second judge run with roughly 1,000 queries and `--runs 5` pushes the test
   split past 100 held-out queries, at which point the sign test stops being a coin flip.
4. **Fixtures, not secrets.** The fixture is 40 hand-authored queries. Either it grows with the
   codebase (a query per major docs change keeps it honest) or is retired in favor of a recorded
   judge run committed as seed data. The latter has its own risks and is not done now.
5. **Candidate-pool diagnostic.** `--candidates 10` is fixed. Reporting pool sparsity per query
   (how many rows end up with `PresentCount == 1`) would show whether the chunker produces
   enough multi-view disagreement to justify four views at all.
6. **Semi-supervised targets.** Training grades are human/judge labels only. Using the
   content-only cosine (or a cross-encoder) as a distillation teacher on unlabeled candidates is
   reasonable and would make the learned model strictly stronger than the judge-only numbers; that
   is the natural next issue rather than part of #915.
7. **CI coverage.** `NivaraChat` is built in tests only because the Docs namespace references it.
   If `--rag-learn` ever gains a new required dependency, the fix lands in
   `tests/Nivara.Tests/Samples` first, then in the sample. The 24-test `RetrievalTrainerTests`
   is that gate.
8. **Optional API promotion.** `IFusionRanker` is sample-local. If reused elsewhere, promote
   `MaskedFusion`, `FusionModel`/`GatedFusionModel` and `RetrievalEvaluation` to a shared helper
   namespace, and decide whether softmax-parameterized late fusion belongs in core as one of the
   promised dynamic-data-frame operations.


---

# 916-A — Reproducible evaluation baseline over `docs/`

## What "reproducible" means here

The same corpus + seed must produce the same chunk ids, the same query split and the same
baseline metrics. Three things pin that:

1. **Chunk identity** — `ChunkId = "{relativePath}#{ordinal}"` (see `MultiViewChunker`).
2. **Corpus identity** — `CorpusIdentity` (content hash, file count, chunk count, chunker
   version, chunk size, docs dir, backend, dimension) captured after the index resolves.
3. **Split** — `QuerySplit.Create` uses FNV-1a over UTF-8 bytes with the seed mixed in, so it
   is process-independent.

## Chunk identity and corpus-hash contract

- Identity is `(SourcePath, ordinal, ChunkerVersion, ChunkSize)`.
- Ordinals are positional on purpose: judge labels in `relevance.tsv` key on `path#ordinal`.
- Any change to chunker output requires a `ChunkerVersion` bump.
- There is no path-keyed escape hatch: `relevance.tsv` is keyed by chunk id, so any docs edit
  that renumbers chunks invalidates labels and the coverage gate must catch it.
- `RagIndexCache.ComputeContentHash(mdFiles, repoRoot)` orders files by repo-relative path and
  hashes `relativePath \0 fileBytes \0` per file. This fixes enumeration-order dependence (F1)
  and makes renames visible (F2). Absolute paths are not part of the hash.
- F3 (machine-specific cache filenames) is a documented limitation: normalising it needs
  `SchemaVersion++` and is not required by the acceptance criteria.

Changing the hash format invalidates every existing `rag_*_cache` and `rag_multiview_cache`
file; the next run rebuilds.

## Label sources

| Source | File | Key | Grade scale | Stability under docs edits | Leakage |
| --- | --- | --- | --- | --- | --- |
| judge | `samples/data/nivarachat/eval/relevance.tsv` | chunk id (`path#ordinal`) | 0/1/2 per chunk | invalidates on renumber | none (human/LLM grades) |
| auto | `samples/data/nivarachat/queries.tsv` | chunk id (`path#ordinal`) | 1 per relevant chunk | invalidates on renumber | Title/Section score ~1 (leakage diagnostic) |

Precedence: an explicit `--labels` value wins; otherwise judge is used when `relevance.tsv`
exists, else the run refuses (no silent fallback).

`--labels fixture` is a deprecated alias for `--labels judge` (both read `relevance.tsv`).

Coverage gate: `RetrievalLabels.MinimumCoverage = 0.5`. Below it the run prints a `REFUSED:`
marker and sets exit code 2.

## Baseline schema

`RetrievalBaseline` (JSON, indented, no timestamps) records: schema version, corpus identity,
label kind/description/source path, matched/total/coverage/is-complete, seed, k, candidates,
split counts, and the two fixed baselines (`fixed(default)`, `fixed(content-only)`) over both
scopes — all resolved usable label queries (wide, stable starting point) and the held-out test
split (the scope the learned comparisons use).

Default write path: `samples/data/nivarachat/eval/baseline-{kind}-s{seed}-k{k}.json`
(gitignored), overridable with `--baseline <path>`.

Full-set fixed metrics are not leakage: the fixed weights never see labels. Learned models are
scored on the test split only.

## Regenerating the artifacts

- `--regenerate queries` — builds the corpus, runs `EvalGroundTruth.DeriveFromChunks`, writes
  `queries.tsv` (sorted by query text, chunk ids within a row sorted, LF, no BOM, single trailing
  newline). Prints the count and the corpus content hash.
- `--regenerate judge` — runs `--rag-eval` grading (`--ollama --model` required) and writes
  `relevance.tsv` in canonical order with a provenance header
  `# corpus=<contentHash> chunker=<ChunkerVersion> chunkSize=<n> backend=<backend> dimension=<n>`.
  Refuses to overwrite an existing cache unless `--force` is given.
- `--regenerate all` — runs `judge` then `queries`, and writes
  `samples/data/nivarachat/eval/regeneration-manifest.json`.

Missing-artifact guidance: when `queries.tsv` or `relevance.tsv` cannot be found, the sample
prints a copy-pasteable regeneration command (and, when stdin is a TTY, offers an interactive
`Regenerate now? [y/N]` prompt). No artifact is overwritten without `--regenerate`/`--force` or
explicit confirmation.

Determinism: no timestamps, sorted rows, LF line endings, UTF-8 without BOM, single trailing
newline. Re-running on the same corpus + model must be byte-identical. Model-dependent output
(judge grades) is deterministic only for a pinned model + temperature (default 0.0 for
regeneration); the model name is recorded in the manifest.

Review each regenerated artifact as a diff and commit it separately from code changes.

## What "same corpus + seed" pins (and does not)

Pins: chunk ids, split, baseline metrics.
Does not pin: model files, backend, chunker version, chunk size, label file, seed (the seed is
an input, not a pinned output).

## Known limits

- No edit-stable escape hatch now that the fixture is retired: `relevance.tsv` is keyed by chunk
  id, so any docs edit that renumbers chunks invalidates labels.
- Stale judge cache: detected by the 50% coverage heuristic (plus the provenance header).
- Silk numerical drift and the uncommitted model (`samples/data/minilm`) make a committed
  baseline non-verifiable in CI.
- With a single label file there is no offline fallback on a fresh checkout: `--labels judge`
  refuses until `--rag-eval --ollama` has produced `relevance.tsv`.


---

# Section 7 - How 916-A / 980 supports 915

916-A turns 915 from "a demo that runs" into "a result that can be reproduced and trusted":

- **Reproducible chunk identity and corpus hash.** 915 trains and evaluates on chunk ids (`path#ordinal`). 916-A makes the corpus hash path-aware and order-independent, so two checkouts of the same corpus yield the same chunk row order, candidate indices, and tie-breaks. Without this, 915 metrics drift across machines for reasons unrelated to the model.
- **Coverage gate.** 915 already refused to train below 50% label coverage; 916-A formalises it (`RetrievalLabels.Accepts`, `REFUSED:` marker, exit code 2) and adds tests, so a stale label file fails loudly instead of silently shrinking the relevant set and lowering every ranker.
- **Structured baseline.** 915 computed `fixed(default)` / `fixed(content-only)` on the test split only and wrote them only with `--report`. 916-A adds `RetrievalBaseline` (JSON, byte-stable, no timestamps) recording schema version, corpus identity, label source, coverage, split counts and both fixed baselines over the full usable set and the held-out split. This is the "same corpus + seed -> same baseline" assertion 915 lacked.
- **Regeneration.** 916-A adds `--regenerate <queries|judge|all>` with canonical sorted output and a provenance header on `relevance.tsv`. 915 depends on the judge cache; 916-A makes it rebuildable and staleness detectable, which is prerequisite to any honest 915 comparison.
- **Split stability.** 915 used `QuerySplit` with FNV-1a; 916-A documents and tests it, and the baseline records the seed, so 915 held-out results are reproducible.

In short: 915 provides the learning-to-rank machinery; 916-A provides the reproducibility, baseline recording, and refusal discipline that make 915 results defensible. 915 should consume `RetrievalBaseline` as its reference point and print the corpus identity line (already done in 916-A) on every run.
