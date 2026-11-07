# Plan — #915 Differentiable Retrieval & Learning-to-Rank (`--rag-learn`)

Issue: https://github.com/khurram-uworx/Nivara/issues/915

## Problem

#914 landed four independently scored embedding views (Content, Title, Section, Code) with a
hand-picked late-fusion weight vector `(0.5, 0.2, 0.2, 0.1)`. Nothing in the sample learns those
weights, so the flagship claim — "Nivara is a differentiable DataFrame: data → query → numerical
computation → loss → gradients → optimization" — is demonstrated only by separate samples
(AutoDiff, MicroGpt) that never touch the retrieval stack.

#915 closes that gap with a new `samples/NivaraChat` mode that learns the fusion weights and a
query-gated mixture-of-experts reranker **over the #914 multi-view dataset**, on CPU, with the
comparison against the fixed baseline made on held-out queries.

Scope is sample-only: no `src/Nivara` changes. Candidate generation stays outside the gradient
path; the trainable part only reranks retrieved candidates. Non-goals: metric learning, learned
similarity kernel, ANN index, PDF extraction, Spearman/Kendall.

## Design

Per query, candidate pool of `C` rows, `S ∈ R^(C×4)` cosine scores (0 where the view is absent),
`M ∈ {0,1}^(C×4)` presence mask.

```
w     = softmax(theta)                 # theta ∈ R^4, strictly positive, sums to 1
fused = ((S ⊙ M) @ w) / (M @ w)        # [C,1] / [C,1]
```

This is exactly what `MultiViewScoring.Fuse` computes per row (numerator = Σ w_j·s_j over present
views, denominator = Σ w_j over present views). The denominator cannot be zero: softmax weights are
strictly positive and every pooled row has `PresentCount >= 1`, so `ReverseGradOperations.Divide`'s
zero-divisor pre-check cannot fire.

**Parity property (load-bearing).** `FusionWeights.Default` sums to exactly 1.0, so initialising
`theta = ln(0.5, 0.2, 0.2, 0.1)` makes `softmax(theta)` return the defaults exactly and the learned
model is numerically identical to the #914 baseline at step 0. The gate gets the same treatment:
`V = 0`, `b = ln(defaults)`, so it is invariant to its input at initialisation. Every later delta is
therefore attributable to learning rather than to a different fusion formula, and both properties
are asserted by tests.

**Gate (query-gated MoE).** The four cosine scorers are the experts; the gate produces their weights
per query. Default feature vector is `z = [mean_j, max_j]` for each view over the candidate pool,
missing views excluded from the mean (8 dims → 4 logits, 36 params). `--gate embed` swaps in
`Linear<float>(384, 4)` over the raw MiniLM query embedding (1540 params) for comparison; the
36-parameter default is the one expected to survive ~100 training queries.

**Loss.** Listwise softmax cross-entropy over the candidate pool:

```
p_i    = 2^grade_i / Σ_c 2^grade_c           # constant per query
loss_q = -Σ_i p_i · logsoftmax(fused)_i      # LogSoftmax(dim 0), Multiply, Sum, Subtract
```

`--loss pairwise` selects the gather-free `A @ fused` variant (constant ±1 matrix, targets = 1,
`BCEWithLogitsLoss`), since `ReverseGradOperations` has no row-gather op and neither path needs one.

**Candidate generation is weight-agnostic.** Pool = `∪_j top-K_j` by each view's own score, plus
every labeled positive. Generating candidates from the fixed baseline's own top-K would let the
baseline pick its own competitors.

### Label sources and leakage

- `samples/NivaraChat/queries.tsv` (committed) is 100% auto-derived (51 `auto: title`, 449
  `auto: heading`, zero graded). A heading-derived query makes Title/Section score ≈ 1.0 for its own
  positive, so training and evaluating on auto labels lets the model "win" by relearning the label
  generator. Auto labels are therefore never a headline: only an explicit `--labels auto` and an
  explicit leakage warning in the report.
- `samples/data/nivarachat/eval/relevance.tsv` (local, gitignored) is real LLM-judge data: 452
  queries, 2060 chunks, grades 0/1/2. Default source.
- New committed `samples/NivaraChat/relevance-fixture.tsv`, keyed by **document path** rather than
  `docs/X.md#N` chunk id, so it cannot rot when docs are edited (this is exactly how the judge cache
  already went stale after the 11/7 `docs/SILK.md` change). Expanded to chunk grades at load time by
  matching `MultiViewChunk.SourcePath`. Fallback when the judge cache is absent.
- The mode prints `matched/total` label coverage and refuses to train when positive coverage falls
  below threshold, instead of silently scoring a shrunken relevant set.

### Query split

80/10/10 train/validation/test by stable hash of the query string with a seed, so the split is
reproducible without RNG state and identical across processes. Checkpoint selection is validation
NDCG@10 with patience; the test split is touched once, at the end.

## Blast radius

| Change | Affected | Covered by |
| --- | --- | --- |
| `MultiViewScoring.ScoreAll` extraction + `Fuse` made public | `Rank` calls both; no behaviour change | existing `MultiViewScoringTests` (17 KB fixture) must pass untouched |
| `MultiViewIndex`: expose `Matrices`, `Metadata`, throwing `EmbedQuery` | new members only; `Retrieve` unchanged | `MultiViewIndexCacheTests`, `RagSearchTests` |
| `RetrievalTrainingData` / `RetrievalTrainer` / `RetrievalEvaluation` | new files, sample-local | new `RetrievalTrainerTests` |
| `RagLearnMode` + `ModeContext` + `Program` flags | new mode, new optional record parameters with defaults | `InferenceCliArgumentTests` untouched (separate sample); new mode has no CLI test yet — tracked below |
| `docs/EMBEDDINGS.md` new subsection | citation pins | `Nivara.Tests.Docs` namespace |
| `DocsCoveragePins` citation counts | docs gate | `DocsCoverageTests` |

Downstream of `MultiViewScoring`: `RagMultiViewMode`, `RagEvalMode`, `MultiViewExplorer`. None
consume the private helpers directly, so the refactor is internal to the file.

`DocSnippet.MarkdownFiles()` scans only repo-root `*.md` plus `docs/**/*.md`, so
`samples/NivaraChat/README.md` and `relevance-fixture.tsv` move no documentation pin. Only
`docs/EMBEDDINGS.md` citations move `ExpectedCitationCount` / `ExpectedCodeCitationCount`.

## Planned commits

1. `docs: plan #915 differentiable retrieval in TODO.md`
2. `refactor(NivaraChat): expose per-view scoring and fusion from MultiViewScoring`
3. `feat(NivaraChat): expose multi-view matrices and query embeddings on the index`
4. `feat(NivaraChat): add retrieval training data, candidate pool and query split`
5. `feat(NivaraChat): add the fusion and gated-fusion trainers`
6. `feat(NivaraChat): add retrieval evaluation and paired comparison`
7. `feat(NivaraChat): add the --rag-learn mode`
8. `test(NivaraChat): cover fusion parity, gradients and candidate selection`
9. `docs: document learning the fusion weights`
10. `test(docs): bump citation pins for the fusion-weights section`
11. `docs: remove TODO.md — plan executed`

## Verification

- `dotnet build Nivara.slnx`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~RetrievalTrainerTests"`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~MultiViewScoringTests"`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- No benchmarks: this is a sample, and no timing assertion is added.

## GitHub issues log

- [ ] Pending discovery — `--rag-learn` has no CLI argument test; decide whether to extend
  `InferenceCliArgumentTests`-style coverage for `NivaraChat` or track it separately.
