# NivaraChat retrieval evaluations

**Status:** plan, not yet executed. Written 2026-11-08.
**Scope:** `samples/NivaraChat` only. No `src/` changes.

This document plans the evaluation that was never run: whether the learned fusion from
issue #915 (`--rag-learn`) actually retrieves better than the fixed fusion from issue #914
(`--rag-multiview`). It also plans the machinery needed to make that answer reproducible,
because the current judge harness is not deterministic and the current ground truth is
mostly self-generated.

---

## Honest limits (read these first)

These are not caveats at the end of the report. They bound what any number produced by
this plan can mean, and they are listed first so a reader cannot reach the results without
passing them.

1. **The judge only sees `chunk.Content`.** The relevance grades describe how well a
   chunk's *body* answers the query. Title, Section and Code views can only contribute
   signal that correlates with content relevance. A learned weight vector that collapses
   toward Content is a plausible and legitimate outcome, not a bug.
2. **The query set is dominated by heading-derived queries.** Of the 852 lines in the
   committed query file, 795 are `auto: heading`, 51 are `auto: title`, and 6 are curated.
   Heading-as-query pairs have heavy lexical overlap with the chunk that contains the
   heading, so most rankers score well on them. That makes them useful for detecting
   regressions and useless for claiming an improvement. Every result is therefore reported
   per stratum, and the headline never pools the auto strata with the curated one.
3. **Determinism is provided by the cache, not by the seed.** A temperature-0, seeded LLM
   call is reproducible in practice, not guaranteed across model digests, quantizations or
   Ollama versions. What actually makes a rerun identical is the committed label file plus
   its provenance header. The seed only reduces how often a re-grade is needed.
4. **Generator and judge share a model family.** If queries are generated with the same
   model that grades them, their errors correlate: the generator produces questions the
   judge finds easy. The curated stratum exists partly to expose that correlation.
5. **Chunk ids are positional.** Labels are keyed by `{relativeSourcePath}#{ordinal}`, so
   editing a section renumbers every later chunk in that document. The current cache is
   already 24.5% stale by row count (13 documents edited after it was written). A corpus
   fingerprint in the label header is the only defence that does not depend on memory.
6. **Statistical power has a floor.** With 200 out-of-fold queries this plan can detect a
   paired effect of about `d = 0.20` at 80% power. A smaller true effect will read as "no
   significant difference", and that is a statement about the sample, not about the model.
7. **Recall is relative, not absolute.** Judge labels cover roughly 50 candidate chunks per
   query, not the full corpus of ~2,389 chunks. `--grade-all` would require on the order of
   45,000 judge calls and is out of scope. Recall@K here means "fraction of the labelled
   relevant set recovered", and the report must say so every time it prints the number.
8. **One corpus, and plan documents are not in it.** Everything is measured on this
   repository's `docs/`, minus `docs/plan/`: those are working notes and
   roadmaps, so they are not chunked, not indexed, and never generate queries. The ~2,389-chunk
   figure quoted elsewhere includes them; the run-of-record re-measures the chunk count on the
   filtered corpus. A result here does not transfer to another corpus, language or chunking
   strategy.
9. **The model is tiny.** The learned ranker is four logits (plus an optional 36- or
   1,540-parameter gate). It cannot fix a representation problem, and a null result is a
   realistic outcome that the report is designed to state plainly.
10. **Existing numbers are superseded.** The `--rag-eval` headline quoted from #945
    (`fused nDCG@5 = 0.795` vs `content-only = 0.507`) was measured on **two** curated
    queries. The blog draft under `docs/blog/rag/` claims `-0.028 nDCG, p = 0.88` on 50
    held-out queries and a chunk count that does not match the corpus (2,060 vs 2,389).
    Neither has a committed run behind it. This work replaces both, and the blog claim is
    corrected or removed in the same change that publishes the run-of-record.

---

## 1. What exists, and what was never measured

| Issue / PR | What shipped | How it was evaluated |
| --- | --- | --- |
| #912 | Offline RAG over `docs/*.md`, MiniLM CPU/Silk embeddings | Demo output |
| #914 / #941 | Multi-view retrieval: four embedding views, fixed late fusion `0.50/0.20/0.20/0.10`, `BestSpace`, divergence, agreement buckets | Demonstration on the repo's docs |
| #943 / #945 | `--rag-eval`: LLM-as-judge harness, cached labels, fused vs content-only plus leave-one-view-out ablations | Run on 2 curated queries; headline `fused 0.795` vs `content-only 0.507` |
| #915 / #978 | `--rag-learn`: trainable global fusion weights and a query-gated mixture, AutoDiff training, held-out comparison | Committed report uses the 40-query **fixture** with a **2-query** test split: `learned 1.000` vs `fixed 0.855`, `p = 1.0` |

The gap is structural, not just a missing run:

- `--rag-learn` measures a **reranking** comparison over a candidate pool. Every ranker
  reorders the same pool, so the numbers say "these weights order these candidates
  differently", not "retrieval got better".
- `--rag-eval` measures **end-to-end retrieval**, but only for hard-coded fixed-weight
  configurations. It cannot score a learned model.
- There is no weights export. A trained model exists only for the lifetime of a process and
  as prose in a markdown report.
- There is no split concept in `--rag-eval`. Evaluating the learned weights on the queries
  they trained on would be leakage.

### Where "453" came from

`relevance.tsv` is the local, gitignored judge cache written by the #943 run. It contains
**453 ordinal-unique queries** (452 case-insensitive: one pair differs only by
`Window Functions` / `window functions`) and 21,092 graded rows. The run was made against
the query file as it stood then; the committed file was extended to 852 lines / 688 unique
queries nineteen minutes after the cache was written. So 453 is not a designed sample size,
it is "how far the run got before the file grew". Nothing downstream should treat it as a
target.

---

## 2. Recon findings that shape the plan

### 2.1 The label cache is a quarter stale

Thirteen documents under `docs/` were modified after `relevance.tsv` was written
(2026-11-07 18:11):

```text
ACCELERATION.md  BFLOAT16.md  CRAZY-ROADMAP.md  DISTELBERT.md  HALF-F16.md
ILGPU.md         LAYA.md      MODERNBERT.md     QWEN.md        ROADMAP-SUGGESTION.md
SILK.md          SMOLLM.md    TENSORS.md
```

They account for 5,170 of 21,092 graded rows (24.5%), and all 453 queries are touched.
Because chunk ids are positional, a patched cache would silently mix two id generations.
The plan therefore rebuilds labels in a new file rather than editing the old one.

### 2.2 The query mix is the bigger threat

| Stratum | Lines | Discriminative value |
| --- | --- | --- |
| `auto: heading` | 795 | Low: lexical overlap with the source chunk |
| `auto: title` | 51 | Low: title view scores its own source near 1.0 |
| `curated` | 6 | The only cross-document, non-self-fulfilling set |

A significant delta on 447 easy queries can be real and still be uninteresting. The plan
adds a generated cross-document set, keeps the curated set separate, and reports three
strata rather than one mean.

### 2.3 The judge is nondeterministic by construction

- `--temperature 0.2`, `--runs 3`, majority vote.
- Candidate batches are built from a `HashSet<string>` with randomized string hashing, so
  **the same passages are batched with different neighbours in different processes**. LLM
  grades depend on batch context, so the labels move.
- `RagEvalJudge.GradeBatchAsync` catches every exception and returns all-zero grades, and
  `ParseGrades` pads missing values with `0`. A judge outage therefore becomes a corpus of
  confidently irrelevant chunks instead of a failed run.
- The cache records no model, prompt, temperature, seed or corpus version. Two generations
  of labels can be silently mixed in one file.

### 2.4 A result is already claimed without a run

`docs/blog/rag/2-differentiable-rag-from-fixed-weights-to-learned-fusion.md` states that
the gated model reports `-0.028 nDCG` with `p = 0.88` on 50 held-out queries, and lists
"452 queries, 2,060 chunks, 21,092 rows". The 21,092 rows are real; the rest is not backed
by a committed report, and 2,060 disagrees with the ~2,389 chunks reported elsewhere. The
run-of-record either reproduces those figures or the blog text is corrected.

---

## 3. Evaluation design

### 3.1 Two levels, one label set

1. **Reranking (isolates the weights).** Every ranker reorders the same weight-agnostic
   pool: the union of each view's own top-K plus every labelled positive. This is what
   `--rag-learn` already does. A difference here is attributable to the fusion weights.
2. **End-to-end retrieval (the headline).** The learned weights drive
   `MultiViewScoring.Rank` over the full corpus and produce a top-K. This is what a user
   experiences, and it is directly comparable to the #943 table.

Both levels use the same queries, the same labels, the same metric code and the same
corpus. If the pool comparison improves but retrieval does not, that is a finding, not an
inconsistency.

### 3.2 Query set: 200, in three strata

| Stratum | Target | Source |
| --- | --- | --- |
| Curated | ~30 | Existing 6 plus drafted cross-document questions, human-reviewed |
| Generated cross-document | ~120 | `--rag-queries`, human-reviewed, committed |
| Sampled `auto: heading` | ~50 | Deterministic sample from the existing file |

Generation requirements: answerable from at least two documents, must not reuse a heading
verbatim, and must carry a one-line rationale. Duplicates of existing queries are rejected.
Generated queries enter the committed file only after a human review step.

### 3.3 Statistics: 5-fold cross-validation, not a single split

At the current 80/10/10 split, 50 held-out queries require roughly 500 total queries. Cross-
validation removes that constraint: each query is held out exactly once, so `N` queries
yield `N` paired out-of-fold comparisons while the model still never sees its own test
query. Training is four parameters, so five retrains cost seconds.

| Out-of-fold n | Detectable paired effect `d` (80% power, alpha 0.05) |
| --- | --- |
| 50 | 0.40 |
| 100 | 0.30 |
| **200** | **0.20** |
| 453 | ~0.13 |

The sign test floor is six decisive queries regardless of n (6-0 gives `p = 0.031`).
Hyperparameters stay fixed at the #915 defaults (`lr 0.05`, `weight decay 0.01`, 200 epochs,
patience 20); tuning them on the evaluation set would invalidate the comparison.

### 3.4 Decision rule, fixed before the run

Call it an improvement only when all four hold on out-of-fold results:

1. mean `nDCG@10` rises;
2. the paired sign test gives `p < 0.05`;
3. at least six decisive (non-tied) queries;
4. the sign of each learned weight is stable across three seeds.

Otherwise report a null result with its power statement. A learned model that loses is a
valid outcome, and #915's own documentation already says so.

---

## 4. Work items

### P1 — Deterministic judge

Rewrite `RagEvalJudge` to call the Ollama client directly instead of through the thin
`IChatClient` wrapper:

- `ChatRequest` with `Options = new RequestOptions { Temperature = 0, Seed = <fixed> }`
  and `Format = "json"` (or a JSON schema for `{ "grades": [int, ...] }`).
- Sort candidate chunk ids ordinally before batching, and use a fixed batch size, so the
  batch composition is stable across processes.
- Retry with backoff on transport or parse failure. After the retry budget, **fail the
  run**; never persist a fabricated zero.
- Count and report calls, retries and parse failures per pass.
- Add a provenance header to the label file: schema version, model, model digest, prompt
  version, temperature, seed, runs, corpus fingerprint (sorted path plus content hash of
  every scanned markdown file), and date.
- `--offline` refuses to load a label file whose header does not match the requested
  configuration. A new generation is written to a new path, never merged into the old one.
- Default `--runs 1`; keep the flag for an explicit label-noise audit.

Tests: stable prompt and batch order, parse and padding behaviour, header round-trip,
mismatch rejection, and a malformed-response case that must not produce zeros.

### P2 — Query generation mode

New `--rag-queries` mode:

- Reads the docs directory and the existing query file.
- Proposes cross-document queries with the same deterministic request settings as P1.
- Deduplicates against existing queries and against itself.
- Writes `queries.candidate.tsv` with the proposed query, source documents and rationale.
- Never writes the committed query file directly. A human promotes candidates (Gate G1).

Tests: prompt construction, dedupe, and that the mode cannot write the committed file.

### P3 — Folds and split persistence

Add `QueryFolds`: a deterministic 5-fold assignment using the same FNV-1a hash and seed as
`QuerySplit`, persisted to `folds.tsv` and committed. The persisted assignment, not a
recomputed hash, is what the trainer and evaluator read, so a future change to the hashing
code cannot silently move the folds.

### P4 — Label refresh

Grade each query's candidate union with the deterministic judge:

```text
candidates(query) = union(
    top-20 by Content, top-20 by Title, top-20 by Section, top-20 by Code,
    top-20 by fixed fusion, top-20 by content-only,
    20 seeded random non-retrieved chunks)
```

Roughly 10,000 passages, about 1,250 judge calls at eight passages per call. Write
`relevance-v2.tsv`, keep the v1 file as history, and report row/query/per-document coverage
plus v1-to-v2 label agreement (Gate G2). After training, a small pass-B grades the chunks
the learned ranker retrieves that pass A did not cover.

### P5 — Weights export

`--rag-learn` writes `learned-weights.json` next to its markdown report:

- schema version, seed, fold assignment, label-file fingerprint;
- global weights and, when trained, gate kind and parameters;
- hyperparameters, initialisation parity, training metrics.

Tests: round-trip, schema rejection, and that the exported weights reproduce the in-process
scores exactly.

### P6 — Learned-model support in `--rag-eval`

- `--weights-file <path>` adds `learned-global` to the configuration table.
- `--fold <n|all>` selects the out-of-fold queries from `folds.tsv`.
- The learned configuration joins the judged candidate union, so its new hits are graded
  rather than silently counted as irrelevant.
- Generic paired comparisons (each challenger against `fused`), per-config counts of
  retrieved-but-unlabelled chunks, and per-stratum metric tables.
- A golden test asserts that the fixed-weight path produces bit-identical numbers to the
  current implementation.

### P7 — Run-of-record and reconciliation

Train on four folds, score the fifth, repeat; aggregate all out-of-fold per-query results.
Run three seeds. Produce the pool comparison and the end-to-end comparison, then write
`samples/NivaraChat/915-EVAL.md` with method, coverage, strata, tables, paired statistics,
learned weights, falsifiers and exact commands.

Correct the blog draft: replace its unsupported `-0.028 nDCG, p = 0.88` claim and the
2,060-chunk figure with the run-of-record, or remove the claim. Editing anything under
`docs/blog/` means running the Docs namespace tests in the same change.

---

## 5. Determinism contract

A rerun of the run-of-record is identical when all of the following are committed and
unchanged:

| Element | Stored in |
| --- | --- |
| Query text and order | `samples/data/nivarachat/eval/queries.tsv` |
| Fold assignment | `samples/data/nivarachat/eval/folds.tsv` |
| Judge labels | `samples/data/nivarachat/eval/relevance-v2.tsv` |
| Judge provenance | Header of the label file |
| Learned weights | `learned-weights.json` |
| Corpus version | Fingerprint in the label header |

The seed and temperature reduce label churn. They are not the guarantee.

---

## 6. Artifacts to commit

- `samples/data/nivarachat/eval/queries.tsv` — updated, with the three strata.
- `samples/data/nivarachat/eval/folds.tsv` — new.
- `samples/data/nivarachat/eval/relevance-v2.tsv` — new, roughly 1.5 MB.
- `samples/NivaraChat/learned-weights.json` — new (or under the eval directory).
- `samples/NivaraChat/915-EVAL.md` — the report.

`relevance.tsv` is currently gitignored because `.gitignore` excludes
`samples/data/nivarachat`. The ignore rule changes to exclude the cache and learn
directories while re-including the `eval` directory, so the labels and query files are
tracked. Committing the labels is what saves the cloud cost on every later run and what
makes the evaluation reproducible for anyone who checks out the repository.

---

## 7. Gates

| Gate | Question | Blocks |
| --- | --- | --- |
| G1 | Is the 200-query set reviewed and committed? | Any labelling spend |
| G2 | Does label coverage and v1-to-v2 agreement look sane? | Training |
| G3 | Are the out-of-fold results reviewed? | Any conclusion, README change or blog correction |

Verification for the code changes:

```text
dotnet build Nivara.slnx -c Release
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~RetrievalTrainerTests"
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~RagEval"
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"
dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore
```

The Docs filter is required because the plan touches `docs/` (the RAG corpus and the blog draft).
Plan documents themselves live under `docs/plan/`, which `DocSnippetExtractor.IsScanned` excludes
from the document, citation and snippet pins, so this plan does not move them. The one scope change
this work does make is excluding `docs/plan/` from the scan, which is why the pins move once, in
`DocsCoveragePins`, with the reason recorded there.

---

## 8. Commands

```text
# 1. generate candidate queries (deterministic, temperature 0)
dotnet run --project samples/NivaraChat -- --rag-queries --ollama --model beam:cloud

# 2. human review: promote candidates into the committed query file

# 3. grade the candidate union with the deterministic judge
dotnet run --project samples/NivaraChat -- --rag-eval --ollama --model beam:cloud --fold all

# 4. train and export weights, per fold
dotnet run --project samples/NivaraChat -- --rag-learn --labels judge --fold 0

# 5. score the learned model offline, per fold
dotnet run --project samples/NivaraChat -- --rag-eval --offline --weights-file <path> --fold 0
```

The exact flags are settled during P1-P6; the sequence is not.

---

## 9. Risks and follow-ups

- **Re-grade cost.** About 1,250 cloud calls for 200 queries. The run is resumable through
  the label cache, and the result is committed so it is paid once.
- **Label noise.** A `--runs 3` audit at the original temperature can measure how much the
  judge disagrees with itself. That number belongs in the report as a ceiling on any
  measurable gain; it is not the headline.
- **Corpus drift during the work.** Any edit to a labelled document invalidates that
  document's chunk ids. The fingerprint gate turns this into a refusal rather than a
  silent regression.
- **Auto strata.** The sampled heading queries stay in the report as a regression check.
  They must never be pooled into the headline.
- **Out of scope.** ANN indexing, metric learning, a learned similarity kernel,
  Spearman/Kendall kernels, PDF corpora, and Silk training parity. Retrieval stays an exact
  scan plus top-K.

---

## 10. Acceptance criteria

- A committed 200-query set with curated, generated and sampled strata, plus committed
  folds.
- A committed judge label file with a provenance header and a passing offline replay.
- Learned weights exported to JSON and scored end-to-end without retraining.
- Out-of-fold `nDCG@10`, `P@10`, `R@10`, `MRR`, win/loss/tie, sign test and Cohen's d for
  `learned-global` against `fixed(default)`, per stratum.
- The pool comparison reported alongside the retrieval comparison, with the difference
  between the two stated.
- A written verdict that follows the decision rule, including a null result if that is what
  the numbers say.
- The blog draft's unsupported figures corrected or removed.
- `dotnet build`, the targeted test filters, the Docs filter and the whitespace gate green.
