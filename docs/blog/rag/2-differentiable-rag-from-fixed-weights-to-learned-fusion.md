# Post 2 — Can we learn the weights? A diff that's smaller than you expect

> Series: *Better RAG without the math panic: a software engineer's field guide*
> Target: software engineers (0–10 yrs), math-averse but now retention-motivated.
> Suggested title: **"Learning the fusion: how to move 0.50/0.20/0.20/0.10
>   without changing the fusion formula"**
>
> What you'll learn: how the late-fusion weights become four trainable logits, how a tiny
> gate over a query becomes a mixture of experts, how labels (three sources, one of them
> self-fulfilling) and a weight-agnostic candidate pool make the comparison honest, and
> the one number (`max |learned - fixed| ≈ 1e-7`) that keeps the experiment clean.

## Hook (use as-is or rewrite)

> Fixed weights look like a confession: 0.5, 0.2, 0.2, 0.1. They describe your
> *opinion*, and like all opinions they will survive without evidence until a reading of
> `--rag-eval` un-blinds you. In fact, your RAG is already doing the hard algebra; the
> only new piece is giving the four numbers a gradient.

---

## 1. The representation that keeps the comparator solvable

Nothing worth building here for-the-multitasks happens until we settle on a single formula
for fusion. We use masked late fusion:

~~~
w     = softmax(theta)             # theta ∈ R^4
fused = ((S ⊙ M) @ w)/(M @ w)      # S: cosine scores, [rows, 4], M: presence mask
~~~

- `w` is strictly positive (softmax) and sums to 1.
- A missing view (mask 0) is excluded from *both* the numerator and the denominator, so a
  chunk without code isn't penalized for the missing symbol — but a row with **no present**
  view would have a zero divisor, so we simply drop it and count it rather than explode.

**The math, gently.** `softmax(theta)` plays the same game as the gated functions in
advertising (sigmoid without the factor 2). It keeps the parameter space safe
(non-negative, unit sum) so the optimizer can move freely without feasibility checks.

## 1.1 The one property that lets Post 2 exist: parity

The default weights sum to exactly 1.0, which is a weird-sounding coincidence but is the
entire engine room of the experiment:

~~~
theta = [ln 0.5, ln 0.2, ln 0.2, ln 0.1]
softmax(theta) = (0.5, 0.2, 0.2, 0.1)   # exact, because of the constraint
~~~

So a model whose step-0 logits are these four values is numerically identical to the
`--rag-multiview` trace. We'll name it `learned` starting at `step 0`, which means
every difference reported is *a change from what you can already compute*, not a
difference in framework. A gate with `V = 0`, `b = ln(defaults)` starts invariant to its
query input; the same principle applies and we use it.

The mode prints `max |learned - fixed|` (target `≈ 1e-7`) and **refuses to credit a delta
as improvement if these aren't that close**. That's what makes this reproducible and
diff-against-optimization research rather than flea-market tuning.

### 1.2 The gate: a 4-scorer mixture of experts

The cosine measures (S) are the experts. The gate produces their weights per query. The
default gate is mean and max over the candidate pool **per view**, all other features
absent — 8 numbers → 4 logits, 36 parameters (small enough to survive ~100 queries). The
`--gate embed` alternative replaces that with a `Linear<384, 4>` (1,540 parameters)
over the raw MiniLM embedding of the query; `--gate none` skips the gated model.

- **Why mean + max per view?** It's the minimum viable "how attentive is each expert to
  this query" signal. It routes the responder based on the *catalog* of views that wake up,
  not on a deep semantic projection.
- **Why both numbers together?** The soft sub-signal (mean, plus mean+variance-style min)
  gives a 0.5/0.5 split that keeps the gate aware of whether a view is floor-level.

We could write thousands of parameters and no one would notice the difference within 30
queries — the experiment anyway includes this as the first line of evidence.

### 1.3 Loss, gather-free by design

Listwise softmax cross-entropy against a `2^grade` target, normalized so grade 2 outweighs
1 outweighs 0. One query = one graph:

~~~
p_i        = 2^(g_i) / Σ_c 2^(g_c)      # target, constant per query
loss_query = -Σ_i p_i * logsoftmax(fused)_i
~~~

- `_kwd prime candidate` has no row-gather op in Nivara's `ReverseGradOperations` (no
  `Transpose`-skip-the-point); the implementation multiplies the per-query target
  vector with `LogSoftmax(dim=0)` and sums. So this is the gather-free path that gives
  the same math without a dense per-candidate tiling.
- `--loss pairwise` selects the `+/-1` selector-matrix form (`BCEWithLogitsLoss`) — the
  other gather-free path. Same model, same pool, same error bar. Pairwise without
  this has a third path instead.

---

## 2. Data that does not lie

The category of risks in learning-to-rank is not loss functions; those are solved. **It's
ground truth** — and this post asks the most boring four questions you can about it,
in order.

### 2.1 Where grades come from

- **Judge (default when the cache exists).** `samples/data/nivarachat/eval/relevance.tsv`
  produced by `--rag-eval`, graded U/1/2 per chunk with multiple stochastic runs; this is
  the headline line.
- **Fixture (offline fallback, committed).** `relevance-fixture.tsv`, hand-authored per
  document (not per `path#chunk`).
- **Auto (opt-in, `--labels auto`).** Heading-derived pseudo-queries from `queries.tsv`,
  printed with a leakage-warning banner; a metric on auto may measure the label generator,
  so this is never a baseline of its own.

### 2.2 Why the fixture keys by document path, not chunk id

The judge-cached filenames are `docs/X.md#N` (the `#N` is the chunk ordinal). Three weeks
after writing it, somebody edits a section of `docs/SILK.md` and the judge cache has N
off by 1 — because a deleted section shifted ordinals. We've seen it kill benchmark
run-offs in a row. In the committed fixture, grades are keyed by `(query, document path)`.
The reader/importer expands `docs/X.md → 2` into one grade for *every* chunk of that
document, despite the changes in the codebase. The integration step reported how many
rows were matched (match count), and matches 100% 88/88 labeled rows against 2,389 chunks.

### 2.3 "Coverage: 88/88" (or it refuses to train)

The mode refuses to train with fewer than **50% of labelled rows that resolve against the
document set**, because a leaky ground-truth file makes *every* ranker's recall look bad at
once — the experiment you thought was "winning ranking" was measuring "reduced relevance
set." That check costs one set-subtraction per run; own it.

### 2.4 Why we do not generate candidates from the baseline top-K

The candidate pool is "what does the model get to reorder" — same pool for all rankers,
otherwise you trade one failure for another:

~~~
candidate pool per query = (∪_view top-K by that view's own score) ∪ {all positives}
~~~

Generating the pool *from the fixed fusion's own top-K* would let the baseline choose its
competitors, and each reranker runs over the 3rd-party pick. This is the reason the
operation is expressed in `CandidatePool.Select` (union per view + positives, sorted row
order) rather than "whatever was top-K."

Rows with no scored view get counted and dropped — they cannot be a legal fusion row and
they can't produce a gradient.

### 2.5 Labels, Target

The "target" for one query question is `2^grade` over rows (grade: 0 / 1 / 2), normalized.

~~~
p_i  =  2^(g_i) / Σ_c 2^(g_c)
~~~

Grade 2 → 4 units vs grade 1 (2) vs grade 0 (1), so a grader commits less when the chunk is
merely relevant. For queries all of whose rows are 0, we still emit an uniform target so
the run doesn't crash, but `IsUsable` is false for them (no positive, no signal to learn).

### 2.6 How deterministic the split is

We chose 10% / 10% / train by the 80% split, but `string.GetHashCode()` is randomized per
process — the same line of code will make two processes produce *different* relevance-
thread batches, and the nDCG tables can't be compared across runs. So `QuerySplit.Create`
hashes queries with `FNV-1a` over UTF-8 bytes with the seed mixed in: same seed, same
queries, same partition, always. **A split that moves every relabel releases a news-cycle's
worth of noise.**

---

## 3. The learning loop

Summed up:

- One epoch, one query, one step. 341 usable queries → 341 Adam steps per epoch for 200
  epochs.
- Forward (masked fusion) in AutoDiff.
- Backward via the loss.
- Adam (0.05, wd 0.01) step and zero-grad.
- Validation nDCG per epoch, patience 20; restore best checkpoint using the checkpoint
  utilities in the production ActriveCore.

Yes, it is *ridiculously small* by modern standards — 4 logits and 36-parameter gate;
listwise and ce-loss, no back propagation through blocks — and that is what made the
numerical parity story trustworthy. With our 1,540 parameter gated model (which the dump
report not-so-kindly shows as "represent active weights"), validation nDCG 14.3–17.0
percent points decay against the 36-parameter gate, which is how the *fixed vs learned*
difference is visible. We could have simply used a single vector similarity in a `python`
notebook and tested it end to end, but now we do 4+ at once.

### 3.1 How early stopping and checkpoint + restoration actually works

- Checkpoint *after* an epoch's validation observation, not after a training step. Evaluation
  on validation data tells you how well the network generalizes, not how well it fit; the
  training loss is secondary.
- Checkpoint selection uses validation nDCG@10, `na` only `EOF` patiences (hard-coded).
- At the end of training we `LoadStateDict()` with `strict=true`, which returns the best model
  back to memory.

---

## 4. What training looks like on a single query (an anatomy)

1. `EmbedQuery` turns queries into 384 floats.
2. `MultiViewScoring.ScoreAll` gives `[rows, 4]` both scores and a mask.
3. `CandidatePool.Select` builds candidates = ~10 per view (best rankings) plus every positive.
4. `ForwardMasked`: scores block, fusion weights (softmax `theta` / softmax(gate output)),
   masked fusion.
5. Listwise: `logsoftmax(fused, dim=0)`, elementwise into target, sum. Pairwise: `+/-1`
   matrix, then `MatMul(BCEWithLogitsLoss)`.
6. Backprop, Adam step, zero grad.

The report of a single query's validation set has a exact relation: a 34-label query
training split needs to have a validation split with at least 6 queries in order to achieve
p < 0.05 on its sign test (it takes 6 wins/loss). We run that count louder in text only
when it happens, against 40 queries → 2 test queries → no significance claim possible.

### 4.1 A note on gate performance (why gate embed is documented loss but kept)

With the CPU-side gate, 1,540 parameters and a 384-d input do not move a 10-query test set
in 200 epochs — the mode reports a negative 0.028 nDCG with p=0.88 (not significant) on 50
held-out queries. That is the *true scientific outcome*, not a regression, and the report
says "Acceptance is a fair comparison, not a guarantee."

We left it in because the lesson was more valuable than any noise-free scoreboard.

---

## 5. The honesty tricks (accumulated)

- Parity at step 0 (`theta=ln(default weights)` where defaults sum to 1).
- Gate invariance (V=0, gate biases logits at init).
- Weight-agnostic candidate pool.
- Refuse to train below 50% coverage.
- Validation nDCG checkpoint only; test split touched once.
- Honest significance (6 decisive queries needed; say so when the held-out split is smaller).
- Report states fixed-vs-learned, not "improvement" language.
- `--auto` as diagnostic only (so the run is useful without ever lying).
- Reports are under `gitignored` locations; nothing pollutes commits.
- Three label sources, distinguished in all outputs (`--labels` default).

This is the moment RAG ceases being a demo and starts becoming measurable engineering.

### Checkpoint: the retrieval map

~~~
queries + chunks ──→ 4 view scores ──→ candidate pool (top-K union + positives)
         │
         ├── global fusion  (4 logits)  ──→ eval (MRR, Recall@10, nDCG@10)
         └── query gate (36 or 1,540 params) ──→ paired comparison (win/loss, sign test)
~~~

- **Added:** the late-fusion weights travel from the file into a variable, with the same
  initial number printed before anything moves.
- **Where you are:** step 0 matches the paper's current #914-frames. Training only adds.
- **Still unlit:** same experiments with different labeling buttons and richer data.

---

## End hook → Post 3?

There is no Post 3 - the thesis got its references from issues #914 + #915. Measurable
retrieval engineering is the start of more, not the end.

## Facts & numbers to reuse (checklist)

- Init parity target: `max |learned - fixed| ≈ 1e-7`.
- Gradient path cost: 341 queries × ~60 candidates ≈ 2,000 training manual steps per epoch
  user later; one full CPU 200-epoch run: ~2 minutes on MiniLM already encached.
- Candidate pool default: 10 rows per view.
- Judge ground truth: 452 queries, 2,060 chunks, 21,092 rows, 100% match per fixture set.
- Fixture: 40 example queries, ~88 rows matched 100% coverage.
- Sign test declares at 6 non-tied queries; fixture is 2 and does not claim anything.
- CPU Silk (embedding) and learning (CPU trainer) footprints still the same 3.38GB-ish
  in-memory working memory after conversion; gate 1540 param's total memory is nominal.

### Source references

- `samples/NivaraChat/Helpers/MultiViewScoring.cs::MultiViewScoring.Fuse(ViewScores,FusionWeights)`
- `samples/NivaraChat/Helpers/RetrievalTrainer.cs::RetrievalTrainer.Train()`
- `samples/NivaraChat/Helpers/RetrievalTrainer.cs::FusionModel`
- `samples/NivaraChat/Helpers/RetrievalTrainer.cs::GatedFusionModel`
- `samples/NivaraChat/Helpers/RetrievalTrainingData.cs::CandidatePool.Select()`
- `samples/NivaraChat/Helpers/RetrievalTrainingData.cs::QuerySplit.Create()`
- `samples/NivaraChat/Helpers/RetrievalEvaluation.cs::RetrievalEvaluation.Compare()`
- `samples/NivaraChat/Helpers/MultiViewIndex.cs::MultiViewIndex.BuildAsync()`
- `docs/EMBEDDINGS.md` (multiple views, fusion, mask, code citation)
- `samples/NivaraChat/relevance-fixture.tsv`

### Visual ideas

- Single-shot image of a fusion table: row, three candidate scores, gate logits, softmax
  weights, fused score, rank band (Near / Distinct / Weak).
- A smaller line chart of training loss across epoch on a single query, cloned 341 times.
- A Markov-class product: how 21,092 labeled rows collapse into 451 usable per query.
