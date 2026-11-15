# Post 3 — When the ranker learns to say "no"

> Series: *Better RAG without the math panic: a software engineer's field guide*
> Target: software engineers (0–10 yrs), math-averse. · Suggested title:
> **"Your RAG learned the weights. Now teach it to refuse."**
>
> What you'll learn: a learned fusion weight is still only a sorting rule. The next step is a
> tiny typed decision layer that accepts, falls back, or abstains per chunk, with a
> calibrated confidence; and the reproducibility discipline that makes any of those numbers
> survive a rerun. By the end, you can run `--rag-adaptive-eval`, read the before/after
> calibration table, and know which rows moved and which refused to.

## Hook (use as-is or rewrite)

> Your `--rag-learn` run finished. The learned weights moved from `(0.5, 0.2, 0.2, 0.1)` to
> something that looks like progress, and the paired sign test on 50 held-out queries printed
> "not significant." You stare at the table and feel cheated: the model did learn, the
> metric did not move. The missing piece is not a bigger model — it is a question nobody
> asked the ranker yet: *should this chunk be used at all?* Sorting is not deciding. A chunk
> can be top-1 and still be the wrong kind of evidence, and no fusion weight can express
> "abstain" because every weight is strictly positive by construction.

Post 2 ended on init parity (`max |learned - fixed| ≈ 1e-7`). Post 3 keeps that number as
the floor and adds a second one above it: a calibrated confidence that a chunk deserves to
be in the answer at all.

---

## 1. The gap between ranking and answering

Post 1 turned one embedding per chunk into four views fused late with a mask. Post 2 turned
the hand-picked weights into parameters and trained them with AutoDiff. Both posts stop at
the same place: a sorted list. The list is better sorted, but it is still a list, and a
list has no opinion about its own top row.

Three things a sorted list cannot say:

1. **"This chunk is relevant but thin."** A 40-word stub that happens to match the query
   can outrank a 400-word section that answers it. The score carries no length or coverage
   signal.
2. **"Two views disagree and I don't know which to trust."** Post 1 already computes that
   disagreement; Post 2's fusion hides it inside one number again.
3. **"Nothing here is good enough."** A retrieval system with no abstain path always
   answers, and an always-answering system is the one that hallucinates.

The fix is not another view. It is a second, smaller model that sits *after* fusion and
reads the per-chunk evidence plus a few cheap features, then emits a typed decision:
`accept`, `fallback`, or `abstain`, with a confidence. That is what `--rag-adaptive` adds.

### 1.1 What "adaptive" means here

- **Retrieve** multi-view top-K per query, exactly as before. Candidate generation stays
  outside the gradient path, so the comparison stays a reranking comparison.
- **Understand** each candidate with the frozen MiniLM views plus a small set of
  ModernBERT features.
- **Decide** with a Laya choice head: accept / fallback / abstain, with a calibrated
  confidence per chunk.
- **Learn** online on the train split, using simulated feedback derived from judge labels.
- **Persist** the head (`laya_head.json`) unless `--no-persist`.
- **Evaluate** on the held-out split and report decision accuracy and calibration before
  vs after.

The head is small on purpose. It is a 3-way classifier over frozen features, not a new
encoder, and the whole point is that it can be retrained on a laptop in seconds.

---

## 2. The decision layer in one paragraph

`--rag-adaptive` takes the same candidate pool as `--rag-multiview` and, for each chunk,
builds a feature vector from the four view scores (mean and max per view, missing views
excluded from the mean) plus the fused score and the disagreement band. The Laya head maps
that to three logits. `softmax` over the three gives the decision probabilities; the
argmax is the decision; the max probability is `AnswerConfidence`. Nothing in this path
touches the encoder or the fusion weights — the head is a pure function of already-computed
retrieval evidence, which is why it can be swapped, retrained, or deleted without
invalidating anything in Posts 1 or 2.

### 2.1 Why three decisions and not a threshold

A threshold on the fused score (`if score < 0.3: skip`) is the obvious first idea and it
fails in two ways. It cannot distinguish "low score because the chunk is irrelevant" from
"low score because only the Code view matched and Code is sparse", and it has no calibrated
probability to report to the caller. A 3-way head keeps both failure modes visible: a chunk
with a low fused score but a high Code score lands in `fallback` (use it, but ask the LLM
to be careful), and a chunk with uniformly low scores lands in `abstain` (do not use it at
all). The confidence is what downstream code keys on, not the label.

### 2.2 Calibration, stated plainly

A confidence of 0.8 that is right 80% of the time is calibrated. A confidence of 0.8 that
is right 55% of the time is not — it is just a number that feels good. `--rag-adaptive-eval`
reports two calibration numbers before and after training:

- **ECE** (expected calibration error) over 10 equal-width bins of `AnswerConfidence`
  against choice correctness.
- **Brier** score on the same.

The useful reading is the delta, not the absolute. A head that starts overconfident and
ends less overconfident is a real improvement even if its accuracy barely moves, because
the downstream code can finally trust the number it is given.

---

## 3. The end-to-end loop (`--rag-adaptive-eval`)

This is the mode that actually runs the whole thing, and it is the one to point a colleague
at when they ask "does any of this work?"

```
dotnet run --project samples/NivaraChat -- --rag-adaptive-eval --labels judge
dotnet run --project samples/NivaraChat -- --rag-adaptive-eval --labels judge --silk
dotnet run --project samples/NivaraChat -- --rag-adaptive-eval --labels judge --no-persist --max-queries 5
```

What it does, in order:

1. Build (or load) the multi-view index.
2. Retrieve multi-view top-K per query.
3. Extract features (MiniLM views + frozen ModernBERT features).
4. Decide with the Laya head.
5. Learn online on the train split, using simulated feedback from judge labels.
6. Persist the head (`laya_head.json`) unless `--no-persist`.
7. Evaluate on the held-out split.
8. Write the report and metrics.

Two things to notice in the output:

- **`adaptive-rerank` preserves top-K membership**, so its Precision@K and Recall@K are
  invariant by construction. Only `adaptive-filter` can move them. If a table shows a
  Precision change under `adaptive-rerank`, that is a bug, not a result.
- **Feedback is simulated from judge labels and is stated as such.** The loop is honest
  about this; do not read the decision accuracy as if a human had clicked "use this chunk."

### 3.1 Candidate set and scoring

The candidate set is the index's top-K per query; the comparison is a within-run reranking
comparison, not a retrieval comparison. That distinction matters for the same reason it
mattered in Post 2: if every ranker reorders the same pool, a win is a sorting win, and
sorting wins are cheaper and less interesting than retrieval wins. Post 3 keeps the same
honest framing.

### 3.2 Timings, split on purpose

The report splits time into three buckets:

- **Feature extraction** (the encoder) — the expensive part, shared by every configuration.
- **Head updates** (AutoDiff) — the cheap part, which is the whole point of keeping the
  head small.
- **Persistence** — writing `laya_head.json`, which is about 147 MB; use `--no-persist`
  or `--head-weights <temp>` for read-only runs.

Index build times are recorded for both CPU and Silk when `--silk` is given. Silk scope here
is deliberately narrow: MiniLM embedding generation and parity only. Decisions and head
training are CPU in both backends. Hand-written Silk weight steps are a follow-up, not a
shipped feature, and the report says so.

---

## 4. Reproducibility: the thing that makes any of this believable (#916-A)

Posts 1 and 2 both quoted numbers. Post 3 is where the numbers either survive a rerun or
do not. The #916-A work adds the discipline that was missing:

### 4.1 Chunk identity

`ChunkId = "{relativePath}#{ordinal}"` is positional on purpose. Judge labels key on
`path#ordinal`, so any docs edit that renumbers chunks invalidates labels, and the coverage
gate catches it. There is no path-keyed escape hatch.

### 4.2 Corpus identity

`CorpusIdentity` captures content hash, file count, chunk count, chunker version, chunk
size, docs dir, backend and dimension. It is printed after the index resolves on both the
cache-hit and rebuild paths. Two runs on the same corpus print the same line; two runs on
different corpora do not pretend to agree.

### 4.3 Split stability

The 80/10/10 split uses FNV-1a over the UTF-8 bytes of the query string with the seed
mixed in, because `string.GetHashCode()` is randomized per process. The same query plus the
same seed lands in the same split on every machine.

### 4.4 Coverage gate

Training refuses when fewer than 50% of label rows resolve against the current corpus
(`RetrievalLabels.MinimumCoverage = 0.5`, `Accepts`). A stale label file shrinks the
relevant set and lowers every ranker silently; the gate makes it loud with a `REFUSED:`
marker and exit code 2. This is the single most important line in the whole pipeline,
because without it every other number is uninterpretable.

### 4.5 Baseline artifact

`RetrievalBaseline` (JSON, indented, no timestamps) records schema version, corpus
identity, label kind/description/source path, matched/total/coverage/is-complete, seed, k,
candidates, split counts, and the two fixed baselines (`fixed(default)`,
`fixed(content-only)`) over both scopes — all resolved usable label queries and the
held-out test split. Default path:
`samples/data/nivarachat/eval/baseline-{kind}-s{seed}-k{k}.json` (gitignored),
overridable with `--baseline <path>`.

Full-set fixed metrics are not leakage: the fixed weights never see labels. Learned models
are scored on the test split only.

### 4.6 Regeneration

`--regenerate <queries|judge|all>` rebuilds label artifacts deterministically (sorted rows,
LF, no BOM, single trailing newline), with a provenance header on `relevance.tsv` and a
manifest for `all`. It refuses to overwrite without `--force`. Missing artifacts print a
copy-pasteable regeneration command, and offer an interactive prompt when stdin is a TTY.

---

## 5. How to read a result without lying to yourself

- **`adaptive-rerank` cannot move P@K/R@K.** If it does, that is a bug, not a win.
- **A non-significant result is a valid result.** With a handful of held-out queries the
  sign test cannot discriminate either way, and saying so is the honest output.
- **Auto labels (`--labels auto`) are a declared leakage diagnostic.** Title and Section
  score their own positive near 1.0. Never quote them as a headline.
- **Calibration deltas beat accuracy deltas.** A head that becomes less overconfident is
  useful even if its accuracy is flat, because downstream code can finally trust the
  number.
- **Silk is embedding + parity only.** Decisions and head training are CPU in both
  backends. Do not read the timings as a GPU win for the head.

### 5.1 Commands

```
dotnet run --project samples/NivaraChat -- --rag-adaptive --text "What is Nivara?"
dotnet run --project samples/NivaraChat -- --rag-adaptive-eval --labels judge
dotnet run --project samples/NivaraChat -- --rag-adaptive-eval --labels judge --silk
dotnet run --project samples/NivaraChat -- --rag-adaptive-eval --labels judge --no-persist --max-queries 5
```

Key flags: `--labels`, `--seed`, `--k`, `--max-queries`, `--max-examples`, `--head-epochs`,
`--head-lr`, `--update-batch`, `--resume`, `--head-weights`, `--no-persist`, `--report`,
`--silk`, `--docs-dir`, `--threshold`.

Reports and metrics land in `samples/data/nivarachat/adaptive/` (gitignored):
`rag-adaptive-report.md`, `metrics-{kind}-s{seed}-k{k}.json`. `laya_head.json` is about
147 MB; use `--no-persist` or `--head-weights <temp>` for read-only runs.

---

## End hook → close of series

Post 1 was views and masking. Post 2 was learning the fusion. Post 3 is deciding whether
to trust the result, and pinning everything so the decision can be re-checked. If you want
to remember one number from the whole notebook, it is still the init parity:
`max |learned - fixed| ≈ 1e-7`. Everything after it is measured against that floor.

## Facts & numbers to reuse (checklist)

- Adaptive decision: accept / fallback / abstain, with calibrated `AnswerConfidence`.
- Calibration: ECE (10 equal-width bins) and Brier, before and after.
- `adaptive-rerank` preserves top-K membership by construction.
- Feedback is simulated from judge labels and is stated as such.
- Silk scope: MiniLM embedding + parity only; decisions and head training are CPU.
- `laya_head.json` is about 147 MB.
- Reports/metrics: `samples/data/nivarachat/adaptive/`.
- Coverage gate: 50% (`RetrievalLabels.MinimumCoverage = 0.5`), `REFUSED:` + exit code 2.
- Split: FNV-1a over UTF-8 bytes with seed mixed in.
- Baseline: `samples/data/nivarachat/eval/baseline-{kind}-s{seed}-k{k}.json`.

### Source references

- `samples/NivaraChat/Modes/RagAdaptiveMode.cs`
- `samples/NivaraChat/Modes/RagAdaptiveEvalMode.cs`
- `samples/NivaraChat/Helpers/RetrievalEvaluation.cs::RetrievalEvaluation.Compare()`
- `samples/NivaraChat/Helpers/MultiViewIndex.cs::MultiViewIndex.BuildAsync()`
- `samples/NivaraChat/Helpers/MultiViewScoring.cs::MultiViewScoring.Fuse()`
- `samples/NivaraChat/Helpers/RetrievalTrainer.cs::RetrievalTrainer.Train()`
- `samples/NivaraChat/Helpers/RetrievalLabels.cs::RetrievalLabels.Accepts`
- `samples/NivaraChat/Helpers/RetrievalLabels.cs::RetrievalLabels.MinimumCoverage`
- `docs/RAG.md` — Sections 10 (adaptive decision loop) and 7 (reproducibility)
- `docs/EMBEDDINGS.md` — "Multiple views of the same corpus"

### Visual ideas

- A three-column table per chunk: fused score, decision, confidence, before vs after.
- A reliability diagram (predicted confidence vs observed accuracy) for the head, two curves.
- A timeline of the adaptive loop: retrieve → decide → learn → persist → evaluate → report.
- One screenshot of the `REFUSED:` coverage-gate message.