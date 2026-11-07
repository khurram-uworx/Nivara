# Post 1 — Everyone's first RAG passes its demo

> Series: *Better RAG without the math panic: a software engineer's field guide*
> Target: software engineers (0–10 yrs), math-averse. · Suggested title:
> **"Your RAG demo scored 91%. Why did it pick the wrong answer anyway?"**
>
> What you'll learn: why a single average embedding silently blurs the signal in a chunk,
> and the three engineering moves (views, section-scoped chunks, late fusion) that turn a
> pretty search into one you can trust. By the end, you can put four views of the same
> Markdown file into a frame and produce a query table that lists the winning view per hit.

## Hook (use as-is or rewrite)

> Your RAG demo scores 91% on twenty questions you wrote by hand. Send it to staging and call
> it a win. Three weeks later a reader asks "where's the null mask?" and your bot points at
> `docs/ROADMAP-SUGGESTION.md` because the chunk that mentioned "null masks" also carried the
> word budget, which your single embedding averaged all the way to 0.4, against a 0.41 for
> a different chunk. The retrieval stack did everything right; **given one number to score
> the chunk with, it had no honest way to decide**. Retrieval silently inherited 20% of the
> damage and the chat wrapper never got a chance.

We're going to do the boring part our ancestor PDP-8 engineers called *gain staging* and
*inspection* — four blocks worth knowing today, and no derivatives in sight.

---

## 1. What the simple design looks like on your actual filesystem

Start with the most boring and most common case: a company has a `docs/` tree full of
Markdown, plus a README, and wants a chatbot.

- 100 pages of markdown, about 2,400 section-scoped chunks of 500 characters each.
- One sentence model: `all-MiniLM-L6-v2` (MiniLM, 22M parameters, 384-d vector space,
  distilled from a BERT encoder). It converts a sentence to a float array of 384 numbers.
- RAG asks: *"How do I use BLAS or MKL?"* and expects `docs/BLAS-MKL.md`.
- The baseline: embed each chunk once as an "embedding" of 384 floats, embed the query, take
  cosine similarity, take top-K.

That baseline is honest, measurable, and is the thing most articles ship. It's also wrong in
a subtler way than a wrong model — it's a wrong **document shape**, three chapters before
you get to the model.

### 1.1 The four views per chunk

We introduce four views of each chunk instead of one:

| View | What gets embedded | Missing when |
|------|--------------------|--------------|
| Content | prose plus fenced blocks, as written | never (if the chunk is nonempty) |
| Title | document title, constant per file | never (file-name stem fallback) |
| Section | heading breadcrumb `H1 > H2 > H3` | never (depth stops at 3) |
| Code | only the fence bodies inside the chunk | the chunk has no fenced blocks |

**Note where "Code" can be empty.** A file like `docs/STREAMING.md` is mostly prose; many of
its chunks have no fenced blocks at all. Code is the *conditional* view; everything else is
always present. That single asymmetry—three constant signals, one that can drop out—is the
thing every later block has to account for.

### 1.2 The single-vector penalty (the souvenir)

Collapse `Content` + `Code` into one array and feed it to the encoder: the chunk whose prose
says "don't be clever" and whose fence says `throw new Exception` leads the model with a
single "interesting" vector whose geometry sits somewhere between the two signals. On your
nearest-neighbor query `"how do I throw an error"` it scores high; on the criterion
`"should the containment policy chapter be surfaced"` it also scores higher than it deserves.
The chunk won; your reader lost.

Once Content and Code are scored **independently**, the same result carries a second number —
"Code 0.94, Content 0.71 — the fence, not the prose, is what matched" — and that sentence
is worth more retrieval authority than a score column ever gives. We had to first split the
views to see that sentence existed.

---

## 2. Section-scoped chunks, and why the fence owns a field on its own

A chunk is a `section-scoped` piece: a maximal run of *one section's prose and whole fenced
blocks*, up to a size cap (500 characters). The unit matches the way a section writes —
paragraphs and complete fences — which puts both flavors in one crumb subset instead of
spreading each fence at the wrong granularity to match the number of chunks.

Fences are never split (a fence less-than-max can be one chunk). Paragraphs longer then a
paragraph limit are split at a whitespace boundary. You get de-duplication "for free":
Title and Section texts repeat heavily (a single "Content" file moves 2-3 files ahead of
its constant parts over all documents), so the view texts are deduplicated per-document
before embedding; instead of 9,556 view slots we embed roughly 3,400 *unique* view texts and
use the result for every matching chunk — a 2.8× win over a per-chunk un-cached pipeline.

**Design decision, the boring one.** One chunk is a section-sized unit, not one fence unit.
With that choice, `Content` and `Code` won't coincide: a chunk of "I can see the lights of
the city" with a fence would otherwise give a `Code` view (a `[1, 0, 0]` pattern) that is
too close to the same surface as `Content` (read the sentence once and the fence once).
Chunking by section keeps the two distinct; the whole sentence under 26 lines even though
it has one sentence and at one point "because we split at 500 chars, the logic is simple
but it might produce rough boundaries around closing fences."

---

## 3. Late fusion, with a mask (the math, gently)

Almost all RAG samples do something like `FinalScore = 0.4 * contentScore + 0.2 * titleScore`.
The almost-always-missing piece is the **mask**: when Code view isn't available, it has no
score. A zero-handled score sneaks zero into the sum and lowers the result — but if you just
drop the code term, a chunk with no code gets *artificially punishes* the final number.

The fix is a per-row normalized fusion. Weights are `w = softmax(theta)` where theta is learned
later (or, as in Post 2, a fixed `ln(0.5, 0.2, 0.2, 0.1)` output);

~~~
fused = ((S ⊙ M) @ w) / (M @ w)     # S: cosine scores, shape [rows, 4]
                                         # M: presence mask, 1 if the view exists, 0 otherwise
~~~

- `⊙` elementwise multiply.
- `@` matrix-vector product.
- `w` always strictly positive (softmax), so the denominator (`M @ w`) can never be zero
  when at least one view passed.
- Anything not grown into `M` gets excluded — the row is evaluated against the views that
  actually exist.

This is the same per-chunk renormalization Nivara's `MultiViewScoring.Fuse` does (per row,
weights renormalized over the views that have a score). The reason the RAG reader gets to
knows which view won is that we also run the same fusion for a `title` and `section` only, on
the same ground and keep those around.

### 3.1 Rank bands, vage disagreement made arithmetic

The winning view (defined as the observation, not the gain) is the four-in-hand test,
`BestSpace` — the view that won ties broke declaration order: Content, Title, Section,
Code. `MaxDivergence` between Content and any other view signals what kind of ambiguity the
query fell into:

| Band | Condition | What to read it as |
|------|-----------|-------------------|
| `Near` | gap ≤ 0.02 | the views mostly agreed |
| `Distinct` | gap ≤ 0.08 | the views give different signals but coexist |
| `Weak` | otherwise | one view's spikes (likely Code) dominate the result |
| `InsufficientViews` | fewer than 2 scored views | either the chunk has no section OR only one view scored — agreement-based stats are meaningless here |

A query that picks up and *diverges* hits the `Weak` column; in the next post we use this
distribution as the reporting grammar itself.

---

## 4. Measurement etiquette: the one you actually need

We run `--rag-multiview` (the runnable mode — see `README.md` and `docs/EMBEDDINGS.md`) to
see what *normal* multi-view retrieval looks like:

~~~
documents -> chunk (MultiViewChunker, fence-aware)
   -> four views (content, title, section, code)
   -> scored per view (Cosine with RowCosineSimilarity)
   -> fused per row ((S . M) @ w / M @ w)
   -> ranked by fused score
~~~

The table you get is straightforward: fused score, rank band (gap to top hit), winning view, the
4 raw scores, the max divergence greatest between content and the others.

Enable `--explore` and you'll bucket the whole corpus by absolute Content-Title/section/Code
difference (`Agree <= 0.05`, `Mild <= 0.15`, `Strong` otherwise), which is the closest thing
to a quick "sanity check" that any retrieval engineer owns.

Most terms there will sound like buzzwords, so treat them all as production metric and
nothing else — right now it's enough *to have metric*. Post 2 comes back and asks the
Question: given this measurement, can we *move* it.

### 4.1 Do read the "early stop" shaped moment

Early eval (early #943 entry-level run, 100 queries LLM-judge) showed fused retiring
content-only 0.698 → 0.596 nDCG across 68 wins / 31 losses / 1 tie, sign test p=0.0003,
Cohen's d 0.422. But in a production system, the *Winning Views* were Title 557, Content 348,
Section 69, Code 26, and agreement buckets were 134 agree, 354 mild, 512 strong. Note that
fused multi-view retrieval didn't show any degradation when individual tiles were dropped
(0.688-0.693) and stopping at just COde retuned the ranking; it's the **explainability**
that the four views provide *on top* of the agreement and divergence statistics.

Why I'm giving you that observation before learning: it keeps us from second-guessing the
model's result for *ourselves*. In Post 2, the learning loop proves the *direction* of
improvement against the fixed baseline, not against the user's eye.

---

## 5. What you've built at the end of Post 1

- Four per-chunk view embeddings held in one convention frame (``float[]`` columns; the
  convention is described in `docs/EMBEDDINGS.md`).
- Late fusion, per row, with a mask for the views a chunk doesn't have.
- A disagreement dial (rank band + divergence band + winning view) that accompanies every hit.
- A way to test tuning, head-to-head, before anybody buys a reranker.

### Checkpoint: the retrieval map

~~~
q text ──→ embedding(q) ──→ per-chunk: content · title · section · code
      │                          │
      │              chunk views ──→ fuse per row (masked) ──→ sorted by fused score
      ▼
      ask (chat visible, not this telemetry-backed timeline)
~~~

- **Added:** treats one number per chunk as a baseline, then 4 numbers as the correct wiring.
- **Where you are:** through the static part (fuse with fixed weights `(0.5, 0.2, 0.2, 0.1)`).
- **Still unlit:** the 0.5/0.2/0.2/0.1 itself — it's in the file because someone wrote it.

---

## End hook → Post 2

Post 2 takes the same fuse expression and asks the model to **move those four numbers**, with
parity at step 0 and an honest comparison in the end. The contents are clear — what's not
covered is the ability to push the baseline from a fixed-fusion mode to a learnable one
without assuming anything about the retrieval failure.

## Facts & numbers to reuse (checklist)

- 4 views: Content, Title, Section, Code; weights 0.5 / 0.2 / 0.2 / 0.1 (Content ·
  Title · Section · Code).
- Chunk size (max chars / words / section / limit): 500 chars max, fences never split.
- MiniLM: 384 d, 6 layers × 12 heads, b dimension 32.
- Early eval: nDCG@10 0.698 vs 0.596; 68 wins / 31 losses / 1 tie; p=0.0003; d 0.422.
- Agree / Mild / Strong divergence buckets: 134 / 354 / 512 (of the 1,000 queries).
- Top hits' winning views: Title 557 / Content 348 / Section 69 / Code 26.
- Used as 2.8× reduction in total views during the embedding budget.

### Source references

- `samples/NivaraChat/Helpers/MultiViewScoring.cs::MultiViewScoring.BuildFrame()` — builds
  the convention frame with 4 view columns + coherence.
- `samples/NivaraChat/Helpers/MultiViewScoring.cs::MultiViewScoring.Fuse(ViewScores,FusionWeights)` —
  the one fusion arithmetic every reader, trainer and evaluator agree on.
- `samples/NivaraChat/Helpers/MultiViewChunk.cs::MultiViewChunker.ChunkMarkdown()` —
  section-scoped, fence-aware chunker.
- `samples/NivaraChat/Helpers/MultiViewIndex.cs::MultiViewIndex.BuildAsync()` — chunk,
  dedup views, embed 4 views, optional cache.
- `docs/EMBEDDINGS.md` — "Multiple views of the same corpus" (the convention).
- `samples/NivaraChat/README.md` — the `--rag-multiview` mode spec.

### Visual ideas

- Pipeline diagram: docs tree → section-scoped chunks → 4 view embeds → 4 scores per chunk
  → masked fusion → sorted hits.
- Table of a single top hit: fused 0.69, winning view Code, (title 0.84, content 0.70,
  section 0.55, code 0.79), divergence 0.09 (Distinct).
- Skirt-of-buckets heatmap (Agree / Mild / Strong) per query.
