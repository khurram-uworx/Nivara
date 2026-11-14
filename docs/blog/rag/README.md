# "Better RAG without the math panic: a field guide to late-fusion retrieval"

Source material for a 2-post blog series. The posts are **not** "introducing Nivara"
marketing — they are a software-engineer's walkthrough of what makes a retrieval layer
useful beyond "embed both text and query, cosine, top-K," using a chatbot over a
directory of Markdown files as the running example (think "our team's `docs/` plus `README.md`").

The series is grounded in #914 and #915 (multi-view retrieval and differentiable
retrieval) and in the runnable `--rag-multiview` / `--rag-learn` modes in `samples/NivaraChat`.
Every number, formula, and snippet can be traced back to a file reference at the end of
each post.

---

## Audience & voice

- **Audience:** software engineers, 0–10 years of experience, who keep a polite distance from
  the maths of ML but have written RAG, embedding, or vector-store code.
- **Front half voice:** plain conversation — "go look, this is what the file already had, you
  are reading booleans and arrays."
- **Back half voice:** gains permission carefully. The series is allowed to go deeper past
  "linear algebra 101" once the system engineer has seen the data. Each jump is flagged with
  a `Concept check: 1 line` banner, never with a proof.
- **The one-sentence thesis of the whole series:** *Embed the useful views separately, fuse
  them late with a mask for the missing ones, and let a tiny model learn fusion after you are
  sure step-0 reproduces the baseline you already trust.*

## Running example (used in every post)

A tiny company has a `docs/` tree plus `README.md`. The chatbot is grounded on them:

- "How do I implement ILGPU probe kernel?" → should pull `docs/SILK.md` or `docs/ILGPU.md`,
  not `README.md`.
- "How do I serve SmolLM as IChatClient?" → `docs/SMOLLM.md`, maybe the README quickstart.
- "What is the default streaming budget?" → `docs/STREAMING.md`; a heading-only chunk might
  match better than the section body.

Queries become 384-d MiniLM vectors (a sentence model with ~22M parameters, heavily
distilled from a BERT-like encoder), chunks become the same; retrieval is `cosine`. No LLM is
needed in the posts 1–2: rank quality is measured with ordinary formulas.

## The one diagram every reader should have by the end

~~~
query: "How does streaming work?"
   │
   ▼
embedding(query)   (MiniLM, 384 floats)
   │
   ▼
per-chunk:   contentScore  = cos(view.Content, q)
             titleScore    = cos(view.Title,    q)   // (missing where no vector)
             sectionScore  = cos(view.Section,   q)
             codeScore     = cos(view.Code,     q)
   │
   ├── late fusion per row:   fused = ((S ⊙ M) · w) / (M · w)     // missing views excluded
   │
   ▼
top-k sorted by fused → show the chunk, the winning view, the divergence band
~~~

## Format conventions reused from the BERT series

- Hook paragraphs are allowed to be rewritten ("use as-is or rewrite").
- Datasheet tables (Input / Output / Cost / Analogy).
- The "math, gently" two-step pattern: equation first only when it's friendlier than code,
  code second only when it's friendlier than math, plain English always.
- `Source references` section at the end of every post: `file::symbol()` backticks so the
  reader, or the writer, can jump.
- Postings-style "Visual ideas" notes, not diagrams (this is source material, not layout).

## Series map

1. `1-effective-rag-from-chunks-to-four-views.md` — the mechanical symptoms of bad RAG (average-vision, missing
   code, headings won't win) and three engineering fixes: views, section-scoped chunks, late
   fusion with masking.
2. `2-differentiable-rag-from-fixed-weights-to-learned-fusion.md` — the late-fusion move from constant to
   learned: parity at step 0, the gated mixture, three label sources, and honesty about
   small test sets.

## Model facts reused (do not re-prove)

- `all-MiniLM-L6-v2`: 384-d, 6 layers, 12 heads (head dim 32).
- Multi-view: 4 views per chunk; weights default to 0.5 / 0.2 / 0.2 / 0.1 over
  Content/Title/Section/Code.
- Split / answer budget: a streaming default of 10,000 rows or a context-budget-derived
  fallback; not central to the posts.

## Series wrap-up planned

Post 2 closes with *"if you want to remember one number from this whole notebook, it's the
init parity: `max |learned - fixed| ≈ 1e-7`."*

## Source references (series-level)

- `samples/NivaraChat/Helpers/MultiViewScoring.cs::MultiViewScoring.BuildFrame()`
- `samples/NivaraChat/Helpers/MultiViewScoring.cs::MultiViewScoring.Fuse()`
- `samples/NivaraChat/Helpers/MultiViewChunk.cs::MultiViewChunker.ChunkMarkdown()`
- `samples/NivaraChat/Helpers/RetrievalTrainer.cs::RetrievalTrainer.Train()`
- `samples/NivaraChat/Helpers/RetrievalEvaluation.cs::RetrievalEvaluation.Compare()`
- `docs/EMBEDDINGS.md` — "Multiple views of the same corpus"

### Visual ideas

- One diagram: single-document tree, four chunks, scores boxed side-by-side.
- One table: default weights vs learned weights vs gate weights.
- One screenshot of the console table from `--rag-learn`.
