# NivaraChat

A sample project demonstrating Nivara-trained domain-specific models as first-class participants in a `Microsoft.Agents.AI.Workflows` graph, mixed with an Ollama-backed `ChatClientAgent` node. This is the showcase example for Nivara's value proposition: **deterministic, lightweight, fast models working alongside an LLM in a production workflow.**

**Target audience:** .NET developers building AI workflows, integrating ML models into agent pipelines, exploring hybrid deterministic + stochastic architectures.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the architecture overview, component map, and design notes.
## What it does

NivaraChat trains four small domain-specific models (sentiment classifier, entity extractor, workflow validator, agents validator) and wires them into a workflow graph. Two execution paths are available:

- **`--workflow`** — classic executor-based graph with fan-out/fan-in topology
- **`--agents` / `--interactive`** — each model wrapped as an `IChatClient` via `NivaraChatClient`, participating as `ChatClientAgent`s through `AsAIAgent()`

With `--ollama`, an Ollama-backed LLM agent is appended after the validator for fluent response generation.

## Quick start

```bash
# Train all four models (overwrites existing)
dotnet run --project samples/NivaraChat -- --train

# Run workflow (Nivara nodes only — no LLM needed)
dotnet run --project samples/NivaraChat -- --workflow

# Single-shot test
dotnet run --project samples/NivaraChat -- --workflow --text "I love this product!"

# Multi-word entity examples
dotnet run --project samples/NivaraChat -- --workflow --text "John Smith from Acme Corp reported great work on January 15"
dotnet run --project samples/NivaraChat -- --workflow --text "Acme Corp in New York announced on March 3"

# Run workflow with Ollama LLM
dotnet run --project samples/NivaraChat -- --workflow --ollama --model llama3.2

# Agents mode (Nivara-only, single-shot)
dotnet run --project samples/NivaraChat -- --agents --text "Jane Doe at TechStart Inc reported issues in San Francisco"

# Agents mode with Ollama LLM
dotnet run --project samples/NivaraChat -- --agents --ollama --text "Acme Corp in New York announced on March 3"

# Interactive agents mode (Nivara-only)
dotnet run --project samples/NivaraChat -- --interactive

# Interactive agents mode with Ollama LLM
dotnet run --project samples/NivaraChat -- --interactive --ollama

# Confidence handoff — Nivara decides if LLM is needed (requires --ollama)
dotnet run --project samples/NivaraChat -- --handoff --ollama --text "I love this product!"
dotnet run --project samples/NivaraChat -- --handoff --ollama --text "This product is interesting but I'm not sure"

# Tool calling — LLM orchestrates Nivara models as AIFunction tools (requires --ollama)
dotnet run --project samples/NivaraChat -- --tools --ollama --text "John Smith from Acme Corp reported great work"

# Writer-critic loop — LLM writes, Nivara scores, retry if poor (requires --ollama)
dotnet run --project samples/NivaraChat -- --critic --ollama --text "Explain quantum computing to a 5-year-old"

# Embedding search (index documents, retrieve context via IEmbeddingGenerator)
dotnet run --project samples/NivaraChat -- --embed

# RAG pipeline: chunk docs, retrieve context (Ollama optional for LLM answer)
dotnet run --project samples/NivaraChat -- --rag --text "How does embedding search work?"
dotnet run --project samples/NivaraChat -- --rag --ollama --text "How does embedding search work?"

# RAG agent: same retrieval, TextSearchProvider auto-context injection (Ollama optional)
dotnet run --project samples/NivaraChat -- --rag-agent --text "What is NivaraChat?"
dotnet run --project samples/NivaraChat -- --rag-agent --ollama --text "What is NivaraChat?"

# Multi-view retrieval (#914): four views per chunk, late fusion, disagreement report (no LLM)
dotnet run --project samples/NivaraChat -- --rag-multiview --docs-dir docs/adr --text "How does a symbol citation resolve?"
dotnet run --project samples/NivaraChat -- --rag-multiview --docs-dir docs/adr --text "How does a symbol citation resolve?" --explore --report multiview.md
dotnet run --project samples/NivaraChat -- --rag-multiview --docs-dir docs/adr --text "How does a symbol citation resolve?" --weights content=0.7,section=0.3

# Differentiable retrieval (#915): learn the fusion weights and a query gate via AutoDiff, then
# compare against the fixed baseline on held-out queries (no LLM)
dotnet run --project samples/NivaraChat -- --rag-learn --labels fixture
dotnet run --project samples/NivaraChat -- --rag-learn --gate embed --epochs 100 --loss pairwise
dotnet run --project samples/NivaraChat -- --rag-learn --labels judge --report learned.md --k 10

# RAG evaluation (#943): LLM-as-judge paired comparison (requires Ollama)
dotnet run --project samples/NivaraChat -- --rag-eval --ollama --model beam:cloud --queries samples/data/nivarachat/eval/queries.tsv --top-k 10

# Online learning from LLM feedback (requires --ollama)
dotnet run --project samples/NivaraChat -- --online-learning --ollama

# TinyShakespeare — batched transformer served as an IChatClient (no LLM needed)
dotnet run --project samples/NivaraChat -- --tinyshakespeare

# Smoke run: smaller vocab is ~3x faster (full options: --tinyshakespeare --help)
dotnet run --project samples/NivaraChat -- --tinyshakespeare --vocab-size 1200 --prompt "ROMEO:"

# SmolLM — serve the pretrained SmolLM-135M-Instruct causal LM as an IChatClient (no LLM needed)
# Model files must be present under samples/data/smollm-135m (see "SmolLM" section below)
dotnet run --project samples/NivaraChat -- --smollm plain --text "The capital of France is"

# Interactive multi-turn chat, token-streamed (full options: --smollm --help)
dotnet run --project samples/NivaraChat -- --smollm chat

# Qwen — native function calling with Qwen2.5-0.5B-Instruct:
# Model emits <tool_call>, GetWeather runs in-process, result feeds back, clean final answer
dotnet run --project samples/NivaraChat -- --qwen tools-weather --text "What's the weather in Paris?"

# Plain transcript streaming (full options: --qwen --help)
dotnet run --project samples/NivaraChat -- --qwen plain --text "The capital of France is"
```

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--train` | — | Mode: train all four models (overwrites existing) |
| `--workflow` | — | Mode: executor-based workflow pipeline with fan-out/fan-in |
| `--interactive` | — | Mode: agents pipeline with live interactive input |
| `--agents` | — | Mode: same as `--interactive`, supports `--text` for single-shot |
| `--handoff` | — | Mode: confidence-based handoff — Nivara decides if LLM is needed |
| `--tools` | — | Mode: LLM orchestrator calls Nivara models as AIFunction tools |
| `--critic` | — | Mode: writer-critic loop — LLM writes, Nivara scores, retry if poor |
| `--embed` | — | Mode: embedding search — index documents, retrieve context via `IEmbeddingGenerator` |
| `--rag` | — | Mode: RAG pipeline — chunk markdown docs, retrieve via vector search, LLM generates answer |
| `--rag-agent` | — | Mode: RAG agent — retrieval (Ollama optional) plus `TextSearchProvider` auto-context injection when `--ollama` |
| `--rag-multiview` | — | Mode: multi-view retrieval (#914) — four views per chunk, late fusion, winning view and disagreement report (no LLM) |
| `--rag-eval` | — | Mode: RAG evaluation (#943) — LLM-as-judge paired comparison of fused vs content-only and ablations |
| `--rag-learn` | — | Mode: differentiable retrieval (#915) — learn the fusion weights and a query gate with AutoDiff, compare against the fixed baseline on held-out queries (`--silk` runs fusion training on the device for listwise loss) |
| `--online-learning` | — | Mode: online learning from LLM feedback — incremental retrain with validated examples |
| `--intent-train` | — | Mode: train intent classifier (5 classes) |
| `--intent` | — | Mode: intent routing — classify input and route to specialist executor |
| `--tinyshakespeare` | — | Mode: train/serve a batched TinyShakespeare transformer as `IChatClient` (see `--tinyshakespeare --help` for its options) |
| `--smollm` | — | Mode: serve the pretrained SmolLM-135M-Instruct causal LM as `IChatClient` (`chat`/`plain` sub-modes; see `--smollm --help`) |
| `--qwen` | — | Mode: native function calling with Qwen2.5-0.5B-Instruct (`tools-weather`/`tools-nivara`/`chat`/`plain` sub-modes; see `--qwen --help`) |
| `--text <message>` | — | Single-shot: run pipeline on one message and exit |
| `--ollama [url]` | — | Flag: enable Ollama LLM agent (optional URL, default: `http://localhost:11434`) |
| `--model <name>` | `llama3.2` | Ollama model name |
| `--threshold <float>` | `0.8` | Confidence threshold for `--handoff` mode |
| `--docs-dir <path>` | `docs/` | Documents directory for `--rag`, `--rag-agent`, `--rag-multiview` and `--rag-learn` modes |
| `--top-k <int>` | `3` | Number of chunks to retrieve for RAG modes |
| `--silk` | — | Use Silk OpenCL backend for `--rag`/`--rag-agent`/`--rag-multiview`/`--rag-learn` (MiniLM embeddings plus, with #915 follow-up, the fusion training loop and Adam step via `SilkFusionTrainer`) and for `--smollm` chat/plain generation |
| `--acl-group <name>` | — | ACL filter for `--rag`/`--rag-agent`/`--rag-multiview` (e.g. `adr`, `docs`, `plan`) |
| `--roundtrip` | — | Show embedding round-trip details for `--rag` |
| `--store <frame\|inmemory>` | `frame` | Retrieval backend for `--rag`/`--rag-agent` |
| `--index <path>` | hashed default | Cache index parquet path |
| `--rebuild` | — | Force rebuild of the cached index |
| `--no-cache` | — | Disable index caching |
| `--chunk-size <int>` | `500` | Max chunk size (chars) for RAG chunking (`--rag`/`--rag-agent`/`--rag-multiview`) |
| `--explore` | — | `--rag-multiview`: bucket every visible chunk by view disagreement, not just the top-K |
| `--report <path>` | — | `--rag-multiview`: write the retrieval report as markdown. `--rag-learn`: write the learned-vs-fixed report as markdown. Relative paths land under the gitignored `samples/data/nivarachat/` |
| `--baseline <path>` | — | `--rag-learn`: write the structured fixed-vs-baseline JSON (default `samples/data/nivarachat/eval/baseline-{kind}-s{seed}-k{k}.json`) |
| `--regenerate <queries\|judge\|all>` | — | Rebuild a label artifact deterministically (`queries` = `queries.tsv`, `judge` = `relevance.tsv`, `all` = both + manifest). Use `--force` to overwrite. |
| `--force` | — | Allow `--regenerate` to overwrite an existing artifact |
| `--weights <spec>` | `content=0.5,title=0.2,section=0.2,code=0.1` | `--rag-multiview`: late-fusion weights, renormalized per chunk over the views it has. Unknown names, negative values and an all-zero total are rejected |
| `--queries <path>` | `queries.tsv` (copied to output) | `--rag-eval`: ground-truth TSV path | 
| `--runs <n>` | `3` | `--rag-eval`: judge runs per query | 
| `--temperature <f>` | `0.2` | `--rag-eval`: judge sampling temperature | 
| `--grade-all` | — | `--rag-eval`: grade every visible chunk (true recall; slower) | 
| `--verbose` | — | `--rag-eval`: print per-query chunk movement | 
| `--offline` | — | `--rag-eval`: use cached labels only; no judge calls |
| `--labels <judge\|auto>` | `judge`, else `judge` | `--rag-learn`: label source. `judge` reads the single `relevance.tsv` (generated by `--rag-eval --ollama`, human-editable), `auto` derives labels from headings/titles and is a leakage diagnostic only. (`fixture` is a deprecated alias for `judge`.) |
| `--epochs <int>` | `200` | `--rag-learn`: training epochs |
| `--lr <float>` | `0.05` | `--rag-learn`: Adam learning rate |
| `--weight-decay <float>` | `0.01` | `--rag-learn`: Adam weight decay |
| `--patience <int>` | `20` | `--rag-learn`: early-stopping patience on validation nDCG |
| `--gate <stats\|embed\|none>` | `stats` | `--rag-learn`: query-gate features. `stats` is per-view mean/max (36 params), `embed` the raw MiniLM query vector, `none` skips the gated model |
| `--loss <listwise\|pairwise>` | `listwise` | `--rag-learn`: ranking loss |
| `--candidates <int>` | `10` | `--rag-learn`: rows each view contributes to the candidate pool, plus every labeled positive |
| `--seed <int>` | `42` | `--rag-learn`: query-split and shuffle seed |
| `--train-queries <int>` | all | `--rag-learn`: cap the training split |
| `--k <int>` | `10` | `--rag-learn`: metric cut-off |
## Modes of use

### Training (`--train`)
Trains all four models on synthetic data: sentiment classifier, entity extractor, workflow validator, and agents validator. Each model follows the same pattern: generate data → tokenize → build frame → train with `TrainingLoop<T>` → save with `ModelSerializer`. No external datasets required.

### Workflow (`--workflow`)
Classic executor-based pipeline with fan-out/fan-in topology. `TextRouter` broadcasts input to `SentimentExecutor` and `EntityExtractor` in parallel; results merge at `ValidatorExecutor` via barrier. Without `--ollama`, runs Nivara nodes only. With `--ollama`, appends an LLM agent after the validator.

### Agents (`--agents`)
Sequential pipeline where each trained model is wrapped as an `IChatClient` via `NivaraChatClient` and participates as a `ChatClientAgent`. Supports `--text` for single-shot execution. With `--ollama`, an Ollama LLM agent is appended after the validator.

### Interactive (`--interactive`)
Same as `--agents` but with live input. Type `quit` to exit. With `--ollama`, the LLM agent is appended after the validator.

### Confidence handoff (`--handoff`)
Demonstrates the hybrid deterministic/stochastic pattern. Nivara models run first; if both sentiment and entity extraction are confident (>= `--threshold`, default 0.8), the result is returned without calling the LLM. If either is uncertain, the partial Nivara results are forwarded to the LLM for enrichment. Requires `--ollama`.

```
Input text
    │
    v
[TextRouter] --fan-out--> [SentimentExecutor, EntityExtractor]
                               │
                          fan-in barrier
                               │
                               v
                        [ConfidenceRouter]
                         /           \
              confident (>=0.8)    uncertain (<0.8)
                    │                    │
                    v                    v
            Nivara result          [Ollama LLM]
```

Tested examples:

| Input | Threshold | Path taken | Why |
|-------|-----------|------------|-----|
| `"I love this product!"` | 0.8 (default) | LLM | Sentiment 0.58, entity 0.27 — both below threshold |
| `"I love this product!"` | 0.7 | LLM | Entity confidence 0.27 still below 0.7 |
| `"John Smith from Acme Corp reported great work on January 15"` | 0.8 | LLM | Entity confidence 0.798 just below 0.8 |
| `"John Smith from Acme Corp reported great work on January 15"` | 0.7 | Nivara only | Both above 0.7 — no LLM needed |

The entity model's average per-token confidence tends to cap around 0.8 for multi-entity inputs. Use `--threshold 0.7` for a more practical cutoff.

### Tool calling (`--tools`)
Flips the architecture: the LLM *decides* when to call Nivara models. Nivara models are wrapped as `AIFunction` tools via `AIFunctionFactory` with `[Description]` attributes. The LLM receives tool definitions and chooses when to invoke sentiment analysis, entity extraction, or response validation. Requires `--ollama`.

Tested examples:

| Input | Tools called | Notes |
|-------|-------------|-------|
| `"John Smith from Acme Corp reported great work"` | ExtractEntities, AnalyzeSentiment | LLM chose both tools, summarized results |
| `"Acme Corp in New York announced on March 3"` | ExtractEntities, AnalyzeSentiment | Multi-entity extraction works well |

The LLM decides which tools to call based on the `[Description]` attributes. Tool results are fed back automatically by the `ChatClientAgent` framework.

### Writer-critic loop (`--critic`)
The LLM generates a response, a Nivara validator model scores it for quality/consistency, and the LLM re-generates if the score is below threshold. Bounded to 3 iterations with structured feedback. Demonstrates Nivara models evaluating LLM output, not just generating their own. Requires `--ollama`.

Tested examples:

| Input | Result | Score |
|-------|--------|-------|
| `"Explain quantum computing to a 5-year-old"` | PASS on attempt 1 | 0.98 |

The validator model was trained on `"original || response"` format for consistency checking. High scores indicate the response is consistent with the query. Max 3 iterations — if all fail, the last attempt is returned with a notice.

### Embedding search (`--embed`)
**This mode is deliberately the primitive path.** It keeps the 8 documents and their vectors in two parallel arrays and ranks them with a hand-written `TensorPrimitives.CosineSimilarity` loop (`OrderByDescending(...).Take(4)`). There are no labels, no null rows, no ACL filter, no round-trip and no cache, and the top-K is fixed at 4. It is the "before" state for the embedding convention documented in [the embedding convention](../../docs/EMBEDDINGS.md): compare `EmbeddingMode.cs` with `RagMode.cs` side by side. Issue #919 originally proposed migrating this mode to the convention; the decision was to keep it as the friction baseline so the cost stays visible, and let `--rag` carry the convention.

Indexes 8 knowledge documents using `IEmbeddingGenerator` backed by a local MiniLM transformer, then runs an interactive REPL. Type a query and the system retrieves the top-4 most relevant documents ranked by cosine similarity. This demonstrates the retrieval step for RAG (Retrieval-Augmented Generation) — in a full pipeline, retrieved context would be injected into the LLM prompt via `TextSearchProvider`. Uses `NivaraEmbeddingGenerator<string>` from `Nivara.Extensions`, the same interface as OpenAI/Ollama embedding providers.

### RAG pipeline (`--rag`)
Full Retrieval-Augmented Generation pipeline. This is the **convention example**: documents are landed as one `NivaraFrame` carrying metadata columns (`ChunkId`, `Source`, `Title`, `SectionPath`, `AclGroup`, `Content`) plus an `Embedding` column of `float[]`, and retrieval runs through `RowCosineSimilarity` + `TopKDescending` — the #867 convention documented in [the embedding convention](../../docs/EMBEDDINGS.md). The index is cached to Parquet in the dimension-frame form (`--index`, `--rebuild`, `--no-cache`); `--store inmemory` keeps the legacy `InMemoryVectorStore` path for comparison. Loads real Nivara documentation (markdown files from `docs/` + `README.md`), chunks them into paragraphs, embeds with the local MiniLM model (CPU or Silk via `--silk`), then runs an interactive REPL. ACL filtering (`--acl-group`) is a `FilterByMask` row op before scoring. User questions are matched against stored chunks via cosine similarity, top-K chunks are injected into a manually constructed prompt, and the LLM generates a grounded answer when `--ollama` is supplied; without `--ollama` the mode lists the retrieved chunks and their scores. Shows retrieval time and LLM time separately. Ollama is optional.

```
Documents (docs/*.md, README.md)
    │
    v
ChunkText (paragraph splitting, ~500 chars)
    │
    v
NivaraFrame (metadata + Embedding) + MiniLMEmbedder
    │  embeds each chunk; NivaraFrame + RowCosineSimilarity retrieval
    v
User query
    │
    v
collection.SearchAsync(query, top: K)  →  ranked chunks
    │
    v
Manual prompt: "Answer based on context:\n{chunks}\n\nQuestion: {query}"
    │
    v
Ollama LLM  →  grounded response
```

Tested examples:

| Input | Top-K | Retrieval | LLM | Answer quality |
|-------|-------|-----------|-----|----------------|
| `"How does embedding search work?"` | 3 | 371ms | 27.5s | Describes MiniLM → NivaraFrame → RowCosineSimilarity pipeline |
| `"What is NivaraChat?"` | 3 | 538ms | 17.2s | Correctly identifies RAG pipeline with MiniLM + TextSearchProvider |

Uses: `MiniLMEmbedder.Create()`, `NivaraFrame` (metadata + `Embedding`), `RowCosineSimilarity`, `TopKDescending`, `FilterByMask`, `RagIndexCache` (Parquet dimension-frame cache), `DocumentChunker.ChunkText()`. Legacy comparison: `--store inmemory` (`InMemoryVectorStore`, `collection.SearchAsync()`).

### Multi-view retrieval (`--rag-multiview`)
A sample-only demonstration of issue #914. Instead of embedding each chunk once, every chunk is embedded **four times**, once per representation, all in the same 384-d MiniLM space:

| View | Text | Missing when |
|------|------|--------------|
| Content | The chunk as written: prose plus any fenced blocks | Never |
| Title | Document title (the H1, else the file-name stem) — constant per file | Never |
| Section | Heading breadcrumb, `H1 > H2 > H3`; H4 and deeper bound a chunk but do not extend the breadcrumb | Never |
| Code | Only the fence bodies inside the chunk | The chunk has no fenced blocks |

Chunks are **section-scoped**: a contiguous run of one section's prose and whole fenced blocks, bounded by `--chunk-size`. That is what keeps Content and Code from being the same text — one chunk per fence would make their divergence ~0 for every code chunk — so a hit whose winning view is `Code` says the fence, not the prose, is what matched. A fence is never split; a paragraph longer than `--chunk-size` is split at whitespace.

Each view is scored independently with the frame's `RowCosineSimilarity` kernel, then fused late. The four weights are renormalized per row over the views that actually have a score, so a chunk without code is not penalized for its missing view: a missing view is dropped from fusion, from `BestSpace` and from divergence. The same rule treats a non-finite score as missing — a zero-magnitude vector makes `TensorPrimitives.CosineSimilarity` return `NaN`, which must never poison the fused score.

The table prints the fused score, the **rank band** (gap from the query's top hit: `Near` ≤ 0.02, `Distinct` ≤ 0.08, else `Weak`), the winning view (`Content`/`Title`/`Section`/`Code`, ties broken in that declaration order), the four raw scores, and the widest **divergence** (absolute difference between the Content score and each other view's score). `--explore` ranks the whole visible corpus and buckets it by absolute divergence — `Agree` ≤ 0.05, `Mild` ≤ 0.15, `Strong` above that, plus an `InsufficientViews` bucket for chunks with fewer than two scored views — and prints a one-line explanation of each example. `--report <path>` writes the same table plus the buckets as markdown; relative paths resolve under the gitignored `samples/data/nivarachat/`.

```
Documents (docs/**/*.md, README.md)
    │
    v
MultiViewChunker (section-scoped chunks; fence-aware; title in a pre-scan)
    │  Content / Title / Section / Code per chunk
    v
Deduplicated view texts  →  MiniLMEmbedder  →  MultiViewScoring.BuildFrame
    │  one frame: metadata + 4 × float[] + Content/Title coherence
    v
MultiViewMatrices (4 flat [rows, 384] frames, built once)  →  MultiViewIndexCache
    │
    v
Query  →  4 × RowCosineSimilarity  →  per-row renormalized fusion
    │  fused score, BestSpace, divergences, max divergence, rank band
    v
Console table  +  --explore buckets  +  --report markdown
```

`Title` and `Section` repeat heavily across a corpus, so view texts are deduplicated before embedding — a full-corpus run embeds roughly 2,000 unique texts rather than one per chunk per view. The multi-view index has its own cache type and its own `SchemaVersion`/`ChunkerVersion` salt (`samples/data/nivarachat/rag_multiview_cache/`), so it can never be mistaken for the single-view #912 cache; `--index`, `--rebuild` and `--no-cache` behave as they do for `--rag`.

### RAG evaluation (`--rag-eval`)

Runs an LLM-as-judge experiment: builds the multi-view index, evaluates `fused` against
`content-only` and leave-one-view-out configurations, and prints paired metrics, win/loss/tie,
diagnostics, and cost. Requires `--ollama` (for example `--model beam:cloud`) unless `--offline`
is used with cached labels in `samples/data/nivarachat/eval/relevance.tsv`.

Interpretation:
- If fused MRR/nDCG are not significantly better than content-only, multi-view is only an
  explainability layer.
- A `BestSpace` distribution dominated by Content means other views rarely win.
- A small Strong disagreement bucket means the views largely agree.

Two documented limits: MiniLM truncates at 128 tokens on both backends, so a long Content or Code view loses its tail; and the mode reports retrieval signal only — it does not call an LLM.

Uses: `MultiViewChunker`, `MultiViewScoring` (`BuildFrame`, `Rank`, `FusionWeights`, `MultiViewMatrices`), `MultiViewExplorer` (`Bucketize`, `Explain`), `MultiViewIndex`/`MultiViewIndexCache`, and the same `RowCosineSimilarity`/`TopKDescending` convention as `--rag`.

### Differentiable retrieval (`--rag-learn`)

Trains the #914 fusion weights — and optionally a query gate over them — with AutoDiff, then scores
the result against the fixed baseline on queries neither model saw. This is the sample that closes
the gap between the AutoDiff and retrieval stories: the corpus is the multi-view dataset, the query
becomes a score vector, the loss is a ranking loss, and the gradient reaches four numbers that used
to be a constant.

The trainable part reranks a **weight-agnostic candidate pool**: the union of each view's own top-K
(`--candidates`) plus every labeled positive. Drawing the pool from the fixed baseline's top-K would
let the baseline choose its own competitors, so candidate generation stays outside the gradient path
and every ranker in the table reranks the same pool.

Two models are trained:

| Model | Parameters | What it learns |
|-------|-----------|----------------|
| `learned-global` | 4 | Four fusion logits, `w = softmax(theta)` |
| `learned-gated(stats)` | 36 | A gate from per-view mean/max cosine (8 dims) to the same four logits, per query |
| `learned-gated(embed)` | 1,540 | `--gate embed`: the gate reads the raw 384-d MiniLM query vector instead |

Both start numerically **equal to the fixed baseline**: `theta = ln(0.5, 0.2, 0.2, 0.1)` reproduces
the default weights exactly (they sum to 1.0), and the gate's weight matrix starts at zero so it is
invariant to its input. The mode prints the measured `max |learned - fixed|` at initialisation, which
is how a drifted fusion path would show up instead of being blamed on stochastic training.

The loss is listwise softmax cross-entropy against a `2^grade` target normalized per query;
`--loss pairwise` swaps in a gather-free logistic loss over ordered pairs. Checkpoint selection uses
validation nDCG@K with `--patience`, and the 80/10/10 split is hashed from the query string with
`--seed`, so two processes produce the same partition.

Labels decide what the numbers mean:

| `--labels` | Source | Use it for |
|-----------|--------|-----------|
| `judge` (default when present) | `samples/data/nivarachat/eval/relevance.tsv` from `--rag-eval` | The headline comparison |
| `auto` | `--queries` heading/title-derived rows | Leakage diagnostics only — Title and Section score their own positive near 1.0 |

The mode refuses to train when fewer than half the label rows resolve against the corpus, because a
stale label file silently shrinks the relevant set and lowers every ranker without any of them being
worse. It also prints the decisive-query count when the held-out split is too small for the sign
test to reach p < 0.05.

Reproducibility: chunk ids, the query split and the fixed-vs-baseline metrics are reproducible from
the same corpus + seed. The mode writes a structured, byte-stable baseline JSON (see `--baseline`)
and prints a `REFUSED:` marker with exit code 2 when coverage is below 50%. Regenerate the label
artifacts with `--regenerate <queries|judge|all>` (see [`./916-A.md`](./916-A.md)).

`--report <path>` writes the metric table, the paired comparisons, the learned weights and a
"reading this report" section as markdown under the gitignored `samples/data/nivarachat/learn/`.
Acceptance is a fair comparison, not a guarantee: a learned model that loses on held-out queries is
a valid outcome and the report says so. With `--silk`, the MiniLM embeddings and — for `--loss listwise` — the fusion
training loop and Adam step run on the device via `SilkRetrievalTrainer`; the
query gate trains on the device too. `--loss pairwise` still trains on CPU,
because the Silk trainer is listwise-only.

Uses: `RetrievalTrainingData` (`CandidatePool`, `TrainingSetBuilder`, `QuerySplit`), `RetrievalLabels` (`LabelKind`, `ResolvedLabels`, `SelectSource`, `Accepts`),
`RetrievalTrainer` (`FusionModel`, `GatedFusionModel`, `TrainingOptions`), `RetrievalEvaluation`
(metrics, ranking, `PairedComparison`), and the `MultiViewScoring`/`MultiViewIndex` pipeline from
`--rag-multiview`.

Full design note: [`./915.md`](./915.md).

### RAG agent (`--rag-agent`)
Same retrieval pipeline as `--rag`, but uses `TextSearchProvider` from the Agent Framework for automatic context injection instead of manual prompt construction. `TextSearchProvider` intercepts each LLM call, performs a search, and injects the retrieved context before the LLM sees the query. This is the standard ecosystem pattern for RAG and composes with other `AIContextProvider` implementations. **Ollama is optional**: without `--ollama` the mode behaves like `--rag` in list-only mode (prints `#rank score [source] "snippet..."` per query); with `--ollama` it runs the agent workflow and prints the retrieved set for each provider search. `--roundtrip` prints the frame ↔ matrix ↔ `ToGeneratedEmbeddings` diagnostics once after the index builds (not per provider search). ACL filtering (`--acl-group`) applies to the frame store only; `--store inmemory` ignores it.

```
Documents (same as --rag)
    │
    v
NivaraFrame (metadata + Embedding) + MiniLMEmbedder
    │
    v
TextSearchProvider (SearchTime = BeforeAIInvoke)
    │  auto-searches before every LLM call
    │  injects top-K chunks as additional context
    v
ChatClientAgent + Ollama LLM
    │
    v
Grounded response with source citations
```

Tested examples:

| Input | Answer quality |
|-------|----------------|
| `"How does embedding search work?"` | Describes MiniLM → InMemoryVectorStore → auto-embedding pipeline with code example |
| `"What is NivaraChat?"` | Identifies as Nivara project component for RAG pipeline |

### Intent routing (`--intent`)
5-class intent classifier routes user input to specialist executors using conditional edges. Requires `--ollama` for specialist executors (except escalation). Training produces `models/intent_model.json` and `models/intent_tokenizer.json`.

```
User input
    │
    v
[IntentClassifier]           Nivara TextClassifierModel, 5 classes
    │
    ├── "factual"      ──> [FactualExecutor]       RAG retrieval + LLM generation
    ├── "question"     ──> [QuestionExecutor]      General Q&A via Ollama
    ├── "command"      ──> [CommandExecutor]       LLM with AIFunction tools
    ├── "complaint"    ──> [EscalationExecutor]    Human-in-the-loop (no LLM)
    └── "chitchat"     ──> [ChitchatExecutor]      Casual conversation via Ollama
```

Tested examples:

| Input | Intent | Response quality |
|-------|--------|------------------|
| `"I'm unhappy with the service"` | complaint | Escalation message with timestamp |
| `"What is the capital of France?"` | question | LLM answer: Paris |
| `"Hello!"` | chitchat | Friendly greeting |

Uses: `IntentClassifier`, `FactualExecutor`, `QuestionExecutor`, `CommandExecutor`, `EscalationExecutor`, `ChitchatExecutor`, `AddEdge<string>` conditional routing.

### Online learning (`--online-learning`)
Demonstrates online learning from LLM feedback. The intent classifier runs first; when confidence is below the threshold (default 0.8), the LLM provides a corrected intent which is added to a training buffer. When the buffer reaches 10 examples, the model is incrementally retrained using `IntentTrainer.TrainIncremental()` with a lower learning rate (0.0005). The updated model is saved and continues classifying. Requires `--ollama`.

```
User input
    │
    v
[IntentClassifier]           Nivara TextClassifierModel, 5 classes
    │
    ├── confidence >= 0.8    ──> Return Nivara classification (no LLM needed)
    │
    └── confidence < 0.8     ──> [Ollama LLM]
                                    │
                                    v
                              LLM provides corrected intent
                                    │
                                    v
                              Add (text, intent) to training buffer
                                    │
                                    v
                              Buffer full (10 examples)?
                                    │
                               yes ──> IntentTrainer.TrainIncremental()
                                    │   loads checkpoint (weights + optimizer state)
                                    │   trains 5 epochs at lr=0.0005 via Continue()
                                    │   saves updated model + checkpoint
                                    v
                              Continue with updated model
```

Tested examples:

| Input | Threshold | Result |
|-------|-----------|--------|
| `"hello there"` | 0.8 | chitchat (confidence: 0.934) — no LLM needed |
| `"what is the capital of France"` | 0.8 | LLM-corrected to question (confidence: 0.550) |
| `"turn on the lights"` | 0.8 | LLM-corrected to command (confidence: 0.493) |

Uses: `FeedbackCollector`, `IntentTrainer.TrainIncremental()`, `TrainingLoop.Continue()`, `TrainingLoop.SaveCheckpoint()/LoadCheckpoint()`, `Optimizer.StateDict()/LoadStateDict()`.

### TinyShakespeare (`--tinyshakespeare`)
Trains a **word-level batched causal transformer** on the TinyShakespeare corpus with Nivara's AutoDiff, then serves it through the standard `Microsoft.Extensions.AI.IChatClient` interface and wires it up via DI. No LLM needed — this mode proves Nivara can train a real transformer and serve it in an ecosystem-compatible way. Training runs when no `--load` is given; a saved model skips straight to generation and the DI demo. See the dedicated [TinyShakespeare section](#tinyshakespeare--batched-transformer-ichatclient-mode---tinyshakespeare) below for the full option list, architecture, and the how-it-differs-from-MicroGpt comparison.

```
TinyShakespeare.txt → word-level TextTokenizer → batched causal transformer training
    → ModelSerializer.Save/Load → BatchedChatClient : IChatClient
    → services.AddChatClient(factory) → console/ASP.NET/MAUI
```

Tested examples:

| Command | Model | Result |
|---------|-------|--------|
| `--tinyshakespeare` | 2L × 96D, 4 heads, vocab 8000 (defaults) | Full train on the corpus, 5 generated replies, DI demo reply |
| `--tinyshakespeare --vocab-size 1200 --prompt "ROMEO:"` | 2L × 96D, 4 heads, vocab 1200 | Smoke run, ~3x faster; generates Shakespeare-style continuations for the prompt |
| `--tinyshakespeare --data <small.txt> --vocab-size 800 --n-embd 32 --n-layer 1 --block-size 32 --n-head 2 --epochs 1 --samples 2 --prompt "ROMEO:"` | 1L × 32D, 2 heads, vocab 592 (31584 params) | Trained a 249-line slice in 1.3s (1350 tok/s, loss 6.17) and replied via `IChatClient` |
| `--tinyshakespeare --load models/ts.json --prompt "KING LEAR:" --no-di-demo` | Matches saved model | Skips training, loads weights + tokenizer, generates directly |

Uses: `BatchedTransformer<T>`, `BatchedChatClient` (`IChatClient`), `TextTokenizer`, `ModelSerializer.Save/Load`, `ReverseGradOperations.BatchedMultiHeadAttention`, `Embedding`/`Linear`/`LayerNorm`/`Activation.Gelu`, `CrossEntropyLoss<T>`, `Adam<T>`, `services.AddChatClient()`.

### SmolLM (`--smollm`)
Serves the **pretrained SmolLM-135M-Instruct causal LM** through the standard `Microsoft.Extensions.AI.IChatClient` interface — no training, no LLM server, no Python runtime. The model runs in-process on Nivara's zero-dependency tensor engine: `LlamaForCausalLM<T>` (autoregressive) + the GPT-2 byte-level BPE tokenizer. The conversation is rendered into SmolLM's Hermes/ChatML format (`<|im_start|>role\n...<|im_end|>`) and the reply is **token-streamed** via `GetStreamingResponseAsync`. Two sub-modes:

```
config.json + model.safetensors + vocab.json + merges.txt
    → SafeTensorsLoader.Read → LlamaLoader.Load → LlamaForCausalLM
    → SmolLMChatClient : IChatClient (greedy/sampled, token-streamed, KV-cached)
    → --smollm plain/chat
```

- `--smollm plain --text "..."` — single-shot reply, no REPL.
- `--smollm chat [--text "..."]` — interactive multi-turn REPL (default when no `--text`); a prompt skips straight to one turn. In the interactive menu (`RunInteractive`), generation options are prompted with sensible defaults.

Options after `--smollm`: `chat|plain` sub-mode, `--model-dir <path>` (default `samples/data/smollm-135m`), `--precision f32|bf16` (default `f32`), `--max-new-tokens <n>` (default 64), `--text <string>`, `--temperature <t>` (0 = greedy, >0 = sampling), `--top-p <p>` (nucleus cutoff, 0–1, default 1), `--seed <n>` (RNG seed for reproducible sampling), `--kv-cache` / `--no-kv-cache` (default: cached). Run `--smollm --help` for the full list.

Model files must be present under `samples/data/smollm-135m` (already downloaded in this repo). To re-download:

```
hf download HuggingFaceTB/SmolLM-135M-Instruct config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json special_tokens_map.json --local-dir samples/data/smollm-135m
```

Tested examples:

| Command | Precision | Result |
|---------|-----------|--------|
| `--smollm plain --text "The capital of France is" --max-new-tokens 12` | f32 | Streams "The capital of France is Paris, which is the largest city" |
| `--smollm chat` | f32 | Interactive multi-turn REPL; reply streamed token-by-token (KV-cached by default) |
| `--smollm chat --temperature 0.6` | f32 | Interactive REPL with sampling; varied replies across turns |
| `--smollm plain --text "The capital of France is" --precision bf16` | bf16 | Same reply, ~half the memory footprint |

This is **Stage A** of the SmolLM two-demo plan: plain causal-LM chat only. Tool calling (the SmolLM2 Hermes `<tool_call>`/`<tool_response>` format wiring Nivara's trained models as `AIFunction` tools) is a later stage and is intentionally not included here.

Uses: `LlamaForCausalLM<T>`, `LlamaLoader.Load`, `LlamaKVCache<T>`, `SafeTensorsLoader.Read<T>`, `Gpt2BpeTokenizer` (`Encode`/`Decode`/`TokenId`), `SmolLMChatClient<T>` (`IChatClient`, temperature/top-p sampling, KV-cached generation), `SmollmChatTemplate` (Hermes ChatML rendering).

### Qwen (`--qwen`)

Native **function calling** with the pretrained **Qwen2.5-0.5B-Instruct** checkpoint, still fully in-process on Nivara's zero-dependency tensor engine (no LLM server, no Python). The conversation is rendered **byte-identically** to HuggingFace's `apply_chat_template` for this checkpoint (`QwenChatTemplate`, pinned against Torch ground-truth fixtures), and the reply is decoded autoregressively with a KV cache. Three sub-modes:

```
config.json + model.safetensors + vocab.json + merges.txt + tokenizer.json
    → SafeTensorsLoader.Read → LlamaLoader.Load → LlamaForCausalLM
    → QwenChatClient : IChatClient (greedy/sampled, token-streamed, KV-cached, <tool_call> parsing)
    → FunctionInvokingChatClient (tools-weather loop, cap 3) or --qwen chat/plain
```

- `--qwen tools-weather --text "What's the weather in Paris?"` — the native tool-calling demo: the model emits `<tool_call>\n{"name": "getWeather", "arguments": {"city": "Paris"}}\n</tool_call>`, the `GetWeather` `AIFunction` runs, the result feeds back as `<tool_response>` inside a `user` turn, and the model closes with a clean natural-language answer (loop capped at 3 iterations so a model that never answers still exits cleanly). Interactive REPL when `--text` is omitted.
- `--qwen tools-nivara --text "Analyze the sentiment of: 'I absolutely love this product!'"` — real Nivara-backed tools (`NivaraAIFunctions` building blocks: `analyzeSentiment`, `extractEntities`, `validateResponse`) invoked by the Qwen loop; subset via `--nivara-tools`. Interactive REPL when `--text` is omitted.
- `--qwen chat` / `--qwen plain` — interactive multi-turn REPL / single-shot plain-text reply, token-streamed.

Options after `--qwen`: `tools-weather|tools-nivara|chat|plain` sub-mode, `--model-dir <path>` (default `samples/data/qwen2.5-0.5b-instruct`), `--nivara-tools <csv>` (subset for tools-nivara: sentiment,entities,validate,all; default all), `--precision f32|bf16` (default `f32`; bf16 keeps the BF16 weights native — BF16-on-disk is read as BFloat16 with no widen), `--max-new-tokens <n>` (default 128), `--text <string>`, `--temperature <t>` (0 = greedy, >0 = sampling), `--top-p <p>` (nucleus cutoff, 0–1, default 1), `--seed <n>` (RNG seed), `--kv-cache` / `--no-kv-cache` (default: cached). Run `--qwen --help` for the full list.

Model files must be present under `samples/data/qwen2.5-0.5b-instruct`:

```
hf download Qwen/Qwen2.5-0.5B-Instruct config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json special_tokens_map.json --local-dir samples/data/qwen2.5-0.5b-instruct
```

Tested examples:

| Command | Precision | Result |
|---------|-----------|--------|
| `--qwen tools-weather --text "What's the weather in Paris?"` | f32 | `[assistant → getWeather(city: Paris)]` → `[tool] Partly cloudy, 18°C. …` → "The weather in Paris is partly cloudy with a temperature of 18°C. …" (loop closes inside the cap) |
| `--qwen plain --text "The capital of France is" --max-new-tokens 24` | f32 | Streams "The capital of France is Paris." then stops on `<\|im_end\|>` |

Ground truth, the format findings, the BF16 loader benchmark, the PyTorch
parity evidence, and the full performance work (Qwen-fast: batched prefill,
fused decode attention/decoder block, bf16 tradeoffs and the P2 backlog) live
in `docs/QWEN.md`.

Uses: `LlamaForCausalLM<T>`, `LlamaLoader.Load`, `LlamaKVCache<T>`, `SafeTensorsLoader.Read<T>` (`Read<float>` fused BF16→F32 at load), `Gpt2BpeTokenizer` (Qwen Split-regex pretokenization), `QwenChatClient<T>` (`IChatClient`), `QwenChatTemplate` (byte-exact renderer), `QwenToolCallParser` (`<tool_call>` → `FunctionCallContent`), `FunctionInvokingChatClient` (MEAI tool loop), `AIFunctionFactory` (`GetWeather`).

## Nivara APIs demonstrated

| API | Where | Purpose |
|-----|-------|---------|
| `TextClassifierModel<T>` | `Nivara.Samples` (shared with NivaraClassifier) | Document-level classification (sentiment, validator) |
| `TokenClassifierModel<T>` | `Nivara.Samples` | Token-level classification (NER) |
| `TextTokenizer` | `Nivara.Samples` (shared with NivaraClassifier) | Word-level tokenization, vocab, encode/decode |
| `Module<T>` | All models | Model base class |
| `Embedding<T>` | All models | Learned word embeddings |
| `Linear<T>` | All models | Fully connected layers |
| `CrossEntropyLoss<T>` | Training | Classification loss |
| `Adam<T>` | Training | Optimizer |
| `GradientUtils.Grad()` | `--rag-learn` | Opt-in reverse-mode training scope |
| `Module<T>.StateDict/LoadStateDict` | `--rag-learn` | Best-validation checkpoint restore |
| `ReverseGradOperations.LogSoftmax/MatMul/Sum` | `--rag-learn` | Gather-free listwise and pairwise ranking losses |
| `BCEWithLogitsLoss<T>` | `--rag-learn` | Pairwise logistic ranking loss |
| `TrainingLoop<T>` | Training | Training orchestration |
| `DataLoader<T>` | Training | Batched data loading |
| `TensorDataset<T>` | Training | Frame-backed dataset |
| `ModelSerializer.Save/Load` | Training + inference | JSON model persistence |
| `Executor<TInput, TOutput>` | Executors (`--workflow`) | Workflow node with type-safe routing |
| `WorkflowBuilder` | Program.cs | Workflow graph construction with fan-out/fan-in |
| `AddFanOutEdge` | Program.cs | Broadcast input to multiple executors in parallel |
| `AddFanInBarrierEdge` | Program.cs | Wait for all parallel executors before proceeding |
| `InProcessExecution.RunAsync` | Program.cs | Static workflow execution |
| `AIFunctionFactory.Create` | NivaraToolFunctions.cs | Wrap static methods as LLM-callable tools |
| `IChatClient` | NivaraChatClient.cs | Microsoft.Extensions.AI chat abstraction |
| `AsAIAgent()` | Program.cs | Convert `IChatClient` to `ChatClientAgent` |
| `ChatClientAgent` | Program.cs | Agent Framework participant from `IChatClient` |
| `NivaraEmbeddingGenerator<T>` | Nivara.Extensions | `IEmbeddingGenerator<TInput, Embedding<float>>` implementation for local models |
| `NivaraFrame.RowCosineSimilarity` / `RowDot` | RagMode.cs, RagAgentMode.cs | #867 convention retrieval over a `float[]` column (converts to the dimension frame internally) |
| `TopKDescending` | RagMode.cs, RagAgentMode.cs | labeled top-K retrieval; nulls last and excluded |
| `ToEmbeddingMatrix` / `ToEmbeddingFrame` / `ToGeneratedEmbeddings` | RagMode.cs (`--roundtrip`), RagIndexCache.cs | round-trip + Parquet cache persistence in the dimension-frame form |
| `FilterByMask` | RagMode.cs, RagAgentMode.cs | ACL row selection before scoring |
| `TrainingLoop.Run(startEpoch)` | IntentTrainer.cs | Resume training from checkpoint |
| `Optimizer.StateDict()/LoadStateDict()` | IntentTrainer.cs | Save/restore optimizer state for incremental training |
| `ModelSerializer` (optimizer state) | IntentTrainer.cs | Persist optimizer state in checkpoints |

## Requirements

- Ollama (optional — only when `--ollama` flag is used; install from [ollama.com](https://ollama.com))

## Library gaps this example resolved

### Library additions driven by this example

| New API | Location | Purpose |
|---------|----------|---------|
| `TextClassifierModel<T>` | `samples/Nivara.Samples/TextClassifierModel.cs` | Embedding → MeanPool → MLP document classifier. |
| `TokenClassifierModel<T>` | `samples/Nivara.Samples/TokenClassifierModel.cs` | Embedding → MLP per-token classifier for NER and sequence labeling. |
| `TextTokenizer` | `samples/Nivara.Samples/TextTokenizer.cs` | Word-level tokenizer with vocab, encode/decode, special tokens, save/load. |
| `MiniLMEmbeddingGenerator` | `src/Nivara.Extensions/AI/BertModel.cs` | Factory wiring MiniLM weights + BertTokenizer into `NivaraEmbeddingGenerator<string>`. |

## Limitations

- **Word-level tokenization** — no subword (BPE) support. Out-of-vocabulary words map to UNK. Sufficient for synthetic data.
- **Synthetic training data** — entity extraction and validation use template-based synthetic data. Real applications would use annotated corpora.
- **No LLM streaming** — the workflow runs non-streaming. The LLM response is collected in full before validation.
- **Sequential agents** — the agents pipeline runs sequentially (Sentiment → Entity → Validator). Fan-out parallelism is only available in `--workflow` mode.
