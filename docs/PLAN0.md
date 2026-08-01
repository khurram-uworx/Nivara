# Plan: DistilBERT Base Model Inference (Baby Step)

> **Status: COMPLETE** (2026-08-01). Tasks 1–3 done and verified on branch `khurram/bert`.
> One plan-driven discovery beyond the original scope: **exact GELU** (see below).

## Purpose

Run the existing `distilbert-base-uncased` encoder in `samples/NivaraInference` to validate that all Nivara neural network layers work correctly with DistilBERT dimensions (768 hidden, 12 heads, 6 layers) before tackling the fine-tuned SST-2 model in `PLAN-DISTILBERT.md`.

This is a prerequisite investment: the token-type embedding fix and encoder weight mapping work done here directly enables both the fine-tuned DistilBERT inference and improves the existing `NivaraFineTuning` sample.

## Background

`samples/data/distilbert/` already contains the base pre-trained `distilbert-base-uncased` model:

| Field | Value |
|-------|-------|
| Model id | `distilbert-base-uncased` |
| Architecture | `DistilBertForMaskedLM` (base encoder + MLM head) |
| Config | `dim=768`, `n_layers=6`, `n_heads=12`, `hidden_dim=3072`, `vocab_size=30522`, `max_position_embeddings=512`, `activation=gelu` |
| Weight files | `model.safetensors` (~268 MB), `vocab.txt`, `config.json` |
| HF tensor prefixes | `distilbert.embeddings.*`, `distilbert.transformer.layer.{0-5}.*` |

This is the **same encoder** that `distilbert-base-uncased-finetuned-sst-2-english` was fine-tuned from. The fine-tuned model adds `pre_classifier.*` and `classifier.*` heads on top.

### One correctness gap to close first

**Token-type embedding contamination.** `BertEncoder<T>.ForwardBatched` always adds a randomly-initialized `tokenTypeEmbed` output. MiniLM has real token-type embeddings; DistilBERT has none. The random contribution corrupts pre-trained inference. Fix: add `includeTokenTypeEmbedding: bool = true` to `BertEncoder<T>`; DistilBERT passes `false`.

## Decisions (locked)

1. **Output scope:** Hidden states only — no MLM classification head, no sentiment classification head. Just tokenize → encoder → print output stats.
2. **No new downloads** — use existing `samples/data/distilbert/`.
3. **Token-type fix first** — must be done before any DistilBERT inference can produce correct results.
4. **Exact (erf) GELU is the BERT-family activation** — see "GELU discovery" below.

## GELU discovery (implemented in this plan)

`DistilBERT` `config.json` declares `"activation": "gelu"`, which HuggingFace maps to the
**exact erf-based** GELU (`x·Φ(x)`). Nivara previously had only the tanh approximation
(`ReverseGradOperations.Gelu`, matching HF `gelu_new`/GPT-2). Because MiniLM has no
`activation` field either, it defaults to exact GELU too.

Resolution:
- Added `ReverseGradOperations.GeluExact<T>` + `Activation.GeluExact<T>` (erf via
  Abramowitz–Stegun 7.1.26; forward + backward).
- `BertLayer<T>` (used by both MiniLM and DistilBERT encoders) now calls `GeluExact`.
- `Gelu` (tanh) is retained for GPT-family `TransformerBlock` (correct there) and
  backward compatibility.
- Added NivaraTorch parity fixtures `gelu_exact_1d/4d` (PyTorch `F.gelu`, default exact)
  in `samples/NivaraTorch/gen_reference.py` (dedicated RNG seed 101 so regeneration does
  not churn other fixtures). Existing `gelu_*` fixtures are tanh (`F.gelu(..., approximate="tanh")`)
  and were stale/undocumented as such — now explicit in the generator.
- Test results: 59/59 `NivaraTorch` tests pass (17 activation incl. 4 new exact-GELU).

Verification impact on the base encoder (`dotnet run --project samples/NivaraInference -- distilbert compare`):

| Metric | Before (tanh) | After (exact) |
|--------|---------------|---------------|
| max abs diff vs HF | 0.028578 | 0.000005 |
| mean abs diff vs HF | 0.000736 | 0.00000018 |
| cosine similarity | 0.99999595 | 0.99999988 |

The model now reproduces HuggingFace `last_hidden_state` to float32 precision.

## Task Breakdown

## Task 1: Fix token-type embedding in BertEncoder

### Priority

High

### Goal

Add `includeTokenTypeEmbedding` support to `BertEncoder<T>` so DistilBERT inference does not add a random token-type embedding.

### Why this exists

The encoder always sums `wordEmb + posEmb + tokenTypeEmb`. MiniLM has real token-type weights; DistilBERT has none, so the random `tokenTypeEmbed` (row 0) corrupts every pre-trained forward pass.

### Scope

- Add `bool includeTokenTypeEmbedding = true` ctor param to `BertEncoder<T>` (`samples/Nivara.Samples/BertModel.cs`).
- When `false`: do not create `tokenTypeEmbed`; skip it in `Forward`, `ForwardWithMask`, and `ForwardBatched`.
- MiniLM path unchanged (default `true`).

### Suggested implementation path

- In the constructor (line 260), gate `tokenTypeEmbed` creation and `RegisterModules` on the flag.
- In `Forward` (line 275), `ForwardWithMask` (line 301), `ForwardBatched` (line 327): conditionally compute and add `ttEmb`.
- Use a field `readonly bool _includeTokenTypeEmbedding` to store the flag.

### Acceptance criteria

- `dotnet build Nivara.slnx` succeeds.
- MiniLM forward output unchanged vs. baseline (existing behavior preserved — default `true`).
- A `BertEncoder` constructed with `includeTokenTypeEmbedding: false` does not create or use `tokenTypeEmbed`.

### Files likely involved

- `samples/Nivara.Samples/BertModel.cs`

## Task 2: Add `distilbert` mode to NivaraInference

### Priority

High

### Goal

Add a `distilbert` model entry to `samples/NivaraInference` that loads the base model encoder and runs forward inference on tokenized text.

### Why this exists

This is the baby-step showcase: run the base DistilBERT encoder in pure .NET to validate all layers work at the correct dimensions.

### Scope

- `Program.cs`:
  - Add `distilbert` to the usage/help text and model dispatch switch.
  - New `RunDistilBertInference(tensors)` method:
    - Parse `config.json` via `DistilBertConfig.FromJson` → `ToBertConfig()`.
    - Build `BertEncoder<float>` with `includeTokenTypeEmbedding: false`.
    - Map weights from `model.safetensors` using `distilbert.embeddings.*` and `distilbert.transformer.layer.{i}.*` key patterns.
    - `model.Eval()`.
    - Tokenize sample sentence via `MiniLMTokenizer`.
    - Forward through encoder → print output shape, stats (min/max/mean), first 10 values.
  - New `RunDistilBertBenchmark(tensors)` method:
    - 3 warmup + 10 timed forward passes.
    - Print avg/min/max ms, parameter count, weight MB.

### Weight mapping detail (encoder only)

```
distilbert.embeddings.word_embeddings.weight       → encoder.wordEmbed
distilbert.embeddings.position_embeddings.weight    → encoder.posEmbed
distilbert.embeddings.LayerNorm.weight/bias         → encoder.embedLn
distilbert.transformer.layer.{i}.attention.q_lin    → encoder.layers[i].attn.qProj
distilbert.transformer.layer.{i}.attention.k_lin    → encoder.layers[i].attn.kProj
distilbert.transformer.layer.{i}.attention.v_lin    → encoder.layers[i].attn.vProj
distilbert.transformer.layer.{i}.attention.out_lin  → encoder.layers[i].attn.oProj
distilbert.transformer.layer.{i}.sa_layer_norm      → encoder.layers[i].ln1
distilbert.transformer.layer.{i}.ffn.lin1           → encoder.layers[i].fc1
distilbert.transformer.layer.{i}.ffn.lin2           → encoder.layers[i].fc2
distilbert.transformer.layer.{i}.output_layer_norm  → encoder.layers[i].ln2
```

### Constraints

- Inference only — no `GradientUtils.Grad()` scope, no graph creation (leaf tensors).
- Reuse shared `MiniLMTokenizer`; do not add a new tokenizer.
- `DistilBertConfig` parsing is already implemented in `NivaraFineTuning/DistilBertModel.cs`; reference its `FromJson`/`SnakeKey` pattern or reuse if promoted.

### Acceptance criteria

- `dotnet run --project samples/NivaraInference -- distilbert` prints hidden state output with shape `[seqLen, 768]`.
- `distilbert benchmark` prints per-pass times + average.
- `dotnet build Nivara.slnx` succeeds.

### Files likely involved

- `samples/NivaraInference/Program.cs`

## Task 3: Verify correctness

### Priority

Medium

### Goal

Confirm the base model forward pass produces non-degenerate output and the token-type fix eliminates corruption.

### Why this exists

Validates that the baby step actually works before building on it in PLAN-DISTILBERT.md.

### Scope

- Run `dotnet run --project samples/NivaraInference -- distilbert` and verify:
  - Output shape is `[seqLen, 768]`.
  - Output stats are reasonable (not all zeros, not NaN/Inf, mean near zero, std in expected range).
  - No token-type corruption: output should be deterministic and stable across runs.
- Run `dotnet run --project samples/NivaraInference -- distilbert benchmark` and record baseline numbers.

### Acceptance criteria

- Forward pass completes without errors.
- Output values are in a reasonable range for normalized hidden states.
- Baseline benchmark numbers are recorded.

### Files likely involved

- None (verification only, run existing code)

## Coordination Notes

- **Decision gates:** none — all decisions are locked above.
- **Parallel-safe:** Tasks 1 and 2 are sequential (Task 2 depends on Task 1). Task 3 depends on Task 2.
- **Shared-file conflicts:** Task 1 touches `BertModel.cs`, Task 2 touches `Program.cs` — no overlap.
- **Follow-up:** This plan feeds into `PLAN-DISTILBERT.md` (fine-tuned SST-2 inference). The token-type fix (Task 1) and weight mapping pattern (Task 2) are directly reusable.

## Suggested Agent Handout Batches

### Batch A: prerequisite (sequential)

- Task 1 (token-type fix)

### Batch B: implementation (depends on A)

- Task 2 (NivaraInference `distilbert` mode)

### Batch C: verification (depends on B)

- Task 3 (correctness check)

## Final Checklist

- Every task has a single-agent-sized scope.
- Every task has acceptance criteria.
- No decision-gate tasks — all decisions locked upfront.
- Likely files listed to reduce agent search time.
- Execution order reflects real dependencies (token-type fix → inference mode → verification).
