# LoRA Fine-Tuning vs RAG — Design Note (#1273)

Status: **design note** (no implementation yet). Scopes a LoRA fine-tuning demo for the
pretrained causal LMs in NivaraChat (SmolLM-135M, Qwen2.5-0.5B) and a head-to-head
comparison with the RAG pipeline in [RAG.md](RAG.md), using the **same markdown corpus** as
the knowledge source.

Closes the "design note first" ask in `docs/plan/NIVARACHAT-SILK-GAPS.md` (Gap S5, #1273 /
#1283 / #1284). Not a release blocker.

## 1. Goal

Answer one practical question for .NET developers, entirely in C#:

> I have a pile of markdown docs. Should I retrieve from them (RAG), bake them into the
> model (fine-tuning), or both?

Deliverables:

1. A reusable `LoRALinear<T>` module and an adapter save/load path.
2. A `--ft-lora` mode that fine-tunes a small causal LM on Q&A derived from `docs/**/*.md`.
3. A `--ft-vs-rag` mode that scores four configurations on held-out questions with the same
   paired statistics as `--rag-eval` / `--rag-learn`.

## 2. What exists today

| Piece | State |
|---|---|
| `LlamaForCausalLM<TModel,TWeight>` (SmolLM, Qwen) | Core AutoDiff supports backward; sample only does inference |
| `Module<T>`, `Parameter<T>`, `StateDict`/`LoadStateDict`, `AdamW`, `TrainingLoop`, `ModelSerializer` | Available (see [AUTODIFF.md](AUTODIFF.md), [RETRAINING.md](RETRAINING.md)) |
| RAG index, chunker, judge labels, split, paired metrics | Available (`samples/NivaraChat/Helpers/*`) |
| LoRA, freeze mechanism, SFT loss with prompt masking | **Missing** |

`samples/NivaraChat/ARCHITECTURE.md` currently lists "Fine-tuning/LoRA" under *Not doing*;
that line is updated when this lands.

## 3. LoRA module

`LoRALinear<T>` wraps a frozen base `Linear` and adds a low-rank update:

```
y = base(x) + (alpha / r) * ((x @ A) @ B)      A: [in, r]   B: [r, out]
```

- `A` is Kaiming-initialized, **`B` is zero**. At step 0 the adapted model is numerically
  the base model. This is the same parity rule as RAG.md §1: every later delta is
  attributable to learning. Unit test: `max |adapted - base| == 0` before any step.
- Default targets: attention `q_proj` and `v_proj`; optional `k_proj`, `o_proj`, FFN
  projections via a flag. Default `r = 8`, `alpha = 16`.
- Trainable parameters for SmolLM-135M at r=8 on q/v: on the order of 0.3-0.5M (well under
  1% of the base). Exact count is printed by the mode.
- Adapter-only persistence: save/load just the `A`/`B` tensors (small file), plus a
  `merge()` that folds the update into the base weight for inference with no overhead.

### Freezing the base — the main technical risk

Base weights are created with `requiresGrad: true` (`Linear` constructor) and are often BF16
mixed-typed parameters (`RegisterMixedParameters`). Passing only adapter parameters to the
optimizer is not enough: AutoDiff would still compute and hold gradients for every base
weight, which wastes memory and time and defeats the point of LoRA.

Needed before the demo is practical (pick one, decide in review):

- a loader option that creates base weights with `requiresGrad: false`, or
- a `Freeze()` on `Module<T>` that rebuilds leaf tensors without grad tracking.

The BF16 storage / F32 compute path (`LlamaLoader.Load<float, BFloat16>`) should be kept for
the frozen base; adapters stay F32.

## 4. Training data

Same corpus and identity rules as RAG.md §2 and §7: every `docs/**/*.md` plus the sample
README, `docs/plan/` excluded, chunk identity `path#ordinal`, corpus hash recorded.

**Q&A generation.** For each chunk an LLM writes 1-3 question/answer pairs grounded in that
chunk (extends the `--rag-eval --regenerate` judge path). Each pair keeps its source
`ChunkId`. Output is a deterministic TSV/JSONL (sorted, LF, provenance header), refusing to
overwrite without `--force`.

**Split by document, not by question.** RAG.md splits by query hash. For fine-tuning that
leaks: two questions about the same chunk in train and test let the model memorize rather
than generalize. Split on the source file path (FNV-1a with seed mixed in, 80/10/10), so
test questions come from documents the adapter never saw. A second report line covers
"seen-document" questions, which is the case FT is *supposed* to win.

**Format.** Chat template of the target model (`SmollmChatTemplate`, `QwenChatTemplate`),
one example = system + question + answer. Loss is next-token cross-entropy over the
**answer tokens only** (prompt masked). Max sequence length 256-512, configurable.

## 5. Training setup

- Optimizer `AdamW`, lr ~2e-4 (adapter-only), few epochs, early stopping on validation loss,
  best-checkpoint restore via `StateDict`/`LoadStateDict` as in RAG.md §5.
- Gradient accumulation to keep per-step memory small on CPU; batch size 1-4.
- CPU AutoDiff, f32 compute. Silk training for causal LMs does not exist (#1283) and is a
  non-goal here; the CPU path stays the reference.
- Seeded and reproducible: same corpus + seed gives same split, data order and loss curve
  (to a stated tolerance).
- Memory budget table (weights, activations, adapter + optimizer state) is printed at start
  and checked against a `--max-memory` guard.

## 6. The comparison: four arms

All arms answer the same held-out questions with the same decoding settings (greedy, same
max tokens).

| Arm | Model | Context given |
|---|---|---|
| A. Base | pretrained | none |
| B. RAG | pretrained | top-K chunks from the multi-view index |
| C. FT | base + LoRA | none |
| D. FT + RAG | base + LoRA | top-K chunks |

Retrieval for B and D reuses the existing index and fusion (fixed weights, optionally the
learned ones from `--rag-learn`) so the retrieval side is held constant.

**Metrics** (per question, then aggregated):

- Answer correctness against the reference answer, LLM-judged against the source chunk
  (same judge plumbing as `--rag-eval`), plus a cheap lexical check (key-term recall) so the
  result does not depend on the judge alone.
- Grounding: does the answer contradict the source chunk?
- Abstention / hallucination on **out-of-corpus** questions (a small fixed set): does the
  model say it does not know, or invent an answer?
- Cost: tokens in/out, latency per answer, one-time training cost, index build cost.
- Staleness test: edit one doc, then measure RAG after a reindex vs FT without retraining
  (and FT after retraining), reported with wall-clock time.

Statistics follow RAG.md §8: paired sign test and Cohen's d on per-question scores. A
non-significant result is a valid result and is reported as such. Test split scores once.

## 7. Expectations (to be confirmed by the run, not assumed)

Stated up front so the demo is honest:

- A 135M-parameter model with a small rank-8 adapter will not memorize a large doc set.
  Expect RAG to beat FT on factual recall.
- FT is expected to help with answer style, format, terminology and refusing off-topic
  questions; D (FT + RAG) is the most likely overall winner.
- RAG wins on freshness: a doc edit is live after reindexing, while FT needs retraining.
- Qwen2.5-0.5B (about 4x larger) is a stretch target; CPU training time may make it
  impractical, and the mode reports measured cost either way.

If the measured results contradict these, the report says so.

## 8. CLI surface (proposed)

```
# 1. Generate grounded Q&A from docs (needs an LLM, e.g. --ollama)
dotnet run --project samples/NivaraChat -- --ft-data --regenerate --ollama --force

# 2. Fine-tune a LoRA adapter (CPU)
dotnet run --project samples/NivaraChat -- --ft-lora --model smollm --rank 8 --epochs 3

# 3. Compare Base / RAG / FT / FT+RAG on held-out questions
dotnet run --project samples/NivaraChat -- --ft-vs-rag --model smollm --report ft-vs-rag.md
```

Key flags: `--model smollm|qwen`, `--rank`, `--alpha`, `--targets q,v[,k,o,ffn]`, `--lr`,
`--epochs`, `--batch-size`, `--max-len`, `--seed`, `--adapter <path>`, `--merge`,
`--docs-dir`, `--report`, `--force`.

## 9. Proposed file layout

- `src/Nivara/AutoDiff/Nn/LoRALinear.cs` — module, merge, adapter save/load (core, reusable)
- `samples/NivaraChat/FineTuning/` — data builder, SFT trainer, `FtLoraMode`, `FtVsRagMode`
- `tests/Nivara.Tests/AutoDiff/LoRALinearTests.cs` — step-0 parity, gradient check vs
  finite differences, merge equivalence, frozen-base has no grad
- `docs/LORA.md` (this file) updated with measured results

## 10. Test plan

1. **Parity:** adapted == base at init (exact), after `merge()` equals unmerged output within 1e-5.
2. **Grad check:** adapter gradients match finite differences on a tiny Llama.
3. **Frozen base:** base weights unchanged after N steps; no base gradients allocated.
4. **Overfit sanity:** a 10-example set drops loss near zero within a few epochs.
5. **Determinism:** same seed reproduces split and first-epoch loss.
6. **Round trip:** save adapter, reload on fresh base, identical logits.

## 11. Open questions

- Core (`src/Nivara`) or sample-scoped `LoRALinear`? Core is more reusable; sample-scoped
  matches how `LlamaForCausalLM` itself lives today.
- Which LLM writes the Q&A: local Ollama, or the in-repo Qwen so the whole demo runs
  offline? Offline is friendlier but the data quality is lower.
- Freeze mechanism (see §3).
- Whether to also expose adapters through the `IChatClient` wrappers so any NivaraChat mode
  can serve a tuned model.

## 12. Non-goals

- Silk / GPU training for causal LMs (#1283), bf16/fp16 training, QLoRA and quantization.
- Full-parameter fine-tuning and general-purpose LLM training beyond this minimal SFT loop
  (#1284).
- Replacing RAG: the point is a measured comparison, not a verdict in advance.

## 13. References

- [RAG.md](RAG.md) — corpus, chunking, judge labels, split, paired metrics
- [AUTODIFF.md](AUTODIFF.md), [RETRAINING.md](RETRAINING.md) — training stack, checkpoints
- [SMOLLM.md](SMOLLM.md), [QWEN.md](QWEN.md) — the target models
- `samples/Nivara.Samples/LlamaForCausalLM.cs`, `src/Nivara/AutoDiff/Nn/Linear.cs`
- `samples/NivaraFineTuning/` — existing DistilBERT fine-tuning sample (classification head)
