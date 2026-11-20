# LoRA Fine-Tuning vs RAG — Design Note (#1273)

Status: **design note** (no implementation yet). Scopes a LoRA fine-tuning demo for the
pretrained causal LMs in NivaraChat (SmolLM-135M, Qwen2.5-0.5B) and a head-to-head
comparison with the RAG pipeline in [RAG.md](RAG.md), using the **same markdown corpus** as
the knowledge source.

Consolidates the design asks in `docs/plan/NIVARACHAT-SILK-GAPS.md` (Gap S5, #1273 /
#1283 / #1284). Full-parameter LLM fine-tuning is intentionally not scoped (see §12);
the supported SFT path is LoRA. Not a release blocker.

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
| Silk causal-LM trainer (SmolLM/Qwen), `rms_norm_backward` | **Missing** (tracked in §14, #1283) |

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

**Format.** Chat template of the target model (`SmollmChatTemplate`, `QwenChatTemplate`,
both `internal static` renderers in the sample; the fine-tuning code lives in the same
assembly so this is fine), one example = system + question + answer. Loss is next-token
cross-entropy over the **answer tokens only** (prompt masked). Max sequence length 256-512,
configurable.

## 5. Training setup

- Optimizer `AdamW`, lr ~2e-4 (adapter-only), few epochs, early stopping on validation loss,
  best-checkpoint restore via `StateDict`/`LoadStateDict` as in RAG.md §5.
- Gradient accumulation to keep per-step memory small on CPU; batch size 1-4.
- CPU AutoDiff, f32 compute. Silk training for causal LMs is tracked in §14 (#1283) and is
  a non-goal for this CPU demo; the CPU path stays the reference.
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

- Full-parameter fine-tuning of causal LMs — infeasible as a shipped CPU workflow (memory +
  wall-clock); out of scope. LoRA is the supported SFT path. Tracked separately as an
  opt-in experiment only, not a mode.
- QLoRA / quantization.
- Silk / GPU training for causal LMs in v1 (see §14); CPU path stays the reference.
- bf16/fp16 *training* (CPU mixed storage is inference-only; Silk training is f32 only).
- Replacing RAG: the point is a measured comparison, not a verdict in advance.

## 13. References

- [RAG.md](RAG.md) — corpus, chunking, judge labels, split, paired metrics
- [AUTODIFF.md](AUTODIFF.md), [RETRAINING.md](RETRAINING.md) — training stack, checkpoints
- [SMOLLM.md](SMOLLM.md), [QWEN.md](QWEN.md) — the target models
- `samples/Nivara.Samples/LlamaForCausalLM.cs` — model, loader, mixed-dtype guard
- `src/Nivara/AutoDiff/Nn/Linear.cs`
- `samples/NivaraFineTuning/` — existing DistilBERT fine-tuning sample (classification head)
- `docs/plan/NIVARAINFERENCE-SILK-GAPS.md` — umbrella for Silk gaps (N5/N6)

## 14. Silk counterpart (future, from #1283)

LoRA on Silk is *not* a v1 deliverable: there is no Silk trainer for causal LMs yet, and a
Silk LoRA path would reuse the same device trainer with frozen base weights plus adapter
buffers. This section consolidates the Silk/causal-LM scoping that was previously a separate note
so there is one source of truth.

**What exists today (Silk).** Trainers: `SilkDistilBertTrainer` (+ BF16 true, FP16),
`SilkLayaHeadTrainer` (+ BF16 narrow), `SilkModernBertHeadTrainer`, `SilkTinyShakespeareTrainer`.
Inference-only runners with no backward/Adam: `SilkSmolLMGpuRunner`, `SilkQwenGpuRunner`,
`SilkMobileNetV2GpuRunner`, `SilkResNet18GpuRunner`. Precedent for a causal-LM-on-Silk
trainer: `SilkTinyShakespeareTrainer` + `BatchedTransformer` (device forward/backward over a
transformer block, Adam step, export back to CPU weights), with parity tests in
`tests/Nivara.Tests/Gpu/SilkTinyShakespeareTrainerTests.cs`.

**Kernel inventory.** Available: `gemm_row4_*`, `col_sum`, `relu`/`relu_backward`,
`gelu`/`gelu_backward`, `layer_norm`/`layer_norm_backward`, `rms_norm` (forward only),
`silu`/`swiglu`, `causal_gqa_attention`, `causal_attention_backward_prep`,
`attention_backward_dq`/`attention_backward_dkv`, `rotary`, `conv2d` +
`conv2d_backward_{weight,input,bias}`, `embedding_*`, `softmax_cross_entropy_backward`,
`adam_update`/`adam_update_bf16`. **Missing for causal LMs:** `rms_norm_backward`
(Tier 1/2). Vision (Tier 3) additionally needs `batchnorm_backward`, `maxpool2d_backward`,
`adaptive_avgpool2d_backward`.

**CPU reference.** `LlamaForCausalLM<float,float>` (existing) via `GradientUtils.Grad()`.
The device trainer must match it; tied embeddings mean the tokEmb buffer is shared with the
LM head and gradients accumulate into the same buffer.

**Tiers.**
- Tier 1: `SilkSmolLMTrainer` (sealed, `IDisposable`) over the Llama family (RoPE, GQA,
  SwiGLU, RMSNorm, tied embeddings); new `SilkSmolLMWeights.cs`, `rms_norm_backward` kernel
  + launcher. Single-sequence steps first; batching deferred.
- Tier 2: Qwen2.5-0.5B reusing Tier 1 with config differences (GQA with qkv bias, larger
  vocab/hidden, different RoPE theta); thin subclass or shared trainer parameterized by
  config + bias flag.
- Tier 3: vision (MobileNetV2, ResNet-18) head-only first (freeze backbone, train
  classifier head), mirroring `SilkModernBertHeadTrainer`; full backbone deferred.

**Precision lanes.** Training f32 only (release constraint, consistent with `docs/SILK.md` §13
and the #1274 text-classifier decision). Inference keeps existing lanes (BF16 widen for
SmolLM/Qwen). Narrow (BF16 weights / F32 compute) training is out of scope unless #1149
lands.

**Parity gate** (mirror `SilkTinyShakespeareTrainerTests`): seeded tiny fixture (small
hidden size, few layers, small vocab); forward logits within 1e-4; gradients within 1e-3
(max-abs) for all parameter groups; one Adam step within 1e-2; checkpoint export → reload
round trip; seeded weights (avoid the unseeded-baseline problem of the first Laya gate).
Skip gracefully when no OpenCL device.

**Non-obvious points.**
- Tied embeddings: tokEmb buffer shared with the head projection; gradients accumulate.
- RoPE is position-dependent: recompute cos/sin per sequence length, save for backward.
- Single sequence per step (`hidden` is `[seqLen, hidden]`); batching deferred.
- For causal LMs the trainer owns the cross-entropy loss internally (like
  `SilkTinyShakespeareTrainer.Backward(targets, scale)`); head-only vision takes
  caller-supplied upstream gradients.
- No activation-level BF16 unless #1149 lands first.

**Open decisions.** Shared `SilkCausalLmTrainer` vs. separate `SilkSmolLMTrainer`/
`SilkQwenTrainer`; CPU AutoDiff as the sole parity reference for v1; primitives-first
surface (no convenience `Train(...)` loop in v1, matching DistilBERT); vision head-only first;
whether a `Conv2d`/`BatchNorm` AutoDiff module is needed in `Nivara.Samples` for a vision
CPU parity reference (recommended, separate item).

**Relation to LoRA.** A Silk LoRA path would build on Tier 1/2 above: frozen base weights
plus adapter buffers, with the freeze mechanism from §3. It is not a v1 deliverable and the
CPU LoRA path (§3-§10) remains the reference and the shipped demo.
