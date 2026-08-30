# Add SmolLM2‑135M as the 5th HuggingFace model (causal LM) in NivaraInference

## Goal & philosophy
The point of this example is not just to run a model — it is to **surface gaps in the core
Nivara library and fix them**, assess performance, and showcase what pure‑managed .NET can do.
SmolLM2‑135M (`HuggingFaceTB/SmolLM2-135M`) is deliberately chosen because it is the smallest
member of the TinyLlama/SmolLM class and yet forces every missing building block:

- **Genuinely BF16‑trained** (HF "Precision: bfloat16") → exercises the lossless
  `SafeTensorsLoader.Read<BFloat16>` path (`docs/BFLOAT16.md`).
- **Decoder‑only LLaMA architecture**: RoPE, GQA, SwiGLU, RMSNorm, weight tying, SentencePiece
  (`LlamaTokenizer`). None of RoPE/GQA/SwiGLU exist in core today → these become core improvements.
- Size ≈ 135M params (~270 MB BF16, ~540 MB F32) — comparable to DistilBERT, fits the sample.

Scope now: **inference + autoregressive generation**. **Training is a follow‑up** (see last section).

## Why not the other candidates
- `TinyLlama-1.1B` — true LLaMA but ~1.1 GB BF16 / 2.2 GB F32: too big for the sample's download + CPU story.
- `GPT-2` — reuses MicroGpt's GPT/Karpathy blocks almost 1:1, but is NOT a TinyLlama/SmolLM-class
  model and is not BF16‑native; it would surface far fewer core gaps, contradicting the goal.

## SmolLM2‑135M architecture (from config + LlamaForCausalLM)
- `vocab_size=49152`, `hidden_size=576`, `num_hidden_layers=30`, `num_attention_heads=9`,
  `num_key_value_heads=3`, `head_dim=64`, `intermediate_size=1536`, `rms_norm_eps=1e-5`,
  `rope_theta=10000`, `rope_interleaved=false`, `tie_word_embeddings=true`, `max_position_embeddings=2048`.
- Per layer: `input_layernorm` (RMSNorm) → GQA attention (Q/K/V proj → RoPE → causal attn with
  KV‑head repeat 3→9 → `o_proj`) → residual → `post_attention_layernorm` (RMSNorm) → SwiGLU FFN
  (`silu(gate_proj(x)) * up_proj(x)` → `down_proj`) → residual. Final `model.norm` (RMSNorm) →
  weight‑tied `lm_head`.

## Repo layout
- New model class: `samples/Nivara.Samples/SmolLmModel.cs`
  (`Nivara.Samples` namespace, mirrors `DistilBertModel.cs`). Generic `SmolLMForCausalLM<T>`.
- New tokenizer wiring: sample‑side helper using `Microsoft.ML.Tokenizers.LlamaTokenizer`
  (`LlamaTokenizer.Create(stream)` on the repo's `tokenizer.model`). The package is already a
  sample dependency (used for `BertTokenizer`).
- New dispatch + modes in `samples/NivaraInference/Program.cs` (modelType key `smollm`).
- New Python reference generator: `samples/NivaraInference/Python/smollm_compare.py`
  (mirror `distilbert_sst_compare.py`).
- README: add model to the supported‑models table + quick‑start + narrow‑precision + capabilities.

## Core Nivara improvements this example drives (the real deliverable)
Each gap below is found *because* we implement SmolLM2; fix in `src/Nivara/AutoDiff` and add tests.
1. **RoPE (RotaryEmbedding)** — new op/helper in `AutoDiff/Nn` + `ReverseGradOperations`.
   Compute cos/sin tables (θ=10000, non‑interleaved / GPT‑NeoX style) and rotate Q/K per head.
   First build it model‑side with existing `Slice`/`Multiply`/`Add` to validate, then promote to a
   vectorizable core op. (Confirm `ReverseGradOperations.Slice`/`Concat` exist; if not, add them.)
2. **GQA** — extend `MultiheadAttention<T>` + `ReverseGradOperations.MultiHeadAttention` + the fused
   `AttentionKernels` to accept `numKeyValueHeads` (repeat‑interleave K/V to match Q heads). Currently
   MHA‑only. This is the highest‑value core fix and lets future LLaMA‑class models reuse it.
3. **SiLU / Swish activation** — new core op in `Activation` (+ gradient path for the training follow‑up).
   Needed for SwiGLU; `SwiGLU(x) = SiLU(W_gate·x) ⊙ (W_up·x)`, then `W_down`.
4. **SwiGLU FFN module** — compose `SiLU` + `Linear` (or a fused op) in `AutoDiff/Nn`.
5. **(Optional, perf)** targeted BF16 matmul widen for the causal‑LM path — DEFERRED (separate from this
   plan per user: the `docs/BFLOAT16-TRANSFORMER.md` widen‑SIMD work is later). Note in validation that
   BF16 generation will be slow (~issue #363); that itself is a finding to feed into the widen work.

## Model implementation (`SmolLMForCausalLM<T>`)
- **LoadWeights**: map LlamaForCausalLM safetensors keys → modules. Inspect
  `samples/Nivara.Samples/DistilBertModel.cs` for the exact `Linear` weight layout convention
  (`[out,in]` vs `[in,out]`) and mirror it. Key map:
  - `model.embed_tokens.weight [vocab,hidden]` → `wte` (`Embedding<T>`)
  - `model.layers.{i}.input_layernorm.weight` / `post_attention_layernorm.weight` → `RMSNorm<T>` (eps 1e‑5)
  - `self_attn.q_proj/k_proj/v_proj/o_proj.weight` → `Linear<T>` (bias=false). k/v are `[192,576]`.
  - `mlp.gate_proj/up_proj/down_proj.weight` → `Linear<T>`.
  - `model.norm.weight` → final `RMSNorm<T>`.
  - `lm_head.weight` → reuse `wte` weight (transposed) since `tie_word_embeddings=true`
    (MicroGpt's weight‑tying pattern).
- **Forward (prefill, full sequence)** for `compare`/`benchmark`: embed token ids (exact `int[]`,
  no narrow dtype), run 30 layers (RMSNorm → GQA+RoPE causal attn → residual → RMSNorm → SwiGLU →
  residual), final RMSNorm, lm_head → logits `[seq, vocab]`. Read logits at last position.
- **Generate (autoregressive, KV cache)** for interactive/demo: reuse MicroGpt's per‑position KV‑cache
  pattern (`keys[layer]`/`values[layer]` lists). `Sampler<T>` (already exists) provides temp/top‑k decode.
  Expose `Generate(prompt, maxNewTokens, temperature, topK)`.
- **Token‑ID correctness**: pass token ids as exact `int[]` to `Embedding<T>.Forward(int[])` (already
  solved, `docs/BFLOAT16.md`). RoPE positions are exact ints. No new risk.

## Tokenizer (sample side)
- `hf download HuggingFaceTB/SmolLM2-135M tokenizer.model config.json` → `samples/data/smollm2/`.
- Load via `LlamaTokenizer.Create(File.OpenRead(tokenizer.model))`. Encode prompt → `int[]` ids
  (addBos=true per SmolLM convention); decode generated ids → text.
- Fallback: if `LlamaTokenizer` needs the `.model` (SentencePiece) file specifically, document the exact
  `hf download` flag in README.

## Program.cs integration
- New modelType `"smollm"`. Extend `--precision f32|bf16|fp16` (already parsed) — bf16 reads the
  on‑disk BF16 weights via `Read<BFloat16>` (native/lossless); f32 widens BF16→F32; fp16 widens
  BF16→F32→Half (note this nuance in README since SmolLM2 ships BF16, unlike the F32‑shipped others).
- Modes: interactive generate REPL (default), `benchmark` (3 warmup + 10 timed, avg/min/max, weight MB),
  `compare` (diff vs Python HF reference: argmax agreement + max abs logit diff), `--precision` honored.
- Reuse the existing `tensorsBf16`/`tensorsHalf`/`tensorsF32` load branches.

## Python reference (`smollm_compare.py`)
- Load `HuggingFaceTB/SmolLM2-135M` with `transformers`, run a fixed prompt (e.g. "Gravity is"), save
  logits `[vocab]` (and optionally the last‑position hidden state) to
  `samples/data/compare_smollm2_py.bin`. Mirror `distilbert_sst_compare.py`.

## Validation
- `dotnet build Nivara.slnx` clean; new `BFloat16`/`SiLU`/`RoPE`/`GQA` unit tests.
- `smollm compare`: argmax agreement **expect 8/8** (token‑id safe), max abs logit diff vs HF ref
  ~`1e-6` in F32, and a small genuine BF16 diff (~0.3, like DistilBERT SST‑2) — must preserve prediction.
- Interactive `smollm`: generated text is coherent/continues the prompt (sanity, not exact match).
- `smollm benchmark` reports timings for f32/bf16; documents BF16 slowness as a finding for the widen work.
- Update `samples/NivaraInference/README.md` (supported‑models table, quick start, narrow‑precision
  results, capabilities matrix).

## Open questions / follow‑ups
- **Training (deferred):** fine‑tune SmolLM2‑135M via `TrainingLoop`/`GradientUtils.Grad()` + AdamW;
  requires backward through RoPE/GQA/SwiGLU (gradients added in step 1–4) and a full‑sequence
  (no KV cache) training forward. Plan as a separate change after inference lands.
- Confirm `LlamaTokenizer` ships in the Microsoft.ML.Tokenizers version pinned in the sample; if only
  `GPT2Tokenizer`/`BertTokenizer` are available, add the package or fall back to a bundled tokenizer.
- Decide final command name (`smollm` vs `smollm2_135m`); keep consistent with existing short keys.
