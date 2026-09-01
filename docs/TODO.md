# Plan — Phase 2: SmolLM-135M causal-LM inference in `samples/NivaraInference`

Branch: `khurram/smollm-2` · Base `main`

## Problem

Add a working SmolLM-135M (Llama-family causal LM) inference path to
`samples/NivaraInference` — the 5th model. The BF16-native checkpoint is already
downloaded (`samples/data/smollm-135m/`) and a PyTorch reference generator exists
(`samples/NivaraInference/Python/smollm_generate_reference.py`, fixture
`samples/data/compare_smollm_py.bin` + `compare_smollm_logits_py.bin`). Running greedy
generation in Nivara requires several engine/ops that don't exist yet: standalone
`RMSNorm` module with affine gamma, `Activation.Silu`, RoPE (RotaryEmbedding), GQA
(9 Q / 3 KV heads), gated-SiLU FFN, tied-embedding LM head, a GPT-2 byte-level BPE
tokenizer, and a greedy generation loop. The README also mislabels the SmolLM tokenizer
as SentencePiece (it is GPT-2 byte-level BPE).

Phase 1 (`NivaraPrimitives.UseWidenSimd`, `WidenPrimitives`, `NarrowFloatKernels`,
`TensorsHelper.MultiplyCore<T>`) shipped and is the SIMD surface the BF16 weights flow
through; this phase routes the model through it but keeps the toggle off (scalar) by
default — the A/B/microbenchmark switch is Phase 3.

Scope is **only SmolLM-135M**. No retrofit of the existing 4 models onto widen kernels.
No true shared-KV GQA kernel (KV-repeat per issue #367).

## SmolLM-135M config (from `samples/data/smollm-135m/config.json`)

hidden_size=576, num_attention_heads=9, num_key_value_heads=3 (GQA), num_hidden_layers=30,
intermediate_size=1536, vocab_size=49152, max_position_embeddings=2048, rope_theta=10000,
rms_norm_eps=1e-5, hidden_act=silu, attention_bias=false, mlp_bias=false,
tie_word_embeddings=true, bos=1, eos=2, pad=2.

Weight keys (HF Llama convention): `model.embed_tokens.weight`,
`model.layers.{i}.input_layernorm.weight`, `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`,
`model.layers.{i}.mlp.{gate,up,down}_proj.weight`,
`model.layers.{i}.post_attention_layernorm.weight`, `model.norm.weight`, plus
`lm_head.weight` only if present (tied → reuse embed).

## Blast radius

- **New ops/modules in `src/Nivara/AutoDiff/Nn/`:** `RMSNorm.cs` (new module, affine gamma),
  `Activation.cs` (add `Silu`), `RotaryEmbedding.cs` (new), `LlamaAttention.cs` / `LlamaBlock.cs`
  / `LlamaForCausalLM.cs` (new), mirror JVP ops in `Operations/ForwardGradOperations.cs` and
  VJP ops in `Operations/ReverseGradOperations.cs`. These are additive; no existing modules are
  modified (the Ops files gain new methods only).
- **`src/Nivara/AutoDiff/Nn/Activation.cs`:** additive `Silu`; no existing activations change.
- **`samples/NivaraInference/Program.cs`:** new `smollm` case in the model-type switch + a
  `generate` mode; existing 4 model modes must still build/run unchanged — no behavior drift.
- **`samples/NivaraInference/README.md`:** new SmolLM usage lines + a "gaps found & fixed"
  retrospective; correct the tokenizer mislabel (GPT-2 byte-level BPE, not SentencePiece).
- **`tests/Nivara.Tests/AutoDiff/`:** new `RMSNormTests.cs`, `SiluTests.cs`,
  `RotaryEmbeddingTests.cs`, `GqaKvRepeatTests.cs`, `LlamaBlockTests.cs` — additive.
- **Depending symbols / tests:** existing AutoDiff tests must remain green (no contract change).
  The `Debug.Assert` in `ComputationGraph.AddNode` (inference-default guard) must NOT fire on
  the generation path — that is the inference-default regression test.
- **Downstream impact:** none outside `src/Nivara` + the sample project + its tests.

## Design decisions

1. **Dedicated Llama modules, not reuse of Bert paths.** `MultiheadAttention<T>` /
   `TransformerBlock<T>` are BERT-encoder shaped (no RoPE, no GQA, no gated SiLU). Build a
   purpose-built `LlamaForCausalLM<T>`; reuse shared pieces (`Linear`, `Embedding`,
   `MatMulTransposedB`, softmax, `TensorsHelper.MultiplyCore`).
2. **GQA via KV-repeat** (per issue #367): split Q into 9 heads, repeat 3 KV heads ×3 so all
   9 align, run the same fused head loop. Deterministic; KV is small relative to weights.
3. **RoPE as a fused op.** `RotaryEmbedding<T>` precomputes cos/sin `[maxSeqLen, headDim/2]`
   from `inv_freq = theta^{-2i/dim}`, rotates Q/K pairwise per position. Single graph op with
   VJP + JVP; headDim=64 (even) so pairwise rotate is clean.
4. **Tied LM head.** Reuse the input `Embedding<T>` weight parameter for the output
   `hidden @ embedWeight^T` projection when `tie_word_embeddings`; loader binds the shared
   `[49152,576]` embedding once.
5. **Inference-default discipline.** Generation runs outside any `GradientUtils.Grad()` scope
   with `requiresGrad:false` → no graph nodes; `Debug.Assert` in `ComputationGraph.AddNode`
   is the guard. The `smollm` path never enters `Grad()`.
6. **Tokenizer.** SmolLM uses a **GPT-2 byte-level BPE** tokenizer (`tokenizer_class:
   GPT2Tokenizer`; `vocab.json` + `merges.txt`, `add_prefix_space:false`), not SentencePiece
   (README currently wrong). **Verified against the installed `Microsoft.ML.Tokenizers`
   3.0.0-preview.26160.2** (already a transitive dep of `Nivara.Samples`):
   - Exposes `BpeTokenizer.Create(vocabFile, mergesFile, preTokenizer, normalizer, specialTokens, ...)`,
     `RobertaPreTokenizer`, `CompositePreTokenizer`, `BpeOptions`.
   - Exposes **no byte-level normalizer** (only `LowerCase/UpperCase/SentencePiece`) and **no
     byte-level pretokenizer**. HF's SmolLM tokenizer applies a byte-level normalizer +
     byte-level pretokenizer that produces the `Ġ`-style byte-mapped token keys. Because
     Microsoft.ML.Tokenizers operates on raw Unicode while SmolLM's vocab keys are byte-encoded,
     the `BpeTokenizer` path is **unlikely to reproduce SmolLM's exact token IDs** (space → `Ġ`
     mismatch alone breaks parity).
   - **Decision (empirical, unit-7 gate):** Unit 7 first tries `BpeTokenizer.Create` +
     `RobertaPreTokenizer` + `specialTokens` and diffs the prompt's token IDs against the
     `compare_smollm_py.bin` prefix. If (as expected) byte-level encoding diverges, fall back to
     a **small byte-level BPE reader** in the sample (`vocab.json`+`merges.txt`, HF byte↔unicode
     map, byte-level regex split, `add_prefix_space=false`). Whichever reproduces the reference
     token ids wins. Prompt is plain ASCII, so a fallback is small and low-risk.
   - This is flagged because tokenizer parity gates the argmax-agreement acceptance criterion.

## Proposed changes (incremental units)

- **Unit 1 — `RMSNorm<T>` module.** New `Module<T>` with affine gamma + `eps` (default 1e-5),
  forward + backward using `RMSNormKernel`, `LoadStateDict("Weight")`. Test: forward vs a
  scalar reference.
- **Unit 2 — `Activation.Silu` / `ReverseGradOperations.Silu`.** Elementwise `x*sigmoid(x)`,
  forward + VJP + JVP. Test: forward/back vs scalar.
- **Unit 3 — `RotaryEmbedding<T>`.** Precompute cos/sin from `rope_theta`; rotate Q/K `[L,
  headDim]` per position; forward + VJP + JVP; inference path graph-free. Test: pairwise
  rotation matches HF formula.
- **Unit 4 — Llama attention + GQA KV-repeat.** `LlamaCausalAttention<T>`: QKV proj → RoPE →
  split 9 Q / repeat 3 KV ×3 → fused causal head loop → o_proj. Test: GQA repeat ==
  hand-computed grouped reference.
- **Unit 5 — `LlamaDecoderBlock<T>` + gated SiLU FFN.** Wire Units 1–4 with residual adds;
  MLP = `silu(gate(x))*up(x)` then `down(x)`. Test: block forward vs scalar reference.
- **Unit 6 — `LlamaForCausalLM<T>` + tied LM head.** embed → 30 blocks → final RMSNorm →
  `hidden @ embed^T` logits. Loader binds all keys (StateDictLoader-style), reusing embed weight
  for the tied head. Test: logits shape `[L,49152]`, no double-count of tied head.
- **Unit 7 — BPE tokenizer binding.** Ordered (a)→(b): try Microsoft.BpeTokenizer, verify
  token ids vs reference prefix, fall back to byte-level BPE reader (see design decision 6).
- **Unit 8 — `smollm` mode in `Program.cs` + greedy generation.** Add `smollm` case + `generate`
  mode: decode up to 32 new tokens from the fixed prompt, print token ids + decoded text, and
  compare token-id stream + final-position logits vs the PyTorch fixtures. Run for F32 and BF16.
- **Unit 9 — README gap-retrospective + tokenizer correction.** Update
  `samples/NivaraInference/README.md`: `smollm` usage lines, "gaps found & fixed" section for
  this 5th model (RMSNorm+gamma, SiLU, gated SiLU FFN, RoPE, GQA KV-repeat, tied-embedding
  head, byte-level BPE tokenizer), fix the SentencePiece mislabel, note numeric-precision
  caveats (BF16 greedy near-tie flips vs PyTorch BF16).

## Acceptance criteria

- `dotnet run --project samples/NivaraInference -c Release -- smollm --precision bf16 generate`
  loads SmolLM-135M and prints a generated sequence.
- Greedy token IDs match `compare_smollm_py.bin` (argmax agreement) for the fixed prompt
  `"The capital of France is"`, max_new_tokens=32.
- Final-position logits vs `compare_smollm_logits_py.bin` within a documented BF16/MHA
  tolerance (numeric precision diff, not bit-exact).
- Runs in F32 and BF16 compute; BF16 is the native on-disk path.
- No `GradientUtils.Grad()` scope on the inference path; no graph nodes built during generation.
- Existing 4 model modes still build & run unchanged.

## Verification steps

- Build `dotnet build Nivara.slnx` after each unit (fast, no test run).
- `dotnet run --project samples/NivaraInference -c Release -- smollm --precision bf16 generate`
  for the end-to-end accept check.
- Focused NUnit runs for Units 1–5 requested from the user (per repo guidance, `dotnet test`
  needs explicit confirmation).

## Planned commits

1. `docs: plan Phase 2 SmolLM-135M causal-LM in TODO.md`
2. `feat: add RMSNorm module with affine gamma` (Unit 1)
3. `feat: add Silu activation (forward + VJP + JVP)` (Unit 2)
4. `feat: add RotaryEmbedding (RoPE) op` (Unit 3)
5. `feat: add Llama attention with GQA KV-repeat` (Unit 4)
6. `feat: add Llama decoder block with gated SiLU FFN` (Unit 5)
7. `feat: add LlamaForCausalLM with tied LM head` (Unit 6)
8. `feat: add GPT-2 byte-level BPE tokenizer binding for SmolLM` (Unit 7)
9. `feat: add smollm generate mode + greedy loop` (Unit 8)
10. `docs: SmolLM gaps-found retrospective + tokenizer correction in NivaraInference README` (Unit 9)

## GitHub issues log

- [x] #368 — causal-LM/llama-family ops (RoPE, SiLU FFN, causal mask, greedy generation, tied
  embedding LM head) — **implemented this phase** (RMSNorm+gamma, SiLU, RoPE, GQA attention,
  gated SiLU FFN, tied-embedding LM head, byte-level BPE tokenizer, greedy generation loop);
  no new gap surfaced beyond the two findings below.
- [x] #367 — GQA support (KV-repeat recommended) — **approach adopted and implemented** via
  `GqaRepeatKV`/`GradKernels.HeadRepeat`; shared-KV kernel is a Phase 3+ follow-up.
- [x] *(new)* — the Microsoft.ML.Tokenizers BPE path **cannot** reproduce SmolLM byte-level
  token IDs (empirically confirmed in Unit 7: every MS pre-tokenizer variant returns
  `[504, 29721, 1714, 33488, 271]` vs. the expected `[504, 3575, 282, 4649, 314]`), so a
  sample-local byte-level BPE reader (`Gpt2BpeTokenizer`) was implemented. An upstream
  MS.ML.Tokenizers feature request (byte-level normalizer/pretokenizer) could be filed later;
  not blocking this phase.
- [x] *(Unit 8 finding)* — RoPE layout bug: `RotaryEmbedding` implemented the GPT-NeoX-style
  **interleaved-pairwise** rotation, but Llama/SmolLM uses the **`rotate_half` (half-split)**
  layout (HF `LlamaRotaryEmbedding.apply_rotary_pos_emb`). A wrong layout rotated Q/K
  incorrectly → logits near-anti-correlated with the reference (cosine −0.92). Fixed
  `GradKernels.RotaryForward`/`RotaryBackward` to half-split and updated
  `RotaryEmbeddingTests`; end-to-end F32 went from 4/32 → 30/32 generated-token match with
  byte-identical decoded text.
- [x] *(Unit 8 finding)* — BF16 generation needs the Phase-1 SIMD widen path: with
  `NivaraPrimitives.UseWidenSimd` off, BFloat16 matmul falls to scalar `TensorPrimitives.Dot`
  (~100× slower), making a 32-token BF16 generation impractical (30+ min). `Program.cs`'s
  `smollm` mode now enables `UseWidenSimd` for the narrow (BFloat16/Half) runs (restoring the
  prior global value afterward so other modes are unaffected). BF16 generation then completes
  in ~17 s with 22/32 generated-token match and final-logits cosine 0.94.
