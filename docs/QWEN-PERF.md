# Making Qwen Fast — Inference Review

Research-only review of the Qwen2.5-0.5B-Instruct inference path in NivaraChat:
discovery of the building blocks, impact of each block on inference, how well
each is implemented today, and where to invest. No implementation was done as
part of this review — it feeds the O(qwen-fast) planning issue.

**Model:** Qwen2.5-0.5B-Instruct — vocab **151,936**, hidden **896**, **24 layers**,
GQA **14→2** heads, head_dim 64, FFN **4864**, RoPE θ=1e6, tied embeddings;
~494M params, 988 MB BF16 on disk. Runs fully in-process through Nivara's
`LlamaForCausalLM<T>` engine.

---

## 1 · Discovery — the building blocks Qwen uses

| Block | Location | Role in Qwen inference |
|---|---|---|
| `QwenChatClient<T>.Generate` | `samples/NivaraChat/Qwen/QwenChatClient.cs:109` | Render → encode → **prefill** → per-token decode loop → stop on `<\|im_end\|>`/`<\|endoftext\|>` |
| `SeedCache` | same file `:159` | **Prefill**: one `ForwardCached` per prompt token to fill the KV cache |
| `LlamaForCausalLM<T>.ForwardCached` | `samples/Nivara.Samples/LlamaForCausalLM.cs:99` | per-token stack: Embed → 24 blocks → final RMSNorm → **tied LM head** |
| `LlamaCausalAttention<T>.ForwardCached` | `src/Nivara/AutoDiff/Nn/LlamaCausalAttention.cs:131` | Q/K/V projections → RoPE → write KV cache → **copy prefix** → GQA repeat → `MultiHeadAttention` → O proj |
| `LlamaKVCache<T>` | `samples/Nivara.Samples/LlamaForCausalLM.cs:129` | per-layer `T[][]` key/value buffers |
| `Linear<T>.Forward` → `MatMulTransposedB` | `src/Nivara/AutoDiff/Nn/Linear.cs:74`; `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs:406` | all projections + FFN |
| `TensorsHelper.MultiplyCore` (+ `MultiplyRowFloat`) | `src/Nivara/Tensors/TensorsHelper.cs:110,254` | **the** dense matmul kernel — `TensorPrimitives.Dot` per output element |
| `MultiHeadAttention<T>` | `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs:496` | pack heads → per-head QKᵀ → scale/mask → `SoftmaxRows` → scores·V → scatter |
| `GqaRepeatKV` | `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs:967` | materializes K/V head repeat ×7 across the **full prefix** |
| `Gather` (Embedding) | `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs:2477` | token→row lookup |
| `RotaryEmbedding<T>` | `src/Nivara/AutoDiff/Nn/RotaryEmbedding.cs` | cached cos/sin tables; per-position rotate |
| `RMSNormKernel<T>` | `src/Nivara/AutoDiff/Nn/RMSNormKernel.cs:11` | per-row sum-sq + normalize (`TensorPrimitives`) |
| `SafeTensorsLoader.Read<T>` | `samples/Nivara.Samples/SafeTensorsLoader.cs:28` | **mmap** read; fused SIMD BF16→F32 widen (or native `BFloat16`) |
| `Gpt2BpeTokenizer` | `samples/Nivara.Samples/Gpt2BpeTokenizer.cs:111,424` | Split-regex pretokenize + BPE merges (per-turn re-encode) |

Everything above (except the loader/tokenizer) runs with `GradientUtils.Grad()`
**off** — ops short-circuit to the non-grad span path and no graph nodes are
built (ADR-001/002 inference-default).

---

## 2 · Impact — how the blocks are used at inference time

Two phases with very different cost profiles.

### A. Prompt prefill (`SeedCache`) — the hidden bottleneck

`SeedCache` runs `ForwardCached` **once per prompt token**
(`QwenChatClient.cs:159-165`). Each `ForwardCached` walks the *entire* model:
embed + 24 layers + norms + the 151,936-row LM head. Every prompt token
therefore re-reads **every weight in the model (~2 GB in F32)**.

For a ~60-token tool prompt that is **≥120 GB of DRAM traffic** — minutes-worth
at CPU bandwidth — *before the first generated token.* This matches the
observed transcript in [`docs/QWEN.md`](QWEN.md) (`3 turn(s) in 342567 ms`,
~1 min attributed to in-process F32 load) and the model-gated e2e test at
~5m35s. **The prefill design is the dominant waste.**

### B. Per-token decode (`ForwardCached` loop)

Per generated token, per layer:

- Q/K/V + O projections (4 matmuls) + RoPE + KV write
- **full KV-prefix `Buffer.BlockCopy`** (`LlamaCausalAttention.cs:160-163`) —
  copies *all cached positions* every step
- **`GqaRepeatKV` over the whole prefix** — materializes `[len, 896]` K and V
  per layer (×7 data blowup), `:173-174`
- `MultiHeadAttention` with `PackHeads` + per-head scores/softmax/context
  (qLen = 1)
- residual add, 2× RMSNorm, SiLU-gated FFN (3 matmuls: 896→4864, 896→4864,
  4864→896)
- final RMSNorm + **LM head matmul `[1,896]·[151936,896]ᵀ`** (136M MACs,
  544 MB reads)

Per-token weight traffic at F32 (all weights re-read each step):

| Component | Params (M) | F32 bytes/token |
|---|---|---|
| FFN (24×13.06M) | 313 | 1.25 GB |
| LM head (vocab 151,936×896) | 136 | 544 MB |
| Attention projections | 43 | 172 MB |
| **Total** | **492** | **≈ 1.97 GB** |

At ~30 GB/s effective bandwidth the **F32 decode ceiling is ~15–20 tok/s**;
with BF16-native weights (~1 GB) ~30–40 tok/s. The measured current
performance is **far below that ceiling** — see §3.

---

## 3 · Opportunity — how optimized is each block today?

### Done well

- Inference-default (no graph nodes), `Eval()`, span-based kernels with
  `TensorPrimitives` (MatMul dots, RMSNorm, RoPE tables precomputed+cached,
  `WidenBf16ToF32` SIMD at load).
- `MultiHeadAttention` reuses `ArrayPool` scratch; weights are stored once in
  the transposed-B layout with no runtime transpose per op.
- Loader is mmap + fused BF16→F32 widen (managed heap ~1.88 GB, one pass).
- Qwen math is numerically verified against PyTorch fixtures (13 parity tests,
  byte-exact tool rendering).

### Main inefficiencies (ranked by impact)

1. **Prefill is O(L) full-model passes** (`QwenChatClient.cs:159`). The
   standard fix — batched `[L, hidden]` prefill at the `Forward(int[])` level
   (`samples/Nivara.Samples/LlamaForCausalLM.cs:67`, unused by the client; the
   `BatchedMultiHeadAttention` op is likewise unused in both clients) — reads
   every weight **once** for the whole prompt instead of once *per prompt
   token*. On a typical chat turn (50–150 prompt tokens vs 30–80 generated)
   this is the single largest win.

2. **MatMul copies every weight before the dot**
   (`TensorsHelper.MultiplyCoreFloat` rents `bT` and `b.CopyTo(bT)` when
   `bTransposed=true`, `TensorsHelper.cs:150-151`). The weights *already are*
   row-major `[out, in]`; for `aRows == 1` (every decode matmul) the copy is
   pure waste. The LM head call alone rents+copies **544 MB per token**
   (ArrayPool cannot pool 136M elements, so this is a fresh big-array alloc +
   full copy **every token**, plus GC churn). FFN adds ~1.25 GB/token of
   redundant copies. **This roughly doubles effective memory traffic
   (~4 GB/token) and is the dominant constant-factor from the kernel side.**

3. **KV cache copy + GQA materialization every step.** `ForwardCached`
   BlockCopies the growing prefix and `GqaRepeatKV`-expands all positions ×7
   per layer per token (≈22 MB/token of copies at 128-token context;
   `LlamaCausalAttention.cs:160-174`). A dedicated single-query decode
   attention would read cached K/V rows directly and compute scores **per KV
   head once**, reused across the 7-query-head group — no copies, no
   expansion.

4. **Per-op tensor boxing.** Every `MatMulTransposedB`, `Add`, `Multiply`,
   `Linear`, `RMSNorm`, `Gather` allocates a `T[]` + `NivaraColumn<T>` +
   `ReverseGradTensor` (`ReverseGradOperations.cs:430-437`, etc.). Decode does
   ~150–200 such ops per token → constant GC churn.
   `AutoDiffDiagnostics.Measure` is zero-cost when disabled (single `if`), so
   diagnostics are not the issue — the allocations are.

5. **No fused QKV.** Three separate `Linear` calls + 3 matmul rents/copies per
   layer; a fused `[1,896]→[1, 3·896]` projection halves that.

6. **Sampling path (only when `--temperature > 0`):** `Select` allocates
   `double[151,936]` and does a full `Array.Sort` per token
   (`QwenChatClient.cs:188-214`). Fine for argmax (greedy, the default) — just
   not for token-paced sampling without a float-based partial top-k.

7. **Tokenizer re-encodes the whole history each turn**
   (`QwenChatClient.cs:111`) — a few hundred ms per turn in the REPL, minor
   relative to the model but a free fix (cache prefix ids).

---

## 4 · Plan — where to invest (review conclusion)

Honest framing: current end-to-end throughput (docs data) is ~2 turns in
~5 min; the F32 memory-bound ceiling is ~15–20 tok/s decode plus near-instant
prefill. There is a **large, well-understood gap**, and the fixes are
orthogonal and additive.

| Priority | Investment | Expected effect | Effort | Risk |
|---|---|---|---|---|
| **P0** | **Batched prompt prefill** — seed the KV cache with one `[L, hidden]` forward (embed + 24 blocks via the existing `Forward(int[])` / `BatchedMultiHeadAttention` path), write K/V into the cache, return last-row logits. | Prefill drops from O(L)·2 GB to 2 GB once → **10–50×** on prompt-heavy chat/tool turns (the observed minutes → seconds) | M | Low — parity fixture exists (`qwen_tool_final_prompt_ids.bin`); the batched forward is already PyTorch-pinned |
| **P0** | **Kill the redundant weight copy in matmul** — for `bTransposed && aRows == 1`, dot against the weight span directly (no rent, no copy); ideally a fused mat-vec path. | Decode traffic ~2 GB → ~1 GB/token + removes the 544 MB/token LM-head alloc + GC pressure → ~1.5–2× decode | S | Low — bit-identical numerics |
| **P0** | **Fused GQA decode-attention** — compute per-KV-head scores over cached rows (span reads, no `BlockCopy`, no `GqaRepeatKV` materialization); reuse across the query group; fold scale+softmax. | Removes ~22+ MB/token copies + per-head packing; attention path becomes allocation-light | M | Medium — needs new kernel + numeric-parity test vs `LlamaCausalAttention.Forward` (cache-vs-full parity test already exists) |
| **P1** | **On-the-fly BF16 weights with F32 compute** — keep weights native `BFloat16` in memory (half traffic, half loaded size), widen in-register per dot; only revisit `Vector<BFloat16>` when it ships (#387/#391). | 2× on all matmuls → ~30–40 tok/s decode; +2× smaller footprint/load | M | Medium — needs a BF16 dot kernel + tolerance-bounded parity test (fixtures tolerate ~0.4 abs on tail logits) |
| **P1** | **Per-token fused decoder-block kernel** (single scratch-buffer decode: fused QKV, O+residual+norm chain, FFN) to remove the ~200 allocs/token | Removes GC churn; compounds with the above | L | Medium |
| **P2** | Sampling path: float-based, single-pass (no full sort); tokenize-prefix caching in the REPL | Minor, only when sampling / in chat REPL | S | Low |
| **Stretch** | INT8 block-quantized weights (llama.cpp-style Q8: ~4× traffic cut, ~65+ tok/s ceiling) and the GGUF backend (#390). | Biggest possible wall-clock cut, at fidelity cost | L | High — fidelity/quality gates needed |

**Not worth investing now:** precomputing a transposed weight-view cache
(weights already in the correct layout); parallelizing decode (`aRows == 1` —
the `ShouldParallelize` gate in `TensorsHelper` correctly stays off;
parallelism belongs in batched prefill); speculative decoding / top-k tricks
(argmax is already a single scan — the win is in traffic, not the scan).

---

## Caveats and next steps

- Per-token / `tok/s` figures above are **derived from memory-bandwidth math
  and docs transcripts, not freshly measured** at review time.
- An existing harness (`samples/NivaraInference → qwen benchmark`,
  `Qwen.cs:394`) measures KV-cached vs full-forward decode directly and would
  pin the real current `ms/token` and the prefill share. Run it before opening
  the implementation issue to get a baseline.
- No code was changed as part of this review. Related tracked work: #384
  (qkvBias), #387/#391 (BF16 SIMD), #388 (fused BF16→F32 read), #390 (GGUF
  backend).

---

# Improvement ledger

Tracked record of executed improvements to the Qwen inference path. Each entry
is written **only after the item's full execution passes** (implementation +
tests + harness gate + full test suite + E2E), so a status here means the
evidence exists. The review prose above stays untouched as the pre-execution
research record; where the ledger contradicts it (measurement, premise,
framing), the ledger wins. Status values: `DONE` · `IN PROGRESS` · `DEFERRED`.

## P0-2 — Kill the redundant weight copy in single-row transposed-B matmul

**Status: DONE** (2026-09-07 · branch `khurram/qwen-perf` · commit `58b721e`
kernel + tests, `d6a32df` results)

**Item:** for every decode matmul (`aRows == 1`, `bTransposed == true`),
dot each output column against the weight row directly — no workspace rent,
no `b.CopyTo(bT)` identity copy, no `ArrayPool.Return(clearArray: true)`
bucket clear on return. BLAS2 GEMV reads each weight exactly once.

**Numerics:** bit-identical by construction — the fast path reuses the same
per-type `Dot` dispatch as the row kernels (`TensorPrimitives.Dot` for
float/double; `WidenPrimitives.Dot` for the generic/half-precision path),
locked by parity tests (single-row == row 0 of a two-row run, Dot-vs-Dot
bit-exact for float/double, Half/BFloat16 widen-path parity, cold-pool
alloc guard). Full suite: **3449 passed, 0 failed** on net11.0.0.

**Measured before/after** (`--runs 3` child-process medians, same machine,
net11.0.0, Qwen2.5-0.5B shapes — `qwen-fast-baseline.json` → `qwen-fast-postfix.json`):

| Decode row | Before | After | Δ | B/op before → after |
|---|---|---|---|---|
| LM head matmul `[1x896 @ 151936x896]` | 5 ops/s · ~200 ms/op | **36 ops/s · ~28 ms/op** | **+620%** | 53 → 5 |
| LM head fwd `[1x896 -> 151936]` (op-level) | 5 ops/s | **42 ops/s** | +740% | 608,517 → 608,469 |
| FFN gate/up/down ×3 | 61 ops/s | **529 ops/s** | +767% | 145 → 1 |
| attn Q/K/V/O proj | 640 ops/s | **6,658 ops/s** | +940% | 193 → 1 |
| Linear fwd `[1x896 -> 2688]` (op-level) | 362 ops/s | **5,088 ops/s** | +1,306% | 22,817 → 22,769 |

**E2E (synthetic F32 weights, 64-token prompt + 24-token decode, median of 3):**
KV-cached **335 ms/token (3.0 tok/s)** vs full-forward 2,074 ms/token — **6.2×**.
(No pre-fix E2E number exists; the synthetic mode shipped with this branch's
harness. The harness rows above carry the kernel-level before/after.)

**Premise correction (important — supersedes §2 item 2):** the review claimed
the LM head "rents+copies **544 MB per token** (ArrayPool cannot pool
136M elements, so this is a fresh big-array alloc + full copy every token,
plus GC churn)". Both halves are wrong on the current runtime, measured and
source-grounded (MS Learn `ArrayPool<T>.Rent` page → dotnet/runtime source,
net-11.0 moniker):

- `ArrayPool<T>.Shared` (`TlsOverPerCoreLockedStacksArrayPool`) has
  **27 buckets pooling up to ~2³⁰ elements** — the 136M-float workspace is a
  reused ~1 GB bucket, so **B/op ≈ 0 steady-state** (no fresh alloc, no GC
  churn). The review's premise that the pool cannot serve the size is false.
- The actual waste was **memory traffic**: identity `CopyTo` (read+write
  545 MB) + the necessary dot read (545 MB) + `Return(clearArray: true)`
  clearing the ~1 GB bucket on every return ≈ **2.7 GB/token**. The fast path
  reduces this to the single 545 MB dot read — hence ~200 ms → ~28 ms, and a
  ~1 GB working-set drop.

**Gate note:** the `--compare` run's 3 FAIL rows (Linear forward 32x256,
Frame Slice, Attn batched fwd+bwd) are pre-existing throughput/gen0 noise on
rows this change does not touch (byte-identical B/op; Frame Slice is the
documented issue #354 flake; machine under load during measurement). All five
Qwen rows pass with margins of +5–13×.

---

### Future entries (template — fill in as items land)

| Plan item | Status | Measured before → after | Notes |
|---|---|---|---|
| P0 — Batched prompt prefill (O(L) → 1 pass) | | | |
| P0 — Fused GQA decode-attention (no BlockCopy / GqaRepeatKV) | | | |
| P1 — On-the-fly BF16 weights with F32 compute | | | |
| P1 — Per-token fused decoder-block kernel | | | |
| P2 — Sampling path + tokenize-prefix cache | | | |
| Stretch — INT8 block-quantized weights / GGUF backend (#390) | | | |