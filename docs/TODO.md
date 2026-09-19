# M2 — GPU kernel fusion + launch-overhead reduction (issue #437)

Branch: `khurram/lazystream-gpu` off `khurram/minilm-gpu`. PR targets
`khurram/minilm-gpu`. Scope agreed with the human: **full fusion + micro-opts**,
gated by all three GPU compares after every step.

## Problem

The ILGPU runner is correctness-first: **one kernel launch per op** —
~114 dispatches/forward for MiniLM, ~119 for DistilBERT (incl. SST head). On the
iGPU at 128-row shapes each launch costs ~0.18–0.2 ms end-to-end:

- MiniLM (1.36 GMAC): GEMM legs ~4–6 ms of the 26.8 ms total → ~21 ms overhead.
- DistilBERT (5.44 GMAC): ~20–30 ms GEMM legs of the 65.3 ms total → ~35 ms overhead.

The runner **already** enqueues everything on one stream (`runtime.Stream`) with a
single `Synchronize()` before readback — the "lazy stream" half of #437 is
effectively already true. The real lever is **dispatch count**. Remaining
stage-2 leftovers are small: `posIds` rebuilt+re-uploaded every call (it is
seqLen-deterministic), and 3 separate `CopyFromCPU` payload uploads per forward.

## Proposed changes (all sample-scoped — `samples/Nivara.Samples/Gpu/` + `samples/NivaraInference`)

Dispatch inventory today → after:

| stage | now | after |
|---|---|---|
| embeddings (gather×2, add, [token-type], LN) | 4–5 | 2 (EmbeddingSum + LN) |
| per layer: q/k/v GEMM+bias (6), attention (1), o+bias (2), add+LN1 (2), fc1+bias+GELU (3), fc2+bias (2), add+LN2 (2) | 18 | 7 |
| ×6 layers | 108 | 42 |
| head (SST) | 6 | 2–3 |
| **total** | ~114–119 | **~44–47** |

### 1. GEMM epilogue fusion — `GemmKernels.cs` (`TiledGemmKernelRow4`)
Add a `bias` `ArrayView<float>` + `activation` byte (0 none / 1 GELU / 2 ReLU) to
the Row4 signature. The four register accumulators add `bias[col]` before the
epilogue write; GELU uses the same A–S 7.1.26 polynomial port as
`ElementwiseKernels.Gelu` (`XMath.Exp`). Folds `bias` into every projection
(q/k/v/o/fc1/fc2 + head) and GELU into fc1 — kills 7 bias/activation launches per
layer. Bit-identical elementwise (register `acc + bias` vs stored-then-added by
`AddBias`; same GELU polynomial). The 1×1 `TiledGemmKernel` and the `TiledGemm`
probe harness stay untouched.

### 2. QKV-concat — `BertEncoderGpuRunner.cs` (upload) + launch site
Upload `[Wq|Wk|Wv]` as one pre-transposed `[hidden × 3·hidden]` buffer (per-layer),
plus one `[3·hidden]` bias. One GEMM → packed `C[rows × 3·hidden]`; q/k/v become
three contiguous `SubView(0|r|2r, rows·hidden)` slices feeding the **unchanged**
attention kernel. 6 → 1 launch/layer. Summation order per output column unchanged
(same K-loop; only N grows), so parity holds at the existing floor.
Both naming styles already map q/k/v roles via `BertGpuKeys`.

### 3. residual+LN + embedding-sum — `ElementwiseKernels.cs`
- `LayerNormResidual1D(a, b, w, bias, y, rows, cols, eps)`: row-reduce over `(a+b)`
  (LN already row-reduces; read two inputs). Kills both residual `Add` launches per
  layer. Preserves the buffer-rotation trick (`LN2` still writes into `x`).
- `EmbeddingSum(ids, posIds, wordEmb, posEmb, [tokenTypeEmb], y, hidden, rows)`:
  `y[i·h+j] = wordEmb[ids[i]·h+j] + posEmb[posIds[i]·h+j] + (hasTokenType ? tt[j] : 0)`
  (ILGPU views can't be null — pass a `hasTokenType` bool, branch on it). Replaces
  gather(word) + gather(pos) + add + [token-type bias] → 1 launch.

### 4. Micro-opts — `BertEncoderGpuRunner.cs`
Cache the `posIds` array per `seqLen` (deterministic: `row % seqLen`), skip
recompute + re-upload when unchanged. Batch mask/ids payload uploads where trivial.
**Benchmark path stays byte-identical (readback kept)** so the measured number is
apples-to-apples with the 26.8 ms baseline — no no-readback variant.

### Expected effect (math, AC power)
Linear-in-count overhead model: MiniLM `~44 × 0.19 ≈ 8.4 ms` + 4–6 ms GEMM legs
≈ **13–14 ms (~1.9–2.1×)**; DistilBERT `~47 × 0.19 ≈ 9 ms` + ~25 ms ≈
**~32–35 ms (~1.9×)**. The #437 ">2× for MiniLM" acceptance sits right at the
fusion-only edge; micro-opts add margin. If the acceptance check lands short, we
report the honest number and optionally pull GEMM tile-32 headroom (separate item,
not in this plan).

## Verification steps (gates after every step)

- After each step: `minilm --gpu compare`, `distilbert --gpu compare`,
  `distilbert_sst --gpu compare` — all three must keep GATE PASS at the **existing**
  ~1.5e-5 maxAbs class (every fused op is bit-identical elementwise, so the bound
  must not loosen).
- `minilm --gpu` (embedding L2 norm) + f32-only rejection path — unchanged.
- Final measurement: `minilm --gpu benchmark` + `distilbert --gpu benchmark` on
  **AC power** (inform the human first), same-session numbers → update README GPU
  table + `docs/BERT-GPU.md` (item 1 "65 → ~25 ms class" gets measured truth;
  MiniLM ~2×) + `docs/ROADMAP-SUGGESTION.md` M2 row.
- Quick `--gpu benchmark` between steps (AC) to watch the improvement curve.

## Planned commits

1. `docs: plan M2 GPU kernel fusion (launch-overhead reduction) in TODO.md`
2. `samples: fuse bias (+GELU/ReLU) into the Row4 GEMM epilogue`
3. `samples: concat Q/K/V into one GEMM with sliced views for attention`
4. `samples: fuse residual-add into LayerNorm and embedding sums into one kernel`
5. `samples: cache positional ids per seqLen in BertEncoderGpuRunner`
6. `docs: M2 benchmark numbers + README/BERT-GPU/ROADMAP updates`
7. `docs: remove TODO.md — M2 plan executed (after G2)`

## Blast radius

- `samples/Nivara.Samples/Gpu/GemmKernels.cs` — Row4 signature gains bias +
  activation params; 1×1 kernel untouched; `TiledGemm` probe harness unchanged.
- `samples/Nivara.Samples/Gpu/ElementwiseKernels.cs` — adds `LayerNormResidual1D`
  + `EmbeddingSum`; existing kernels untouched.
- `samples/Nivara.Samples/Gpu/BertEncoderGpuRunner.cs` — qkv concat at upload,
  epilogue-aware launch call sites, posIds cache. Core forward logic unchanged.
- `samples/NivaraInference/Program.cs` — no functional change expected; GPU gate
  run-modes unchanged.
- No `src/Nivara`, `Nivara.Extensions`, `src/Nivara.Gpu`, or `tests/` changes.
- GPU gates are run-mode verification (`minilm|distilbert|distilbert_sst --gpu
  compare`), not NUnit — no new unit tests needed; existing compares are the
  regression net.

## GitHub issues log

- [#437](https://github.com/khurram-uworx/Nivara/issues/437) — M2 GPU kernel
  fusion / lazy stream (this plan's tracked work; scope-comment it when starting).