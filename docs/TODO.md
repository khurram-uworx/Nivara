# M2 — GPU kernel fusion + launch-overhead reduction (issue #437)

Branch: `khurram/lazystream-gpu` off `khurram/minilm-gpu`. PR targets
`khurram/minilm-gpu`. Scope agreed with the human: **full fusion + micro-opts**,
gated by all three GPU compares after every step.

## Problem

The ILGPU runner is correctness-first: **one kernel launch per op** —
~114 dispatches/forward for MiniLM, ~119 for DistilBERT (incl. SST head). On the
iGPU at 128-row shapes each dependent kernel costs **~29 µs** measured (not the
~0.18–0.2 ms first estimated — see Expected effect):

- MiniLM (1.36 GMAC): 26.8 ms total; ~113 launches in the baseline.
- DistilBERT (5.44 GMAC): 65.3 ms total; ~119 launches.

The runner **already** enqueues everything on one stream (`runtime.Stream`) with a
single `Synchronize()` before readback — the "lazy stream" half of #437 is
effectively already true. The real lever is **dispatch count**. Remaining
stage-2 leftovers are small: `posIds` rebuilt+re-uploaded every call (it is
seqLen-deterministic), and 3 separate `CopyFromCPU` payload uploads per forward.

## Proposed changes (all sample-scoped — `samples/Nivara.Samples/Gpu/` + `samples/NivaraInference`)

Dispatch inventory today → after:

| stage | now (post‑step‑1) | after |
|---|---|---|
| embeddings (gather×2, add, [token-type], LN) | 4–5 | 3 (EmbeddingSum + word gather + LN) |
| per layer: q/k/v GEMM+bias (3), attention (1), o+bias (1), add+LN1 (2), fc1+GELU (1), fc2 (1), add+LN2 (2) | 11 | 7 |
| ×6 layers | 66 | 42 |
| head (SST) | 3 (gather + 2 fused GEMMs) | 3 |
| **total** | ~71–72 (was 113–119) | **~45–48** |

Measured so far (2026-09-19, AC): step 1 (epilogue fusion) took 42 + head
dispatches out; MiniLM 26.8 → **25.6 ms**, DistilBERT ~66 ms. Per-dispatch
dependency latency ≈ **29 µs**, so the remaining −24 dispatches buy ~0.7 ms.

### 1. GEMM epilogue fusion — `GemmKernels.cs` (lean sibling kernels)
Bias + activation folded into the Row4 GEMM (DONE, `b1cd089`, gated all three
compares). The existing `TiledGemmKernelRow4` stays untouched because the
`TiledGemm` probe harness loads it by signature; the runner switched to fused
launches via an `activation` byte (0 none / 1 GELU / 2 ReLU) with the A–S 7.1.26
GELU port. Folds `bias` into every projection (q/k/v/o/fc1/fc2 + head) — 42
dispatches/forward removed. Bit-identical elementwise; gates held at the
existing maxAbs floor.

**Refinement (post-measurement):** the single fat kernel inlines all three
epilogues plus `XMath.Exp` for every launch — a suspected register-pressure cost
on the lean (identity) GEMM legs. Split into three lean siblings
`TiledGemmKernelRow4Bias` / `TiledGemmKernelRow4Gelu` / `TiledGemmKernelRow4Relu`
(shared tile+K-loop device helper, hard-coded epilogue per method) and load all
three; q/k/v/o/fc2 + head-cls take the bias-only kernel, fc1 the GELU kernel, head
pre-classifier the ReLU kernel. Re-measure after the split to see how much of the
missing gain was epilogue overhead vs the corrected per-kernel latency floor.

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

### Expected effect (measured truth, AC power)
Per-dispatch dependency latency is **~29 µs** (measured: −42 dispatches → −1.2 ms
on MiniLM 26.8 → 25.6 ms), not the ~0.19 ms first modelled — the model was off
~6.5× because the GEMM legs and non-GEMM compute dominate. Remaining micro-opts
(−24 dispatches) buy only **~0.5–0.7 ms**: MiniLM **~24–25 ms (~1.1×)**,
DistilBERT **~65 ms**. The #437 ">2× for MiniLM" acceptance is **not reachable
via fusion on this iGPU** (would need ~420 more kernels removed). The >2× lever
is kernel throughput (GEMM tile-32/2×2 — the deferred separate item); even an
optimistic 2–3× GEMM speedup lands MiniLM ~22 ms, not 13.4. Acceptance is
reported honestly as unmet with the measured numbers, and a follow-up issue tracks
the GEMM-throughput item.

### 5. DistilBERT fine-tuning measurements (CPU, Nivara) — past methodology

Separate from the GPU inference track, the human wants a fresh DistilBERT training
measurement recorded, following the **established `NivaraFineTuning` slice
harness** (its README §Performance benchmarks):

- Run: `dotnet run -c Release --project samples/NivaraFineTuning -- --mode train
  --epochs 1 --batch-size 2 --max-examples 25` — 13 batches; the full
  67,349-example epoch is ~32 h and out of reach, so the slice is the documented
  compromise. **First (JIT warmup) batch is excluded** from steady-state ms/batch.
  AC power for stable thermals. Inform the human before the run.
- Record steady-state ms/batch and compare to the last recorded Nivara number
  (2026-08-21: **1540 ms/batch**, ~3× vs PyTorch 499 ms/batch on the same slice).
- Optional PyTorch A/B via `samples/NivaraFineTuning/Python/benchmark_timing.py`
  when the Python env is available (same-session, same methodology).
- **Detail the measurement in the NivaraInference README** per the human's
  direction (date, machine, exact command, per-side numbers, full-epoch
  extrapolation), with a pointer to the NivaraFineTuning methodology. Training is
  CPU-side (no ILGPU involvement) and is unaffected by the fusion work.

## Verification steps (gates after every step)

- After each step: `minilm --gpu compare`, `distilbert --gpu compare`,
  `distilbert_sst --gpu compare` — all three must keep GATE PASS at the **existing**
  ~1.5e-5 maxAbs class (every fused op is bit-identical elementwise, so the bound
  must not loosen).
- `minilm --gpu` (embedding L2 norm) + f32-only rejection path — unchanged.
- Final measurement: `minilm --gpu benchmark` + `distilbert --gpu benchmark` on
  **AC power** (inform the human first), same-session numbers → update README GPU
  table + `docs/BERT-GPU.md` (item 1 gets measured truth: MiniLM ~24–25 ms,
  DistilBERT ~65 ms — acceptance documented as unmet) +
  `docs/ROADMAP-SUGGESTION.md` M2 row.
- File a follow-up GitHub issue for the GEMM-throughput item (tile-32/2×2) that
  the >2× target actually requires — the honest lever identified by measurement.
- DistilBERT fine-tuning slice measurement (see Proposed changes §5) — Nivara
  number + optional PyTorch A/B, recorded in the NivaraInference README.
- Quick `--gpu benchmark` between steps (AC) to watch the improvement curve.

## Planned commits

1. `docs: plan M2 GPU kernel fusion (launch-overhead reduction) in TODO.md`
2. `samples: fuse bias (+GELU/ReLU) into the Row4 GEMM epilogue`
3. `samples: split the fused GEMM epilogue into lean bias/GELU/ReLU siblings`
4. `samples: concat Q/K/V into one GEMM with sliced views for attention`
5. `samples: fuse residual-add into LayerNorm and embedding sums into one kernel`
6. `samples: cache positional ids per seqLen in BertEncoderGpuRunner`
7. `docs: M2 benchmark numbers + README/BERT-GPU/ROADMAP updates`
8. `docs: record DistilBERT fine-tuning slice measurement in NivaraInference README`
9. `docs: remove TODO.md — M2 plan executed (after G2)`

## Blast radius

- `samples/Nivara.Samples/Gpu/GemmKernels.cs` — adds fused GEMM siblings
  (row4 + lean bias/GELU/ReLU forms); original Row4 + 1×1 + `TiledGemm` probe
  harness untouched.
- `samples/Nivara.Samples/Gpu/ElementwiseKernels.cs` — adds `LayerNormResidual1D`
  + `EmbeddingSum`; existing kernels untouched.
- `samples/Nivara.Samples/Gpu/BertEncoderGpuRunner.cs` — qkv concat at upload,
  epilogue-aware launch call sites, posIds cache. Core forward logic unchanged.
- `samples/NivaraInference/Program.cs` — no functional change expected; GPU gate
  run-modes unchanged.
- `samples/NivaraInference/README.md` — gains a DistilBERT fine-tuning measurement
  note (§5 of this plan); inference tables updated with fusion numbers.
- `samples/NivaraFineTuning` + `samples/data/sst2|distilbert` — **used read-only**
  for the training measurement (no code change to the sample).
- No `src/Nivara`, `Nivara.Extensions`, `src/Nivara.Gpu`, or `tests/` changes.
- GPU gates are run-mode verification (`minilm|distilbert|distilbert_sst --gpu
  compare`), not NUnit — no new unit tests needed; existing compares are the
  regression net.

## GitHub issues log

- [#437](https://github.com/khurram-uworx/Nivara/issues/437) — M2 GPU kernel
  fusion / lazy stream (this plan's tracked work; scope-comment it when starting).