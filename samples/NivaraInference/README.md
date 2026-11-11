# Nivara HuggingFace Inference Sample

Load pre-trained HuggingFace models (MobileNetV2, ResNet-18, MiniLM, DistilBERT, DistilBERT SST-2, SmolLM-135M-Instruct, Qwen2.5-0.5B-Instruct, ModernBERT-large, Laya) into Nivara's zero-dependency tensor engine and run forward inference in pure managed .NET — no Python runtime, no CUDA, no third-party ML framework.

The same architecture is also implemented in PyTorch (`samples/NivaraInference/Python/`) for direct CPU performance comparison.

## Quick start

```bash
# Download model weights via HuggingFace CLI
hf download google/mobilenet_v2_1.0_224 --local-dir samples/data/mobilenet_v2
hf download microsoft/resnet-18 model.safetensors config.json --local-dir samples/data/resnet18
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir samples/data/minilm
# (distilbert-base-uncased already present under samples/data/distilbert)
hf download distilbert/distilbert-base-uncased-finetuned-sst-2-english config.json model.safetensors vocab.txt tokenizer_config.json --local-dir samples/data/distilbert_sst
# SmolLM-135M-Instruct — BF16-native Llama-family causal LM (GQA, SiLU, RoPE)
hf download HuggingFaceTB/SmolLM-135M-Instruct config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json special_tokens_map.json --local-dir samples/data/smollm-135m
# Qwen2.5-0.5B-Instruct — function calling + teacher distillation (BF16 on disk)
hf download Qwen/Qwen2.5-0.5B-Instruct config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json special_tokens_map.json --local-dir samples/data/qwen2.5-0.5b-instruct
# ModernBERT-large — 28-layer hybrid (10 full + 18 sliding-window) encoder, 1.5 GB
hf download answerdotai/ModernBERT-large --local-dir samples/data/modernbert
# Laya — ModernBERT-large encoder plus a typed decision head, 843 MB F16
hf download convaiinnovations/laya model.safetensors encoder/config.json rl_agent_config.json tokenizer/tokenizer.json tokenizer/tokenizer_config.json rl_common.py README.md --local-dir samples/data/laya

# Run inference
dotnet run --project samples/NivaraInference -c Release -- mobilenet_v2
dotnet run --project samples/NivaraInference -c Release -- resnet18
dotnet run --project samples/NivaraInference -c Release -- minilm
dotnet run --project samples/NivaraInference -c Release -- distilbert
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst
dotnet run --project samples/NivaraInference -c Release -- smollm                 # greedy causal-LM generation (F32)
dotnet run --project samples/NivaraInference -c Release -- smollm --precision bf16  # native BF16 (256.6 MB)
dotnet run --project samples/NivaraInference -c Release -- qwen tools            # function calling (KV cache on)
dotnet run --project samples/NivaraInference -c Release -- qwen distill --teacher-examples 12  # teacher distillation
dotnet run --project samples/NivaraInference -c Release -- qwen --silk distill --teacher-examples 12  # same, teacher on the OpenCL GPU
dotnet run --project samples/NivaraInference -c Release -- qwen benchmark        # KV-cached vs full re-forward
dotnet run --project samples/NivaraInference -c Release -- modernbert            # ModernBERT-large forward (28-layer hybrid)
dotnet run --project samples/NivaraInference -c Release -- laya                 # decision head over the fixture questions (CPU reference)

# TypeSafe decision-model demo — no weights; calls a decision-model engine over HTTP
dotnet run --project samples/NivaraInference -c Release -- typesafe             # Ollaya (laya:en)
dotnet run --project samples/NivaraInference -c Release -- typesafe ollama      # Ollama (tev1:0.8b)

# Benchmark (10 passes each)
dotnet run --project samples/NivaraInference -c Release -- mobilenet_v2 benchmark
dotnet run --project samples/NivaraInference -c Release -- resnet18 benchmark
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst benchmark
dotnet run --project samples/NivaraInference -c Release -- modernbert benchmark  # seq 128 + 256; 1 warmup + 3 timed; --seq 256 times one length
dotnet run --project samples/NivaraInference -c Release -- modernbert --silk benchmark    # seq 128 + 512 + 2048 + 4096

# MiniLM on the OpenCL iGPU (Silk only -- hand-authored OpenCL kernels; F32-only)
dotnet run --project samples/NivaraInference -c Release -- minilm --silk

# BERT-family encoders on the OpenCL iGPU.
# NOTE: the former GPU sample half was removed in #814. The live GPU flag is now
# `--silk` (hand-authored OpenCL); older GPU snippets below are historical
# baselines retained for the performance tables.
# Historical first-backend command lines (removed in 1.5.0; use --silk now):
#   (no live command for the old backend; the equivalents are --silk / --silk compare)
#   dotnet run --project samples/NivaraInference -c Release -- modernbert --silk
#   dotnet run --project samples/NivaraInference -c Release -- laya --silk            # encoder + decision head
#
# GPU-vs-CPU parity gates (live gates use --silk compare):
#   dotnet run --project samples/NivaraInference -c Release -- modernbert --silk compare
#   dotnet run --project samples/NivaraInference -c Release -- laya --silk compare

# BERT-family encoders on the second OpenCL backend: hand-authored OpenCL C via Silk.NET
# (MiniLM / DistilBERT / DistilBERT SST-2; --silk now accepts --precision bf16 (widened to F32 on device
#  for the encoder; SmolLM/Qwen use true BF16 kernels). fp16 is rejected for Qwen (Half range/precision insufficient for Qwen2.5-0.5B; use f32 or bf16). Mutually exclusive with the
#  (the first GPU flag was removed in 1.5.0); --silk is the live GPU flag.
#  see docs/SILK.md Part 2 for the backend and docs/ACCELERATION.md §1 for the cross-backend view)
dotnet run --project samples/NivaraInference -c Release -- minilm --silk
dotnet run --project samples/NivaraInference -c Release -- distilbert --silk
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst --silk
dotnet run --project samples/NivaraInference -c Release -- minilm --silk compare            # Silk-vs-CPU parity gate
dotnet run --project samples/NivaraInference -c Release -- distilbert --silk compare
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst --silk compare    # logits vs CPU + argmax 8/8

# Narrow-precision inference (half weight memory; see model docs for details)
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst bf16
dotnet run --project samples/NivaraInference -c Release -- distilbert bf16
dotnet run --project samples/NivaraInference -c Release -- minilm bf16
```

## Supported models

| Model | Type | Weight size | Tensors | Parameters | Output |
|-------|------|-------------|---------|------------|--------|
| MobileNetV2 | Vision (classification) | 13.5 MB | 262 | 3.4M | 1001 classes |
| ResNet-18 | Vision (classification) | 44.6 MB | 102 | 11.7M | 1000 classes |
| MiniLM (L6-v2) | Text (embedding) | 91 MB | 104 | 22.7M | 384-dim embedding |
| DistilBERT (base-uncased) | Text (encoder) | 255.5 MB | 105 | 67.0M | `[seqLen, 768]` hidden states |
| DistilBERT SST-2 (fine-tuned) | Text (classification) | 255.4 MB | 104 | 66.9M | 2-class sentiment (`NEGATIVE`/`POSITIVE`) |
| SmolLM-135M-Instruct | Text (causal LM) | 269 MB | 272 | 134.5M | token ids (generation) |
| Qwen2.5-0.5B-Instruct | Text (causal LM + function calling) | 989 MB | 290 | ~494M | token ids (tool-assisted generation) |
| ModernBERT-large | Text (hybrid encoder) | 1510 MB | 170 of 173 | 394.8M | `[seqLen, 1024]` hidden states |
| Laya | Text (decision head on ModernBERT-large) | 843 MB F16 | 206 | 421.3M | typed answer per question |

## Model details

For architecture, verification, performance, and GPU-path details per model:

| Model | Doc |
|-------|-----|
| MobileNetV2, ResNet-18 | [docs/VISION.md](../../docs/VISION.md) |
| DistilBERT, MiniLM, SST-2 | [docs/DISTELBERT.md](../../docs/DISTELBERT.md) |
| ModernBERT-large | [docs/MODERNBERT.md](../../docs/MODERNBERT.md) |
| SmolLM-135M-Instruct | [docs/SMOLLM.md](../../docs/SMOLLM.md) |
| Qwen2.5-0.5B-Instruct | [docs/QWEN.md](../../docs/QWEN.md) |
| Laya | [docs/LAYA.md](../../docs/LAYA.md) |
| GPU acceleration (all models) | [docs/ACCELERATION.md](../../docs/ACCELERATION.md) |

## TypeSafe decision-model demo (`typesafe`)

`typesafe` loads no weights. It is an HTTP client for the **TypeSafe "System One"** decision-model
API, which both **Ollaya** and **Ollama 0.35+** expose at the same `POST /v1/systemone` route. The
command line is `typesafe [engine] [mode] [--url <url>] [--model <name>]`:

| Engine | Default URL | Default model | How to select |
|--------|-------------|---------------|---------------|
| Ollaya | `http://localhost:11435` | `laya:en` | default, or `ollaya` |
| Ollama | `http://localhost:11434` | `tev1:0.8b` | `ollama` |

```bash
dotnet run --project samples/NivaraInference -c Release -- typesafe                            # Ollaya, laya:en
dotnet run --project samples/NivaraInference -c Release -- typesafe ollama                     # Ollama, tev1:0.8b
dotnet run --project samples/NivaraInference -c Release -- typesafe ollama --model tev1:0.8b   # model made explicit
dotnet run --project samples/NivaraInference -c Release -- typesafe benchmark                  # time the round trip
dotnet run --project samples/NivaraInference -c Release -- typesafe ollama benchmark
dotnet run --project samples/NivaraInference -c Release -- typesafe benchmark --warmup 2 --iters 10  # more samples (#676)
```

It sends six AI/ML news headlines through one typed question set — a category choice, three yes/no
questions and two score questions — and prints the answer and confidence for each, exiting
non-zero when every assessment failed (the visible symptom of an engine that is not running).
`--url` and `--model` are `typesafe`-only options; any other model rejects them, and not every
model an engine holds is a decision model — the engine itself answers `400 … is not a decision
model` for the rest.

`--warmup N` and `--iters N` on `typesafe benchmark` raise the sample count above the default
1 warmup + 3 timed passes (the defaults are unchanged); `--seq` remains `modernbert benchmark`-only,
since it names a sequence shape, not a sample count.

The engines, what a decision model is, `typesafe benchmark`'s protocol and its measured numbers
are in **[docs/DECISION-MODELS.md](../../docs/DECISION-MODELS.md)**.

## Usage

### C# (Nivara)

**Vision models:**
```bash
dotnet run --project samples/NivaraInference -- mobilenet_v2
dotnet run --project samples/NivaraInference -- resnet18
dotnet run --project samples/NivaraInference -- mobilenet_v2 benchmark
dotnet run --project samples/NivaraInference -- mobilenet_v2 path/to/image.jpg
```

**Text models:**
```bash
dotnet run --project samples/NivaraInference -- minilm
dotnet run --project samples/NivaraInference -- minilm benchmark
dotnet run --project samples/NivaraInference -- minilm similarity
dotnet run --project samples/NivaraInference -- distilbert
dotnet run --project samples/NivaraInference -- distilbert benchmark
dotnet run --project samples/NivaraInference -- distilbert_sst
dotnet run --project samples/NivaraInference -- smollm
dotnet run --project samples/NivaraInference -- smollm --precision bf16
dotnet run --project samples/NivaraInference -- qwen tools
dotnet run --project samples/NivaraInference -- qwen distill --teacher-examples 12
dotnet run --project samples/NivaraInference -- qwen benchmark
dotnet run --project samples/NivaraInference -- modernbert
dotnet run --project samples/NivaraInference -- modernbert benchmark
dotnet run --project samples/NivaraInference -- laya
```

### Argument validation

Every option is validated before the first weight file is touched: an unrecognised model or mode, a
flag whose consuming path is not selected, or a narrow `--precision` / `--simd-widen` that the
selected run does not read exits `1` with a message naming the requirement instead of silently
running something else. In particular:

- Each model accepts a fixed mode set — `minilm` `benchmark`/`similarity`, `qwen` `tools`/`distill`/`benchmark`, and so on; a vision model also accepts the path of an existing image. Anything else (e.g. `minilm garbag`, `qwen garbag`) is rejected rather than falling through to the model's default runner.
- `--precision f32|bf16|fp16` is accepted only where the dispatch selects a BFloat16/Half runner (`minilm`, `distilbert`, `distilbert_sst`, `smollm`, `modernbert`, `laya`; `qwen` for `benchmark`). `f32` is always accepted. The vision models reject bf16/fp16: they have no narrow path. Even where a bf16 runner exists, `--silk --precision fp16` is rejected (no measured speedup); use bf16.
- `--simd-widen` is an A/B lever for those narrow runs only. `smollm` and `qwen` enable widen for
  their narrow runs themselves (use `smollm ab`), and an F32 run is unchanged, so the flag is
  rejected there.

```bash
dotnet run --project samples/NivaraInference -c Release -- minilm --precision bf16          # accepted (narrow runner)
dotnet run --project samples/NivaraInference -c Release -- minilm --precision bf16 --simd-widen  # accepted (A/B lever)
dotnet run --project samples/NivaraInference -c Release -- mobilenet_v2 --precision bf16    # rejected: no narrow path
dotnet run --project samples/NivaraInference -c Release -- smollm --simd-widen              # rejected: auto-enabled
```

### Python (PyTorch reference)

```bash
cd samples/NivaraInference/Python
pip install -r requirements.txt

python mobilenet.py
python resnet18.py
python minilm_benchmark.py
python distilbert_benchmark.py
python modernbert_compare.py
python modernbert_benchmark.py   # --seq N[,N...] singles out one length (default 128,256)
python laya_compare.py
python laya_benchmark.py
python generate_input.py
```

## SafeTensors loader

The sample includes a custom zero-dependency `SafeTensorsLoader` that parses the HuggingFace SafeTensors binary format directly:

- **Memory-mapped file loading** (`MemoryMappedFile` + `CreateViewAccessor` + `AcquirePointer`): the string-path loads memory-map the safetensors file and read each tensor's bytes directly from the mapped view — no full-file `byte[]` is materialized, so peak managed memory is just the widened tensors (~1.88 GB for Qwen, ~0.94 GB below the old copy-into-`byte[]` load). Parses the JSON header from the first 8 bytes + offset table via `System.Text.Json`.
- **Dtype support** — loads **F32** (native `float`), **F16** (`System.Numerics.Half`), and **BF16** (`System.Numerics.BFloat16`) tensors, converting each to the requested result type `T` via `T.CreateChecked`. Narrow on-disk dtypes are widened when `T` is wider (e.g. a BF16 checkpoint read as `float[]` widens losslessly), and a wider on-disk dtype is narrowed when `T` is `BFloat16` (e.g. the `bf16` run mode reads the on-disk F32 weights as `BFloat16`, truncating to genuine 7-bit mantissa). Any other dtype raises `NotSupportedException` with guidance.

See [docs/SAFETENSORS.md](../../docs/SAFETENSORS.md) for additional format details.

## OpenBLAS acceleration (opt-in)

OpenBLAS is an opt-in GEMM provider in `Nivara.Extensions.Blas` (see `docs/BLAS-MKL.md`). Default off; the managed path remains the fallback. Requires `lib/libopenblas.dll` at the repo root.

`--threads N` is the Nivara row-block count. `--blas-threads N` is `openblas_set_num_threads`. Both are optional and default to 1, so `--blas` alone is one row-block and one OpenBLAS thread. The same switch covers single-row GEMV (`aRows == 1`); that call has no row-blocks, so `--blas-threads` is its only pool. `--blas-threads` was not swept on the sentence benchmarks; the default stays 1. Both above 1 is two pools and is printed, not rejected.

```bash
# managed (default)
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark
# BLAS (row-blocks; OpenBLAS pool stays 1 unless --blas-threads is set)
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark --blas --threads 2
dotnet run --project samples/NivaraInference -c Release -- modernbert benchmark --blas --threads 4
# OpenBLAS pool, one Nivara block
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark --blas --threads 1 --blas-threads 4
```

Recommended `--threads` (measured on Intel Core Ultra 7 255H, 16 logical cores, AC power, 2026-11-11):

| Model | Recommended `--threads` | Notes |
|-------|------------------------|-------|
| MiniLM (128 tok) | 2 (or skip BLAS) | ~1.0x at 2; ≥4 slower |
| ModernBERT (seq=128) | 4 | 2.6x vs managed |
| DistilBERT | 4 (not swept) | 1.15x at 1 thread |
| Laya (max_len=512) | 4 (not swept) | ~1.0x at 1 thread |

End-to-end results (managed vs BLAS, ms, same machine):

| Model | managed | BLAS (best threads) | Speedup |
|-------|---------|---------------------|---------|
| MiniLM (128 tok) | 127.8 | 130.3 (2 thr) | ~1.0x |
| DistilBERT (128 tok) | 321.6 | 278.8 (1 thr) | 1.15x |
| ModernBERT (seq=128) | 2422.5 | 927.8 (4 thr) | 2.61x |
| ModernBERT (seq=256) | 4079.0 | 1599.7 (4 thr) | 2.55x |
| Laya (max_len=512) | ~8.3-8.8 s | ~8.4-8.7 s | ~1.0x |

Interpretation: end-to-end improvement 1.0-2.6x depending on model, below the issue #1119 acceptance bar (2x) for most. The ~6x GEMM-only win (docs/BLAS-MKL.md §5.1) does not translate to end-to-end because GEMM is only ~1/6 of the gap (#458). Issue #1119 part 1.

## Performance benchmarks

Measured on the same machine (CPU-only, no GPU): Intel Core Ultra 7 255H (16 logical processors), Nivara in Release mode, PyTorch with MKL-optimized kernels. Batch size 1, 3-pass warmup + 10 timed passes, median reported - except the SmolLM and ModernBERT rows, which use 1 warmup + 3 timed on the Nivara side (see the note under the table). Both frameworks are always measured in the same session.

| Model | Input | PyTorch | Nivara CPU | Slowdown (CPU) | Nivara BLAS | Slowdown (BLAS) | Nivara GPU | Slowdown (GPU) |
|-------|-------|---------|------------|----------------|-------------|-----------------|------------|----------------|
| **MobileNetV2** | 1x3x224x224 | 22.4 ms | 463.5 ms | **~20.7x** | — (conv) | — |  |  |
| **ResNet-18** | 1x3x224x224 | 14.1 ms | 123.3 ms | **~8.7x** | — (conv) | — |  |  |
| **MiniLM-L6** | 128 tokens | 11.0 ms | 78.7 ms | **~7.2x** | 130.3 ms (2 thr) | **~11.8x** | 24.1 ms | **~2.2x** |
| **DistilBERT** | 128 tokens | 26.9 ms | 200.6 ms | **~7.5x** | 278.8 ms (1 thr) | **~10.4x** | 63.3 ms | **~2.4x** |
| **DistilBERT SST-2** | 128 tokens | 32.8 ms | 179.4 ms | **~5.5x** | — | — | 63.8 ms | **~1.9x** |
| **SmolLM-135M** (F32 greedy gen) | 5 prompt + 32 new tokens | 881 ms | 814 ms | **~0.9x** | — | — |  |  |
| **SmolLM-135M** (BF16 greedy gen) | 5 prompt + 32 new tokens | 757 ms | 1041 ms | **~1.4x** | — | — |  |  |
| **ModernBERT-large** | 128 tokens | 312.4 ms | 1457.5 ms | **~4.7x** | 927.8 ms (4 thr) | **~3.0x** |  |  |
| **ModernBERT-large** | 256 tokens | 574.4 ms | 2084.6 ms | **~3.6x** | 1599.7 ms (4 thr) | **~2.8x** |  |  |

*Cur* = 2026-11-07, both frameworks measured in one session from the captured `ni-*` / `py-*` `%TEMP%` outputs, same machine, Nivara .NET 11.0.0 Release. Transformer rows: 128-token single forward pass; SmolLM rows report 32 generated tokens (median of 3 runs - 1 warmup + 3 timed Nivara, 3 warmup + 3 timed PyTorch); ModernBERT rows are medians under 1 warmup + 3 timed passes. PyTorch MobileNetV2/ResNet-18/SST-2 figures are carried from the 2026-09-27 capture for unchanged-weight runs; no fresh capture was available for them in this prep. `modernbert benchmark --warmup 3` matches the Python side's protocol. SmolLM F32 = BF16 checkpoint widened to F32 (513.1 MB); SmolLM BF16 = BF16-native on disk (256.6 MB).

`--seq N[,N...]` times a chosen subset of the benchmark's lengths (`--warmup N` and `--iters N` set the pass counts) on both C# paths, and the PyTorch script takes the same `--seq N[,N...]` (default `128,256`), so one row can be re-measured on both sides of the CPU comparison; the defaults reproduce the tables above unchanged. The CPU path no longer builds a dense `[L, L]` mask (#476), so `--seq` is not capped at 2048 on the CPU side.

```bash
dotnet run --project samples/NivaraInference -c Release -- modernbert benchmark --seq 512
dotnet run --project samples/NivaraInference -c Release -- modernbert --silk benchmark --seq 2048,4096 --iters 10 --warmup 3
python modernbert_benchmark.py --seq 256
```

GPU/OpenCL (`--silk`) rows: fill the Silk column from a same-session capture of the
commands in `## Release Benchmark` above (AC power). Do not carry forward battery
measurements or prior one-off numbers; if no capture is available, leave the cells
as `—` and state the reason in the PR. See `docs/ACCELERATION.md` and `docs/SILK.md`
for published iGPU baselines. Record the OpenCL device/vendor and driver version
alongside the machine line.



These three run 3 warmup + 10 timed passes and report an **average**; `modernbert --silk benchmark` reports a **median** of 3, so its rows are not directly comparable to them (the run prints this too, and `--warmup 3 --iters 10` adopts their protocol). ModernBERT's own GPU rows are in [docs/ACCELERATION.md](../../docs/ACCELERATION.md) §1.

The MiniLM row above is the AC-powered historical `` baseline (backend removed in 1.5.0). The second MiniLM backend (`--silk`) was compared to it **on battery, both legs in one session** (per-kernel in one process, whole forward back-to-back, same 3 + 10 protocol): no separation (whole forward ≈0.94×, per-kernel 0.79–1.19×). Battery numbers are not quotable rows — the figures and the caveat are in [docs/ACCELERATION.md](../../docs/ACCELERATION.md) §1.

See [docs/ACCELERATION.md](../../docs/ACCELERATION.md) for full GPU details.


## Sample data

| File | Purpose |
|------|---------|
| `samples/data/mobilenet_v2/model.safetensors` | MobileNetV2 weights (~13.5 MB) |
| `samples/data/resnet18/model.safetensors` | ResNet-18 weights (~44.6 MB) |
| `samples/data/minilm/model.safetensors` | MiniLM weights (~87 MB) |
| `samples/data/distilbert/model.safetensors` | DistilBERT weights (~255.5 MB, 105 tensors) |
| `samples/data/distilbert_sst/model.safetensors` | Fine-tuned DistilBERT SST-2 weights (~255.4 MB, 104 tensors) |
| `samples/data/smollm-135m/model.safetensors` | SmolLM-135M-Instruct weights (~269 MB, 272 tensors, all BF16) |
| `samples/data/qwen2.5-0.5b-instruct/model.safetensors` | Qwen2.5-0.5B-Instruct weights (~989 MB, 290 tensors, BF16 on disk) |
| `samples/data/modernbert/model.safetensors` | ModernBERT-large weights (~1510 MB F32, 170 encoder tensors) |
| `samples/data/laya/model.safetensors` | Laya weights (842.6 MB F16, 206 tensors; encoder under `encoder.*`) |

Reference fixtures (generated by `Python/*_compare.py`, not checked in) are listed in each model's doc.

## Nivara capabilities exercised

See the per-model docs for full capability tables:

- [docs/VISION.md](../../docs/VISION.md) — MobileNetV2, ResNet-18
- [docs/DISTELBERT.md](../../docs/DISTELBERT.md) — DistilBERT, MiniLM, SST-2
- [docs/MODERNBERT.md](../../docs/MODERNBERT.md) — ModernBERT-large
- [docs/SMOLLM.md](../../docs/SMOLLM.md) — SmolLM-135M-Instruct
- [docs/QWEN.md](../../docs/QWEN.md) — Qwen2.5-0.5B-Instruct
- [docs/LAYA.md](../../docs/LAYA.md) — Laya

## Release Benchmark

Run this during release prep (step 5 of `RELEASING.md`). Requires Python, PyTorch, and HuggingFace model weights (see Quick start for `hf download` commands).

Run both sides in the same session for fair comparison:

```powershell
# Nivara (C#) — one pass per model (CPU)
dotnet run --project samples/NivaraInference -c Release -- mobilenet_v2 benchmark
dotnet run --project samples/NivaraInference -c Release -- resnet18 benchmark
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst benchmark
dotnet run --project samples/NivaraInference -c Release -- smollm benchmark                     # F32
dotnet run --project samples/NivaraInference -c Release -- smollm --precision bf16 benchmark   # native BF16
dotnet run --project samples/NivaraInference -c Release -- qwen benchmark                      # KV-cached vs full re-forward
dotnet run --project samples/NivaraInference -c Release -- modernbert benchmark                 # seq 128 + 256

# Silk (OpenCL) — required in the same session as the CPU run, on AC power
dotnet run --project samples/NivaraInference -c Release -- minilm --silk benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert --silk benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst --silk benchmark
dotnet run --project samples/NivaraInference -c Release -- laya --silk benchmark
dotnet run --project samples/NivaraInference -c Release -- modernbert --silk benchmark             # seq 128 + 512 + 2048 + 4096; --seq selects a subset
dotnet run --project samples/NivaraInference -c Release -- qwen --silk benchmark                  # KV-cached vs full re-forward
dotnet run --project samples/NivaraInference -c Release -- qwen --silk --plain benchmark          # plain-prompt decode path
dotnet run --project samples/NivaraInference -c Release -- smollm --silk benchmark                # F32 generation

# PyTorch (Python) — run immediately after on the same machine
cd samples/NivaraInference/Python
python minilm_benchmark.py
python distilbert_benchmark.py
python modernbert_benchmark.py                 # --seq N[,N...] selects a subset of 128,256
python smollm_generate_reference.py --dtype float32
python smollm_generate_reference.py --dtype bfloat16
```

Keep both dtype pairs same-dtype on CPU so the ratio and the F32-vs-BF16 memory/performance tradeoff are meaningful.

**Update the tables:**
1. CPU tables: shift existing **Current** values to **Prev**, place fresh measurements in
   **Current**, compute **Ratio** and **Δ%**.
2. Silk tables: use the same Prev/Current/Ratio/Δ% treatment for the Silk column, in the
   same units as the CPU column, from the same machine and revision.
3. Keep `—` only where no captured `--silk benchmark` output exists, with the reason in
   the table footer.
4. Update the machine/recording line to include AC power and the OpenCL device/vendor and
   driver version for every Silk row.


## TypeSafe (in-process)

`typesafe nivara` answers the same news question set as the HTTP engines, but runs the Laya head on the Silk backend in-process.

```powershell
dotnet run --project samples/NivaraInference -c Release -- typesafe nivara
dotnet run --project samples/NivaraInference -c Release -- typesafe nivara benchmark
dotnet run --project samples/NivaraInference -c Release -- typesafe nivara verify
```
