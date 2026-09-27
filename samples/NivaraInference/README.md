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
dotnet run --project samples/NivaraInference -c Release -- qwen benchmark        # KV-cached vs full re-forward
dotnet run --project samples/NivaraInference -c Release -- modernbert            # ModernBERT-large forward (28-layer hybrid)
dotnet run --project samples/NivaraInference -c Release -- modernbert compare    # last_hidden_state parity gate vs PyTorch
dotnet run --project samples/NivaraInference -c Release -- modernbert compare_diag  # per-stage embeddings→layer 27→final norm
dotnet run --project samples/NivaraInference -c Release -- laya                 # decision head over the fixture questions (CPU reference)
dotnet run --project samples/NivaraInference -c Release -- laya compare         # prompt + logit + decision gate vs the laya 0.3.20 wheel

# Benchmark (10 passes each)
dotnet run --project samples/NivaraInference -c Release -- mobilenet_v2 benchmark
dotnet run --project samples/NivaraInference -c Release -- resnet18 benchmark
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst benchmark
dotnet run --project samples/NivaraInference -c Release -- modernbert benchmark  # 3 warmup + 10 timed; --seq 256 doubles the length

# BERT-family encoders on the OpenCL iGPU (ILGPU 1.5.3; --gpu is F32-only in this phase)
dotnet run --project samples/NivaraInference -c Release -- minilm --gpu
dotnet run --project samples/NivaraInference -c Release -- distilbert --gpu
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst --gpu
dotnet run --project samples/NivaraInference -c Release -- modernbert --gpu

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
| GPU acceleration (all models) | [docs/GPU-ACCL.md](../../docs/GPU-ACCL.md) |

## Usage

### C# (Nivara)

**Vision models:**
```bash
dotnet run --project samples/NivaraInference -- mobilenet_v2
dotnet run --project samples/NivaraInference -- resnet18
dotnet run --project samples/NivaraInference -- mobilenet_v2 benchmark
dotnet run --project samples/NivaraInference -- mobilenet_v2 compare
dotnet run --project samples/NivaraInference -- mobilenet_v2 compare_diag
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
dotnet run --project samples/NivaraInference -- distilbert_sst compare
dotnet run --project samples/NivaraInference -- smollm
dotnet run --project samples/NivaraInference -- smollm --precision bf16
dotnet run --project samples/NivaraInference -- qwen tools
dotnet run --project samples/NivaraInference -- qwen distill --teacher-examples 12
dotnet run --project samples/NivaraInference -- qwen benchmark
dotnet run --project samples/NivaraInference -- modernbert
dotnet run --project samples/NivaraInference -- modernbert compare
dotnet run --project samples/NivaraInference -- modernbert compare_diag
dotnet run --project samples/NivaraInference -- modernbert benchmark
dotnet run --project samples/NivaraInference -- laya
dotnet run --project samples/NivaraInference -- laya compare
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
python modernbert_benchmark.py
python laya_compare.py
python generate_input.py
```

## SafeTensors loader

The sample includes a custom zero-dependency `SafeTensorsLoader` that parses the HuggingFace SafeTensors binary format directly:

- **Memory-mapped file loading** (`MemoryMappedFile` + `CreateViewAccessor` + `AcquirePointer`): the string-path loads memory-map the safetensors file and read each tensor's bytes directly from the mapped view — no full-file `byte[]` is materialized, so peak managed memory is just the widened tensors (~1.88 GB for Qwen, ~0.94 GB below the old copy-into-`byte[]` load). Parses the JSON header from the first 8 bytes + offset table via `System.Text.Json`.
- **Dtype support** — loads **F32** (native `float`), **F16** (`System.Numerics.Half`), and **BF16** (`System.Numerics.BFloat16`) tensors, converting each to the requested result type `T` via `T.CreateChecked`. Narrow on-disk dtypes are widened when `T` is wider (e.g. a BF16 checkpoint read as `float[]` widens losslessly), and a wider on-disk dtype is narrowed when `T` is `BFloat16` (e.g. the `bf16` run mode reads the on-disk F32 weights as `BFloat16`, truncating to genuine 7-bit mantissa). Any other dtype raises `NotSupportedException` with guidance.

See [docs/SAFETENSORS.md](../../docs/SAFETENSORS.md) for additional format details.

## Performance benchmarks

Measured on the same machine (CPU-only, no GPU): Intel Core Ultra 7 255H (16 logical processors), Nivara in Release mode, PyTorch with MKL-optimized kernels. Batch size 1, 3-pass warmup + 10 timed passes. Both frameworks are always measured in the same session.

| Model | Input | PyTorch cur | Nivara cur | Slowdown cur |
|-------|-------|-------------|-------------|--------------|
| **MobileNetV2** | 1×3×224×224 | 22.4 ms | 731.7 ms | **~33×** |
| **ResNet-18** | 1×3×224×224 | 14.1 ms | 249.4 ms | **~18×** |
| **MiniLM-L6** | 128 tokens | 10.9 ms | 71.2 ms | **~6.5×** |
| **DistilBERT** | 128 tokens | 32.8 ms | 209.7 ms | **~6.4×** |
| **DistilBERT SST-2** | 128 tokens | 32.8 ms | 199.9 ms | **~6.1×** |
| **SmolLM-135M** (F32 greedy gen) | 5 prompt + 32 new tokens | 947 ms | 7404 ms | **~7.8×** |
| **SmolLM-135M** (BF16 greedy gen) | 5 prompt + 32 new tokens | 806 ms | 11499 ms | **~14×** |
| **ModernBERT-large** | 128 tokens | 287.3 ms | 1249.8 ms | **~4.3×** |
| **ModernBERT-large** | 256 tokens | 504.0 ms | 1922.4 ms | **~3.8×** |

*Cur* = 2026-09-27 on **AC power**, both frameworks in one session, same machine, Nivara .NET 11.0.0 Release, PyTorch 2.13.0+cpu. Transformer rows: 128-token single forward pass (3 warmup + 10 timed); SmolLM and ModernBERT report the median of 3 runs. SmolLM F32 = BF16 checkpoint widened to F32 (513.1 MB); SmolLM BF16 = BF16-native on disk (256.6 MB).

**GPU (iGPU, `--gpu`)** — same machine, Arc 140T-class iGPU via ILGPU 1.5.3 (OpenCL), F32 only, AC power:

| Model | GPU Nivara | CPU Nivara | vs CPU |
|-------|-----------|------------|--------|
| **MiniLM** | 24.1 ms | 71.2 ms | **~3.0× faster** |
| **DistilBERT** | 63.3 ms | 209.7 ms | **~3.3× faster** |
| **DistilBERT SST-2** | 63.8 ms | 199.9 ms | **~3.1× faster** |

See [docs/GPU-ACCL.md](../../docs/GPU-ACCL.md) for full GPU details.

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
# Nivara (C#) — one pass per model
dotnet run --project samples/NivaraInference -c Release -- mobilenet_v2 benchmark
dotnet run --project samples/NivaraInference -c Release -- resnet18 benchmark
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst benchmark
dotnet run --project samples/NivaraInference -c Release -- smollm benchmark                     # F32
dotnet run --project samples/NivaraInference -c Release -- smollm --precision bf16 benchmark   # native BF16
dotnet run --project samples/NivaraInference -c Release -- qwen benchmark                      # KV-cached vs full re-forward
dotnet run --project samples/NivaraInference -c Release -- modernbert benchmark                 # seq 128 + 256

# PyTorch (Python) — run immediately after on the same machine
cd samples/NivaraInference/Python
python minilm_benchmark.py
python distilbert_benchmark.py
python modernbert_benchmark.py
python smollm_generate_reference.py --dtype float32
python smollm_generate_reference.py --dtype bfloat16
```

Keep both dtype pairs same-dtype on CPU so the ratio and the F32-vs-BF16 memory/performance tradeoff are meaningful.
