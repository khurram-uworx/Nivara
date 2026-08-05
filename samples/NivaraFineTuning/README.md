# NivaraFineTuning — DistilBERT Fine-Tuning on GLUE SST-2

A sample project demonstrating fine-tuning a pre-trained DistilBERT model for sequence classification using Nivara's AutoDiff engine — entirely in C#, no Python runtime required for inference.

**Target audience:** .NET developers exploring transfer learning and fine-tuning pipelines with Nivara.

## What this sample does

1. Loads pre-trained DistilBERT weights (distilbert-base-uncased) from SafeTensors format
2. Builds a sequence classification head on top of the frozen encoder
3. Loads the GLUE SST-2 sentiment dataset via Parquet files
4. Fine-tunes the classification head (full model) using AdamW optimizer
5. Evaluates on the dev set and reports loss + accuracy
6. Saves and reloads the fine-tuned model via `ModelSerializer`
7. Interactive sentiment prediction mode

## Architecture

```
Input: [B×L] token IDs + [B×L] attention mask
  │
  ├─ wordEmbed:     Embedding(vocabSize=30522, dim=768)
  ├─ posEmbed:      Embedding(maxPos=512, dim=768)  ← per-sequence repeated [0..L-1]
  ├─ embedLn:       LayerNorm(768)
  │
  ├─ BertEncoder × 6 layers
  │   ├─ ln1 → BertSelfAttention (12 heads, block-diagonal mask) → residual
  │   └─ ln2 → Linear(768→3072) → GELU → Linear(3072→768) → residual
  │
  ├─ Extract [CLS] tokens: gather positions [0, L, 2L, ...] → [B, 768]
  ├─ preClassifier: Linear(768→768) → ReLU
  └─ classifier:    Linear(768→numLabels) → [B, numLabels]
```

The classification head uses **ReLU** (matching HF `DistilBertForSequenceClassification`, which applies `nn.ReLU()` after `pre_classifier`) via `ReverseGradOperations.Relu`. The shared `DistilBertForSequenceClassification<T>` (in `Nivara.Samples`) is used by both this sample and the `distilbert_sst` inference showcase.

### Batched attention

The encoder uses `BertEncoder<T>.ForwardBatched` with a block-diagonal attention mask that:
- Isolates cross-sequence positions (masked to `-inf`) so each sequence only attends within itself
- Applies padding mask to ignore pad tokens within each sequence
- Position IDs are `[0, 1, ..., L-1]` repeated `B` times (not `[0..B×L-1]`)

## Dataset

[GLUE SST-2](https://huggingface.co/datasets/nyu-mll/glue) — Stanford Sentiment Treebank v2, binary sentiment classification:

| Split | Examples | Format |
|-------|----------|--------|
| Train | 67,349 | Parquet (`train-00000-of-00001.parquet`) |
| Dev   | 872     | Parquet (`validation-00000-of-00001.parquet`) |

Labels: `0` = negative, `1` = positive.

The data is loaded via `Nivara.IO.ParquetReader` from `Nivara.Extensions` (Parquet.Net 6.0.3).

## Setup

### Prerequisites

- .NET 10.0 SDK
- HuggingFace CLI (`hf`): `pip install huggingface-hub`
- Python 3.10+ (for data downloads and reference script)

### Download DistilBERT model

```bash
hf download distilbert-base-uncased --local-dir samples/data/distilbert
```

This downloads:
- `samples/data/distilbert/config.json` — model configuration
- `samples/data/distilbert/model.safetensors` — pre-trained weights (6 layers, 67M params)
- `samples/data/distilbert/vocab.txt` — BERT tokenizer vocabulary

### Download SST-2 dataset

```bash
hf download --repo-type dataset nyu-mll/glue sst2/train-00000-of-00001.parquet sst2/validation-00000-of-00001.parquet --local-dir samples/data/sst2
```

This downloads Parquet files to `samples/data/sst2/`.

## Usage

> **Always run with `-c Release`.** The default `dotnet run` configuration is Debug
> (no JIT optimization), which runs the numeric kernels several times slower than
> Release. Use `dotnet run -c Release --project samples/NivaraFineTuning -- ...`.
> The sample enables Server GC and Tiered PGO to reduce GC pauses under the
> ~1 GB/batch allocation churn of a 67M-parameter training step.

### Train

```bash
dotnet run -c Release --project samples/NivaraFineTuning -- --mode train --epochs 3 --batch-size 4
```

Trains for 3 epochs with AdamW (lr=2e-5, weight_decay=0.01), reports per-epoch training loss and dev accuracy. Saves the fine-tuned model to `samples/data/distilbert/finetuned_model.json`.

```bash
# Quick smoke test (1 epoch, batch size 2)
dotnet run -c Release --project samples/NivaraFineTuning -- --mode train --epochs 1 --batch-size 2
```

### Evaluate

```bash
dotnet run -c Release --project samples/NivaraFineTuning -- --mode eval
```

Loads the fine-tuned model from `samples/data/distilbert/finetuned_model.json`, runs inference on the SST-2 dev set, reports loss and accuracy.

### Interactive prediction

```bash
dotnet run -c Release --project samples/NivaraFineTuning -- --mode predict
```

Type sentences and get POSITIVE/NEGATIVE with confidence percentage. Type `quit` to exit.

### Custom options

```bash
dotnet run --project samples/NivaraFineTuning -- \
  --mode train \
  --epochs 5 \
  --lr 3e-5 \
  --batch-size 8 \
  --max-len 256 \
  --model-dir ./my-distilbert \
  --data-dir ./my-data \
  --save-path ./my-model.json
```

## Expected results

On a single CPU core (no GPU):

| Epoch | Training loss (avg) | Dev accuracy |
|-------|-------------------|--------------|
| 1     | ~0.45             | ~78-82%      |
| 2     | ~0.35             | ~80-84%      |
| 3     | ~0.30             | ~81-85%      |

Accuracy target: >75% (above random baseline), with 3 epochs typically reaching 80-85%.

### Expected timing

Fine-tuning runs entirely in managed C# on CPU — this is a correctness/transfer-learning
demonstration, not a throughput benchmark. For `--max-examples 100 --batch-size 2 --epochs 1 -c Release`:
expect roughly 1-3 seconds per batch (forward + backward + AdamW through 67M params),
i.e. on the order of 10-30 minutes per full 67K-example epoch on a modern multi-core CPU.
Keep `--max-len` at the default 128 and use `--max-examples` to validate the pipeline
before committing to a full run. PyTorch (BLAS-accelerated) is typically 10-50x faster on the same hardware.

## Python reference

A PyTorch reference implementation is provided for accuracy comparison:

```bash
cd samples/NivaraFineTuning/Python
pip install -r requirements.txt
python finetune_distilbert.py
```

Uses identical hyperparameters (lr=2e-5, epochs=3, batch_size=4, max_len=128) with `transformers.AutoModelForSequenceClassification` and `datasets` SST-2 loader.

## Key Nivara APIs exercised

| API | Usage |
|-----|-------|
| `BertEncoder<T>.ForwardBatched` | Flattened batch processing with block-diagonal attention mask |
| `Embedding<T>` | Token embedding (gather path) and position embedding |
| `Linear<T>` | Q/K/V/O projections, FFN layers, pre-classifier, classifier head |
| `LayerNorm<T>` | Pre-norm layer normalization with configurable epsilon |
| `ReverseGradOperations.GeluExact` | GELU activation in the encoder FFN intermediate (exact erf) |
| `ReverseGradOperations.Relu` | Classification-head activation (matches HF `nn.ReLU`) |
| `ReverseGradOperations.Gather` | [CLS] token extraction from batched encoder output |
| `AdamW<T>` | Parameter-efficient fine-tuning with weight decay |
| `CrossEntropyLoss<T>` | Multi-class classification loss with integer labels |
| `GradientUtils.Grad()` | Reverse-mode autograd scope for training |
| `ModelSerializer.Save/Load` | Fine-tuned model persistence |
| `SafeTensorsLoader` | Zero-dependency SafeTensors binary parser (via `Nivara.Samples`) |
| `Module<T>.Train/Eval` | Training/eval mode toggle |
| `Nivara.IO.ParquetReader` | Parquet dataset loading (via `Nivara.Extensions`) |

## Core library improvements made during implementation

- **Block-diagonal mask utility**: `BuildBlockDiagonalMask` in `BertSelfAttention<T>` — handles flattened batch encoding with per-sequence attention isolation. Candidate for promotion to `ModuleHelpers<T>` if similar patterns arise.
- **DistilBERT weight mapping**: SafeTensors-to-module parameter mapping with HuggingFace snake_case key translation (`pre_classifier`, `sa_layer_norm`, `ffn.lin1/lin2`, etc.)
- **Loading random-init heads gracefully**: `LoadWeights` checks for classifier/pre_classifier keys before loading, allowing random initialization when fine-tuning from scratch
- **DistilBertConfig JSON parser**: Maps HuggingFace `snake_case` config keys to PascalCase C# properties via reflection-based parsing
- **Shared model promotion**: `DistilBertForSequenceClassification<T>` + `DistilBertConfig` now live in `Nivara.Samples` and are shared by this sample and the `distilbert_sst` inference showcase (`samples/NivaraInference`)
- **No token-type embeddings**: the classifier constructs its encoder with `includeTokenTypeEmbedding: false` (DistilBERT never feeds segment ids); the previous default added a random token-type embedding that degraded frozen-encoder parity
