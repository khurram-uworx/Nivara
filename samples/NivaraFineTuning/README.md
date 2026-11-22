# NivaraFineTuning — Fine-tuning showcase

A sample project demonstrating fine-tuning with Nivara's AutoDiff engine — entirely in
C#, no Python runtime required for inference.

- **Track A — DistilBERT on GLUE SST-2** (sequence classification, full-model fine-tune).
- **Track B — LoRA SFT for Llama-family SLMs** (Qwen2.5-0.5B headline, SmolLM generic
  verification; CPU reference + Silk/iGPU path).

**Target audience:** .NET developers exploring transfer learning and fine-tuning pipelines
with Nivara.

## Guides

- **[BERT-TUNING.md](BERT-TUNING.md)** — DistilBERT/SST-2 track: ML concepts for .NET
  developers, architecture, dataset, setup, usage, expected results, performance
  benchmarks, scope of validation, Python reference, key APIs, core library improvements.
- **[LLAMA-TUNING.md](LLAMA-TUNING.md)** — LoRA SFT track: generic Llama-family LoRA
  trainer (CPU + Silk), CLI, SmolLM verification, report artifacts, adapter format, D2a
  parity status.

## What this sample does

1. Loads pre-trained DistilBERT weights (distilbert-base-uncased) from SafeTensors and
   fine-tunes a classification head on GLUE SST-2 (Track A).
2. Runs generic LoRA SFT on Llama-family causal LMs (Qwen2.5-0.5B / SmolLM) over
   deterministic synthetic worklog data, emitting a validated adapter bundle (Track B).
3. Evaluates base vs LoRA-adapted models and writes structured JSON + Markdown reports.
4. Supports CPU reference and Silk/OpenCL (iGPU) paths; adapter bundles round-trip
   between backends.

## How it helped Nivara the library

DistilBERT track (core improvements in `src/Nivara`):

- Block-diagonal attention mask utility for flattened batch encoding.
- SafeTensors-to-module weight mapping with HuggingFace snake_case key translation.
- Graceful loading of random-init heads when fine-tuning from scratch.
- Reflection-based DistilBertConfig JSON parser (snake_case -> PascalCase).
- Shared `DistilBertForSequenceClassification<T>` promoted to `Nivara.Samples`.
- No token-type embeddings for DistilBERT (`includeTokenTypeEmbedding: false`).

LoRA track (core additions in `src/Nivara`):

- `LoRALinear<T>` / `LoRALinear<TModel, TWeight>`: frozen base, Kaiming A / zero B,
  merge, adapter-only gradients.
- `LoraAdapterSerializer` (`nivara-lora-adapter-v1`): single-adapter save/load with
  rank/alpha/base-model/targets/shape validation.
- Multi-adapter SFT bundle (`nivara-lora-sft-adapters-v1`) shared by CPU and Silk
  trainers, with shape/rank/count validation on load.

## Key Nivara APIs exercised

| API | Where | Purpose |
|-----|-------|---------|
| `BertEncoder<T>.ForwardBatched` | Track A | Flattened batch processing with block-diagonal attention mask |
| `Linear<T>` | Both | Projections, FFN, classifier head, LoRA A/B |
| `CrossEntropyLoss<T>` | Both | Classification / prompt-masked causal-LM loss |
| `AdamW<T>` | Both | Optimizer (full-model and adapter-only) |
| `GradientUtils.Grad()` | Both | Opt-in reverse-mode training scope |
| `ModelSerializer.Save/Load` | Track A | Fine-tuned model persistence |
| `SafeTensorsLoader` | Track A | Zero-dependency SafeTensors parser (via `Nivara.Samples`) |
| `Module<T>.Train/Eval` | Both | Training/eval mode toggle |
| `Nivara.IO.NivaraParquetReader` | Track A | Parquet dataset loading (via `Nivara.Extensions`) |

## Setup

See [BERT-TUNING.md](BERT-TUNING.md) for DistilBERT/SST-2 prerequisites and downloads,
and [LLAMA-TUNING.md](LLAMA-TUNING.md) for LoRA SLM prerequisites (model dirs + worklog data).

> **Always run with `-c Release`.** Debug builds run the numeric kernels unoptimized.

## Quick start

DistilBERT (Track A):

```bash
dotnet run -c Release --project samples/NivaraFineTuning -- --mode train --epochs 1 --batch-size 2 --max-examples 25
```

LoRA SFT (Track B, CPU):

```bash
dotnet run -c Release --project samples/NivaraFineTuning -- --mode ft-lora \
  --model-dir samples/data/qwen2.5-0.5b-instruct \
  --train-data samples/data/worklog/train.jsonl \
  --val-data samples/data/worklog/test.jsonl \
  --template qwen --epochs 1 --max-examples 4 --max-len 128 \
  --rank 4 --alpha 8 --lr 2e-4 --out-dir out/worklog-lora
```

See the guides above for full options, benchmarks, and validation scope.