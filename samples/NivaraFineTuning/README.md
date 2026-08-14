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

## ML concepts for .NET developers

New to language-model training loops? The hyperparameters in this sample map
cleanly onto ideas you already know from C#:

| ML term | What it is | C# mental model | This sample |
|---------|------------|-----------------|-------------|
| **Batch (B)** | Number of examples processed before one weight update | Rows per `foreach` iteration | `--batch-size 4` |
| **Epoch** | One complete pass over the training set | One full `for` loop over all rows | `--epochs 3` |
| **Sequence length (L)** | Fixed width every input row is padded/truncated to | A rectangular array dimension | `--max-len 128` |
| **Token** | A subword unit produced by the tokenizer | A key looked up in `vocab.txt` | word pieces |

- **Batch size (B).** The model does not adjust its weights after every single
  sentence. It processes `B` sentences, aggregates their error, and takes one
  optimization step. `B=2` means 67,349 / 2 ≈ 33,675 weight updates per epoch.
  Bigger batches = fewer steps and better vectorized-kernel utilization, but
  more memory and autograd graph per step.

- **Epoch.** A single pass through the entire training set. The default run uses
  3 epochs, so each sentence is seen 3 times (re-shuffled between passes) —
  which is why a full run costs roughly 3× a single epoch.

- **Token vs. character.** The model never reads raw text. The tokenizer splits
  each sentence into subword units first — e.g. `"flawlessly"` becomes the two
  tokens `["flaw", "##lessly"]`. The average SST-2 sentence is only ~19 tokens,
  but a long tail of examples exceeds 100.

- **Sequence length (L).** A hard cap on the number of tokens per sentence.
  Every row is padded to `L` (typically with `[PAD]` tokens) so a batch is a
  rectangular `[B × L]` matrix the kernels can process in one shot. `128` is the
  GLUE convention: it covers ~99.9% of sentences at roughly 10× less compute
  than BERT's 512-token pretraining width. The trade-off: compute scales with
  `L`, not with actual content, so a 19-token sentence still pays for 128
  positions — most of the attention work is spent on padding.

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

> The accuracy figures below are **targets** from reference runs, not
> measurements this sample establishes — it validates on a small slice rather
> than the full dataset. See
> [Scope of validation](#scope-of-validation-proof-of-concept-not-a-benchmark-claim).

On a single CPU core (no GPU):

| Epoch | Training loss (avg) | Dev accuracy |
|-------|-------------------|--------------|
| 1     | ~0.45             | ~78-82%      |
| 2     | ~0.35             | ~80-84%      |
| 3     | ~0.30             | ~81-85%      |

Accuracy target: >75% (above random baseline), with 3 epochs typically reaching 80-85%.

### Expected timing

Fine-tuning runs entirely in managed C# on CPU. For `--max-examples 25 --batch-size 2 --epochs 1 -c Release`
expect **~3.4 s/batch** steady-state (measured 2026-08-14, see [Performance benchmarks](#performance-benchmarks));
a full 67K-example epoch extrapolates to ~32 hours. Keep `--max-len` at the default 128 and use
`--max-examples` to validate the pipeline before committing to a full run.

## Performance benchmarks

Measured on the same machine (CPU-only, no GPU). Nivara runs in Release mode
with Server GC + Tiered PGO (as configured in the sample project). PyTorch uses
MKL-optimized kernels with `torch_threads = nproc`. Both sides fine-tune
DistilBERT-base (67M params) on the first `--max-examples` rows of SST-2 at
`--batch-size 2 --max-len 128` and report **steady-state ms/batch**: PyTorch
runs 2 untimed warmup epochs before timing; Nivara excludes the first (JIT
warmup) batch. `--seed 0` fixes the training shuffle so A/B comparisons are
reproducible. Numbers vary with machine load — re-measure both sides in the
same session when comparing (run-to-run variance ~±10% per
`tests/Nivara.PerformanceTests/README.md`).

Recorded **2026-08-14** on an 11th-gen Intel i5-1135G7 laptop (4P/8T, Windows
11, `torch_threads=4`). The 2026-08-06 figures were from a different (faster)
machine, so the absolute s/batch below are **not comparable** to that table;
the ~3× ratio holds on both machines.

| Config | PyTorch (CPU) | Nivara (.NET 10) | Slowdown |
|--------|---------------|-------------------|----------|
| Fine-tune B=2, L=128, 25 examples | 1.16 s/batch | 3.4 s/batch | **~3×** |

The gap is far smaller than a naive port suggests: at batch size 2 the per-batch
cost is dominated by 38 small Linear matmuls and the backward pass through 67M
params, not by BLAS peak throughput, so the CPU SIMD kernels and memory
management keep Nivara within ~3× of PyTorch's MKL on this configuration.

### Estimated full-run wall-clock (approximate)

> **Disclaimer:** the figures below are **estimates, not measurements.** They
> extrapolate the 25-example per-batch numbers above to the full 67,349-example
> train set and assume per-batch cost grows linearly with batch size — which is
> approximate. Real runs drift with machine load, GC pressure, and thermals;
> treat them as ballpark planning figures (±30%).

| Config | Framework | 1 epoch (est.) | 3 epochs (est.) |
|--------|-----------|----------------|-----------------|
| B=2, L=128 | PyTorch (CPU) | ~11 h | ~32 h |
| B=2, L=128 | Nivara (.NET 10) | ~32 h | ~96 h |
| B=4, L=128 | PyTorch (CPU) | ~10 h | ~31 h |
| B=4, L=128 | Nivara (.NET 10) | ~30 h | ~90 h |

B=4 figures are not benchmarked end-to-end — they're derived from the measured
B=2 throughput. Batch cost grows sub-linearly in B (better kernel utilization),
so the per-epoch times above are the same order of magnitude, not exact.

### Before/after (2026-08-06)

| Nivara fine-tune B=2, 25 examples | Before (`e031ff0`) | After (HEAD) |
|-----------------------------------|--------------------|--------------|
| Steady-state wall-clock | ~1.3 s/batch | ~1.4 s/batch |
| Per-param optimizer allocation | 268 MB/batch (`new T[n]` per param per step) | eliminated (in-place writes) |

"Before" is the last commit before the PERF-66 performance work
(`e031ff0`); "after" is current HEAD. The work — in-place SGD/Adam/AdamW
steps (write into the parameter's backing array + version bump, no per-param
`new T[n]`), `BatchedMultiHeadAttention` in the encoder (no block-diagonal
mask, no wasted cross-sequence compute), grad-tracking transposed-B matmul in
`Linear` (no per-forward weight transpose), and a tiled `Transpose` kernel —
kept wall-clock flat (~1.3 → ~1.4 s/batch, within the ±10% run-to-run
variance; the A/B was measured with the pre-warmup harness) while removing
the dominant per-batch allocation. The remaining GC pressure is grad-array
churn. Further optimizations to validate against this harness: pooled grad
buffers and a single-pass attention row kernel.

### Re-running

```bash
cd samples/NivaraFineTuning

# Both sides, tee'd to benchmark_results.txt (uses GNU coreutils tee on PATH)
.\benchmark_timing.cmd 25 2 1

# PyTorch side only
python Python\benchmark_timing.py --epochs 1 --batch-size 2 --max-examples 25

# Nivara side only (batch times are printed per batch; batch 1 is JIT warmup)
dotnet run -c Release --project samples/NivaraFineTuning -- --mode train --epochs 1 --batch-size 2 --max-examples 25
```

Record new results in the table above with the measurement date and a note on
what changed. Kernel-level micro-measurements (matmul/transpose/softmax/attention)
live in `tests/Nivara.PerformanceTests`.

## Scope of validation: proof of concept, not a benchmark claim

This sample is deliberately **validated on a tiny slice** of SST-2
(`--max-examples 25`, batch size 2, one epoch). That was an explicit choice, and
it deserves an explicit note.

**What we set out to prove.** That the entire fine-tuning pipeline works
end-to-end in managed C# — SafeTensors weight loading, the frozen-encoder
forward pass, the backward pass through 67M parameters, AdamW parameter updates,
full dev-set evaluation, and save/reload. That is a *correctness* question, and
it does not require the full dataset to answer.

**What was actually measured.** On the 25-example slice, training loss fell from
~0.75 to ~0.70 across the 13 batches of a single epoch, and evaluation on the
*full* 872-example dev set returned **60.09% accuracy** — comfortably above the
50% coin-flip baseline. Those numbers show the loop is learning and the numerics
line up; they are not a claim about final model quality.

**What we deliberately did not run.** A full SST-2 fine-tune on CPU costs ~32
hours per epoch (67,349 examples at ~3.4 s/batch), so a complete 3-epoch run is
~4 days of wall-clock. We judged that against the goal — proving technical
correctness — and chose the 25-example slice instead. The first example of a
training slice already exercises every layer, every gradient, and every
optimizer step the full run would; what a slice cannot tell you is *accuracy*,
which is why the harness still evaluates on the full dev set.

**Where performance stands.** On this machine Nivara is ~3× slower than
PyTorch's MKL kernels at this configuration (3.4 s vs 1.16 s per batch). The
honest summary: *technically correct, not yet performant.* The numerics are
right, the pipeline is complete, and the benchmark harness in the previous
section is the tool for chasing the gap.

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
