# NivaraClassifier — Word-Level Text Classifier

A sample project demonstrating Nivara's autograd engine for text classification. Trains a word-level text classifier using learned embeddings and an MLP head — the natural next step after MicroGpt on the NLP path.

**Target audience:** ML engineers, NLP practitioners, .NET developers exploring .NET-native ML.

## What it does

NivaraClassifier trains a model to predict sentiment (positive/negative) from text. It showcases:

- **Reverse-mode autograd** with structured training (`TrainingLoop<T>`, `DataLoader<T>`)
- **Learned embeddings** (`Embedding<T>`) as a first-class layer for word-level features
- **Batched embedding lookup** via the `Embedding<T>.Forward(ReverseGradTensor<T>)` overload
- **MeanPool** — new core autograd operation for `[B, L, D]` → `[B, D]` sequence reduction
- **CrossEntropyLoss<T>** with integer labels for multi-class classification
- **Model serialization** (`ModelSerializer.Save`/`Load`) for persistence
- **Synthetic data generation** — self-contained, no external datasets required
- **Reusable tokenizer** (`TextTokenizer`) — a word-level tokenizer with vocab building, encoding, and special tokens

## Quick start

```bash
# Interactive wizard (no args) — pick action and options
dotnet run --project samples/NivaraClassifier

# Generate synthetic training data
dotnet run --project samples/NivaraClassifier -- --command generate --num-samples 1000

# Train on generated data (default path: samples/data/sentiment_data.csv)
dotnet run --project samples/NivaraClassifier -- --command train --epochs 20 --save samples/data/classifier.model.json

# Interactive prediction with saved model
dotnet run --project samples/NivaraClassifier -- --command predict --load samples/data/classifier.model.json
```

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--command`, `-c <cmd>` | `train` | Command: `train`, `predict`, `generate` |
| `--data-path <path>` | `samples/data/sentiment_data.csv` | CSV data file (columns: text, label) |
| `--save <path>` | — | Save trained model + tokenizer |
| `--load <path>` | — | Load trained model + tokenizer |
| `--embedding-dim <int>` | 32 | Embedding dimension |
| `--hidden-dim <int>` | 64 | Hidden layer dimension |
| `--max-seq-len <int>` | 20 | Fixed sequence length (pad/truncate) |
| `--max-vocab <int>` | 5000 | Max vocabulary size |
| `--batch-size <int>` | 32 | Batch size |
| `--epochs <int>` | 10 | Training epochs |
| `--num-samples <int>` | 1000 | Samples to generate |
| `--lr <float>` | 0.001 | Learning rate |
| `--seed <int>` | 42 | RNG seed |
| `--help`, `-h` | — | Show CLI help |

## Modes of use

### Generate (`--command generate`)
Produces a CSV with `text` and `label` columns using template-based synthetic data. Saves to `samples/data/sentiment_data.csv` by default. Lets anyone run the example without external data.

### Training (`--command train`)
Trains the model on the CSV at `--data-path`. Prints epoch loss, batch timing, and test accuracy. Saves model with `--save`.

### Prediction (`--command predict`)
Interactive REPL: type sentences, get predicted sentiment. Requires a trained model via `--load`.

## Architecture

```
NivaraClassifier/
├── Program.cs                  # CLI entry, training loop, inference, wizard
├── TextClassifierModel.cs      # TextClassifierModel<T> : Module<T>
├── TextTokenizer.cs            # Reusable word-level tokenizer (vocab, encode/decode, save/load)
├── DataGenerator.cs            # Synthetic sentiment CSV generator
└── NivaraClassifier.csproj     # References Nivara core
```

### Model architecture

```
Input: [batch, seqLen] token IDs (integer)
    │
    ▼
Embedding(vocabSize, embedDim)          → [batch, seqLen, embedDim]
    │
    ▼
MeanPool over seqLen dimension          → [batch, embedDim]
    │
    ▼
Linear(embedDim → hiddenDim) → ReLU    → [batch, hiddenDim]
    │
    ▼
Linear(hiddenDim → numClasses)          → [batch, numClasses] logits
```

### TextTokenizer

A reusable word-level tokenizer with:
- Vocabulary building from training documents (min-freq filtering, max-vocab truncation)
- Special tokens: BOS, EOS, UNK, PAD
- `Encode(string) → int[]` with optional fixed-length padding/truncation
- `Decode(int[]) → string` for round-trip verification
- Save/load vocab mapping for deployment
- Lifecycle: created from training data, frozen for inference

### Data pipeline

1. Generate or load a CSV with `text` and `label` columns
2. Build vocab via `TextTokenizer.FromDocuments(docs)`
3. Tokenize all texts to fixed-length integer sequences
4. Create `NivaraFrame` with token position columns (`tok_0`, `tok_1`, ...) + `label` column
5. `TensorDataset<double>` wraps the frame for batched training
6. `DataLoader<double>` handles shuffling and batching

### Synthetic data generation

Templates with positive/negative patterns, fill nouns and activities from pools, optional adverb modifiers. Pool is large enough to prevent template memorization — the model must learn word associations.

## Nivara APIs demonstrated

| API | Where | Purpose |
|-----|-------|---------|
| `Module<T>` | `TextClassifierModel.cs` | Model base class with parameter registration |
| `Embedding<T>` | `TextClassifierModel.cs` | Learned word embeddings (first-class layer) |
| `Linear<T>` | `TextClassifierModel.cs` | Fully connected layers |
| `ReverseGradOperations.Relu` | `TextClassifierModel.cs` | Non-linearity |
| `ReverseGradOperations.MeanPool` | `TextClassifierModel.cs` | Sequence pooling `[B,L,D]` → `[B,D]` |
| `CrossEntropyLoss<T>` | `Program.cs` | Classification loss with integer labels |
| `Adam<T>` | `Program.cs` | Optimizer |
| `TrainingLoop<T>` | `Program.cs` | Training orchestration |
| `DataLoader<T>` | `Program.cs` | Batched data loading |
| `TensorDataset<T>` | `Program.cs` | Frame-backed dataset |
| `ModelSerializer.Save/Load` | `Program.cs` | JSON model persistence |
| `NivaraColumn<T>.Create` | `TextClassifierModel.cs` | Null-free tensor creation at AutoDiff boundary |

## Requirements

- .NET 10.0 SDK
- Nivara core library (`src/Nivara/Nivara.csproj`)
- No external dependencies

## Library gaps this example exposed and resolved

NivaraClassifier drove several core library additions and fixes. The original spec identified these gaps; most were resolved during earlier example development.

| Gap | Problem | Resolution |
|-----|---------|------------|
| **Embedding\<T\> not a Module\<T\>** | Embedding wasn't a Module — no StateDict, no Train/Eval, no Dispose lifecycle. | `Embedding<T>` now extends `Module<T>` (resolved during NivaraGpt development). |
| **No batched Embedding Forward** | Single-token `Forward(int)` only. Classifier needs batch `[B, L]` → `[B, L, D]`. | `Embedding<T>.Forward(ReverseGradTensor<T>)` handles batched input via one-hot + MatMul (resolved during NivaraGpt development). |
| **No integer-label CrossEntropyLoss** | `CrossEntropyLoss<T>` required one-hot targets. | `Forward(logits, int[])` overload added — builds one-hot internally (resolved during NivaraGpt development). |
| **No word-level tokenizer** | Only char-level tokenizer existed (MicroGpt). | `TextTokenizer` built as example-local reusable utility. Candidate for promotion to `Nivara.Extensions.Text` if useful. |
| **No MeanPool operation** | No built-in mean-over-dimension reduction for `[B, L, D]` → `[B, D]`. | `ReverseGradOperations.MeanPool<T>` added to core with full autograd backward support. |
| **Gap F: No LinearClassifier\<T\>** | Simplest classifier (linear → softmax → CE) requires boilerplate. | **Not yet resolved.** Candidate for core addition: `LinearClassifier<T> : Module<T>`. |

### Core library additions from this example

| New API | Location | Purpose |
|---------|----------|---------|
| `ReverseGradOperations.MeanPool<T>` | `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` | Core autograd mean-pooling with backward gradient distribution |
| `TextTokenizer` | `samples/NivaraClassifier/TextTokenizer.cs` | Reusable word-level tokenizer with vocab building, encode/decode, special tokens |

### Core library performance considerations

- **Embedding forward** uses `Parallel.For` + `TensorPrimitives.Dot` for MatMul (SIMD-accelerated)
- **MeanPool** uses scalar sum + divide over sequence dimension — could benefit from `TensorPrimitives.Sum` for SIMD acceleration on large embeddings
- **TensorDataset.BuildTensor** rents from `ArrayPool<T>` to minimize allocation overhead
- **AccumulateGradient fast path** (from NivaraChess optimizations) benefits all autograd operations in this example

## Performance

| Operation | Expected time | Notes |
|-----------|--------------|-------|
| Data generation | <1s | 1000 samples, template-based |
| Tokenizer build | <1s | Vocabulary from 1000 documents |
| Training (10 epochs) | ~1-2s | Batch size 32, 1000 samples. Embedding + MLP is lightweight. |
| Training (20 epochs) | ~1.5s | Reaches 100% test accuracy on synthetic data |
| Inference | <1ms | Single text classification |

## Limitations

- **Fixed sequence length** — all texts are padded/truncated to `seqLen`. Short texts are padded with PAD tokens; long texts are truncated. Variable-length handling would require a custom `Dataset<T>`.
- **Word-level tokenization** — no subword (BPE/WordPiece) support. Out-of-vocabulary words map to UNK. Sufficient for synthetic data; real-world use would benefit from subword tokenization.
- **Simple MLP head** — no attention, no transformer blocks. This is intentional: the example focuses on the Embedding + DataLoader + TrainingLoop pipeline, not architectural complexity.
- **No pre-trained embeddings** — embeddings are learned from scratch. Loading GloVe/Word2Vec would require file parsing and weight injection logic.
- **Single-threaded data generation** — the synthetic data generator runs sequentially. Not a bottleneck at typical sample counts.
