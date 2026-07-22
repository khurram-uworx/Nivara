# NivaraClassifier — Word-Level Text Classifier

A sample project demonstrating Nivara's autograd engine for text classification. Trains a word-level text classifier using learned embeddings and an MLP head — the natural next step after MicroGpt on the NLP path.

**Target audience:** ML engineers, NLP practitioners, .NET developers exploring .NET-native ML.

## What it does

NivaraClassifier trains a model to predict sentiment (positive/negative) from text. It showcases:

- **Reverse-mode autograd** with structured training (`TrainingLoop<T>`, `DataLoader<T>`)
- **Learned embeddings** (`Embedding<T>`) as a first-class layer for word-level features
- **Batched embedding lookup** via the `Embedding<T>.Forward(ReverseGradTensor<T>)` overload
- **CrossEntropyLoss<T>** with integer labels for multi-class classification
- **Model serialization** (`ModelSerializer.Save`/`Load`) for persistence
- **Synthetic data generation** — self-contained, no external datasets required
- **Reusable tokenizer** (`TextTokenizer`) — a word-level tokenizer with vocab building, encoding, and special tokens

## Quick start

```bash
# Interactive wizard (no args) — pick action and options
dotnet run --project samples/NivaraClassifier

# Train on synthetic data (default)
dotnet run --project samples/NivaraClassifier -- --epochs 10 --save model.json

# Train on a custom CSV
dotnet run --project samples/NivaraClassifier -- --csv sentiment.csv --epochs 15

# Predict sentiment
dotnet run --project samples/NivaraClassifier -- --load model.json --predict "this movie was great"

# Interactive prediction REPL
dotnet run --project samples/NivaraClassifier -- --load model.json --interactive

# Generate synthetic data only
dotnet run --project samples/NivaraClassifier -- --generate-data sentiment.csv
```

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--csv <path>` | — | Path to training CSV (columns: text, label) |
| `--test-csv <path>` | — | Optional test CSV for evaluation |
| `--embed-dim <int>` | 32 | Embedding dimension |
| `--hidden-dim <int>` | 64 | Hidden layer dimension |
| `--seq-len <int>` | 32 | Fixed sequence length (pad/truncate) |
| `--vocab-size <int>` | 5000 | Max vocabulary size |
| `--min-freq <int>` | 2 | Minimum word frequency |
| `--batch-size <int>` | 32 | Batch size |
| `--epochs <int>` | 10 | Training epochs |
| `--lr <float>` | 0.001 | Learning rate |
| `--seed <int>` | 42 | RNG seed |
| `--save <path>` | — | Save trained model + tokenizer |
| `--load <path>` | — | Load trained model + tokenizer |
| `--predict <text>` | — | Predict sentiment for given text |
| `--interactive` | — | Interactive prediction REPL |
| `--generate-data <path>` | — | Generate synthetic sentiment CSV |
| `--help`, `-h` | — | Show CLI help |

## Modes of use

### Training (default)
Trains the model on synthetic data (or `--csv` if provided). Prints epoch loss, batch timing, and test accuracy. Save the model with `--save`.

### Prediction (`--predict`)
One-shot prediction on a text string. Prints class and confidence score.

### Interactive REPL (`--interactive`)
Loops over user-supplied text, printing predicted sentiment and confidence. Type `quit` to exit.

### Data generation (`--generate-data`)
Produces a CSV with `text` and `label` columns using template-based synthetic data. Configurable row count. Lets anyone run the example without external data.

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
4. Create `NivaraFrame` with token position columns (`t0`, `t1`, ...) + `label` column
5. `TensorDataset<float>` wraps the frame for batched training
6. `DataLoader<float>` handles shuffling and batching

### Synthetic data generation

Templates with positive/negative patterns, fill nouns and activities from pools, optional adverb modifiers. 1000 training + 200 test rows by default. Pool is large enough to prevent template memorization — the model must learn word associations.

## Nivara APIs demonstrated

| API | Where | Purpose |
|-----|-------|---------|
| `Module<T>` | `TextClassifierModel.cs` | Model base class with parameter registration |
| `Embedding<T>` | `TextClassifierModel.cs` | Learned word embeddings (first-class layer) |
| `Linear<T>` | `TextClassifierModel.cs` | Fully connected layers |
| `Dropout<T>` | `TextClassifierModel.cs` | Regularization (optional) |
| `Activation.Relu` | `TextClassifierModel.cs` | Non-linearity |
| `CrossEntropyLoss<T>` | `Program.cs` | Classification loss with integer labels |
| `AdamW<T>` | `Program.cs` | Optimizer with weight decay |
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
| **No MeanPool operation** | No built-in mean-over-dimension reduction for `[B, L, D]` → `[B, D]`. | Implemented as a model-level utility using `ReverseGradOperations.Slice` + `Mean`. Could be promoted to core as `MeanPool<T>` if reusable. |
| **Gap F: No LinearClassifier\<T\>** | Simplest classifier (linear → softmax → CE) requires boilerplate. | **Not yet resolved.** Candidate for core addition: `LinearClassifier<T> : Module<T>`. |

### Core library additions from this example

| New API | Location | Purpose |
|---------|----------|---------|
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
| Training (10 epochs) | ~5-15s | Batch size 32, 1000 samples. Embedding + MLP is lightweight. |
| Inference | <1ms | Single text classification |

## Limitations

- **Fixed sequence length** — all texts are padded/truncated to `seqLen`. Short texts are padded with PAD tokens; long texts are truncated. Variable-length handling would require a custom `Dataset<T>`.
- **Word-level tokenization** — no subword (BPE/WordPiece) support. Out-of-vocabulary words map to UNK. Sufficient for synthetic data; real-world use would benefit from subword tokenization.
- **Simple MLP head** — no attention, no transformer blocks. This is intentional: the example focuses on the Embedding + DataLoader + TrainingLoop pipeline, not architectural complexity.
- **No pre-trained embeddings** — embeddings are learned from scratch. Loading GloVe/Word2Vec would require file parsing and weight injection logic.
- **Single-threaded data generation** — the synthetic data generator runs sequentially. Not a bottleneck at 1000 samples.
