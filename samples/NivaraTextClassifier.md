# NivaraTextClassifier — Learned Embedding + MLP for Text Classification

## Goal

A grounded next step from MicroGpt on the NLP path. MicroGpt trains a
full transformer character-level language model. This example trains a
**word-level text classifier** using a learned embedding layer followed by
an MLP head — simpler architecture, but introduces real-world data
pipelines, batch training, and a reusable utility class.

The story: "I have a CSV with text reviews and labels. I want to train a
classifier to predict sentiment."

```
CSV (text, label)
    │
    ▼
Csv.ReadAsFrame  (via Nivara.Extensions)
    │
    ▼
TextTokenizer    (NEW reusable utility — word-level, builds vocab)
    │
    ▼
NivaraFrame with integer token columns + label column
    │
    ▼
TensorDataset<float> + DataLoader<float>  (batched training)
    │
    ▼
TextClassifierModel<float> : Module<float>
    ├── Embedding<float> (vocabSize → embedDim)   * FIX: make it a Module<T> *
    ├── Mean pool over sequence dimension
    └── Linear(embedDim → hiddenDim) → ReLU → Linear(hiddenDim → numClasses)
    │
    ▼
CrossEntropyLoss<float>  (or manual NLL)
    │
    ▼
TrainingLoop<float> + AdamW<float>
    │
    ▼
ModelSerializer.Save / Load
    │
    ▼
Inference: predict sentiment on new text strings
```

## Why This Is the Right Next Step

| What | MicroGpt | NivaraTextClassifier | Why It Matters |
|---|---|---|---|
| **Model architecture** | Full transformer (6+ types of layers) | Embedding → MeanPool → MLP (3 layer types) | Simpler to understand, faster to train |
| **Tokenization** | Char-level, hand-rolled in example | Word-level, **reusable utility** | First reusable NLP primitive in Nivara ecosystem |
| **Embedding usage** | Internal to model, single-token lookup | Full-sequence embedding + mean pool | Exercised as a first-class layer |
| **Training loop** | Manual per-position loop | `TrainingLoop<T>` with real batching | Proves `TrainingLoop` works with sequence models |
| **Data source** | External HTTP download | CSV via `Csv.ReadAsFrame` (Extensions) | Exercises Extensions I/O pipeline |
| **Label type** | Next-token prediction (vocab-size) | Binary/multi-class classification | Introduces `CrossEntropyLoss<T>` |
| **Save/load** | None | `ModelSerializer.Save/Load` with full state dict | Real persistence workflow |
| **Inference** | Generate via sampling | Predict class from raw text | "Enterprise" use case |
| **Evaluation** | Subjective (name quality) | Accuracy / confusion matrix | Quantitative validation |

## Gaps This Will Discover and Fix

### Gap 1: `Embedding<T>` is not a `Module<T>`

**Problem:** `Embedding<T>` implements `IDisposable` directly, not
`Module<T>`. This means:
- `TrainingLoop<T>` cannot manage its parameters via `GetParameters()`
- `ModelSerializer.Save/Load` cannot access its state dict
- No `Train()` / `Eval()` mode propagation

**Fix:** Make `Embedding<T>` extend `Module<T>`. This is backward-compatible
since `MicroGptModel` collects parameters manually via `emb.Parameters`
and doesn't rely on `Module<T>` methods. The change adds `Module<T>`'s
`RegisterParameters`, `StateDict()`, `LoadStateDict()`, `Train()`, `Eval()`,
and `Dispose()`.

The `Forward(int tokenId)` signature (single int) doesn't fit `Module<T>`'s
`Forward(ReverseGradTensor<T> input)`. Two options:
- **Option A:** Add `Forward(ReverseGradTensor<T> tokenIds)` for batch
  lookup, and keep `Forward(int)` as an overload. The batch variant
  does `Gather`-like selection from the weight matrix.
- **Option B:** Keep single-token `Forward(int)` and add a separate
  `EmbeddingBag<T>` module for mean-pooled sequence embedding.

**Recommendation: Option A.** Add batch embedding lookup so the
TextClassifier can embed a full sequence in one forward call.

### Gap 2: No batch embedding lookup

**Problem:** `Embedding<T>.Forward(int)` does one-hot → MatMul for one
token. Training on sequences of length L with batch size B requires
looking up B×L embeddings efficiently.

**Fix:** Add `Forward(ReverseGradTensor<T> tokenIds)` to `Embedding<T>`.
The input is `[batchSize, seqLen]` of integer token IDs. The output is
`[batchSize, seqLen, embedDim]`. Implementation uses one-hot encoding
per position expanded to a 3D tensor then MatMul'd with the weight matrix.

Since Nivara's ops are 2D only (no `BatchMatMul`), the implementation
flattens the batch×seqLen dimension:
```
Input:  [B, L]           token IDs
Flatten → [B*L]          contiguous token IDs
OneHot  → [B*L, V]       one-hot matrix
MatMul  → [B*L, D]       embedded vectors
Reshape → [B, L, D]      per-token embeddings
```

The `MeanPool` (averaging over the L dimension) reduces `[B, L, D]` to
`[B, D]` by summing and dividing by L using selection matrices or
element-wise operations, then feeding into the MLP head.

### Gap 3: No word-level tokenizer in core or extensions

**Problem:** The only tokenizer is MicroGpt's char-level example class.

**Fix:** Build `TextTokenizer` as a reusable class. It goes in the example
initially; if it proves useful, it can be promoted to Nivara core or
Extensions. The tokenizer:
- Builds a vocabulary from a list of documents
- Supports min-freq filtering and max-vocab-size truncation
- Handles BOS/EOS/UNK/PAD special tokens
- `Encode(string) → int[]` with optional padding/truncation to fixed length
- `Decode(int[]) → string` for round-trip verification
- `VocabSize` property
- Lifecycle: created from training data, then frozen for inference

### Gap 4: `Embedding<T>` has zero tests

**Problem:** No unit test anywhere validates `Embedding<T>` forward pass,
gradient flow, dispose, or serialization.

**Fix:** Add tests concurrent with the example development. At minimum:
- Forward single token returns correct shape `[1, D]`
- Gradient flows through embedding lookup (requiresGrad)
- Out-of-range token ID throws
- Dispose works (no double-dispose error)
- Serialization round-trip via `ModelSerializer` (once it's a `Module<T>`)

### Gap 5 (minor): CrossEntropyLoss requires one-hot targets

`CrossEntropyLoss<T>.Forward(logits, targets)` expects both to have the
same shape — `targets` must be one-hot. For the classifier, we can:
- Use one-hot labels (simple for 2-class sentiment)
- Or use the manual `LogSoftmax → NLL` pattern from MicroGpt

We'll use one-hot labels since the number of classes is small (2 for
sentiment). This also exercises `CrossEntropyLoss<T>` which MicroGpt
skipped.

## Architecture

### TextTokenizer (reusable utility)

```csharp
public sealed class TextTokenizer
{
    public int VocabSize { get; }
    public int BosToken { get; }
    public int EosToken { get; }
    public int UnkToken { get; }
    public int PadToken { get; }

    // Build vocab from training documents
    public static TextTokenizer FromDocuments(
        IEnumerable<string> documents,
        int maxVocabSize = 10000,
        int minFreq = 1);

    // Encode: string → token IDs with optional padding/truncation
    public int[] Encode(string text, int? fixedLength = null, bool addBosEos = true);

    // Decode: token IDs → string (for debugging)
    public string Decode(ReadOnlySpan<int> tokens);

    // Save/load vocab mapping for deployment
    public void Save(string path);
    public static TextTokenizer Load(string path);
}
```

### Data pipeline

1. Generate or load a CSV with columns `text` and `label` (0 or 1)
2. `Csv.ReadAsFrame("sentiment.csv")` → `NivaraFrame`
3. Build vocab: pass all text through `TextTokenizer.FromDocuments(docs)`
4. Tokenize all texts → create a `NivaraColumn<int>` per position
   (or a single column of flattened tokens for `TensorDataset`)
5. Create `NivaraFrame` with token columns + label column

For `TensorDataset<T>`, the expected format is:
```csharp
// Each row: [token_0, token_1, ..., token_{L-1}, label]
// Feature columns: "t0", "t1", ..., "t_{L-1}"
// Label column: "label"
```

This means we fix the sequence length L and pad/truncate all sequences
to that length. Each token position becomes a separate column in the
frame. This is the simplest approach and works with `TensorDataset<T>`
as-is.

**Alternative (more efficient):** Store all tokens in a single column
with jagged sequences, but `TensorDataset<T>` doesn't support that
natively. Fixed-length padding is the right trade-off for this example.

### TextClassifierModel<T>

```csharp
public sealed class TextClassifierModel<T> : Module<T>
    where T : struct, INumber<T>
{
    readonly Embedding<T> embedding;     // [vocabSize → embedDim]
    readonly Linear<T> hiddenLayer;      // [embedDim → hiddenDim]
    readonly Linear<T> outputLayer;      // [hiddenDim → numClasses]
    readonly int seqLen;
    readonly int embedDim;

    public TextClassifierModel(
        int vocabSize,
        int embedDim,      // e.g., 32
        int hiddenDim,     // e.g., 64
        int numClasses,    // e.g., 2
        int seqLen,        // fixed sequence length
        double initStd = 0.02);

    // Forward: [batch, seqLen] token IDs → [batch, numClasses] logits
    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> tokenIds)
    {
        // 1. Embed: [batch, seqLen] → [batch * seqLen, embedDim] (flattened)
        //    via Embedding.Forward(tokenIds) — the new batch variant

        // Actually: since Embedding.Forward takes a single int,
        // we need the batch variant. For now, work around:
        // Flatten tokenIds to 1D, embed each position, reshape back.

        // 2. Mean pool: average over sequence dimension
        //    [batch, seqLen, embedDim] → [batch, embedDim]

        // 3. MLP head: [batch, embedDim] → [batch, hiddenDim] → [batch, numClasses]
        //    via hiddenLayer → ReLU → outputLayer

        return output;
    }

    // Predict: token IDs → class index
    public int Predict(ReverseGradTensor<T> tokenIds);
}
```

### Loss and training

```csharp
// Loss: CrossEntropyLoss<T>.Forward(logits, oneHotLabels)
var lossFn = (ReverseGradTensor<T> logits, ReverseGradTensor<T> labels) =>
    new CrossEntropyLoss<T>().Forward(logits, labels);

// Optimizer
var optimizer = new AdamW<T>(learningRate);
optimizer.AddParameterGroup(model.GetParameters().Values, learningRate);

// Training loop
using var loop = new TrainingLoop<T>(model, loader, lossFn, optimizer, epochs);
var result = loop.Run();
```

## CLI Interface

```
--csv <path>              Path to training CSV (columns: text, label)
--test-csv <path>         Optional test CSV for evaluation
--embed-dim <int>         Embedding dimension (default: 32)
--hidden-dim <int>        Hidden layer dimension (default: 64)
--seq-len <int>           Fixed sequence length (default: 32)
--vocab-size <int>        Max vocabulary size (default: 5000)
--min-freq <int>          Minimum word frequency (default: 2)
--batch-size <int>        Batch size (default: 32)
--epochs <int>            Training epochs (default: 10)
--lr <float>              Learning rate (default: 0.001)
--seed <int>              RNG seed (default: 42)
--save <path>             Save trained model + tokenizer
--load <path>             Load trained model + tokenizer
--predict <text>          Predict sentiment for given text
--interactive             Interactive prediction REPL
--generate-data <path>    Generate synthetic sentiment CSV at path
--help, -h                Show this help
```

### Modes

**Default (no `--load`):** Generate synthetic data (or load from `--csv`),
train, print accuracy, optionally save.

**`--load <path>`:** Load model + tokenizer, skip training, run prediction
on `--predict` text or enter `--interactive` REPL.

**`--predict "Some text"`:** Print class and confidence.

**`--interactive`:** REPL: type text, get prediction. Type `quit` to exit.

**`--generate-data <path>`:** Generate a CSV with synthetic text + binary
labels. This lets anyone run the example without external data.

## Data Generation (Synthetic Sentiment)

The `--generate-data path/to/sentiment.csv` flag produces a CSV with
`text` and `label` columns. Generation uses simple templates:

### Positive templates (label=1)
```
"i loved the {noun}"
"the {noun} was absolutely fantastic"
"great {noun} and wonderful {noun}"
"amazing experience with the {noun}"
"{noun} exceeded my expectations"
"highly recommend this {noun}"
"the {noun} is perfect for {activity}"
```

### Negative templates (label=0)
```
"i hated the {noun}"
"the {noun} was terrible and boring"
"awful {noun} complete waste of time"
"very disappointed with the {noun}"
"{noun} broke immediately"
"do not buy this {noun}"
"the {noun} is useless for {activity}"
```

**Nouns pool:** product, movie, book, service, restaurant, hotel,
experience, food, phone, laptop, app, game, course, concert,
album, workout, software, website, delivery, cleaning, coating,
charging, wait, room, view, location, staff, system, tool,
package, version, update, support, quality, design, build,
battery, screen, speed, taste, price, value.

**Activities pool:** daily use, travel, work, gaming, cooking,
fitness, studying, entertainment, relaxation, productivity,
streaming, editing, photography, reading, music, sleeping,
outdoor, commuting, training, teaching, learning.

Each generated sample randomly picks a template, fills nouns and
activities from the pools, and optionally adds adverb modifiers
("very", "really", "extremely", "somewhat", "quite"). Total pool
is large enough to avoid overfitting — the model must learn word
associations, not memorize templates.

### Output format

```csv
text,label
"i loved the movie",1
"the service was terrible and boring",0
"amazing experience with the product",1
...
```

Generate 1000 training + 200 test rows by default (configurable).

## Files

```
samples/NivaraTextClassifier/
├── Program.cs                    # Entry point, CLI parsing, all modes
├── TextClassifierModel.cs        # TextClassifierModel<T> : Module<T>
├── TextTokenizer.cs              # Reusable word-level tokenizer
├── DataGenerator.cs              # Synthetic sentiment CSV generator
└── NivaraTextClassifier.csproj   # Project referencing Nivara
```

### NivaraTextClassifier.csproj

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <ProjectReference Include="..\..\src\Nivara\Nivara.csproj" />
  </ItemGroup>

</Project>
```

## What This Exercises vs. MicroGpt

| Feature | MicroGpt | NivaraTextClassifier | Status |
|---|---|---|---|
| Architecture | Transformer (complex) | Embedding+MLP (lean) | New |
| Tokenization | Char-level (in example) | Word-level (reusable) | New utility |
| Embedding<T> | Used internally | First-class layer, Module<T> | **Gap to fix** |
| Batch embedding lookup | No | Yes (new Forward overload) | **Gap to fix** |
| TrainingLoop<T> | No (manual loop) | Yes (structured, epoch-based) | New |
| DataLoader<T> | No (one doc/step) | Yes (batched, shuffled) | New |
| Data source | HTTP download | CSV via Csv.ReadAsFrame | Exercises Extensions |
| Loss | Hand-rolled NLL | CrossEntropyLoss<T> | New |
| Optimizer | Adam<T> | AdamW<T> | New |
| Model serialization | No | Save/Load full model | New |
| Evaluation | Subjective | Accuracy metric | New |
| Interactive mode | Generate text | Predict sentiment (REPL) | New |
| Synthetic data | No (real names) | Yes (generator) | New |
| Embedding<T> unit tests | No (not tested) | Yes (part of example) | **Gap to fix** |

## Anticipated Gaps & Fixes to Implement

### Fix 1: Make `Embedding<T>` a `Module<T>`

**File:** `src/Nivara/AutoDiff/Nn/Embedding.cs`

Change from `IDisposable` to `Module<T>`. This requires:
- Inherit from `Module<T>` instead of `IDisposable`
- Remove manual `Parameters` list (use `RegisterParameters`)
- Remove manual `Dispose()` (use `Module<T>`'s)
- Add `Forward(ReverseGradTensor<T> tokenIds)` overload
- Keep `Forward(int tokenId)` for backward compatibility with MicroGpt
- Verify MicroGpt still compiles and runs

### Fix 2: Add batch embedding lookup

**File:** `src/Nivara/AutoDiff/Nn/Embedding.cs`

```csharp
// New: batch sequence embedding
// tokenIds: [batchSize, seqLen] of integer token IDs
// Returns:  [batchSize * seqLen, embedDim] (flattened)
public ReverseGradTensor<T> Forward(ReverseGradTensor<T> tokenIds)
{
    // Requires tokenIds to be integer values stored as T
    // Since T is constrained to INumber<T>, we cast via ConvertToInt32

    int totalTokens = tokenIds.Length;
    var result = new T[totalTokens * EmbeddingDim];

    // For each position, look up the embedding row
    // This is a gather operation via one-hot + MatMul
    // ...

    var col = NivaraColumn<T>.Create(result);
    var tensor = new ReverseGradTensor<T>(col, requiresGrad: true);
    tensor.Reshape(totalTokens, EmbeddingDim);
    return tensor;
}
```

Implementation detail: each token ID is a scalar `T` value. We convert
to `int` via `int.CreateTruncating(value)`, build a one-hot vector,
and MatMul with the weight matrix. This is done per-token (looping)
or in batch (building a block-diagonal one-hot matrix).

For the batch case: flatten `[B, L]` → `[B*L]`, build one-hot matrix
`[B*L, V]`, MatMul with weight `[V, D]` → `[B*L, D]`. Reshape to
`[B, L, D]`.

But Nivara MatMul is 2D only. The approach:
1. Flatten token IDs to `[N]` where N = B×L
2. Build one-hot `[N, V]` matrix
3. MatMul `[N, V] @ [V, D]` → `[N, D]`
4. Reshape to `[B, L, D]`

### Fix 3: Embedding<T> unit tests

**File:** `tests/Nivara.Tests/AutoDiff/NnTests.cs` (or new file)

Add tests for:
- Forward single token: shape `[1, D]`, gradient flows
- Forward batch: shape `[N, D]` where N = total tokens
- Out-of-range token throws
- Dispose lifecycle
- StateDict round-trip (once it's Module<T>)
- Parameter count: 1 parameter with V×D elements

### Fix 4 (if needed): CrossEntropyLoss with integer labels

If one-hot encoding is too wasteful for large vocabularies (not an issue
here with 2 classes), we could add a `sparse` flag to `CrossEntropyLoss`.
Not needed for this example — we'll use one-hot labels.

## Not Doing (Stretch / Future)

- **BPE/WordPiece tokenization** — the word-level tokenizer is sufficient
  for this example. Subword tokenization would be a future upgrade.
- **Convolutional text model** (TextCNN) — could be interesting but adds
  complexity without proportional learning value for this step.
- **Pre-trained embedding loading** (GloVe, Word2Vec) — would require
  file parsing and weight injection logic. Future step.
- **Multi-GPU or data-parallel training** — `DataParallelTrainer` exists
  but is not needed for this scale.
- **Attention-based classifier** — more complex than needed for the
  "next step" from MicroGpt. Keep it simple: Embedding → MeanPool → MLP.
- **Real dataset download** — the synthetic data generator makes the
  example self-contained. A future real-data mode could load IMDB.
