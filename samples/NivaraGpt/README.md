# NivaraGpt — Character-Level Transformer (Nivara-Native)

A miniature GPT language model implemented the **Nivara way** — using `Module<T>`, `TrainingLoop<T>`, `DataLoader<T>`, `TensorDataset<T>`, `ModelSerializer`, and batched causal attention. Same task as MicroGpt (character-level name generation on makemore/names.txt), but built on Nivara's high-level APIs instead of manual parameter management and per-position loops.

**Target audience:** .NET developers evaluating Nivara for NLP/sequence modeling tasks.

## What it does

NivaraGpt trains a character-level GPT to generate human names, character-by-character. It showcases:

- **`Module<T>` subclass** with `RegisterModules`/`RegisterParameters` — `StateDict()`, `LoadStateDict()`, `ModelSerializer`, `Train()`/`Eval()` work out of the box
- **`TrainingLoop<T>`** + **`DataLoader<T>`** + **`TensorDataset<T>`** — standard batched training pipeline
- **Batched causal attention** — full-sequence forward with upper-triangular causal mask (not per-position)
- **`TransformerBlock<T>`** — reusable core library building block for multi-head attention + MLP
- **`CrossEntropyLoss<T>`** with integer labels — no manual one-hot + LogSoftmax + NLL
- **`Dropout<T>`** — embedding dropout, attention dropout, residual dropout
- **`ModelSerializer.Save/Load`** — save and reload trained models
- **`Sampler<T>`** — temperature and top-k sampling for generation

## Quick start

```bash
# Train with defaults
dotnet run --project samples/NivaraGpt

# Train with custom hyperparameters
dotnet run --project samples/NivaraGpt -- --n-embd 32 --n-layer 2 --block-size 16 --n-head 4 --epochs 20 --batch-size 64

# Train and save model
dotnet run --project samples/NivaraGpt -- --save nivaragpt.json

# Generate from saved model
dotnet run --project samples/NivaraGpt -- --load nivaragpt.json --samples 20

# Compare with MicroGpt (Karpathy's hyperparameters)
dotnet run --project samples/NivaraGpt -- --block-size 16 --n-embd 16 --n-head 4 --n-layer 1 --lr-decay --temperature 0.5 --samples 20
```

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--n-embd <int>` | 64 | Embedding dimension |
| `--n-layer <int>` | 2 | Number of transformer layers |
| `--block-size <int>` | 32 | Context window / max sequence length |
| `--n-head <int>` | 4 | Number of attention heads |
| `--dropout <float>` | 0.1 | Dropout probability |
| `--epochs <int>` | 20 | Training epochs |
| `--batch-size <int>` | 64 | Batch size |
| `--lr <float>` | 3e-3 | Learning rate |
| `--beta1 <float>` | 0.9 | Adam beta1 |
| `--beta2 <float>` | 0.95 | Adam beta2 |
| `--init-std <float>` | 0.02 | Weight init std dev |
| `--no-weight-tying` | off | Use separate lm_head instead of weight tying |
| `--lr-decay` | off | Linear LR decay to zero over epochs |
| `--temperature <float>` | 0.8 | Sampling temperature |
| `--top-k <int>` | 0 | Top-k sampling (0 = disabled) |
| `--no-dropout` | off | Disable all dropout |
| `--save <path>` | — | Save trained model to JSON |
| `--load <path>` | — | Load model from JSON |
| `--seed <int>` | 42 | RNG seed |
| `--help`, `-h` | — | Show help |

## Architecture

```
NivaraGptModel<T> : Module<T>
├── Embedding<T> tokenEmb        [vocabSize, nEmbd]
├── Dropout<T> (embedding dropout)
├── Embedding<T> posEmb          [blockSize, nEmbd]
├── TransformerBlock<T> × nLayer
│   ├── Pre-LN: RMSNorm
│   ├── Multi-Head Self-Attention (with causal mask)
│   │   ├── Linear Q, K, V projections
│   │   ├── Batched scaled dot-product attention
│   │   ├── Concat heads (via ReverseGradOperations.Concat)
│   │   └── Output projection
│   ├── Dropout (attention dropout)
│   ├── Residual connection
│   ├── Pre-LN: RMSNorm
│   ├── MLP: Linear(nEmbd, 4×nEmbd) → Squared ReLU → Linear(4×nEmbd, nEmbd)
│   ├── Dropout (residual dropout)
│   └── Residual connection
├── RMSNorm (final)
├── Dropout (final)
└── Output projection
    ├── Weight tying: MatMul(x, Transpose(tokenEmb.Weight)) — default
    └── Separate Linear lm_head — when --no-weight-tying
```

### Batched causal attention

Unlike MicroGpt's per-position approach, NivaraGpt processes the entire sequence in one forward pass:

```
Input: [B, L] token IDs
  → Embedding: [B, L, nEmbd]
  → TransformerBlock × N:
      Q = x @ Wq  → [B, L, nEmbd]
      K = x @ Wk  → [B, L, nEmbd]
      V = x @ Wv  → [B, L, nEmbd]
      Reshape to heads: [B, H, L, headDim]
      Attention scores = Q @ K^T / sqrt(headDim)  → [B, H, L, L]
      Apply causal mask (upper-triangular -∞)
      Attention weights = Softmax(scores, axis=-1)
      Attention out = weights @ V  → [B, H, L, headDim]
      Concat heads: [B, H, L, headDim] → [B, L, nEmbd]
      Output projection + residual
      MLP + residual
  → Final RMSNorm
  → Output projection → [B, L, vocabSize]
```

## Training pipeline

```
1. Download names.txt (~32K names)
2. Build char-level vocabulary (BOS, EOS, unique chars)
3. Encode all names → token sequences with BOS/EOS
4. Build NivaraFrame with columns:
   - "input_ids": flattened [total_samples × blockSize] int values
   - "target_ids": shifted [total_samples × blockSize] int values
5. TensorDataset<float>(frame, ["input_ids"], ["target_ids"])
6. DataLoader<float>(dataset, batchSize, shuffle)
7. TrainingLoop<float>(model, CrossEntropyLoss, Adam, epochs)
8. Generate samples with temperature/top-k sampling
```

### Dataset construction

Each name produces multiple training examples via a sliding window:

```
Name: "alice" → tokens: [BOS, a, l, i, c, e, EOS]
blockSize=5:
  input:  [BOS, a, l, i, c]     target: [a, l, i, c, e]
  input:  [a, l, i, c, e]       target: [l, i, c, e, EOS]
```

Short names (< 2 tokens after BOS/EOS) are skipped. Sequences shorter than `blockSize` are right-padded with a pad token (index 0, treated as padding in loss).

## How inference/generation works

```csharp
model.Eval();
var sampler = new Sampler<float>();
var tokens = new List<int> { tokenizer.BOS };
for (int pos = 0; pos < blockSize; pos++) {
    var input = TensorFromTokens(tokens, pos);
    var logits = model.Forward(input);  // [1, L, vocabSize]
    int next = sampler.Sample(logits[^1], temperature, topK);
    tokens.Add(next);
    if (next == tokenizer.EOS) break;
}
```

- Start with `[BOS]` token
- At each step, run full forward pass on the token sequence
- Extract logits at the last position
- Sample with temperature (and optional top-k)
- Append to sequence; stop at `[EOS]` or `blockSize`

## Nivara APIs demonstrated

| API | Where | Purpose |
|-----|-------|---------|
| `Module<T>` | `NivaraGptModel.cs` | Model base class with parameter registration |
| `Embedding<T>` | `NivaraGptModel.cs` | Token and position embeddings |
| `Linear<T>` | `TransformerBlock` (core) | Attention projections and MLP |
| `Activation.Relu` | `TransformerBlock` (core) | Squared ReLU activation |
| `Dropout<T>` | `NivaraGptModel.cs` | Embedding, attention, and residual dropout |
| `RMSNorm` | `TransformerBlock` (core) | Pre-LayerNorm normalization |
| `CrossEntropyLoss<T>` | `Program.cs` | Integer-label classification loss |
| `Adam<T>` | `Program.cs` | Optimizer |
| `TrainingLoop<T>` | `Program.cs` | Training orchestration |
| `DataLoader<T>` | `Program.cs` | Batched data loading |
| `TensorDataset<T>` | `Program.cs` | Frame-backed dataset |
| `ModelSerializer.Save/Load` | `Program.cs` | JSON model persistence |
| `Sampler<T>` | `Program.cs` | Temperature and top-k sampling |
| `ReverseGradOperations.Concat` | `TransformerBlock` (core) | Differentiable head concatenation |
| `ReverseGradOperations.Softmax` | `TransformerBlock` (core) | Batched softmax with axis support |
| `ReverseGradOperations.MatMul` | Throughout | Matrix multiplication |
| `ReverseGradOperations.RMSNorm` | `TransformerBlock` (core) | Normalization |

## Comparison with MicroGpt

| Aspect | MicroGpt | NivaraGpt |
|--------|----------|-----------|
| Model base class | `IDisposable` with manual `List<Parameter<T>>` | `Module<T>` with `RegisterModules`/`RegisterParameters` |
| Training loop | Manual (per-step, per-position) | `TrainingLoop<T>` with epochs and batches |
| Data loading | Single random doc per step | `DataLoader<T>` with `TensorDataset<T>` and `NivaraFrame` |
| Attention | Per-position with KV cache | Batched full-sequence with causal mask |
| Loss | Manual `LogSoftmax` + `MatMul(one_hot)` + `Negate` | `CrossEntropyLoss<T>` with integer labels |
| Head concat | `PadRight`/`PadLeft` selection matrices | `ReverseGradOperations.Concat` |
| Dropout | None | Configurable embedding/attention/residual dropout |
| Save/load | None | `ModelSerializer.Save/Load` + `StateDict` |
| Sampling | Inline softmax + loop | `Sampler<T>` with temperature and top-k |
| Inference mode | Manual (no eval mode) | `model.Eval()` / `model.Train()` |

## Library gaps this example exposed and resolved

NivaraGpt drove several core library additions. The original spec identified these gaps; all were resolved during implementation.

### Core library additions

| New API | Location | Purpose |
|---------|----------|---------|
| `TransformerBlock<T>` | `src/Nivara/AutoDiff/Nn/TransformerBlock.cs` | Reusable pre-LN transformer block with batched multi-head causal attention, MLP, residual connections, and configurable dropout |
| `ReverseGradOperations.Concat<T>` | `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` | Differentiable tensor concatenation along an axis; backward splits gradient along the same axis |
| `Sampler<T>` | `src/Nivara/AutoDiff/Nn/Sampler.cs` | Temperature and top-k sampling from logit tensors |
| `CrossEntropyLoss<T>.Forward(logits, int[])` | `src/Nivara/AutoDiff/Nn/Functional/CrossEntropyLoss.cs` | Integer-label overload — converts targets to one-hot internally |

### Gap analysis (from `samples/README.md`)

| Gap | Status | Resolution |
|-----|--------|------------|
| **E: No batched TransformerBlock** | Resolved | `TransformerBlock<T>` in core — pre-LN, causal mask, dropout, residual |
| **H: No integer-label CrossEntropyLoss** | Resolved | New overload accepts `int[]` targets |
| **K: No Sampler utilities** | Resolved | `Sampler<T>` with `Sample(logits, temperature, topK)` |
| **New: No Concat op** | Resolved | `ReverseGradOperations.Concat<T>` with correct backward |

### What MicroGpt does that NivaraGpt improves

| MicroGpt workaround | NivaraGpt improvement | Core library change |
|---------------------|----------------------|---------------------|
| `PadRight`/`PadLeft` selection matrices for ConcatHeads | `ReverseGradOperations.Concat` | New op: differentiable concatenation |
| `SoftmaxList()` + manual NLL computation | `CrossEntropyLoss<T>` with integer labels | New overload on existing class |
| Manual `List<Parameter<T>>` management | `Module<T>` inheritance | No change (already existed) |
| Inline sampling code | `Sampler<T>` utility | New class in `AutoDiff/Nn/` |
| Per-position forward/backward loop | Batched `TrainingLoop<T>` | No change (already existed) |
| No save/load | `ModelSerializer.Save/Load` | No change (already existed) |
| No dropout | `Dropout<T>` composed in model | No change (already existed) |

## Requirements

- .NET 10.0 SDK
- Nivara core library (`src/Nivara/Nivara.csproj`)

## Performance

Expected to be comparable to or faster than MicroGpt due to:
- Batched MatMul kernels (vs per-position scalar loops)
- SIMD-accelerated TensorPrimitives in attention and MLP
- No per-call identity matrix allocation (Concat vs PadRight/PadLeft)

Performance results will be documented after implementation.

## File map

| File | Purpose |
|------|---------|
| `NivaraGptModel.cs` | GPT model: `Module<T>` subclass with embeddings, transformer blocks, lm_head |
| `Program.cs` | CLI, data loading, training loop, generation, timing |
| `NivaraGpt.csproj` | Project file referencing Nivara core |
| `README.md` | This file |

## Limitations

- **Character-level only** — no BPE or subword tokenization. The vocabulary is small (~30 chars) which limits the model's expressiveness but keeps the example focused on architecture rather than tokenization.
- **No causal inference optimization** — generation runs the full forward pass at each step (no KV cache reuse). For a production model, you'd want incremental decoding. This is acceptable for a sample.
- **Same dataset as MicroGpt** — makemore/names.txt. The comparison is apples-to-apples.
