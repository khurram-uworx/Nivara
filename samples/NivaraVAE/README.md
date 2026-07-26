# NivaraVAE — Variational Autoencoder for Synthetic Pattern Generation

A sample project demonstrating Nivara's autograd encoder–decoder architecture. Trains a VAE to learn latent representations of synthetic 2D patterns (circles, stripes, blobs, checkerboards), then generates, interpolates, and walks the latent space.

**Target audience:** ML practitioners exploring .NET-native generative models, developers evaluating Nivara for unsupervised learning.

## What it does

NivaraVAE trains a variational autoencoder on synthetic binary grid patterns. It showcases:

- **Reverse-mode autograd** with a manual training loop (`GradientUtils.Grad()`, `loss.Backward()`, `optimizer.Step()`)
- **`Module<T>` with individual `Linear<T>` layers** — encoder/decoder composed from `Linear<T>`, `Dropout<T>`, and activation layers
- **VAE-specific operations** — `SampleNormal` (reparameterization trick), `KlDivergence` (ELBO loss), `BCEWithLogitsLoss` (reconstruction)
- **`TensorDataset<T>`** with same-column-as-both-features-and-labels (autoencoding pattern)
- **`ModelSerializer.Save`/`Load`** for checkpoint persistence
- **Latent space exploration** — generation from noise, interpolation between encoded patterns, per-dimension latent walks

## Quick start

```bash
# Interactive wizard (no args)
dotnet run --project samples/NivaraVAE

# Train with defaults (10 epochs, 8x8 patterns, 5000 samples)
dotnet run --project samples/NivaraVAE -- --epochs 10

# Train with custom hyperparameters
dotnet run --project samples/NivaraVAE -- --epochs 20 --latent-dim 16 --hidden-dim 256 --lr 0.0005 --beta 0.5

# Generate 8 samples after training
dotnet run --project samples/NivaraVAE -- --epochs 10 --generate 8

# Save/load model
dotnet run --project samples/NivaraVAE -- --epochs 10 --save vae.json
dotnet run --project samples/NivaraVAE -- --load vae.json --generate 4

# Interpolate between random pairs (5 steps each)
dotnet run --project samples/NivaraVAE -- --load vae.json --interpolate 3

# Walk each latent dimension (-3 to +3, 7 steps)
dotnet run --project samples/NivaraVAE -- --load vae.json --latent-walk

# Evaluate reconstruction quality on held-out set
dotnet run --project samples/NivaraVAE -- --epochs 10 --eval

# Show example patterns from the dataset
dotnet run --project samples/NivaraVAE -- --show-patterns
```

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs <int>` | 10 | Training epochs |
| `--latent-dim <int>` | 8 | Latent space dimension |
| `--hidden-dim <int>` | 128 | Hidden layer size |
| `--batch-size <int>` | 64 | Batch size |
| `--lr <float>` | 0.001 | Learning rate (AdamW) |
| `--pattern-size <int>` | 8 | Pattern grid size (NxN pixels) |
| `--num-patterns <int>` | 5000 | Number of synthetic patterns |
| `--seed <int>` | 42 | RNG seed |
| `--beta <float>` | 1.0 | KL divergence weight (beta-VAE) |
| `--dropout <float>` | 0.2 | Dropout probability |
| `--save <path>` | — | Save trained model to JSON |
| `--load <path>` | — | Load model from JSON |
| `--generate <int>` | — | Generate N samples from random latent vectors |
| `--interpolate <int>` | — | Interpolate between N random latent pairs |
| `--latent-walk` | — | Walk each latent dimension one at a time |
| `--show-patterns` | — | Display example patterns from the dataset |
| `--eval` | — | Evaluate reconstruction on test set |
| `--mode <linear\|conv>` | `linear` | Model architecture: `linear` uses `VaeModel<T>` (MLP), `conv` uses `ConvVAE<T>` (Conv2d encoder/decoder) |
| `--help`, `-h` | — | Show CLI help |

## Modes of use

### Model architecture (`--mode linear|conv`)

**`--mode linear` (default):** MLP-based VAE. See VaeModel layout below.

**`--mode conv`:** Convolutional VAE using `ConvVAE<T>` from the core library. Input is reshaped to `[B, 1, patternSize, patternSize]` (single-channel grayscale). The encoder uses Conv2d with stride downsampling, and the decoder uses ConvTranspose2d with stride upsampling. This mode demonstrates the full Conv2d/ConvTranspose2d pipeline and produces sharper pattern reconstructions due to the spatial inductive bias.

```
--mode conv architecture:
  Encoder: Conv2d(1→32, k=4, s=2, p=1) → Conv2d(32→64, k=4, s=2, p=1) → flatten
  Latent:  mu + logVar (reparameterization trick)
  Decoder: ConvTranspose2d → ConvTranspose2d → reshape to [B, 1, patternSize, patternSize]
  Loss: BCEWithLogitsLoss(reconstruction) + beta * KL(mu, logVar)
```

### Training (default)
Trains the VAE on synthetic patterns. Prints per-epoch loss (ELBO = BCE reconstruction + beta * KL divergence), batch timing, and a generated sample after training.

### Generation (`--generate N`)
After training (or `--load`), samples N latent vectors from N(0,1) and decodes them into pattern grids. Displays ASCII art in the console.

### Interpolation (`--interpolate N`)
Encodes N random pairs from the dataset into latent vectors, linearly interpolates between each pair in 5 steps, and decodes each step. Demonstrates that the latent space is smooth — intermediate codes produce visually intermediate patterns.

### Latent walk (`--latent-walk`)
For each latent dimension, sweeps from -3 to +3 in 7 steps while holding all other dimensions at 0. Reveals what each latent dimension encodes (e.g., one dimension might control pattern orientation, another controls size).

### Evaluation (`--eval`)
Splits data 80/20 train/test, trains, then reports reconstruction BCE loss on the held-out test set. Quantifies how well the VAE generalizes.

### Pattern display (`--show-patterns`)
Generates and displays 8 example patterns from the dataset as ASCII art, showing the variety of synthetic training data.

## Architecture

```
PatternDataset (synthetic 2D patterns on a grid)
    |
    v
Manual batching  (shuffled index array, BuildBatch for features/targets)
    |
    v
VaeModel<T>  (Module<T> subclass)
    ├── Encoder: Linear → LeakyReLU → Dropout → Linear → LeakyReLU → Dropout
    ├── MuHead:  Linear
    ├── LogVarHead: Linear
    ├── Reparameterize: SampleNormal(mu, logVar)
    └── Decoder: Linear → LeakyReLU → Dropout → Linear (raw logits, no Sigmoid)
    |
    v
BCEWithLogitsLoss<T> + KlDivergence (ELBO loss — logits, not probabilities)
    |
    v
Adam<T>  (manual training loop with GradientUtils.Grad())
    |
    v
ModelSerializer.Save / Load  (checkpointing)
```

### VaeModel layout

```
Input: [batch, patternSize * patternSize]  (flattened binary grid)

Encoder:
  Linear(patternSize², hiddenDim) → LeakyReLU(0.01) → Dropout(0.2)
  Linear(hiddenDim, hiddenDim)     → LeakyReLU(0.01) → Dropout(0.2)

MuHead:      Linear(hiddenDim, latentDim)
LogVarHead:  Linear(hiddenDim, latentDim)

Reparameterize:  z = mu + exp(logVar * 0.5) * ε

Decoder:
  Linear(latentDim, hiddenDim)  → LeakyReLU(0.01) → Dropout(0.2)
  Linear(hiddenDim, patternSize²) → (raw logits — BCEWithLogitsLoss applies sigmoid internally)

Output: [batch, patternSize²]  (raw logits, sigmoid applied by loss function)
```

### Loss

```
ELBO = BCEWithLogitsLoss(recon_logits, target) / batchSize + beta * KL(mu, logVar)
```

Where:
- `BCEWithLogitsLoss<T>` computes numerically stable binary cross-entropy from raw logits, reduced to mean via division by batch size
- `KlDivergence(mu, logVar)` computes `-0.5 * sum(1 + logVar - mu² - exp(logVar))`
- `beta` default 1.0, adjustable via CLI

## What this exercises vs. other samples

| Feature | MicroGpt | NivaraGpt | NivaraChess | **NivaraVAE** |
|---|---|---|---|---|
| **Architecture type** | Autoregressive | Autoregressive | Feedforward MLP | **Feedforward encoder–decoder** |
| **Module\<T\> inheritance** | No | Yes | Yes | **Yes** |
| **Sequential\<T\>** | No | No | No | **No** (individual Linear\<T\> fields) |
| **Dropout\<T\>** | No | Yes | No | **Yes** |
| **TrainingLoop\<T\>** | No (raw) | No (raw) | Yes | **No (manual — demonstrates Grad() scope)** |
| **DataLoader\<T\>** | No | Yes | Yes | **No** (manual batching) |
| **TensorDataset\<T\>** | No | Yes | Yes | **No** (manual batching from NivaraFrame) |
| **Loss function** | Hand-rolled NLL | CrossEntropy | MSE | **BCEWithLogits + KlDivergence** |
| **Optimizer** | Adam | Adam | AdamW | **Adam** |
| **ModelSerializer** | No | Yes | Yes | **Yes** |
| **Latent space ops** | No | No | No | **Yes** (SampleNormal, KlDivergence) |
| **Interactive modes** | Generate | Generate | Eval/REPL/UCI | **Generate, Interpolate, Latent Walk, Eval** |
| **Gradient clipping** | No | No | No | **Yes** (ClipGradNorm in manual loop) |
| **Binary reconstruction** | No | No | No | **Yes** (BCE with logits) |
| **Unsupervised (no labels)** | No | No | No | **Yes** (autoencoding) |
| **Data source** | External file | External file | Synthetic positions | **Fully synthetic patterns** |

## Nivara APIs demonstrated

| API | Where | Purpose |
|-----|-------|---------|
| `Module<T>` | `VaeModel.cs`, `ConvVAE<T>` (core) | Model base class with parameter registration |
| `Linear<T>` | `VaeModel.cs` | Fully connected layers (encoder, decoder, heads) |
| `Conv2d<T>` | `ConvVAE<T>` (core, `--mode conv`) | Convolutional encoder layers |
| `ConvTranspose2d<T>` | `ConvVAE<T>` (core, `--mode conv`) | Transposed convolutional decoder layers |
| `Dropout<T>` | `VaeModel.cs` | Regularization during training |
| `Activation.LeakyRelu<T>` | `VaeModel.cs` | Non-linearity (negativeSlope: 0.01) |
| `BCEWithLogitsLoss<T>` | `Program.cs` | Binary cross-entropy from logits |
| `ReverseGradOperations.KlDivergence<T>` | `Program.cs` | KL divergence for VAE regularization |
| `ReverseGradOperations.SampleNormal<T>` | `VaeModel.cs` | Reparameterization trick |
| `Adam<T>` | `Program.cs` | Optimizer with adaptive learning rates |
| `GradientUtils.Grad()` | `Program.cs` | Enables reverse-mode autograd scope |
| `GradientUtils.ClipGradNorm<T>` | `Program.cs` | Global gradient norm clipping |
| `ModelSerializer.Save/Load` | `Program.cs` | JSON model persistence |
| `NivaraFrame` / `NivaraColumn<T>` | `PatternDataset.cs` | DataFrame-backed pattern storage |
| `TensorPrimitives` | `PatternDataset.cs` | SIMD-accelerated pattern generation |

## Files

```
samples/NivaraVAE/
├── Program.cs           # Entry point, CLI parsing, training loop, generation modes
├── VaeModel.cs          # VaeModel<T> : Module<T> with Sequential encoder/decoder
├── PatternDataset.cs    # Synthetic pattern generation + NivaraFrame builder
└── NivaraVAE.csproj     # Project file referencing Nivara core
```

## Requirements

- .NET 10.0 SDK
- Nivara core library (`src/Nivara/Nivara.csproj`)
- No external dependencies

## Library gaps this example exposed and resolved

NivaraVAE drove several core library fixes and improvements. The original spec identified these gaps; all were resolved during implementation.

| Gap | Problem | Resolution |
|-----|---------|------------|
| **`Activation.LeakyRelu` slope default** | `LeakyRelu<T>(input, negativeSlope: default)` where `T` is `float` produces slope=0 (equivalent to ReLU), not 0.01 as intended. `default(float)` is 0. | Changed default parameter handling: when slope is `T.Zero`, use `T.CreateChecked(0.01f)` instead. File: `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`. |
| **`BCEWithLogitsLoss` returns SUM** | Loss scales with batch size, requiring LR tuning per batch size. Not suitable for batched training without manual normalization. | Added `Forward(logits, targets, bool reduceToMean)` overload that divides by element count when `reduceToMean: true`. File: `src/Nivara/AutoDiff/Nn/Functional/BCEWithLogitsLoss.cs`. |
| **ADR-001: null handling in VAE hot paths** | `KlDivergence`, `SampleNormal`, `AccumulateGradient`, and `AdamW` contained ~200 lines of dead null-handling branches (AutoDiff is non-nullable per ADR-001). | Removed null branches from: `ApplyKlElementWise`, `ApplyKlMeanGradient`, `ApplyKlLogVarGradient`, `ApplySampleNormalForward`, `ApplySampleNormalLogVarGradient`, `AccumulateGradient`, `AdamW.applyAdamW`. Single SIMD path via `TensorPrimitives`. Files: `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`, `src/Nivara/AutoDiff/Optimizer/AdamW.cs`. |
| **`TensorPrimitives` SIMD in KL/sample ops** | After null cleanup, the freed code paths were replaced with direct `TensorPrimitives.Exp`, `.Multiply`, `.Add`, `.Subtract` calls for SIMD acceleration. | Replaced scalar loops with `TensorPrimitives` spans in `ApplyKlElementWise`, `ApplySampleNormalForward`, and gradient helpers. |
| **`BCEWithLogitsLoss` backward gradient at x=0** | Forward decomposed into `Relu(x) - x*z + log(1+exp(-|x|))`. At x=0, Relu and Abs subgradients are both 0, producing gradient -1.0 instead of the correct `sigmoid(0) - z = -0.5`. | Replaced primitive-ops decomposition with fused backward that computes `sigmoid(x) - z` directly via a single `OpNode<T>` custom backward. Forward values unchanged. File: `src/Nivara/AutoDiff/Nn/Functional/BCEWithLogitsLoss.cs`. |

### Core library additions from this example

| New API | Location | Purpose |
|---------|----------|---------|
| `BCEWithLogitsLoss<T>.Forward(logits, targets, reduceToMean)` | `src/Nivara/AutoDiff/Nn/Functional/BCEWithLogitsLoss.cs` | Mean-reduced BCE loss for batched training |
| `BCEWithLogitsLoss<T>` fused backward | `src/Nivara/AutoDiff/Nn/Functional/BCEWithLogitsLoss.cs` | Correct gradient `sigmoid(x) - z` via custom `OpNode<T>`, fixing subgradient bug at x=0 |
| `Activation.LeakyRelu` default slope fix | `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` | Correct 0.01 default instead of 0 |

### Core library performance fixes driven by this example

| Fix | What changed | Impact |
|-----|-------------|--------|
| **ADR-001 null cleanup (VAE paths)** | Removed ~200 lines of null-handling branches from `KlDivergence`, `SampleNormal`, `AccumulateGradient`, and `AdamW` internal helpers. Single code path per operation. | Eliminates branch mispredictions on hot training paths. All VAE operations now have single (null-free) SIMD-accelerated paths. |
| **TensorPrimitives in KL/sample ops** | After null branch removal, `ApplyKlElementWise` uses `TensorPrimitives.Multiply`, `.Exp`, `.Add`, `.Subtract` on raw spans. `ApplySampleNormalForward` uses `TensorPrimitives.Multiply`, `.Exp`, `.Add`. | SIMD-vectorized on AVX2/AVX512 hardware. Estimated 2-4x speedup for KL divergence and reparameterization on large batches. |
| **AccumulateGradient simplification** | Reduced to single `TensorPrimitives.Add` call — no null-merge logic, no `WithoutNulls()` copy. | Hottest path in backward pass; every gradient accumulation benefits. |
| **Fused BCE backward** | `BCEWithLogitsLoss` backward replaced Relu/Abs decomposition with single custom `OpNode<T>` computing `sigmoid(x) - z` directly. Eliminates subgradient error at x=0 where Relu and Abs both return gradient 0. | Correct gradient everywhere. Fewer graph nodes in backward — ~5 fewer OpNodes per BCE call. |

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Training (10 epochs, 5K patterns, 8x8) | ~3-5s | Manual loop, Adam, batch size 64 |
| Training (10 epochs, 5K patterns, 16x16) | ~8-12s | 4x more pixels, same architecture width |
| Pattern generation (1000 samples) | <100ms | Synthetic data, no I/O |
| Inference (encode + decode) | <50ms | Forward pass only, no grad tracking |

## Limitations

- **Linear-only architecture** — no convolutional layers. Use `--mode conv` for the ConvVAE alternative that leverages spatial structure.
- **No learning rate scheduling** — fixed LR throughout training. Cosine annealing or warmup would improve convergence.
- **No optimizer state serialization** — `ModelSerializer` saves model weights only, not Adam moment buffers. Continued training after load restarts moments from scratch.
- **Small patterns** — 8x8 and 16x16 grids only. Larger patterns would benefit from convolutional architecture.
- **No quantitative evaluation metrics** — no FID, no reconstruction accuracy percentage. Visual inspection only.

## Future work

1. **Conv2d / ConvTranspose2d modules** — done. `ConvVAE<T>` uses Conv2d encoder and ConvTranspose2d decoder. Accessible via `--mode conv`.
2. **Conditional VAE (CVAE)** — condition on pattern type (circle vs stripe vs blob) by concatenating a one-hot encoding to the input.
3. **Beta annealing** — schedule `beta` from 0 to 1 across epochs for better latent disentanglement.
4. **Export latent vectors to NivaraFrame** — encode all patterns, store latent vectors in a frame, then query by cosine similarity. Would exercise `NivaraFrame.Dot<T>` / `CosineSimilarity<T>`.
5. **MNIST integration** — replace synthetic patterns with MNIST digits for a more realistic benchmark.
