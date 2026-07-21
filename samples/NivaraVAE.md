# NivaraVAE: VAE for Synthetic Pattern Generation

## Goal

Create a second Nivara showcase example that goes beyond what MicroGpt
demonstrates — exercising the module system, training pipeline, serialization,
and a fundamentally different architecture (encoder–decoder vs. autoregressive).

The example is self-contained: no external data downloads. Pattern data is
generated synthetically so anyone can run it immediately.

## Architecture

```
PatternDataset (synthetic 2D patterns on a grid)
    |
    v
TensorDataset<T> + DataLoader<T>  (batched, shuffled)
    |
    v
VaeModel<T>  (Module<T> subclass)
    ├── Encoder: Sequential<Linear → LeakyRelu → Dropout → Linear → LeakyRelu>
    ├── MuHead:  Linear
    ├── LogVarHead: Linear
    ├── Reparameterize: SampleNormal
    └── Decoder: Sequential<Linear → LeakyRelu → Dropout → Linear → Sigmoid>
    |
    v
BCEWithLogitsLoss<T> + KlDivergence (ELBO loss)
    |
    v
TrainingLoop<T> + AdamW<T>  (structured training with epoch callbacks)
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
  Linear(hiddenDim, patternSize²) → Sigmoid

Output: [batch, patternSize²]  (reconstructed binary probabilities)
```

### Loss

```
ELBO = BCEWithLogitsLoss(recon_logits, target) + beta * KL(mu, logVar)
```

Where:
- `BCEWithLogitsLoss<T>` computes numerically stable binary cross-entropy
  from logits (the decoder outputs Sigmoid internally for display but the
  loss operates on logits before the sigmoid)
- `KlDivergence(mu, logVar)` computes `-0.5 * sum(1 + logVar - mu² - exp(logVar))`
- `beta` default 1.0, adjustable via CLI

## CLI Interface

```
--epochs <int>         Training epochs (default: 10)
--latent-dim <int>     Latent space dimension (default: 8)
--hidden-dim <int>     Hidden layer size (default: 128)
--batch-size <int>     Batch size (default: 64)
--lr <float>           Learning rate (default: 0.001)
--pattern-size <int>   Pattern grid size (default: 8 → 8×8 = 64 pixels)
--num-patterns <int>   Number of synthetic patterns to generate (default: 5000)
--seed <int>           RNG seed for reproducibility (default: 42)
--beta <float>         KL divergence weight (default: 1.0)
--dropout <float>      Dropout probability (default: 0.2)
--save <path>          Save trained model to file
--load <path>          Load trained model from file
--generate <int>       Generate N samples from random latent vectors
--interpolate <int>    Interpolate between N random latent pairs (show steps)
--latent-walk          Walk each latent dimension one at a time (show grid)
--show-patterns        Display example patterns from the dataset (console art)
--eval                 Evaluate reconstruction on test set
--help, -h             Show this help
```

### Mode descriptions

**Default (no generation flags):** Train for `--epochs` epochs, print loss
per epoch, then generate a few samples.

**`--generate N`:** After training (or after `--load`), sample N latent
vectors from N(0,1) and decode to show generated patterns.

**`--interpolate N`:** Pick N random pairs from the dataset, encode to latent,
linearly interpolate between them in 5 steps, decode each step. Shows that
the latent space is smooth.

**`--latent-walk`:** For each latent dimension, sweep from -3 to +3 in 7
steps while holding others at 0. Decode each step. Reveals what each
latent dimension encodes.

**`--eval`:** Split data 80/20 train/test, report reconstruction BCE on
test set after training.

## Pattern Generation (PatternDataset)

Patterns are generated on a grid of `patternSize × patternSize` binary pixels.
Each pixel is a `float` in {0, 1}. The dataset stores the flattened grid as a
single column in a `NivaraFrame`. For the VAE, the grid is both input and
target (autoencoding).

### Pattern types (randomly chosen per sample)

| Type | Description | Parameters varied |
|---|---|---|
| Circle | Filled circle at random position | Center x, y, radius |
| Stripes | Horizontal or vertical stripes | Orientation, phase, thickness |
| Blob | Gaussian blob, thresholded | Center x, y, sigma |
| Checkerboard | Checkerboard of random cell size | Cell size |
| Corner | Single filled quadrant | Which quadrant |
| Cross | Cross/plus shape | Center, arm width |

Each pattern is generated as a 2D array, thresholded, then flattened to
`patternSize²` elements stored in a `NivaraColumn<float>`.

The `PatternDataset` class:

```csharp
public sealed class PatternDataset
{
    public NivaraFrame Frame { get; }   // single column "pixels"
    public int Count { get; }
    public int GridSize { get; }
    public int NumPixels { get; }

    public PatternDataset(int numPatterns, int gridSize, int seed = 42);

    // Display a pattern as ASCII art (console output)
    public static string RenderPattern(ReadOnlySpan<float> pixels, int gridSize);
}
```

## What This Exercises vs. MicroGpt

| Feature | MicroGpt | NivaraVAE |
|---|---|---|
| **Architecture type** | Autoregressive (per-token) | Feedforward encoder–decoder |
| **Module<T> inheritance** | No (manual parameter list) | Yes (`VaeModel<T> : Module<T>`) |
| **Sequential<T>** | No | Yes (encoder/decoder chains) |
| **Dropout<T>** | No | Yes (regularization) |
| **TrainingLoop<T>** | No (raw loop) | Yes (epoch callbacks, structured) |
| **DataLoader<T>** | No (single doc per step) | Yes (batched, shuffled) |
| **TensorDataset<T>** | No | Yes (NivaraFrame-backed dataset) |
| **Loss function** | Hand-rolled NLL | BCEWithLogitsLoss<T> + KlDivergence |
| **Optimizer** | Adam<T> | AdamW<T> (decoupled weight decay) |
| **ModelSerializer** | No | Save/Load with CLI flags |
| **Interactive modes** | Generate only | Generate, interpolate, latent walk, eval |
| **GradientUtils** | Grad() only | Grad() + epoch-level management |
| **Sigmoid activation** | No | Yes (decoder output) |
| **LeakyRelu activation** | No | Yes (encoder/decoder hidden) |
| **KlDivergence + SampleNormal** | No | Yes (VAE-specific ops) |
| **Batch tensor ops** | No (single row) | Yes (batched matrix multiply) |
| **Data pipeline** | External download (names.txt) | Fully synthetic, no downloads |
| **Type parameter** | Uses float throughout | Uses float throughout |
| **Gradient clipping** | No | Yes (ClipGradNorm on total loss) |

## Files

```
samples/NivaraVAE/
├── Program.cs           # Entry point, CLI parsing, mode dispatch
├── VaeModel.cs          # VaeModel<T> : Module<T>
├── PatternDataset.cs    # Synthetic pattern generation + NivaraFrame builder
└── NivaraVAE.csproj     # Project file referencing Nivara
```

### NivaraVAE.csproj

Same pattern as MicroGpt:

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

### Program.cs structure

```csharp
using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Training;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Utilities;
using NivaraVAE;

// 1. Parse CLI args (same pattern as MicroGpt)

// 2. Generate synthetic patterns → NivaraFrame → TensorDataset<float>
//    Patterns are stored as a single "pixels" column of float

// 3. Create TensorDataset<float> with feature="pixels", label="pixels"

// 4. Create DataLoader<float> with the dataset

// 5. Create VaeModel<float>

// 6. Create AdamW<float> optimizer, register model parameters

// 7. Define loss function:
//    var bceLoss = new BCEWithLogitsLoss<float>();
//    (reconLogits, targets) => {
//        var bce = bceLoss.Forward(reconLogits, targets);
//        // Extract mu/logvar from model internals...
//        // Actually: require a wrapper that captures mu/logvar
//    }

// 8. Create TrainingLoop<float> with custom epoch callbacks
//    (uses OnEpochStart/OnBatchEnd for logging and checkpointing)

// 9. Run training → TrainingResult<float>

// 10. Dispatch to generation/interpolation/latent-walk modes
```

**Loss function design:** The VAE loss needs both the reconstruction logits
(from decoder) and the mu/logvar (from encoder). Since `TrainingLoop<T>`
accepts `Func<ReverseGradTensor<T>, ReverseGradTensor<T>, ReverseGradTensor<T>>`
(one output, one target), we need the model to return more than just the
reconstruction. Two approaches:

1. **Recommended: Model returns combined loss directly.**
   Override `VaeModel<T>.Forward` to return the ELBO loss scalar when given
   (input, target) — but `Module<T>.Forward(ReverseGradTensor<T> input1, ReverseGradTensor<T> input2)`
   throws by default. We override it:

   ```csharp
   public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input, ReverseGradTensor<T> target)
   {
       var (recon, mu, logVar) = ForwardWithLatent(input);
       var bce = _bceLoss.Forward(recon, target);
       var kl = ReverseGradOperations.KlDivergence(mu, logVar);
       var scaledKl = ReverseGradOperations.Multiply(kl, _betaTensor);
       return ReverseGradOperations.Add(bce, scaledKl);
   }
   ```

   Then `TrainingLoop<T>` calls `model.Forward(batch.Features, batch.Labels)`
   which works because `TrainingLoop.Run()` calls `_model.Forward(batch.Features)`,
   but that's only one argument. **Gap: we need to handle this.** Options:

   - **Option A:** Wrap the model in a lambda for `TrainingLoop<T>`:
     ```csharp
     var loop = new TrainingLoop<float>(
         model,
         loader,
         (features, labels) => {
             // Forward returns (recon, mu, logVar), compute loss
             // Use model's internal forward that returns all three
             var (recon, mu, logVar) = model.ForwardWithLatent(features);
             var bce = bceLoss.Forward(recon, labels);
             var kl = ReverseGradOperations.KlDivergence(mu, logVar);
             return ReverseGradOperations.Add(bce, kl);
         },
         optimizer,
         epochs);
     ```
   - **Option B:** Have `VaeModel<T>.Forward` do full autoencoding:
     `Forward(x) → elbo-loss-scalar` using internal forward pass + BCE loss
     stored as fields. This couples the model to the loss function, which is
     less clean but simplest for an example.

   We'll go with **Option A** — it keeps the model reusable and the loss
   composable, which is the idiomatic Nivara pattern.

## Anticipated Gaps & Fixes

These are things we expect to discover while writing this example:

### Gap 1: Embedding<T> is not a Module<T>

Not relevant for VAE (we don't need embedding lookup), but noted.

### Gap 2: TrainingLoop<T>.Forward only takes one tensor

`TrainingLoop.Run()` calls `_model.Forward(batch.Features)`, not `.Forward(features, labels)`.
VAEs typically need both features and labels (which are the same for
autoencoding). **Workaround:** Use a lambda-based loss function that calls
the model's forward internally. This works without any changes to `TrainingLoop<T>`.

### Gap 3: BCEWithLogitsLoss<T> returns sum not mean

`BCEWithLogitsLoss<T>.Forward` returns `Sum(loss)`, not `Mean(loss)`.
For batched training this means the loss scales with batch size, requiring
LR tuning per batch size. **Fix (in the example):** divide by batch size
manually in the loss lambda. Alternatively, add a `reduction` parameter
to the loss class — worth considering as a Nivara improvement.

### Gap 4: No test/validation split in TrainingLoop

`TrainingLoop<T>` doesn't have a validation callback or test set evaluation.
We handle test evaluation manually after training (`--eval` mode), which
is fine for a showcase.

### Gap 5: No built-in LR scheduler

`Optimizer<T>` only has `SetGroupLearningRate`. For cosine annealing or
warmup, we'd need to implement it outside the optimizer. Not needed for
this example (fixed LR works fine for small VAEs), but noted.

### Gap 6: ModelSerializer.Load does not propagate requiresGrad

`JsonToStateDict<T>` has a `requiresGrad` parameter defaulting to `false`.
When loading a model for continued training, we need `requiresGrad: true`.
**Workaround:** Call `JsonToStateDict<T>(json, requiresGrad: true)` manually
(already supported via the parameter). The `ModelSerializer.Load<T>` helper
doesn't expose this — we may call `LoadStateDict` directly with the JSON
parsed manually.

### Gap 7: NivaraColumn copy behavior with null masks

When copying pixel data that has no nulls (our patterns are always fully
observed), the `nullMask` propagation path is never exercised. This is a
design choice — the example uses dense binary data. The null-mask paths
are tested in unit tests already.

## Relationship to Existing VAE<T>

Nivara already has `Nivara.AutoDiff.Nn.VAE<T>`. The key difference is:

| Aspect | Existing VAE<T> | Our VaeModel<T> |
|---|---|---|
| Architecture | 2 hidden layers encoder, 1 hidden decoder | Same structural pattern |
| Activation | Configurable (default ReLU) | LeakyReLU (fixed) |
| Regularization | None (no dropout) | Dropout(0.2) |
| Loss | `ElboLoss()` method (MSE + KL) | Separate `BCEWithLogitsLoss<T>` + `KlDivergence` |
| Module structure | `Module<T>` subclass, proper registration | Same pattern |
| Forward signature | `Forward(x) → recon` (only) | Dual forward: recon-only + (recon, mu, logVar) |
| CLI wiring | None (library class) | Full interactive CLI |

The existing `VAE<T>` is a library component. Our `VaeModel<T>` is a
showcase example that could be simplified to just subclass `VAE<T>` and
add the dual-forward + CLI, but writing it from scratch as `VaeModel<T>`
with `Sequential<T>` is more instructive and exercises more of the API.

## Testing Strategy

Since patterns are synthetic, we can verify correctness by:

1. **Training convergence:** Loss should decrease monotonically over epochs
2. **Reconstruction accuracy:** After training, encode→decode a test pattern;
   the output should resemble the input
3. **Latent interpolation:** Interpolating between two encoded patterns
   should produce visually intermediate results
4. **Latent walk:** Varying a single latent dimension should produce a
   smooth, interpretable change in the output
5. **Save/load round-trip:** Save then load model, generate same sample with
   same seed → identical output

## Stretch Goals (Post-V1)

If the basic example works and we want to push further:

1. **Add `Conv2d` / `ConvTranspose2d` modules** — the current Linear-only
   VAE ignores spatial structure. A convolutional VAE would be more
   realistic and reveal if Nivara's op set supports conv operations.
   (It does not currently — this would be a new module family.)

2. **Conditional VAE (CVAE)** — condition on pattern type (circle vs stripe
   vs blob) by concatenating a one-hot encoding to the input.

3. **Beta-VAE** with beta annealing schedule across epochs — useful for
   discovering disentangled latent factors.

4. **Export trained patterns to NivaraFrame for CosineSimilarity lookup** —
   encode all patterns, store latent vectors in a frame, then query by
   latent similarity. This would exercise the `NivaraFrame.Dot<T>` /
   `CosineSimilarity<T>` paths.
