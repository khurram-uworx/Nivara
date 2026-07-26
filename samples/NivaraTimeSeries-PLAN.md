# NivaraTimeSeries — Server Monitoring Anomaly Detection

> **Status: IMPLEMENTED** — All files created and building. See `README.md` for usage.

## Resolved Library Gaps

During implementation, three gaps in the core library were exposed and fixed:

| Gap | Problem | Fix | Files |
|-----|---------|-----|-------|
| BatchNorm1d rejects 3D input | `Forward` threw when `input.Rank != 2`, breaking Conv1d→BN pipeline | Accept 2D `[N, C]` or 3D `[B, C, L]`, extract `planeSize` from dimension | `BatchNorm.cs`, `BatchNormKernel.cs` |
| xHat latent allocation bug | `xHat` only allocated when `affine=true`, but SIMD path always writes to it | Always allocate `xHat` regardless of `affine` | `BatchNormKernel.cs` |
| MSELoss sum-only | No `reduceToMean` overload; loss scaled with input shape | Added `Forward(pred, targets, bool reduceToMean)` overload | `MSELoss.cs` |

## Overview

A sample project demonstrating Nivara's AutoDiff engine for practical time series anomaly detection using server monitoring metrics (CPU, memory, disk I/O, network traffic). Trains a Conv1d-based Variational Autoencoder on normal operating patterns, then detects anomalies by measuring reconstruction error.

**Target audience:** .NET engineers who deal with Application Insights, Prometheus, Grafana daily. Shows that ML is infrastructure, not a specialized Python thing.

**Key message:** Train a model on normal traffic in pure C#, let it flag deviations automatically — no Python, no Torch, no external dependencies.

---

## Domain

4-channel multivariate time series (each channel 0-100, normalized):

| Channel | Normal behavior | Anomaly types |
|---------|----------------|---------------|
| **CPU** | Steady 30-70% with diurnal pattern | Spike (burst to 95%+), flatline (stuck at 0%) |
| **Memory** | Gradual growth, periodic GC dips | Step jump (sudden 20%+ increase = leak), sustained growth |
| **Disk I/O** | Low baseline with periodic bursts | Sustained high (disk thrashing), sudden flatline (failure) |
| **Network** | Correlated with CPU, bursty | Spike (DDoS), level shift (new connection pool) |

**Normal patterns** (training data):
- Diurnal CPU cycle (sinusoidal daily pattern + noise)
- Gradual memory growth (linear trend + sawtooth GC)
- Periodic disk I/O bursts (background jobs every N minutes)
- Network correlated with CPU (request-driven)

**Anomaly patterns** (test data, ~15% of test set):
- **Spike**: Sudden jump to 90-100% for 5-15 steps, then return
- **Level Shift**: Step change +20-40% sustained for 20+ steps (memory leak)
- **Trend Change**: Slope change in growth rate (disk filling faster)

---

## Architecture

### Conv1d Encoder + Linear Decoder VAE

```
Input: [batch, 4, windowSize]          (4 metrics, windowSize=64 timesteps)

Encoder (Conv1d feature extraction):
  Conv1d(4 → 32, kernel=7, padding=3)  → BatchNorm1d(32) → LeakyReLU
  Conv1d(32 → 64, kernel=5, padding=2) → BatchNorm1d(64) → LeakyReLU
  Conv1d(64 → 32, kernel=3, padding=1) → BatchNorm1d(32) → LeakyReLU
  Output: [batch, 32, windowSize]      (same length due to padding = kernel//2)

Flatten: [batch, 32 * windowSize]

Latent space:
  mu:     Linear(32 * windowSize → latentDim)
  logVar: Linear(32 * windowSize → latentDim)
  z = mu + exp(logVar * 0.5) * ε      (reparameterization trick)

Decoder (Linear reconstruction):
  Linear(latentDim → 128) → LeakyReLU → Dropout
  Linear(128 → 128) → LeakyReLU → Dropout
  Linear(128 → 4 * windowSize)         (raw output, no activation)

Output: [batch, 4, windowSize]          (reshaped back to channel layout)
```

### Design Choices

- Conv1d with `padding = kernelSize // 2` preserves sequence length — no information loss from downsampling
- BatchNorm1d after each Conv1d — stabilizes training, demonstrates the new BatchNorm module
- Linear decoder (not ConvTranspose1d) — keeps it simple, Conv1d already extracted temporal features
- `MSELoss` for reconstruction (continuous metrics, not binary like VAE patterns)
- `KlDivergence` for regularization (standard VAE ELBO)

### Loss Function

```
ELBO = MSE(reconstruction, original) + beta * KL(mu, logVar)
```

- `MSELoss<T>` for continuous-valued metrics (not BCE — these aren't binary)
- `KlDivergence(mu, logVar)` for latent regularization
- Manual training loop with `GradientUtils.Grad()` (VAE loss needs 4 args)

---

## Files to Create

```
samples/NivaraTimeSeries/
├── NivaraTimeSeries.csproj    # Standard console app, Nivara core only
├── Program.cs                 # CLI, training loop, detection, ASCII viz
├── MetricsGenerator.cs        # Synthetic server metrics with injected anomalies
├── TimeSeriesModel.cs         # Conv1d encoder + Linear decoder VAE
└── README.md                  # Documentation
```

Also update: `samples/README.md` — add entry for NivaraTimeSeries.

---

## File Details

### 1. `NivaraTimeSeries.csproj`

Standard boilerplate (identical to NivaraVAE):

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

### 2. `TimeSeriesModel.cs`

Namespace: `NivaraTimeSeries`

```csharp
public sealed class TimeSeriesModel<T> : Module<T> where T : struct, INumber<T>
```

**Fields:**
- `Conv1d<T> _conv1, _conv2, _conv3` — encoder conv layers
- `BatchNorm1d<T> _bn1, _bn2, _bn3` — batch norm after each conv
- `Linear<T> _muHead, _logVarHead` — latent space projection
- `Linear<T> _dec1, _dec2, _dec3` — decoder FC layers
- `Dropout<T> _drop1, _drop2` — decoder dropout
- `int _windowSize, _latentDim, _convOutputSize`

**Constructor:** `TimeSeriesModel(int numChannels = 4, int windowSize = 64, int latentDim = 16, int hiddenDim = 128, float dropout = 0.2f)`

**Methods:**
- `Forward(input)` → full encode → reparameterize → decode
- `Encode(input)` → `(ReverseGradTensor<T> Mu, ReverseGradTensor<T> LogVar)`
- `Reparameterize(mu, logVar, seed?)` → `ReverseGradTensor<T>`
- `Decode(z)` → `ReverseGradTensor<T>`
- `ReconstructError(input)` → `float` (MSE per element, for anomaly scoring)

**Forward flow:**
```
input: [B, 4, W]
  → Conv1d(4→32, k=7, p=3) → BN → LeakyReLU     [B, 32, W]
  → Conv1d(32→64, k=5, p=2) → BN → LeakyReLU    [B, 64, W]
  → Conv1d(64→32, k=3, p=1) → BN → LeakyReLU    [B, 32, W]
  → Reshape [B, 32*W]
  → mu: Linear(32*W → latentDim)                   [B, latentDim]
  → logVar: Linear(32*W → latentDim)               [B, latentDim]
  → z = SampleNormal(mu, logVar)
  → Dec: Linear(latDim → hidden) → LeakyReLU → Dropout
  → Dec: Linear(hidden → hidden) → LeakyReLU → Dropout
  → Dec: Linear(hidden → 4*W)                       [B, 4*W]
  → Reshape [B, 4, W]
```

### 3. `MetricsGenerator.cs`

Namespace: `NivaraTimeSeries`

```csharp
public sealed class MetricsGenerator
```

**Public API:**
- `MetricsGenerator(int numSamples, int windowSize, int seed = 42, float anomalyRatio = 0.15f)`
- `float[] GetNormalWindow(int index)` — returns `numChannels * windowSize` flattened data
- `float[] GetWindow(int index)` — returns window (may be normal or anomalous)
- `bool IsAnomaly(int index)` — whether this window has injected anomaly
- `int Count` — total windows
- `int NumChannels` => 4
- `int WindowSize` — timesteps per window
- `NivaraFrame Frame` — all windows as a single column frame (for DataLoader compatibility)

**Data generation algorithm:**

```
For each window i:
  1. Generate base channels (normal patterns):
     CPU:     50 + 20*sin(2π*t/96) + noise(σ=3)
     Memory:  40 + 0.05*t - 5*floor(t/20) + noise
     Disk:    5 + 15*max(0, sin(2π*t/24)) + noise
     Network: 30 + 15*sin(2π*t/96) + noise(σ=5)

  2. If anomalyRatio applies (deterministic based on index + seed):
     Spike:     cpu[t:t+10] = 95 + noise(σ=2)
     Level:     mem[t:] += 25 (step change, sustained)
     Trend:     disk[t:] *= 1.5 (slope change)

  3. Normalize all channels to [0, 1] range
  4. Store flattened: [cpu_0, cpu_1, ..., cpu_W, mem_0, ..., net_W]
```

**File I/O:**
- `Save(string path)` — CSV format: header + one row per window
- `Load(string path)` — restore from CSV

### 4. `Program.cs`

Top-level statements entry point. Structure:

```
1. Parse CLI options (Options class, same pattern as NivaraVAE)
2. Generate or load dataset
3. If --show-metrics: display ASCII chart and exit
4. Split into train (normal-only) and test (normal + anomalous)
5. Train VAE on training set
6. Compute baseline reconstruction error statistics
7. If --detect: run anomaly detection on test set, print results
8. If --save: save model
```

**Training loop** (manual, same pattern as NivaraVAE):
```csharp
model.Train();
for (int epoch = 1; epoch <= epochs; epoch++)
{
    Shuffle(indices, rng);
    for (int start = 0; start < dataset.Count; start += batchSize)
    {
        int size = Math.Min(batchSize, dataset.Count - start);
        var features = BuildBatch(dataset, indices, start, size, numChannels, windowSize, requiresGrad: true);
        var targets  = BuildBatch(dataset, indices, start, size, numChannels, windowSize, requiresGrad: false);

        using (GradientUtils.Grad())
        {
            var (mu, logVar) = model.Encode(features);
            var z = model.Reparameterize(mu, logVar);
            var recon = model.Decode(z);

            var reconLoss = new MSELoss<float>().Forward(recon, targets);
            var kl = ReverseGradOperations.KlDivergence(mu, logVar);
            var klMean = ReverseGradOperations.Divide(kl, batchSizeTensor);
            var loss = ReverseGradOperations.Add(reconLoss,
                ReverseGradOperations.Multiply(klMean, betaTensor));

            loss.Backward();
            GradientUtils.ClipGradNorm(model.Parameters().Values, 1.0);
            optimizer.Step();
            optimizer.ZeroGrad();
            lossVal = loss[0];
        }
    }
}
```

**Anomaly detection:**
```csharp
model.Eval();
// 1. Compute reconstruction errors on training set (normal data)
var trainErrors = new List<float>();
foreach (var window in trainSet)
{
    var input = BuildTensor(window, requiresGrad: false);
    var recon = model.Forward(input);
    var error = MseBetween(recon, input);
    trainErrors.Add(error);
}
float mean = trainErrors.Average();
float stddev = StandardDeviation(trainErrors);
float threshold = mean + 2 * stddev;

// 2. Score test windows
foreach (var window in testSet)
{
    var input = BuildTensor(window, requiresGrad: false);
    var recon = model.Forward(input);
    var error = MseBetween(recon, input);
    bool isAnomaly = error > threshold;
    // Print result with channel attribution
}
```

**ASCII visualization:**
```
--- Sample Metrics (first 128 timesteps) ---
     CPU   Mem   Disk  Net
  0: ▃▃▃▃ ▄▄▄▄ ▁▁▁▁ ▃▃▃▃
 16: ▅▅▅▅ ▄▄▄▄ ▁▁▁▁ ▅▅▅▅
 32: ▇▇▇▇ ▅▅▅▅ ▃▃▃▃ ▇▇▇▇
 48: ████ ▅▅▅▅ ▁▁▁▁ ████
```

Uses Unicode block characters: `▁▂▃▄▅▆▇█` (8 levels) to render metric values.

**Options class:**
```csharp
sealed class Options
{
    public int Epochs { get; init; } = 20;
    public int WindowSize { get; init; } = 64;
    public int LatentDim { get; init; } = 16;
    public int HiddenDim { get; init; } = 128;
    public int BatchSize { get; init; } = 64;
    public float LearningRate { get; init; } = 0.001f;
    public float Beta { get; init; } = 0.5f;
    public float Dropout { get; init; } = 0.2f;
    public int NumSamples { get; init; } = 5000;
    public int Seed { get; init; } = 42;
    public float AnomalyRatio { get; init; } = 0.15f;
    public string? SavePath { get; init; }
    public string? LoadPath { get; init; }
    public bool Detect { get; init; }
    public bool ShowMetrics { get; init; }
    public bool Help { get; init; }

    public static Options Parse(string[] args) { ... }
    public static void PrintHelp() { ... }
}
```

---

## Nivara APIs Demonstrated

| API | Purpose in this sample |
|-----|----------------------|
| `Conv1d<T>` | Temporal feature extraction from multivariate metrics |
| `BatchNorm1d<T>` | Stabilize conv feature maps (train/eval modes) |
| `Linear<T>` | Decoder (reconstruction), mu/logVar heads |
| `Dropout<T>` | Regularization in decoder |
| `Activation.LeakyRelu<T>` | Non-linearity throughout |
| `MSELoss<T>` | Reconstruction loss (continuous values) |
| `KlDivergence` | VAE latent regularization |
| `SampleNormal` | Reparameterization trick |
| `Adam<T>` | Optimizer |
| `GradientUtils.Grad()` | Autograd scope |
| `GradientUtils.ClipGradNorm` | Gradient clipping |
| `ModelSerializer.Save/Load` | Model persistence |
| `NivaraColumn<float>` | Metric channel storage |
| `NivaraFrame` | Windowed dataset |

---

## CLI Usage

```bash
# Train with defaults
dotnet run --project samples/NivaraTimeSeries -- --epochs 20

# Custom hyperparameters
dotnet run --project samples/NivaraTimeSeries -- --epochs 30 --window-size 128 --latent-dim 16 --lr 0.0005

# Train + save model
dotnet run --project samples/NivaraTimeSeries -- --epochs 20 --save timeseries.json

# Load model + detect anomalies
dotnet run --project samples/NivaraTimeSeries -- --load timeseries.json --detect

# Show sample metrics as ASCII chart
dotnet run --project samples/NivaraTimeSeries -- --show-metrics

# Control anomaly injection
dotnet run --project samples/NivaraTimeSeries -- --anomaly-ratio 0.15 --seed 42

# Help
dotnet run --project samples/NivaraTimeSeries -- --help
```

---

## Implementation Order

1. `NivaraTimeSeries.csproj` — project file
2. `MetricsGenerator.cs` — synthetic data generation with normal + anomaly patterns
3. `TimeSeriesModel.cs` — Conv1d encoder + Linear decoder VAE
4. `Program.cs` — CLI parsing, training loop, detection logic, ASCII visualization
5. `README.md` — documentation
6. Update `samples/README.md` — add NivaraTimeSeries entry

---

## Differences from NivaraVAE

| Aspect | NivaraVAE | NivaraTimeSeries |
|--------|-----------|-----------------|
| Domain | Binary 2D patterns | Server monitoring metrics |
| Input | 1D flat (pixels) | 3D multivariate (channels × time) |
| Encoder | Linear layers | Conv1d layers (temporal features) |
| Loss | BCEWithLogits (binary) | MSELoss (continuous) |
| Use case | Generation, interpolation | Anomaly detection |
| BatchNorm | No | Yes (BatchNorm1d after each Conv) |
| Practical value | Academic | Real-world monitoring |

---

## Key Implementation Notes

### Conv1d Shape Management

Conv1d expects input `[N, C, L]`. With `padding = kernelSize // 2` and `stride = 1`, output length equals input length:
```
oL = (L + 2*padding - kernelSize) / stride + 1 = L
```

So the three conv layers all preserve `windowSize` in their output. The flatten before the latent heads is `32 * windowSize`.

### Reshaping Between Conv1d and Linear

The Conv1d output `[B, 32, W]` needs to be flattened to `[B, 32*W]` for the Linear heads. Use `tensor.Reshape(B, 32 * W)`.

The decoder output `[B, 4*W]` needs to be reshaped to `[B, 4, W]` for comparison with input. Use `tensor.Reshape(B, 4, W)`.

### Manual Training Loop (Why Not TrainingLoop)

Same reason as NivaraVAE: VAE loss requires 4 arguments (recon, original, mu, logVar) but `TrainingLoop<T>` only supports 2-arg loss `(output, labels)`. Use manual loop with `GradientUtils.Grad()` scope.

### MSELoss vs BCEWithLogitsLoss

Server metrics are continuous [0, 1] values, not binary 0/1. `MSELoss` is appropriate for reconstruction of continuous data. `BCEWithLogitsLoss` is for binary/Bernoulli targets (like the VAE pixel patterns).

### Anomaly Threshold

Use training set statistics: `threshold = mean(reconErrors) + 2 * stddev(reconErrors)`. This captures ~95% of normal reconstruction errors. Windows exceeding this threshold are flagged as anomalies.

For channel attribution: compute per-channel MSE to identify which metric caused the anomaly.

### ASCII Chart Rendering

Use Unicode block characters: `▁▂▃▄▅▆▇█` (8 levels from low to high). Map each value to the nearest level based on its position in the [0, 1] range. Render each channel as a separate column, one row per time step group (16 steps per row).

---

## Dependencies

- .NET 10.0 SDK
- Nivara core library (`src/Nivara/Nivara.csproj`)
- No external NuGet packages

---

## Implementation Notes

All planned files have been created:

- **NivaraTimeSeries.csproj** — Standard console app referencing Nivara core
- **MetricsGenerator.cs** — Generates 4-channel synthetic metrics with SPIKE, LEVEL, and TREND anomalies; per-channel normalization; CSV save/load
- **TimeSeriesModel.cs** — Conv1d encoder (4→32→64→32 channels) with BatchNorm1d + Linear decoder VAE
- **Program.cs** — CLI with full options parsing, manual training loop, anomaly detection, ASCII visualization
- **README.md** — Comprehensive documentation

Training with defaults (20 epochs, 5K windows) completes successfully and detects anomalies with reasonable precision. The sample exercises all target Nivara APIs including the new BatchNorm1d 3D input support and MSELoss `reduceToMean`.

---

## Expected Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Training (20 epochs, 5K windows, W=64) | ~10-20s | Conv1d is more compute than Linear |
| Anomaly detection (1000 windows) | <2s | Forward pass only, no grad tracking |
| Model size | ~200KB JSON | Conv1d weights + BatchNorm params |

---

## README Structure

```markdown
# NivaraTimeSeries — Server Monitoring Anomaly Detection

## What it does
[Description of the anomaly detection use case]

## Quick start
[Basic commands]

## CLI options
[Table of all options]

## How it works
### Data generation
[Normal and anomaly patterns]

### Model architecture
[Conv1d encoder + Linear decoder diagram]

### Training
[Manual VAE training loop]

### Anomaly detection
[Reconstruction error threshold approach]

## What this exercises vs. other samples
[Comparison table]

## Nivara APIs demonstrated
[Table of APIs used]

## Files
[File listing]

## Requirements
[.NET 10.0, no external deps]
```
