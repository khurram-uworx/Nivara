# NivaraTimeSeries — Server Monitoring Anomaly Detection

A sample project demonstrating Nivara's AutoDiff engine for practical time series anomaly detection using server monitoring metrics (CPU, memory, disk I/O, network traffic). Trains a Conv1d-based Variational Autoencoder on normal operating patterns, then detects anomalies by measuring reconstruction error.

**Target audience:** .NET engineers who deal with Application Insights, Prometheus, Grafana daily. Shows that ML is infrastructure, not a specialized Python thing.

**Key message:** Train a model on normal traffic in pure C#, let it flag deviations automatically — no Python, no Torch, no external dependencies.

## What it does

NivaraTimeSeries generates synthetic 4-channel server metrics (CPU utilization, memory usage, disk I/O, network traffic) with realistic diurnal patterns and injected anomalies (spikes, level shifts, trend changes). It trains a Conv1d encoder + Linear decoder VAE on normal data, then detects anomalies by flagging windows with high reconstruction error.

Key capabilities demonstrated:
- **Conv1d temporal feature extraction** — 3-layer convolutional encoder extracts patterns from multivariate time series
- **BatchNorm1d with 3D input** — normalizes across batch and time dimensions for each channel (exercises the new `[B, C, L]` support)
- **Manual VAE training loop** — `GradientUtils.Grad()` scope with MSE reconstruction + KL divergence loss
- **Anomaly detection via reconstruction error** — threshold-based detection with per-window scoring
- **SIMD-accelerated data generation and scoring** — `TensorPrimitives` for normalization and MSE computation

## Quick start

```bash
# Train with defaults (20 epochs, 5K windows, window size 64)
dotnet run --project samples/NivaraTimeSeries

# Quick test (fewer samples, smaller windows)
dotnet run --project samples/NivaraTimeSeries -- --epochs 10 --num-samples 1000 --window-size 32

# Train + detect anomalies
dotnet run --project samples/NivaraTimeSeries -- --epochs 15 --num-samples 2000 --detect

# Show sample metrics as ASCII chart
dotnet run --project samples/NivaraTimeSeries -- --show-metrics

# Save/load model
dotnet run --project samples/NivaraTimeSeries -- --epochs 20 --save timeseries.json
dotnet run --project samples/NivaraTimeSeries -- --load timeseries.json --detect

# Help
dotnet run --project samples/NivaraTimeSeries -- --help
```

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs <int>` | 20 | Training epochs |
| `--window-size <int>` | 64 | Timesteps per window |
| `--latent-dim <int>` | 16 | Latent space dimension |
| `--hidden-dim <int>` | 128 | Decoder hidden layer size |
| `--batch-size <int>` | 64 | Batch size |
| `--lr <float>` | 0.001 | Learning rate (Adam) |
| `--beta <float>` | 0.5 | KL divergence weight |
| `--dropout <float>` | 0.2 | Dropout probability |
| `--num-samples <int>` | 5000 | Number of synthetic windows |
| `--seed <int>` | 42 | RNG seed |
| `--anomaly-ratio <float>` | 0.15 | Fraction of anomalous windows |
| `--save <path>` | — | Save trained model to JSON |
| `--load <path>` | — | Load model from JSON |
| `--save-data <path>` | — | Save dataset to CSV |
| `--load-data <path>` | — | Load dataset from CSV |
| `--detect` | — | Run anomaly detection after training |
| `--show-metrics` | — | Display sample metrics as ASCII chart |
| `--help`, `-h` | — | Show CLI help |

## How it works

### Data generation

Four synthetic server metrics with realistic patterns:

| Channel | Normal behavior | Anomaly types |
|---------|----------------|---------------|
| **CPU** | Diurnal sine (50 +/- 20%, period=96 steps) | Spike (burst to 95%+) |
| **Memory** | Gradual growth + GC sawtooth | Level shift (sudden +25% = leak) |
| **Disk I/O** | Periodic bursts (period=24 steps) | Trend change (x1.5 slope) |
| **Network** | Correlated with CPU, bursty | — |

All channels normalized to [0, 1] per-window using `TensorPrimitives.Min`/`Max`/`Divide`.

### Model architecture

```
Input: [batch, 4, windowSize]          (4 metrics, windowSize=64 timesteps)

Encoder (Conv1d feature extraction):
  Conv1d(4 -> 32, kernel=7, padding=3)  -> BatchNorm1d(32) -> LeakyReLU
  Conv1d(32 -> 64, kernel=5, padding=2) -> BatchNorm1d(64) -> LeakyReLU
  Conv1d(64 -> 32, kernel=3, padding=1) -> BatchNorm1d(32) -> LeakyReLU
  Output: [batch, 32, windowSize]

Flatten: [batch, 32 * windowSize]

Latent space:
  mu:     Linear(32 * windowSize -> latentDim)
  logVar: Linear(32 * windowSize -> latentDim)
  z = SampleNormal(mu, logVar)          (reparameterization trick)

Decoder (Linear reconstruction):
  Linear(latentDim -> 128) -> LeakyReLU -> Dropout(0.2)
  Linear(128 -> 128) -> LeakyReLU -> Dropout(0.2)
  Linear(128 -> 4 * windowSize)         (raw output)

Output: [batch, 4 * windowSize]         (compared with input via MSELoss)
```

### Training

Manual VAE training loop with `GradientUtils.Grad()` scope:
```
ELBO = MSE(reconstruction, original) / elementCount + beta * KL(mu, logVar)
```

- `MSELoss<T>` with `reduceToMean: true` for continuous-valued metrics
- `KlDivergence(mu, logVar)` for latent regularization
- `Adam<T>` optimizer with gradient clipping (`ClipGradNorm`)
- Trained on normal (non-anomalous) windows only

### Anomaly detection

Reconstruction error threshold approach:
1. Compute MSE reconstruction error for all training windows
2. Calculate `threshold = mean(errors) + 2 * stddev(errors)` (~95th percentile)
3. Flag windows exceeding threshold as anomalies
4. Report precision, recall, F1 score

## What this exercises vs. other samples

| Feature | NivaraVAE | NivaraTimeSeries |
|---------|-----------|-----------------|
| **Architecture type** | Feedforward encoder-decoder | **Conv1d encoder + Linear decoder** |
| **Input shape** | 1D flat `[B, W]` | **3D multivariate `[B, C, L]`** |
| **Encoder** | Linear layers | **Conv1d temporal feature extraction** |
| **BatchNorm** | No | **Yes (BatchNorm1d with 3D input)** |
| **Loss** | BCEWithLogits (binary) | **MSELoss (continuous)** |
| **Use case** | Generation, interpolation | **Anomaly detection** |
| **Module\<T\>** | Yes | **Yes** |
| **Conv1d\<T\>** | No | **Yes** |
| **Conv2d\<T\>** | Yes (conv mode) | No |
| **TrainingLoop\<T\>** | No (manual) | **No (manual)** |
| **ModelSerializer** | Yes | **Yes** |
| **Practical value** | Academic | **Real-world monitoring** |

## Nivara APIs demonstrated

| API | Purpose |
|-----|---------|
| `Conv1d<T>` | Temporal feature extraction from multivariate metrics |
| `BatchNorm1d<T>` | Stabilize conv feature maps (supports 3D `[B, C, L]` input) |
| `Linear<T>` | Decoder reconstruction, mu/logVar heads |
| `Dropout<T>` | Regularization in decoder |
| `Activation.LeakyRelu<T>` | Non-linearity throughout |
| `MSELoss<T>` | Reconstruction loss with `reduceToMean` |
| `ReverseGradOperations.KlDivergence<T>` | KL divergence for VAE regularization |
| `ReverseGradOperations.SampleNormal<T>` | Reparameterization trick |
| `Adam<T>` | Optimizer with adaptive learning rates |
| `GradientUtils.Grad()` | Enables reverse-mode autograd scope |
| `GradientUtils.ClipGradNorm<T>` | Global gradient norm clipping |
| `ModelSerializer.Save/Load` | JSON model persistence |
| `NivaraColumn<float>` | Metric channel storage |
| `NivaraFrame` | Windowed dataset |
| `TensorPrimitives` | SIMD-accelerated normalization and MSE scoring |

## Files

```
samples/NivaraTimeSeries/
├── NivaraTimeSeries.csproj  # Console app referencing Nivara core
├── Program.cs               # CLI, training loop, detection, ASCII viz
├── MetricsGenerator.cs      # Synthetic server metrics with anomaly injection
├── TimeSeriesModel.cs       # Conv1d encoder + Linear decoder VAE
└── README.md                # This file
```

## Library gaps this example exposed and resolved

| Gap | Problem | Resolution |
|-----|---------|------------|
| **BatchNorm1d rejects 3D input** | `BatchNorm1d.Forward` throws when `input.Rank != 2`, breaking the Conv1d -> BatchNorm1d pipeline that produces 3D `[B, C, L]` output. | Accept both 2D `[N, C]` and 3D `[B, C, L]` input, extracting `planeSize` from the third dimension. The kernel already supported it via the `planeSize` parameter. File: `src/Nivara/AutoDiff/Nn/BatchNorm.cs`. |
| **BatchNormKernel xHat allocation** | `xHat` was only allocated when `affine=true`, but the SIMD path (`planeSize >= 4`) always writes to `xHat`. Latent crash exposed by 3D input enabling the SIMD path. | Always allocate `xHat` regardless of `affine`. File: `src/Nivara/AutoDiff/Nn/BatchNormKernel.cs`. |
| **BatchNormKernel scalar path xHat not written** | The scalar fallback path (`planeSize < 4` or `T != float`) guarded the xHat write with `if (affine)`, so xHat stayed all-zeros when `affine=false`. `BackwardInput` reads xHat unconditionally, producing silently wrong gradients. | Remove the guard so xHat is always populated with normalized values. File: `src/Nivara/AutoDiff/Nn/BatchNormKernel.cs`. |
| **MSELoss returns sum, not mean** | `MSELoss<T>.Forward` always returns the sum of squared errors, which scales with input shape. No way to get mean MSE like PyTorch's default. | Added `Forward(predictions, targets, bool reduceToMean)` overload matching the `BCEWithLogitsLoss` pattern. File: `src/Nivara/AutoDiff/Nn/Functional/MSELoss.cs`. |

## Requirements

- .NET 10.0 SDK
- Nivara core library (`src/Nivara/Nivara.csproj`)
- No external dependencies
