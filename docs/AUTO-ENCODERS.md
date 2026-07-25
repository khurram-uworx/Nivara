# AutoDiff NN Layers — Plan & Progress

## Context

Nivara is positioned as a **typed, immutable, null-aware DataFrame/query layer for .NET** — not as an AutoDiff framework. The AutoDiff subsystem exists as an internal component in `src/Nivara/AutoDiff/`. VAE work extends AutoDiff with well-scoped operations while **avoiding frame-level tensor wrappers** (Dot, CosineSimilarity, etc. remain deprecated candidates).

**Key architectural decisions:**
- AutoDiff stays in core Nivara (not moving to Extensions)
- AutoDiff is a **non-nullable domain** per ADR-001 — null boundary enforced at `NivaraColumn<T>` → `ReverseGradTensor<T>` conversion; all AutoDiff ops assume non-null data

## Implemented Components

### BatchNorm1d / BatchNorm2d

**Files:** `BatchNorm.cs`, `BatchNormKernel.cs`

Fused span-based kernel with `TensorPrimitives` — single `OpNode` per call. No expanded tensors, no intermediate autograd nodes.

```
BatchNormKernel<T>
├── Forward(input, n, c, hw, gamma, beta, eps, affine) → (Output, XHat, InvStd, Mean)
├── BackwardInput(gradOut, xHat, gamma, invStd, n, c, hw) → gradInput
├── BackwardWeight(gradOut, xHat, n, c, hw) → gradGamma
└── BackwardBias(gradOut, n, c, hw) → gradBeta
```

- Train mode: computes batch statistics, updates running stats via direct span arithmetic
- Eval mode: uses cached running mean/var
- StateDict/LoadStateDict includes `running_mean`, `running_var`, `num_batches_tracked`
- **14 tests** (7 per variant): forward shape, train stats update, eval running stats, affine=false, backward gradients, state dict round-trip, dispose

### Conv2d

**Files:** `Conv2d.cs`, `Im2Col.cs`

Tiled im2col → `TensorPrimitives.Dot` per output channel. Key optimizations:

- **PatchLocation lookup table**: `PatchLocation` struct precomputes `(Batch, OH, OW)` per-tile, eliminates 4 integer divisions per position from all hot loops
- **ConvForward1x1**: bypasses im2col entirely for 1×1 kernels (stride=1, padding=0). Gathers input channels into pooled buffer, TensorPrimitives.Dot per output channel
- **InputGrad specializations**: `InputGrad1x1` (direct MultiplyAdd), `InputGrad3x3` (bounds-checked 9-tap scatter), `InputGradGeneric` (nested loops)
- **Zero-copy via TryGetSpan**: eliminates tensor copy when storage is contiguous
- **Weight gradient**: tiled im2col → `TensorPrimitives.MultiplyAdd`
- **Bias gradient**: `TensorPrimitives.Sum` per channel
- **Grouped convolution**: `groups` parameter splits input/output channels into independent groups. For `groups=1` (common path), zero overhead — full buffers passed directly. For `groups>1`, gather/scatter per group with NCHW layout handled correctly.
- All kernel methods accept `Span<T>`/`ReadOnlySpan<T>` (not `T[]`) for composability with grouped slicing

```
Forward:   Im2ColTile → Dot per output channel (groups=1: direct, groups>1: gather/scatter)
InputGrad: InputGrad1x1 | InputGrad3x3 | InputGradGeneric
WeightGrad: Im2ColTile → MultiplyAdd per output channel
BiasGrad:   Sum per channel
```

- **18 tests**: forward shapes (basic, padding, stride), bias, backward gradients, no-bias backward, 1×1=Linear equivalence, backward with stride+padding, multi-batch backward, large-channel forward, grouped forward shape, grouped multi-batch, grouped identity-matches-ungrouped, grouped backward, grouped invalid-groups-throws, grouped deep, dispose

### ConvTranspose2d

**Files:** Same as Conv2d

Direct scatter kernel (not im2col-based) — forward uses `Col2ImForward`, backward uses `ConvTransposeInputGradKernel` and `ConvTransposeWeightGradKernel`.

```
Forward:     Col2ImForward (scatter) + bias
InputGrad:   ConvTransposeInputGradKernel (scatter with stride check)
WeightGrad:  ConvTransposeWeightGradKernel (reduction over ih, iw)
```

- **8 tests**: forward shapes (basic, padding, stride), backward gradients, backward with stride, bias, large-channel forward, dispose

### Conditional VAE

**Files:** `VAE.cs`

Extended `VAE<T>` with `conditionDim` parameter. Encoder/decoder accept optional condition tensor via `Concat`.

```
VAE<T> : Module<T>
├── Forward(x) → recon
├── Forward(x, condition) → recon
├── Encode(x) → (Mu, LogVar)
├── Encode(x, condition) → (Mu, LogVar)
├── Reparameterize(mu, logVar) → z
├── Decode(z) → recon
├── Decode(z, condition) → recon
└── ElboLoss(recon, x, mu, logVar, lossType) → scalar
```

- **7 tests**: encode shape, null condition handling, forward end-to-end, elbo loss, decode with condition, backward gradients, invalid ctor throws

### ConvVAE

**File:** `ConvVAE.cs`

Fully convolutional VAE with 1×1 Conv2d heads for spatial latent representations. Uses `Conv2d` encoder (stride downsampling), `ConvTranspose2d` decoder (stride upsampling), and `Conv2d(1×1)` for mu/logvar projection.

```
ConvVAE<T> : Module<T>
├── Forward(x) → recon
├── Encode(x) → (Mu, LogVar)       [both spatial, e.g. B×C'×H'×W']
├── Reparameterize(mu, logVar) → z  [spatial reparameterization trick]
├── Decode(z) → recon               [ConvTranspose stack]
└── ElboLoss(recon, x, mu, logVar) → scalar  [MSE + KL divergence]
```

- Configurable encoder channel list, latent channels, kernel/stride/padding
- 1×1 Conv heads preserve spatial structure in latent space
- **8 tests**: forward shape, encode shape, decode round-trip, elbo loss, backward gradient flow, end-to-end loss reduction, invalid args, RGB forward

### DepthwiseSeparableConv2d

**File:** `DepthwiseSeparableConv2d.cs`

Efficient depthwise separable convolution (MobileNet-style): depthwise conv (`groups=inChannels`) + pointwise 1×1 conv. Reuses existing `Conv2d` grouped kernel and `ConvForward1x1`.

```
DepthwiseSeparableConv2d<T> : Module<T>
└── Forward(input) → Conv2d(groups=inChannels) → ReLU → Conv2d(1×1)
```

- Configurable inChannels, outChannels, kernelSize, stride, padding, useBias
- All kernels use existing TensorPrimitives-backed Conv2d paths
- **5 tests**: forward shape, stride, backward gradients, no-bias, equivalence with manual composition

### LayerNorm

**Files:** `LayerNorm.cs`, `LayerNormKernel.cs`

Span-based kernel with `TensorPrimitives`. Normalizes over the last dimension per instance (no running stats, unlike BatchNorm).

```
LayerNormKernel<T>
├── Forward(input, rows, normalizedShape, gamma, beta, eps, affine) → (Output, Mean, InvStd, XHat)
├── BackwardInput(gradOut, xHat, gamma, invStd, rows, normalizedShape, affine) → gradInput
├── BackwardWeight(gradOut, xHat, rows, normalizedShape) → gradGamma
└── BackwardBias(gradOut, rows, normalizedShape) → gradBeta
```

- **Tests**: 6 tests — 2D forward, 4D forward, backward gradients, affine=false, normalized output, dispose

### Dropout

**File:** `Dropout.cs`

Delegates to `ReverseGradOperations.Dropout`. Train mode: zeros out elements with probability `p` and scales by `1/(1-p)`. Eval mode: identity.

### TransformerBlock

**File:** `TransformerBlock.cs`

Full pre-norm transformer block with causal masking:
- Multi-head self-attention (Q/K/V projections, scaled dot-product, output projection)
- RMSNorm (fused per-row TensorPrimitives kernel)
- GELU FFN (fc1 → GELU → fc2)
- Residual connections with optional dropout

### MultiheadAttention

**File:** `MultiheadAttention.cs`

Standalone reusable attention module extracted from TransformerBlock patterns:
- Q/K/V/O linear projections (no bias)
- Scaled dot-product attention with configurable `numHeads` and `headDim = embedDim / numHeads`
- Supports **self-attention** (single input) and **cross-attention** (separate Q, K, V tensors)
- Optional causal masking and dropout
- Works with any sequence length — scale tensor matches (qLen × kvLen) for cross-attention

```
MultiheadAttention<T> : Module<T>
├── Forward(input) → output           [self-attention]
├── Forward(query, key, value) → output  [cross-attention]
└── Forward(query, key, value, causal) → output  [cross-attention with mask override]
```

- **5 tests**: self-attention shape, causal shape, backward gradient flow, cross-attention shape, invalid embedDim throws

### Broadcast Operations

**File:** `ReverseGradOperations.cs`

- `BroadcastMultiply<T>(input, scale)` — channel-wise multiply with 1D scale tensor
- `BroadcastAdd<T>(input, bias)` — channel-wise add with 1D bias tensor
- Used by BatchNorm backward for gamma/beta application
- **11 tests** (as of commit a8907da): 2D/4D forward, backward for input/scale/bias, both-grad, mismatch throws

### Other Pre-existing Components

- `Linear<T>` — fully connected layer
- `Embedding<T>`, `SparseEmbedding<T>` — lookup tables
- `Sequential<T>` — module container
- `Activation` — Relu, Sigmoid, Tanh
- `SGD<T>`, `Adam<T>`, `AdamW<T>` — optimizers
- `TrainingLoop<T>`, `DataParallelTrainer<T>` — training infrastructure
- `Module<T>.StateDict()` / `LoadStateDict()` — serialization (virtual)
- Loss functions: `MSELoss`, `BCELoss`, `BCEWithLogitsLoss`, `CrossEntropyLoss`, `L1Loss`
- `TextClassifierModel<T>`, `TokenClassifierModel<T>` — sample models

## Test Coverage Summary

| Component | Tests | Coverage Notes |
|-----------|-------|----------------|
| BatchNorm1d | 7 | Forward, train/eval modes, affine, backward, state dict, dispose |
| BatchNorm2d | 7 | Same pattern as 1d |
| Conv2d | 18 | Shapes, padding, stride, bias, backward, backward with stride+padding, multi-batch, 1×1, large channels, grouped (6 tests), dispose |
| ConvTranspose2d | 8 | Shapes, padding, stride, backward, backward with stride, bias, large channels, dispose |
| Conditional VAE | 7 | Encode, decode, forward, elbo, backward, null condition |
| ConvVAE | 8 | Forward, encode, decode, elbo, backward, end-to-end loss reduction, invalid args, RGB |
| LayerNorm | 6 | 2D/4D forward, backward, affine=false, normalized output, dispose |
| MultiheadAttention | 5 | Self-attention, causal, backward, cross-attention, validation |
| DepthwiseSeparableConv2d | 5 | Forward, stride, backward, no-basis, manual equivalence |
| BroadcastMultiply | 6 | 2D/4D forward, input backward, scale backward, both-grad, mismatch throws |
| BroadcastAdd | 5 | 2D/4D forward, input backward, bias backward, mismatch throws |
| **Total (NN effort)** | **82** | |
| **Full suite** | **1922** | All passing |

## Known Limitations

- **Null propagation through VAE**: `Linear.Forward` cannot mix tensor-backed parameters with nullable column inputs (storage type mismatch). Null inputs to `VAE.Encode()` or `VAE.Forward()` will fail. Per ADR-001, AutoDiff is non-nullable; this is a storage-layer boundary issue.
- **VAE training loop**: Standard `TrainingLoop<T>` expects 2-arg loss. VAE's `ElboLoss` needs 4 args. VAE training uses manual loops (demonstrated in `VAE_Training_ReducesLoss`).
- **ConvTranspose2d**: No grouped convolution support yet (Conv2d has it).
- **ConvInputGrad1x1**: No bounds checking (safe when output spatial ≤ input spatial, which holds for stride=1 padding=0). If non-standard padding is needed, falls back to generic path.
