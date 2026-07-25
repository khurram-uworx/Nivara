# AutoDiff NN Layers — Plan & Progress

## Context

Nivara is positioned as a **typed, immutable, null-aware DataFrame/query layer for .NET** — not as an AutoDiff framework. The AutoDiff subsystem exists as an internal component in `src/Nivara/AutoDiff/`. VAE work extends AutoDiff with well-scoped operations while **avoiding frame-level tensor wrappers** (Dot, CosineSimilarity, etc. remain deprecated candidates).

**Key architectural decisions:**
- AutoDiff stays in core Nivara (not moving to Extensions)
- AutoDiff is a **non-nullable domain** per ADR-001 — null boundary enforced at `NivaraColumn<T>` → `ReverseGradTensor<T>` conversion; all AutoDiff ops assume non-null data

## Implemented Components

### BatchNorm1d / BatchNorm2d

**Files:** `BatchNorm.cs`, `BatchNormKernel.cs`

Fused span-based kernel with `TensorPrimitives` — single `OpNode` per call (like `PerRowRMSNorm` pattern). No expanded tensors, no intermediate autograd nodes.

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
- **7 tests per variant** (1d/2d): forward shape, train stats update, eval running stats, affine=false, backward gradients, state dict round-trip, dispose

### Conv2d

**Files:** `Conv2d.cs`, `Im2Col.cs`

Tiled im2col → `TensorPrimitives.Dot` per output channel. Key optimizations:

- **PatchLocation lookup table**: `PatchLocation` struct precomputes `(Batch, OH, OW)` per-tile, eliminates 4 integer divisions per position from all hot loops
- **ConvForward1x1**: bypasses im2col entirely for 1×1 kernels (stride=1, padding=0). Gathers input channels into pooled buffer, TensorPrimitives.Dot per output channel
- **InputGrad specializations**: `InputGrad1x1` (direct MultiplyAdd), `InputGrad3x3` (unrolled 9-tap scatter), `InputGradGeneric` (nested loops)
- **Zero-copy via TryGetSpan**: eliminates tensor copy when storage is contiguous
- **Weight gradient**: tiled im2col → `TensorPrimitives.MultiplyAdd`
- **Bias gradient**: `TensorPrimitives.Sum` per channel

```
Forward:   Im2ColTile → Dot per output channel
InputGrad: InputGrad1x1 | InputGrad3x3 | InputGradGeneric
WeightGrad: Im2ColTile → MultiplyAdd per output channel
BiasGrad:   Sum per channel
```

- **8 tests**: forward shapes (basic, padding, stride), bias, backward gradients, no-bias backward, 1×1=Linear equivalence, dispose

### ConvTranspose2d

**Files:** Same as Conv2d

Direct scatter kernel (not im2col-based) — forward uses `Col2ImForward`, backward uses `ConvTransposeInputGradKernel` and `ConvTransposeWeightGradKernel`.

```
Forward:     Col2ImForward (scatter) + bias
InputGrad:   ConvTransposeInputGradKernel (scatter with stride check)
WeightGrad:  ConvTransposeWeightGradKernel (reduction over ih, iw)
```

- **6 tests**: forward shapes (basic, padding, stride), backward gradients, bias, dispose

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

### Broadcast Operations

**File:** `ReverseGradOperations.cs`

- `BroadcastMultiply<T>(input, scale)` — channel-wise multiply with 1D scale tensor
- `BroadcastAdd<T>(input, bias)` — channel-wise add with 1D bias tensor
- Used by BatchNorm backward for gamma/beta application

### Other Prerequisites (pre-existing)

- `Linear<T>` — fully connected layer
- `Sequential<T>` — module container
- `Activation` — Relu, Sigmoid, Tanh
- `SGD<T>`, `Adam<T>`, `AdamW<T>` — optimizers
- `TrainingLoop<T>` — standard training loop
- `Module<T>.StateDict()` / `LoadStateDict()` — serialization (made virtual)

## Test Coverage Summary

| Component | Tests | Coverage Notes |
|-----------|-------|----------------|
| BatchNorm1d | 7 | Forward, train/eval modes, affine, backward, state dict, dispose |
| BatchNorm2d | 7 | Same pattern as 1d |
| Conv2d | 8 | Shapes, padding, stride, bias, backward, 1×1, dispose |
| ConvTranspose2d | 6 | Shapes, padding, stride, backward, bias, dispose |
| Conditional VAE | 7 | Encode, decode, forward, elbo, backward, null condition |
| BroadcastMultiply | 0 | No dedicated tests (exercised via BatchNorm backward) |
| BroadcastAdd | 0 | No dedicated tests (exercised via BatchNorm backward) |
| **Total new tests** | **35** | |

## Known Limitations

- **Null propagation through VAE**: `Linear.Forward` cannot mix tensor-backed parameters with nullable column inputs (storage type mismatch). Null inputs to `VAE.Encode()` or `VAE.Forward()` will fail. Per ADR-001, AutoDiff is non-nullable; this is a storage-layer boundary issue.
- **VAE training loop**: Standard `TrainingLoop<T>` expects 2-arg loss. VAE's `ElboLoss` needs 4 args. VAE training uses manual loops (demonstrated in `VAE_Training_ReducesLoss`).

## Deferred Features

| Feature | Reason |
|---------|--------|
| Conv encoder/decoder for VAE | Conv layers exist now; could build ConvVAE as a composition |
| Grouped/depthwise convolution | Requires `groups` parameter on Conv2d |
| Dropout | Simple to add, no architectural questions |
| LayerNorm | Alternative to BatchNorm; straightforward implementation |
| Residual connections | Already possible via `Sequential` + manual add; could add `Residual<T>` wrapper |
| Multi-head attention / Transformer | Larger effort; would need `MultiheadAttention<T>` module |
| BroadcastMultiply/BroadcastAdd dedicated tests | Currently only exercised indirectly via BatchNorm backward |
| Conv backward tests with stride/padding | Current backward tests only cover basic case |
