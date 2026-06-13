# VAE Foundation — Plan & Progress

## Context: TENSORS.md positioning

Nivara is positioned as a **typed, immutable, null-aware DataFrame/query layer for .NET** — not as an AutoDiff framework. The AutoDiff subsystem exists as an internal/experimental component. VAE work extends AutoDiff with two small, well-scoped operations while **avoiding frame-level tensor wrappers** (Dot, CosineSimilarity, etc. remain deprecated candidates).

## Design decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Placement | `src/Nivara/AutoDiff/` (core) | AutoDiff already lives here; moving it is deferred work |
| Encoder API | `VAE<T>` class with `Encode()` → `(Mu, LogVar)` tuple | C# value tuples are idiomatic; PyTorch-aligned pattern; does not fight `Module<T>.Forward` single-output contract |
| Operation scope | Fused KlDivergence + SampleNormal | Single OpNode each for efficiency; backward closures capture needed intermediates |
| Training integration | Existing `TrainingLoop<T>` + optimizer | No new training infrastructure needed |
| Loss type | `ElboLossType` enum (`KldBeta`, `KldAnnealing`) | Type-safe, discoverable, no magic strings |
| Beta param | `Parameter<float>` on VAE instance | Consistent with Module<T> serialization/optimizer contract; `requiresGrad: false` for fixed β, `true` for learnable β-VAE |
| Default activation | `Activation.Relu` (configurable via constructor) | Matches PyTorch VAE conventions, MSE-compatible |

## Architecture

```text
VAE<T> : Module<T>
├── _encoder: Sequential<T>        (shared: Linear → ReLU → Linear → ReLU)
├── _muHead: Linear<T>             (projects hidden → latentDim, registered as sub-module)
├── _logVarHead: Linear<T>         (projects hidden → latentDim, registered as sub-module)
├── _decoder: Sequential<T>        (Linear → ReLU → Linear, no final activation)
├── _beta: Parameter<T>            (scalar, requiresGrad: false; registered as parameter)
│
├── Forward(x) → recon             (full pipeline: encode → reparam → decode)
├── Encode(x) → (Mu, LogVar)       (shared encoder → muHead → logVarHead)
├── Reparameterize(mu, logVar) → z (static; eval mode returns mu directly)
├── Decode(z) → recon              (decoder forward)
└── ElboLoss(recon, x, mu, logVar, lossType) → scalar (recon + β·KL)
```

## Known limitations

- **Null propagation through VAE**: `Linear.Forward` cannot mix tensor-backed parameters with nullable column inputs (storage type mismatch). Null inputs to `VAE.Encode()` or `VAE.Forward()` will fail. The individual VAE operations (`KlDivergence`, `SampleNormal`) handle nulls correctly — see their tests in `GradOperationsTests`.
- **VAE training loop**: Standard `TrainingLoop<T>` expects a 2-arg loss `(predictions, targets) → scalar`. VAE's `ElboLoss` needs 4 args. VAE training uses manual loops (demonstrated in `VAE_Training_ReducesLoss`).

### Future (deferred)

| Feature | Reason deferred |
|---------|----------------|
| Conditional VAE | Add condition tensor to `Encode(x, condition)` |
| Conv encoder/decoder | No Conv layers exist yet in Nn |
| BatchNorm | Separate effort, not VAE-specific |
| Moving AutoDiff to Extensions | Strategic decision, not tactical |
