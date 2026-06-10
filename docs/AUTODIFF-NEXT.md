# AutoDiff — Next Features (Discussion Draft)

Current state: working autograd engine (`ReverseGradTensor<T>`, computation graph, basic ops, SGD, gradient utils, Nivara frame interop). ~14 files, ~3000 lines, 8 test files.

Goal: evolve from a differentiable-tensor library into something a C# engineer can actually build and train ML models with — cherry-picking the best from PyTorch, NumPy, and Polars.

---

## Priority Stack

### P1 — `nn` Module System (Foundation)

> PyTorch `nn.Module` / `nn.Linear` — column-backed, parameter-registering.

```
src/Nivara/AutoDiff/Nn/
├── Parameter.cs              — wraps ReverseGradTensor<T>, auto-registers requiresGrad=true
├── Module.cs                 — abstract base: Forward(), Parameters(), NamedModules(), Train()/Eval(), ToDevice()
├── Linear.cs                 — y = x @ Wᵀ + b (column-backed, shape-aware, bias optional)
├── Sequential.cs             — ordered module chain, Forward() pipes through each
├── Dropout.cs                — (optional) inverted dropout with train/eval toggle
├── BatchNorm.cs              — (optional) per-channel normalization with running stats
├── Embedding.cs              — (optional) lookup table for categorical features
├── Activation.cs             — ReLU, Sigmoid, Tanh, LeakyReLU wrappers
└── Initializers/
    ├── KaimingUniform.cs     — a = √(6/fan_in)
    ├── KaimingNormal.cs      — a = √(2/fan_in)
    ├── XavierUniform.cs      — a = √(6/(fan_in+fan_out))
    ├── XavierNormal.cs       — a = √(2/(fan_in+fan_out))
    ├── Uniform.cs            — U(-bound, bound)
    └── Normal.cs             — N(mean, std)
```

**Key design decisions:**
- `Parameter<T>` is just a `ReverseGradTensor<T>` with a name and `requiresGrad=true` — no separate storage
- `Module<T>.Parameters()` returns a flat `Dictionary<string, ReverseGradTensor<T>>` for optimizers
- `Module<T>.NamedModules()` returns child modules for recursive parameter collection
- Shape is explicit on `Linear` — no shape inference yet (defer broadcasting)
- Weight initialization is optional (caller can skip, defaults to uniform)

**Example usage:**
```csharp
class MLP : Module<float>
{
    private readonly Linear<float> L1, L2;

    public MLP(int inputDim, int hiddenDim, int outputDim)
    {
        L1 = new Linear<float>(inputDim, hiddenDim);
        L2 = new Linear<float>(hiddenDim, outputDim);
        RegisterModules(L1, L2);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> x)
    {
        var h = GradOperations.Relu(L1.Forward(x));
        return L2.Forward(h);
    }
}

var model = new MLP(784, 256, 10);
KaimingUniform.Init(model.Parameters());  // optional init
```

**Estimated effort:** ~4h (Parameter, Module, Linear, Sequential, Activation wrappers, basic init)

---

### P2 — Loss Functions

> PyTorch `nn.functional.mse_loss`, `cross_entropy`, `binary_cross_entropy_with_logits`

```
src/Nivara/AutoDiff/Nn/
└── Functional/
    ├── MSELoss.cs              — ½(y - ŷ)²
    ├── L1Loss.cs               — |y - ŷ|
    ├── BCELoss.cs              — -(y·log(ŷ) + (1-y)·log(1-ŷ))
    ├── BCEWithLogitsLoss.cs    — fused sigmoid + BCE (numerically stable)
    ├── CrossEntropyLoss.cs     — fused softmax + NLL (numerically stable)
    ├── Softmax.cs              — dim-aware softmax (for CrossEntropy pre-processing)
    └── LogSoftmax.cs           — log-softmax
```

**Null semantics:** all losses propagate null masks. If a target is null, that position contributes zero to the loss (like `reduction='sum'` skipping nan in PyTorch).

**Estimated effort:** ~2h

---

### P3 — Stateful Optimizers

> PyTorch `optim.Adam` / `optim.AdamW` (with weight decay, bias correction)

```
src/Nivara/AutoDiff/Optimizer/
├── Optimizer.cs               — abstract base: Step(), ZeroGrad(), AddParameterGroup()
├── SGD.cs                     — current SgdOptimizer adapted into the pattern
├── Adam.cs                    — bias-corrected, β₁/β₂ defaults 0.9/0.999, ε=1e-8
└── AdamW.cs                   — decoupled weight decay (Loshchilov & Hutter 2019)
```

**Key design decisions:**
- `Optimizer` owns state (momentum buffers, adaptive learning rates) — `Optimizer.Step()` mutates in-place
- `AddParameterGroup(parameters, lr, weightDecay)` for per-layer hyperparams
- Null-skip semantics: null positions don't accumulate moment buffers (no update at those positions)
- State buffers use `ArrayPool<T>.Shared` and are released on `Dispose()`

**Estimated effort:** ~3h (base class + Adam + AdamW)

---

### P4 — Data Loading & Mini-batch Training

> PyTorch `DataLoader` + `Dataset` — mini-batch iteration over `NivaraFrame` rows

```
src/Nivara/AutoDiff/Training/
├── TensorDataset.cs           — wraps NivaraFrame, exposes feature/label tensor slices
├── DataLoader.cs              — batch iteration, shuffling, configurable batch size
├── Batch.cs                   — { Features: ReverseGradTensor<T>, Labels: ReverseGradTensor<T> }
└── TrainingLoop.cs            — epoch iteration, loss logging, model checkpoint hooks
```

**How it works:**
- `TensorDataset` stores a `NivaraFrame` and column name mappings
- `DataLoader` creates row-index batches, slices columns, returns `Batch<T>` instances
- Batching reuses `NivaraColumn<T>.Slice()` — zero-copy read-only views
- `TrainingLoop` wraps the common pattern: `for epoch → for batch → forward → backward → optimizer.step`

**Integration with P1:** `TrainingLoop` accepts a `Module<T>` and `Optimizer`, calling `module.Forward(batch.Features)` and `optimizer.Step()`.

**Estimated effort:** ~3h (Dataset + DataLoader + TrainingLoop)

---

### P5 — Serialization

> PyTorch `torch.save(model.state_dict(), path)`

```
src/Nivara/AutoDiff/Serialization/
├── ModelSerializer.cs         — Save/Load parameter state dicts (JSON + binary)
└── Checkpoint.cs              — Epoch + loss + optimizer state + model params
```

**Format:** JSON for metadata (names, shapes, dtype) + base64-encoded binary for values. Each `ReverseGradTensor<T>` serializes its data array and null mask.

**Estimated effort:** ~1h

---

### P6 — Data Parallel Training

> Already planned in `AUTODIFF-PLAN.md` Phase 2 — `DataParallelTrainer`

Reuses `ParallelExecutionHelper` from the execution engine. Depends on P4 (batching) being available. Moves from planned to implemented when P1–P4 are stable.

**Estimated effort:** ~4h

---

## Phasing & Dependencies

```
P1 (Module System) ──→ P2 (Losses) ──→ P3 (Optimizers) ──→ P4 (DataLoader)
                                                              │
                                                              └──→ P5 (Serialization)
                                                              │
                                                              └──→ P6 (Data Parallel)
```

Each phase feeds the next. P1 is the hardest dependency — without it, losses and optimizers have nothing to operate on.

---

## Open Questions for Discussion

1. **Namespace layout** — Does P1 go into `Nivara.AutoDiff.Nn`, or should the module system live at `Nivara.Nn` to signal it's a first-class API (like PyTorch's `torch.nn`)?

2. **Move to core vs stay in Extensions** — The existing `AUTODIFF-PLAN.md` calls for moving AutoDiff into the core `Nivara` library. Does the module system accelerate that decision, or should it land in Extensions first and move later?

3. **Shape strategy** — `Linear` needs explicit input/output dims. Should we keep it explicit forever, or add an inferred `Linear(inFeatures)` that lazily builds weights on first forward pass?

4. **Null in training labels** — When a label is null in a supervised batch, should that sample be:
   - Skipped entirely (remove row from batch)
   - Included but contribute zero loss (current Sum/Mean skip nulls)
   - Throw (strict mode)

5. **Module.Train() / Eval()** — Needed for Dropout/BatchNorm. Worth implementing the toggle now even if those layers are deferred?

6. **CPU-only forever?** — If we add a `ToDevice()` concept now (even just CPU), it shapes the API surface. Or stay flat and add device later.

7. **Test strategy per phase** — Should each phase ship with its own test file (mirroring the `tests/Nivara.Tests/AutoDiff/` pattern), or batch tests into larger files by area (e.g., `NnTests.cs`, `OptimizerTests.cs`)?
