# AutoDiff — Next Features (Discussion Draft)

Current state: working autograd engine (`ReverseGradTensor<T>`, computation graph, basic ops, SGD, gradient utils, Nivara frame interop). ~14 files, ~3000 lines, 8 test files.

> **Prerequisite completed:** Dead-code cleanup (Phase 0) done — `IAutoGradNumeric.cs`, `AutoGradNumericTypes.cs`, `IGradOperation.cs`, and the source-tree `README.md` removed. AutoDiff files live in core (`src/Nivara/AutoDiff/`) with namespaces `Nivara.AutoDiff.*`.

Goal: evolve from a differentiable-tensor library into something a C# engineer can actually build and train ML models with — cherry-picking the best from PyTorch, NumPy, and Polars.

---

## Priority Stack

### P0 — Performance Foundation (prerequisite for P1, P3)

> Make the hot loops fast before building abstractions on top of them.

Without this phase, `Linear.Forward` runs `MatMul` as a scalar triple-loop and optimizers iterate element-by-element. P1 and P3 would be unusably slow on real data. This phase touches no public APIs — it's purely internal rewrites that P1/P3 then call into.

**Where time currently goes (single 128×256×10 matmul forward):**
```
┌─────────────────────┬───────────┬──────────────────────────┐
│ Phase               │ % of step │ Bottleneck today          │
├─────────────────────┼───────────┼──────────────────────────┤
│ MatMul forward      │  ~40%     │ ❌ scalar triple for-loop │
│ Activations         │  ~10%     │ ✅ partial SIMD           │
│ MatMul backward     │  ~30%     │ ❌ scalar triple for-loop │
│ Optimizer.Step      │  ~10%     │ ❌ scalar element loop    │
│ Gradient clip/norm  │  ~5%      │ ❌ scalar for-loop        │
│ Graph traversal     │  ~5%      │ ❌ sequential (ok for now)│
└─────────────────────┴───────────┴──────────────────────────┘
```
**~90% of training time is currently single-threaded scalar code.** P0 fixes the top four rows using SIMD kernels from `System.Numerics.Tensors` — zero new dependencies, zero API changes.

```
src/Nivara/Tensors/
└── MatMulHelper.cs         — central MatMul (TensorPrimitives.Dot + Parallel.For)
src/Nivara/AutoDiff/
├── Operations/
│   └── GradOperations.cs   — MatMul → MatMulHelper.Multiply
└── Optimizer/
    ├── SgdOptimizer.cs     — TensorPrimitives.Multiply + Subtract
    ├── Adam.cs             — Step() uses TensorPrimitives throughout
    └── AdamW.cs            — same
```

**Work items:**

**0a — SIMD MatMul (~45m)**
Replace the triple `for` loop in `MatMulVectorized<T>()` with the new central `MatMulHelper.Multiply` in `src/Nivara/Tensors/MatMulHelper.cs`. This is the *single file that changes* when `Tensor.MatrixMultiply` ships in a future .NET version.

- **Algorithm (no known Tensor.MatrixMultiply in .NET 10):** Transpose the right matrix into a rented buffer, then `Parallel.For` over output rows with `TensorPrimitives.Dot` per row×column pair. This gives SIMD-accelerated dot products (AVX2, AVX-512, NEON) from the runtime team's `TensorPrimitives` implementation, plus row-level parallelism.
- **No-null fast path (`MatMulHelper.Multiply(ReadOnlySpan<T>, ...)`):** zero-copy via `TensorStorage.GetFlattenedSpan()` when both inputs use `TensorStorage`; otherwise copy to rented `ArrayPool` buffers.
- **Null path:** fill nulls with `T.Zero`, run the dense SIMD kernel on filled arrays, then compute the result null mask via boolean OR propagation (a position is null if any contributing element was null).
- **Future swap:** when `Tensor.MatrixMultiply` ships, only the body of `MatMulHelper.Multiply<T>(Tensor<T>, ...)` changes — every caller (GradOperations, backward gradients, etc.) uses the same API.

**0b — SIMD optimizer kernel (~45m)**
- `SgdUpdate`: replace element `for` loop with `TensorPrimitives.Multiply` + `TensorPrimitives.Subtract` on the no-null path; `ArrayPool` + batched kernel on the null path.
- `Adam.Step()` / `AdamW.Step()`: all element-wise math (bias correction, weight decay, momentum blending) uses `TensorPrimitives` generic overloads. No scalar `for` loops remain for float/double.

**0c — SIMD gradient utilities (~30m)**
- `ClipGradValue` → `TensorPrimitives.Clamp` (available in .NET 10)
- `GetGradientNorm` → `TensorPrimitives.Norm`
- `ClipGradNorm` → `TensorPrimitives.Norm` + `TensorPrimitives.Multiply`

**Estimated effort:** ~2h

**Key design decisions:**
- One new file (`MatMulHelper.cs`) centralizes the MatMul algorithm. When `Tensor.MatrixMultiply` ships in a future .NET version, only this file changes — no callers are modified.
- The algorithm uses `TensorPrimitives.Dot` (SIMD dot product maintained by the runtime team) + `Parallel.For` over output rows. This gives SIMD-accelerated per-element math plus row-level parallelism, even without a full GEMM primitive.
- The null-aware path fills nulls with `T.Zero`, runs the dense SIMD kernel, then computes the result mask via boolean OR propagation (same semantics as existing code).
- `GradientUtils` methods remain sequential across parameters (each parameter's update is an independent SIMD call); cross-parameter parallelism is deferred to P6.

---

### P1 — `nn` Module System (Foundation)

> PyTorch `nn.Module` / `nn.Linear` — column-backed, parameter-registering. **Builds on P0 — `Linear.Forward` calls the SIMD `MatMul` from Phase 0a.**

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
        var h = GradOperations.Relu(L1.Forward(x));  // MatMul = SIMD from P0a
        return L2.Forward(h);                         // MatMul = SIMD from P0a
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

> PyTorch `optim.Adam` / `optim.AdamW` (with weight decay, bias correction). **Builds on P0b — `Step()` uses `TensorPrimitives` for all element-wise math.**

```
src/Nivara/AutoDiff/Optimizer/
├── Optimizer.cs               — abstract base: Step(), ZeroGrad(), AddParameterGroup()
├── SGD.cs                     — SgdOptimizer adapted into the pattern, SIMD via P0b
├── Adam.cs                    — bias-corrected, β₁/β₂ defaults 0.9/0.999, ε=1e-8
└── AdamW.cs                   — decoupled weight decay (Loshchilov & Hutter 2019)
```

**Key design decisions:**
- `Optimizer` owns state (momentum buffers, adaptive learning rates) — `Optimizer.Step()` mutates in-place
- `AddParameterGroup(parameters, lr, weightDecay)` for per-layer hyperparams
- Null-skip semantics: null positions don't accumulate moment buffers (no update at those positions)
- State buffers use `ArrayPool<T>.Shared` and are released on `Dispose()`
- All element-wise math in `Step()` (bias correction, momentum blend, weight decay) uses `TensorPrimitives` generic overloads — no scalar `for` loops for float/double

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

> Bulk-synchronous data parallelism: split data rows across cores, each core builds its own computation graph (shared params by reference), computes local gradients, then sums across chunks.

```
Split data rows into N chunks (N = maxDop × 2 for work-stealing)
  ↓
Parallel.ForEach(chunks):
  ├─ CreateRowSubset(data, chunk.Start, chunk.Length)
  ├─ Build local computation graph (shared params by ref)
  ├─ Forward → loss
  ├─ Backward() → local partial gradients
  └─ Return local gradient columns
  ↓
  (Each chunk's forward/backward pass uses TensorPrimitives for SIMD-vectorised
   arithmetic at the hardware's native width.)
  ↓
Sum partial gradients across chunks:
  grad_i = chunkGrads.Sum(g => g[i])   // NivaraColumn<T>.Add
  ↓
Optimizer.Step(params, lr)              // P3's Adam/AdamW or SGD
  ↓
ZeroGrad(params)
  ↓
Record epoch diagnostics (timing, loss, gradient norm)
```

**Key design choices:**
- **Parameters shared by reference**: `ReverseGradTensor<T>` wraps immutable `NivaraColumn<T>`. Concurrent reads from all chunks are safe.
- **Local gradient accumulation**: Each chunk produces its own gradient columns. No shared mutable state during parallel section. Sum after `Parallel.ForEach`.
- **Reuses `ParallelExecutionHelper`**: Chunk sizing (`CalculateOptimalChunkSize`), DOP capping (`GetRecommendedParallelism`), and row subset slicing (`CreateRowSubset`).
- **Diagnostics integration**: Per-epoch timing, per-chunk timing, strategy info. Pattern matches `ExecutionDiagnostics.GenerateReport()`.
- **Two-level parallelism**: Data is split across cores (coarse-grained, row-level) AND each core's chunk computation is SIMD-vectorised via `TensorPrimitives` (fine-grained, element-level).

**Result types:**
```csharp
public sealed class TrainingResult
{
    public IReadOnlyList<EpochResult> Epochs { get; }
    public Dictionary<string, ReverseGradTensor<T>> TrainedParameters { get; }
    public void PrintSummary();  // console-friendly report
}

public sealed class EpochResult
{
    public int Epoch { get; }
    public double Loss { get; }
    public TimeSpan Elapsed { get; }
    public int Workers { get; }
    public int Chunks { get; }
    public double GradientNorm { get; }
}
```

**Where files go:**
```
src/Nivara/AutoDiff/Training/
├── DataParallelTrainer.cs     — orchestrates chunk-split + parallel compute + gradient sum
└── TrainingResult.cs          — TrainingResult, EpochResult
```

Depends on P4 (batching) for row-subset slicing and P3 (optimizer) for the update step.

**Estimated effort:** ~4h (DataParallelTrainer + TrainingResult)

---

## Phasing & Dependencies

```
P0 (Performance) ──→ P1 (Module System) ──→ P2 (Losses) ──→ P3 (Optimizers) ──→ P4 (DataLoader)
                       │                                                            │
                       │                                                            └──→ P5 (Serialization)
                       │                                                            │
                       └── (Linear.Forward needs P0a MatMul)                        └──→ P6 (Data Parallel)
                                                                                         (needs P4 batching)
```

**P0 must come first** — without SIMD MatMul and SIMD optimizer kernels, P1's `Linear.Forward` and P3's `Adam.Step()` would be scalar loops that make real-world training impractically slow. P0 changes no public APIs and produces no visible features; it's the engine tune-up before the body shop.

**P3 depends on P0b** (TensorPrimitives optimizer math) for acceptable performance at scale. The interface and ergonomics are defined in P3; the fast kernel is delivered by P0.

---

## Open Questions for Discussion

1. **Namespace layout** — Does P1 go into `Nivara.AutoDiff.Nn`, or should the module system live at `Nivara.Nn` to signal it's a first-class API (like PyTorch's `torch.nn`)?

2. **Move to core vs stay in Extensions** — ✅ Already done — AutoDiff lives at `src/Nivara/AutoDiff/` with namespaces `Nivara.AutoDiff.*`. The question is moot. All future phases (P0–P6) will land directly in core.

3. **Shape strategy** — `Linear` needs explicit input/output dims. Should we keep it explicit forever, or add an inferred `Linear(inFeatures)` that lazily builds weights on first forward pass?

4. **Null in training labels** — When a label is null in a supervised batch, should that sample be:
   - Skipped entirely (remove row from batch)
   - Included but contribute zero loss (current Sum/Mean skip nulls)
   - Throw (strict mode)

5. **Module.Train() / Eval()** — Needed for Dropout/BatchNorm. Worth implementing the toggle now even if those layers are deferred?

6. **CPU-only forever?** — If we add a `ToDevice()` concept now (even just CPU), it shapes the API surface. Or stay flat and add device later.

7. **Test strategy per phase** — Should each phase ship with its own test file (mirroring the `tests/Nivara.Tests/AutoDiff/` pattern), or batch tests into larger files by area (e.g., `NnTests.cs`, `OptimizerTests.cs`)?

---

## Performance Targets (after P0)

P0's goal is simple: **no scalar `for` loops in the hot path for float/double.** After P0, a single training step on a 128×256 MLP should be CPU-bound on the matmul pipeline, not on scalar overhead.

### Before vs after (single step, batch=128, 784→256→10)

| Operation | Before P0 | After P0 |
|-----------|-----------|----------|
| MatMul (128×784 @ 784×256) | scalar triple-loop, ~100 µs | `MatMulHelper` (Dot + Parallel.For), ~5–10 µs |
| MatMul (128×256 @ 256×10) | scalar triple-loop, ~25 µs | `MatMulHelper` (Dot + Parallel.For), ~2–4 µs |
| SGD update (10K params) | element for-loop, ~2 µs | `TensorPrimitives`, sub-µs |
| Grad norm (10K params) | manual sum-sq + sqrt, ~1 µs | `TensorPrimitives.Norm`, sub-µs |
| Grad clip (10K params) | element for-loop, ~1 µs | `TensorPrimitives.Clamp`, sub-µs |

Estimates assume AVX2 (256-bit, 4-wide float). Actual speedup varies by hardware — the key is that matmul goes from O(n³) scalar to hardware-tuned SIMD.

### What P0 DOES NOT fix (deferred to later)

- **Parallel backward pass** — sibling subgraphs still run sequentially. Worth doing after P4 when multi-batch pipelining becomes relevant.
- **Parallel parameter updates** — each parameter's `Step()` currently runs serially. P3's `Optimizer.Step()` iterates over parameters one at a time; concurrent per-parameter updates are a P6 concern.
- **SIMD for int/long types** — `TensorPrimitives` has generic overloads that work, but integer ML is rare. Deferred unless a use case appears.
- **GPU** — completely out of scope. C# SIMD on CPU is the selling point.

### End-to-end performance targets (after P0 + P3 + P6)

Aspirational wall-clock targets for a full training run (linear regression, 4096 rows, 100 epochs) combining SIMD matmul, SIMD optimizer, and data-parallel chunking. These become testable benchmarks once P0, P3, and P6 ship.

| Cores | Vector width | Batch rows | Epochs | Target wall time |
|-------|-------------|-----------|--------|-----------------|
| 1 (scalar, no SIMD) | 1 float/op | 4096 | 100 | ~8-10s |
| 1 (baseline) | 8 floats (AVX2) | 4096 | 100 | ~2-3s |
| 4 | 8 floats (AVX2) | 4096 | 100 | ~600ms |
| 8 | 8 floats (AVX2) | 4096 | 100 | ~350ms |
| 16 | 16 floats (AVX-512) | 4096 | 100 | ~150ms |

Target: near-linear scaling up to 8 cores; AVX-512 provides an additional ~1.3× multiplier at 16 cores.

---

## Why This Matters — Real-World Training Scenarios

Below is what P1–P4 collectively unlock, written for someone who doesn't work with ML day-to-day. The goal is to show *what kind of problems become solvable* and *how much ceremony disappears* compared to using the raw autograd engine today.

### Current friction (before P1–P4)

Training any model today (e.g., binary fraud classifier) requires hand-rolling all of this per script:

```
╔══════════════════════════════════════════════╗
║   1. Create weight/bias tensors manually     ║
║   2. Write forward pass with raw operations  ║
║   3. Hand-write loss function gradient       ║
║   4. Call Backward() then per-tensor Sgd     ║
║   5. Zero gradients manually                 ║
║   6. Slice data into batches by hand         ║
║   7. Write the epoch-for-batch loop          ║
╚══════════════════════════════════════════════╝
   ~50–80 lines of repetitive boilerplate
```

Each new model or dataset means repeating all of the above. There is no Adam, no data pipeline, no weight initialization — just the raw `ReverseGradTensor` + `GradOperations`.

### After P1–P4 — the gap closes to a single pattern

```
P1 (Module System)   →  Declare model architecture in a class
P2 (Loss Functions)  →  Pick MSE, CrossEntropy, BCE off the shelf
P3 (Optimizers)      →  Adam/AdamW with momentum, weight decay, bias correction
P4 (Data Loading)    →  Load CSV → TensorDataset → DataLoader → TrainingLoop
```

All four feed into a training pipeline that is the same shape whether you're doing fraud detection, house price regression, or recommendation:

```csharp
// 1. Load data (already works today via Nivara.IO)
var frame = Csv.ReadAsFrame("data.csv");

// 2. Wrap in dataset + loader  (P4)
var loader = new DataLoader<float>(new TensorDataset<float>(frame, features, label), batchSize: 128);

// 3. Define model  (P1)
class MyModel : Module<float> { ... }
var model = new MyModel(...);

// 4. Train  (P1 + P2 + P3 + P4)
TrainingLoop.Run(model, loader, new MSELoss<float>(), new Adam<float>(model.Parameters()), epochs: 20);
```

The pipeline is the same regardless of domain — only the model class and loss function change.

### Concrete examples

**Fraud detection (binary classification)**
```csharp
var frame = Csv.ReadAsFrame("transactions.csv");
var loader = new DataLoader<float>(
    new TensorDataset<float>(frame, ["amount", "hour", "distance"], "is_fraud"), batchSize: 128);

class FraudNet : Module<float>
{
    Linear<float> L1, L2, L3;
    public FraudNet() { L1 = new Linear<float>(3, 64); L2 = new Linear<float>(64, 32); L3 = new Linear<float>(32, 1); RegisterModules(L1, L2, L3); }
    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> x)
        => L3.Forward(GradOperations.Relu(L2.Forward(GradOperations.Relu(L1.Forward(x)))));
}

var model = new FraudNet();
KaimingUniform.Init(model.Parameters());
TrainingLoop.Run(model, loader, new BCEWithLogitsLoss<float>(), new Adam<float>(model.Parameters(), lr: 0.001f), epochs: 20);
```

**House price regression**
```csharp
var frame = Csv.ReadAsFrame("housing.csv"); // nulls in data — handled automatically
var loader = new DataLoader<float>(
    new TensorDataset<float>(frame, ["sqft", "bedrooms", "bathrooms", "year_built"], "price"), batchSize: 64);

class HouseModel : Module<float>
{
    Linear<float> L1, L2;
    public HouseModel() { L1 = new Linear<float>(4, 128); L2 = new Linear<float>(128, 1); RegisterModules(L1, L2); }
    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> x)
        => L2.Forward(GradOperations.Relu(L1.Forward(x)));
}

TrainingLoop.Run(model, loader, new MSELoss<float>(), new AdamW<float>(model.Parameters(), lr: 0.001f, weightDecay: 1e-4f), epochs: 50);
```

**Matrix factorization (recommendation)**
```csharp
var ratings = Csv.ReadAsFrame("ratings.csv");
var loader = new DataLoader<float>(
    new TensorDataset<float>(ratings, "user_id", "item_id", "rating"), batchSize: 256);

class MF : Module<float>
{
    Embedding<float> UserEmb, ItemEmb;
    public MF(int nU, int nI, int dim) { UserEmb = new Embedding<float>(nU, dim); ItemEmb = new Embedding<float>(nI, dim); RegisterModules(UserEmb, ItemEmb); }
    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> u, ReverseGradTensor<float> i)
        => GradOperations.Sum(GradOperations.Multiply(UserEmb.Forward(u), ItemEmb.Forward(i)));
}

TrainingLoop.Run(model, loader, new MSELoss<float>(), new Adam<float>(model.Parameters(), lr: 0.01f), epochs: 10);
```

### What's the same across all three

| Concern | Pattern | Delivered by |
|---------|---------|--------------|
| Data loading | `Csv.ReadAsFrame → TensorDataset → DataLoader` | P4 + existing IO |
| Model definition | `class X : Module<T> { ... RegisterModules(...) }` | P1 |
| Loss function | `new MSELoss<T>()` / `new CrossEntropyLoss<T>()` | P2 |
| Optimizer | `new Adam<T>(parameters, lr)` / `new AdamW<T>(...)` | P3 |
| Training loop | `TrainingLoop.Run(model, loader, loss, optimizer, epochs)` | P4 |

### The .NET advantage — more workers and bigger shovels

This plan leans into what C#/.NET does well that Python ML libraries cannot easily replicate:

**More workers (thread parallelism).** Python's GIL limits single-process parallelism — you need `multiprocessing` or `ray` to use multiple cores, which adds serialization overhead and complexity. C# gets `Parallel.For`, `Task.WhenAll`, and thread-safe state from the language itself:
- P0 keeps each kernel SIMD-single-threaded (safe and cache-friendly)
- P6 adds `Parallel.For` over independent parameter updates in the optimizer
- Parallel backward pass spawns concurrent tasks for independent subgraph branches
- Training loop pipelines forward/backward across batches using `Task.WhenAll`

**Bigger shovels (SIMD acceleration).** `System.Numerics.Tensors` gives CPU-tuned SIMD kernels (AVX2, AVX-512, NEON) maintained by the .NET runtime team:
- **P0a**: `MatMulHelper` (centralized in `src/Nivara/Tensors/MatMulHelper.cs`) uses `TensorPrimitives.Dot` + `Parallel.For` — the single biggest speedup in the plan. When `Tensor.MatrixMultiply` ships in a future .NET version, only the helper body changes.
- **P0b**: `TensorPrimitives.Multiply`/`Subtract`/`Add` replace scalar optimizer loops
- **P0c**: `TensorPrimitives.Norm`/`Clamp` replace manual gradient norm and clipping

Python has NumPy for SIMD but loses it the moment autograd enters the picture (PyTorch's tiny kernels don't vectorize well in eager mode). C# keeps SIMD through the entire autograd pipeline because `TensorPrimitives` operates on raw spans without Python overhead.

### What a non-ML engineer should take away

- **P0** makes everything fast — SIMD matmul and SIMD optimizer math mean training doesn't crawl. This is unique to .NET; Python can't keep SIMD through an autograd graph without a JIT compiler.
- **P1** makes model architecture readable — you can see the layer sizes in the constructor, not hunt through variable declarations.
- **P2** means you don't derive calculus formulas by hand — losses are pre-built and numerically stable.
- **P3** means the optimizer does more than raw SGD — Adam adapts learning rates per-parameter, AdamW adds regularization. State is managed internally, not by the caller.
- **P4** means data pipelines and training loops are not rewritten per project — they're a library call.
- **P6** means your CPU's extra cores actually get used — parameter updates and batch processing parallelize without the GIL fighting you.

Together, P0–P6 move AutoDiff from "a fascinating engine you could build something with" to "something you can actually train a model with by writing ~20 lines of glue code" — while keeping every CPU core fed with SIMD instructions, something Python cannot do without a JIT compiler.

---

## Non-goals (out of scope)

| Feature | Rationale |
|---------|-----------|
| GPU backend | CPU-first. Leverage SIMD across cores. |
| Broadcast / shape inference | Current shape validation is strict. Separate effort. |
| ExecutionEngine / IQueryOperation integration | Training doesn't fit the column-transform model. |
| Operator overloads (`+`, `-`, `*`) | `GradOperations.Add(...)` stays. Ergonomics deferred. |

## Reference: current AutoDiff file layout

AutoDiff already lives in core (`src/Nivara/AutoDiff/`). Current file layout after PLAN Phase 1 (completed):

```
src/Nivara/AutoDiff/
├── GradTensor.cs
├── ReverseGradTensor.cs
├── OpNode.cs
├── ComputationGraph.cs
├── Exceptions/
│   └── AutoGradExceptions.cs
├── Operations/
│   └── GradOperations.cs
├── Optimizer/
│   └── SgdOptimizer.cs
├── Utilities/
│   ├── GradientUtils.cs
│   ├── TypeConverter.cs
│   └── TypeValidator.cs
└── Extensions/
    └── NivaraAutoGradExtensions.cs
```

Namespaces: `Nivara.AutoDiff`, `Nivara.AutoDiff.Operations`, `Nivara.AutoDiff.Optimizer`, `Nivara.AutoDiff.Utilities`, `Nivara.AutoDiff.Extensions`.

Tests: `tests/Nivara.Tests/AutoDiff/` (8 test files).
