# Nivara Automatic Differentiation (AutoDiff)

Nivara provides a PyTorch-inspired reverse-mode automatic differentiation engine built on top of its columnar DataFrame types. Tensors wrap `NivaraColumn<T>` and the computation graph is a DAG of operation nodes built implicitly during forward operations.

Beyond the core autograd engine, Nivara delivers a full training stack: module system, loss functions, stateful optimizers, data loading, training loops, serialization, and data-parallel training — all implemented in ~50 files under `src/Nivara/AutoDiff/`.

---

## Icebreaker: Honest comparison with PyTorch

A cross-framework parity exercise (see `examples/README.md`) trained an
identical 3-layer MLP in both Nivara and PyTorch and compared results.

**Correctness: proven.** Loss curves match within 0.04% relative diff across
50 epochs. The gradient math, optimizer (Adam), and training loop are correct.

**Developer experience: PyTorch still wins comfortably.** The gaps hit during
the exercise:

| Dimension | PyTorch | Nivara | Impact |
|-----------|---------|--------|--------|
| Model registration | `self.l1 = nn.Linear(8, 64)` — auto-registered | `L1 = new Linear<float>(8, 64)` + manual `RegisterModules(...)` | Easy to forget; no compiler error if you do |
| Forward pass | `torch.relu(self.l1(x))` — direct | `GradOperations.Relu(L1.Forward(x))` — extra ceremony | More typing, harder to read |
| Generics | None (dynamic) | `Module<float>`, `Linear<float>`, `ReverseGradTensor<float>` everywhere | Type pollution propagates through all code |
| Weight loading | `param.data.copy_(tensor)` — one-liner | Custom JSON flatten + manual name mapping (PyTorch name → Nivara key) | ~40 lines of fragile boilerplate, easy to mismatch |
| Optimizer API | `optimizer = Adam(model.parameters())` — unambiguous | Optimizers now register owning `Parameter<T>` objects via `model.GetParameters().Values`; tensor dictionaries are for inspection/initialization, not training | Safer than the old API, still more verbose than PyTorch |
| Error messages | Polished after a decade | Generic operation-node stack traces | Harder to debug autograd failures |
| Ecosystem | TensorBoard, torchinfo, etc. | `result.PrintSummary()` — minimal | Fine for small models, insufficient at scale |

**Resolved optimizer trap:** The old tensor-dictionary optimizer overload was
removed. Training registration should use owning parameter wrappers:
`optimizer.AddParameterGroup(model.GetParameters().Values)`. `Parameters()`
continues to return tensor dictionaries for inspection, initialization, and
serialization-oriented workflows.

**Weight divergence is expected but disorienting.** Even with identical
seeded initialization, SGD trajectories diverge between frameworks (different
BLAS kernels, FP accumulation order). The loss curves stay aligned, proving
the gradient computation is correct, but a first-time user comparing weights
directly will see large differences (e.g., L2.Weight 180% relative diff) and
may incorrectly conclude something is broken.

**Verdict for this example:** A for correctness, B for usability. The hard
part (backprop, optimizer, training infrastructure) works. The easy part
(ergonomics, discoverability, documentation) shows PyTorch's decade of
iteration.

---

## Architecture

```
Core Engine
───────────
GradTensor<T>                         ← Base: Data, Shape, Reshape, ToColumn/ToSeries/AsTensor
└── ReverseGradTensor<T>              ← Adds: Grad, RequiresGrad, GradFn, Backward(), Detach()

OpNode<T>                             ← One operation node in the graph
├── OperationName                     ← "Add", "MatMul", "Relu", etc.
├── Inputs                            ← Parent tensors (object[])
├── BackwardFunction                  ← Action<NivaraColumn<T>> (local gradient rule)
├── ShouldSaveForBackward             ← Whether to save values for backward pass
├── SavedValues                       ← Dictionary of saved forward values
└── Apply(gradOutput)                 ← Invokes the backward function

ComputationGraph                      ← Graph traversal engine
├── AddNode(output, opNode)           ← Attaches GradFn to output tensor
├── Backward(tensor, gradient)        ← Topological sort + reverse traversal
├── TopologicalSort(root)             ← DFS with cycle detection
├── ZeroGrad(tensor)                  ← Recursively clears gradients
└── GetGraphInfo(root)                ← Returns diagnostic summary

GradOperations                        ← All forward+backward ops (static methods)
├── Element-wise: Add, Subtract, Multiply, Divide, Clip
├── Matrix: MatMul, Transpose         ← MatMul uses SIMD MatMulHelper with TensorPrimitives.Dot
├── Reductions: Sum, Mean
├── Activations: Relu, LeakyRelu, Sigmoid, Tanh
├── Unary: Negate, Abs, Exp, Log
├── Probability: Softmax, LogSoftmax
└── ...vectorized via TensorPrimitives where available

MatMulHelper                          ← Central SIMD MatMul (TensorPrimitives.Dot + Parallel.For)
src/Nivara/Tensors/MatMulHelper.cs

Module System (Nn)
──────────────────
Parameter<T>                          ← Named ReverseGradTensor<T> with requiresGrad=true
Module<T>                             ← Abstract base: Forward(), Parameters(), Train()/Eval()
├── Linear<T>                         ← y = x @ Wᵀ + b (Kaiming-uniform init, bias optional)
├── Sequential<T>                     ← Ordered module chain
├── Activation<T>                     ← ReLU, Sigmoid, Tanh, LeakyReLU wrappers
├── Dropout<T>                        ← Inverted dropout with train/eval toggle
└── Initializers/                     ← Kaiming, Xavier, Uniform, Normal

Loss Functions (Nn.Functional)
───────────────────────────────
MSELoss<T>                            ← Σ(pred - target)²
L1Loss<T>                             ← Σ|pred - target|
BCELoss<T>                            ← -(y·log(p) + (1-y)·log(1-p))
BCEWithLogitsLoss<T>                  ← Fused sigmoid + BCE (numerically stable)
CrossEntropyLoss<T>                   ← Fused log-softmax + NLL
Softmax<T> / LogSoftmax<T>            ← Dim-aware softmax wrappers

Optimizers
──────────
Optimizer<T> (abstract)              ← Step(), ZeroGrad(), AddParameterGroup()
├── SGD<T>                           ← Momentum + weight decay + TensorPrimitives fast path
├── Adam<T>                          ← Bias-corrected, β₁/β₂ defaults 0.9/0.999
├── AdamW<T>                         ← Decoupled weight decay (Loshchilov & Hutter 2019)
└── SgdUpdate (static)              ← Single-tensor SgdUpdate helper

Training
────────
TensorDataset<T>                      ← Wraps NivaraFrame, exposes feature/label tensor slices
DataLoader<T>                         ← Batch iteration with shuffle
Batch<T>                              ← { Features, Labels }
TrainingLoop<T>                       ← ForEach-epoch/batch: forward → backward → step → zero_grad
DataParallelTrainer<T>                ← Parallel.For over chunks + gradient sum + optimizer step

Serialization
─────────────
ModelSerializer                       ← Save/Load model state dicts (JSON + base64 binary)
Checkpoint<T>                         ← Epoch + loss + optimizer state + model params

Utilities
─────────
GradientUtils                         ← ZeroGrad, Detach, ClipGradValue/Norm, creators, diagnostics
TypeValidator                         ← Runtime type checking (only float/double)
TypeConverter                         ← Cross-type tensor conversion (float ↔ double)

NivaraAutoGradExtensions              ← NivaraColumn/NivaraSeries/NivaraFrame ↔ ReverseGradTensor
```

---

## Key Design Principles

- **Only `float` and `double` are supported** — enforced at runtime by `TypeValidator.ValidateNumericType<T>()`. Other numeric types (int, long, etc.) throw `TypeValidationException`.
- **1D storage, shape metadata** — data is always stored as a flat `NivaraColumn<T>`. Shape is metadata (`int[] shape`) with `Reshape()` validation. Default shape is `[Length]`.
- **Inference is the default** — normal `Forward` and `GradOperations` calls compute values without building a computation graph.
- **Training is explicit** — wrap manual training code in `using (GradientUtils.Grad())`; inside that scope, operations check trainable inputs (`requiresGrad`) and attach `OpNode` history to results.
- **Gradient accumulation** — `AccumulateGradient()` either sets or adds to `Grad` (supports fan-in from multiple paths).
- **Explicit null-mask propagation** — Nivara's nullable semantics flow through gradients. Nulls propagate via mask OR; null positions in gradients are skipped during accumulation.
- **IDisposable** — `GradTensor<T>`, `Parameter<T>`, `Module<T>`, `Optimizer<T>`, and `TrainingLoop<T>` all implement `IDisposable`.

---

## Tensor Classes

### GradTensor\<T\>

The base tensor class holding data and shape metadata:

```csharp
public class GradTensor<T> : IDisposable where T : struct, INumber<T>
```

| Member | Description |
|--------|-------------|
| `Data` | The underlying `NivaraColumn<T>` |
| `Length` | Number of elements |
| `HasNulls` | Whether data contains null values |
| `Shape` | Read-only copy of dimension sizes |
| `Rank` | Number of dimensions |
| `this[int index]` | Element accessor |
| `IsNull(int index)` | Null check |
| `Reshape(params int[] dims)` | Sets shape metadata (product must equal Length) |
| `AsTensor()` | Returns `Tensor<T>` view (throws if nulls present) |
| `ToColumn()` | Returns `NivaraColumn<T>` |
| `ToSeries()` | Returns `NivaraSeries<T>` |

Constructor validates the type via `TypeValidator.ValidateNumericType<T>()`.

### ReverseGradTensor\<T\>

Extends `GradTensor<T>` with gradient tracking and backward pass:

```csharp
public sealed class ReverseGradTensor<T> : GradTensor<T> where T : struct, INumber<T>
```

| Member | Description |
|--------|-------------|
| `Grad` | `NivaraColumn<T>?` — accumulated gradient (null before backward) |
| `RequiresGrad` | Whether this tensor tracks gradients |
| `GradFn` | `OpNode<T>?` — computation graph node (null for leaf tensors) |
| `IsLeaf` | `true` if `GradFn == null` |
| `Backward(gradient?)` | Initiates reverse-mode gradient computation |
| `Detach()` | Returns new tensor without gradient tracking |
| `ZeroGrad()` | Clears gradient |
| `ConvertTo<TTarget>()` | Converts to different numeric type |
| `ToFloat()` / `ToDouble()` | Convenience conversion methods |

**Factory methods:**

| Method | Description |
|--------|-------------|
| `FromColumn(column, requiresGrad)` | Wraps a `NivaraColumn<T>` |
| `FromSeries(series, requiresGrad)` | Wraps a `NivaraSeries<T>` |
| `FromArray(array, requiresGrad)` | Creates from `T[]` |
| `FromMatrix(data, rows, cols, requiresGrad)` | Creates 2D tensor with shape [rows, cols] |

**Backward behavior:**
- Scalar tensors (length 1): `Backward()` with no argument initializes gradient to `[1.0]`
- Non-scalar tensors: `Backward(gradient)` requires an explicit gradient tensor of matching shape
- Throws `InvalidOperationException` if called on tensors without `requiresGrad`
- Wraps graph errors (circular dependency, missing output) in descriptive messages

---

## Computation Graph (OpNode / ComputationGraph)

### OpNode\<T\>

Represents a single operation in the computation graph:

```csharp
sealed class OpNode<T> where T : struct, INumber<T>
{
    string OperationName { get; }           // "Add", "MatMul", "Relu", etc.
    IReadOnlyList<object> Inputs { get; }   // parent tensors
    Action<NivaraColumn<T>> BackwardFunction { get; }
    bool ShouldSaveForBackward { get; }
    Dictionary<string, object>? SavedValues { get; }
    void Apply(NivaraColumn<T> gradOutput);
}
```

The `BackwardFunction` closure captures references to input tensors and any saved forward values (e.g., sigmoid output for sigmoid gradient). It computes the local gradient contribution and calls `AccumulateGradient` on each input that requires `grad`.

### ComputationGraph

Static graph traversal engine:

| Method | Description |
|--------|-------------|
| `AddNode(output, node)` | Attaches GradFn to the output tensor |
| `Backward(tensor, gradient)` | Topological sort + reverse-topological traversal, calling each node's `Apply(gradOutput)` |
| `TopologicalSort(root)` | DFS with cycle detection via visiting/visited sets |
| `ValidateGraph(root)` | Validates no circular dependencies |
| `ZeroGrad(tensor)` | Recursively clears gradients from reachable tensors |
| `GetGraphInfo(root)` | Returns `{ TotalNodes, IsLeaf, RequiresGrad, OperationCounts }` |

**Backward algorithm:**

1. `BuildNodeToOutputMap(tensor)` — maps OpNode → output tensor
2. `TopologicalSort(tensor)` — DFS producing a linear order
3. Iterate nodes **in reverse** (reverse topological order)
4. For each node, look up its output tensor, get `outputTensor.Grad`
5. Call `node.Apply(outputGrad)` which invokes the backward function
6. Each backward function computes local gradients and accumulates them via `AccumulateGradient`

---

## Supported Operations

### Element-wise

| Op | Forward | Backward Rule | Null Semantics |
|----|---------|---------------|----------------|
| `Add(a, b)` | `a + b` | `∂/∂a = grad`, `∂/∂b = grad` | mask OR |
| `Subtract(a, b)` | `a - b` | `∂/∂a = grad`, `∂/∂b = -grad` | mask OR |
| `Multiply(a, b)` | `a * b` | `∂/∂a = grad * b`, `∂/∂b = grad * a` | mask OR |
| `Divide(a, b)` | `a / b` | `∂/∂a = grad / b`, `∂/∂b = -(a/b²) * grad` | mask OR; throws on zero division |
| `Clip(a, min, max)` | `clamp(a, min, max)` | 1 if in-range, 0 outside | mask OR |

### Matrix

| Op | Forward | Backward Rule | Requirements |
|----|---------|---------------|--------------|
| `MatMul(a, b)` | `a @ b` | `∂/∂a = grad @ bᵀ`, `∂/∂b = aᵀ @ grad` | Both tensors rank 2; `a.Cols == b.Rows` |
| `Transpose(a)` | `aᵀ` | `∂/∂a = gradᵀ` | Rank 2 |

MatMul uses `MatMulHelper` (`src/Nivara/Tensors/MatMulHelper.cs`) with `TensorPrimitives.Dot` + `Parallel.For` for SIMD-accelerated forward and backward passes. When `Tensor.MatrixMultiply` ships in a future .NET version, only `MatMulHelper` changes.

### Reductions

| Op | Forward | Backward Rule | Notes |
|----|---------|---------------|-------|
| `Sum(a)` | `∑a` | `broadcast(grad, n)` — fills gradient value to all positions | Expects scalar output |
| `Mean(a)` | `(∑a)/n` | `broadcast(grad/n, n)` — fills gradient/n to all positions | Expects scalar output |

### Activations

| Op | Forward | Backward Rule | Vectorization |
|----|---------|---------------|---------------|
| `Relu(a)` | `max(a, 0)` | `grad * (1 if a > 0 else 0)` | `TensorPrimitives.Max` for forward; manual loop for grad |
| `LeakyRelu(a, slope)` | `max(a, 0) + slope * min(a, 0)` | `grad * (1 if a > 0 else slope)` | Manual loop |
| `Sigmoid(a)` | `σ(a) = 1/(1+e⁻ᵃ)` | `σ(a) * (1-σ(a)) * grad` | Manual loop via `Math.Exp` |
| `Tanh(a)` | `tanh(a)` | `(1 - tanh²(a)) * grad` | Manual loop via `Math.Tanh` |
| `Negate(a)` | `-a` | `-grad` | `TensorPrimitives.Negate` |
| `Abs(a)` | `\|a\|` | `sign(a) * grad` | `TensorPrimitives.Abs` for forward; manual loop for grad |
| `Exp(a)` | `eᵃ` | `eᵃ * grad` | Manual loop via `Math.Exp` |
| `Log(a)` | `ln(a)` | `grad / a` | Manual loop; throws on non-positive |

### Probability

| Op | Forward | Backward Rule | Notes |
|----|---------|---------------|-------|
| `Softmax(a)` | `e^(a - max) / Σe^(a - max)` | Full Jacobian via `diag(s) - s·sᵀ` | Numerically stable subtract-max |
| `LogSoftmax(a)` | `log(softmax(a))` | `grad - Σgrad · softmax(a)` | Fused for CrossEntropy efficiency |

### Vectorization Strategy

All operations follow a two-path approach:

1. **No-null fast path**: Extract `ReadOnlySpan<T>` via `TryGetSpan()`, call `TensorPrimitives` kernel, return `NivaraColumn<T>.Create(result)`.
2. **Null-aware fallback**: Rent buffers from `ArrayPool<T>.Shared`, call `CopyTo()` to fill buffers with `T.Zero` for null positions, run `TensorPrimitives` kernel, merge null masks via OR, return `NivaraColumn<T>.CreateFromSpans(result, nullMask)`.

Operations that lack `TensorPrimitives` support (e.g., Sigmoid, Tanh, Exp, Log) use manual loops in both paths. MatMul uses `MatMulHelper.Multiply` with `TensorPrimitives.Dot` + `Parallel.For`; its null-aware result mask is propagated with row/column null summaries.

---

## Null Handling

Nivara's explicit null-mask semantics flow through all gradient operations:

- **Null in input** → propagates to both forward result and gradient masks (mask OR semantics)
- **Null in gradient** → `AccumulateGradient` skips the position entirely (no zeroing)
- **Null in SGD** → if parameter is null → stays null; if gradient is null → no update at that position (parameter unchanged)
- **Null in MatMul** → position-level null: if any contributing element is null, the corresponding output position is null
- **Null in Adam/AdamW** → null positions skip momentum buffer accumulation; buffers are zeroed for that position

The `MergeNullMasks(a, b, destination)` helper performs the OR operation, handling cases where one or both inputs lack null masks.

---

## Module System (Nn)

### Parameter\<T\>

```csharp
public sealed class Parameter<T> : IDisposable where T : struct, INumber<T>
```

Wraps a `ReverseGradTensor<T>` with a name. Constructors accept array, size, or an existing tensor. `requiresGrad` defaults to `true`. Disposing a parameter disposes its current tensor; module-registered parameters are disposed by their owning module.

### Module\<T\>

```csharp
public abstract class Module<T> : IDisposable where T : struct, INumber<T>
```

| Member | Description |
|--------|-------------|
| `IsTraining` | Current train/eval state |
| `Forward(input)` | Abstract — define model logic |
| `Forward(input, input2)` | Virtual — multi-input forward (throws by default) |
| `Train()` | Sets training mode (recursive) |
| `Eval()` | Sets evaluation mode (recursive) |
| `RegisterModules(...)` | Register child modules for parameter discovery |
| `RegisterParameters(...)` | Register standalone parameters |
| `Parameters()` | Returns flat `Dictionary<string, ReverseGradTensor<T>>` |
| `GetParameters()` | Returns `Dictionary<string, Parameter<T>>` with metadata |
| `NamedModules()` | Returns registered child modules |

### Linear\<T\>

```csharp
public sealed class Linear<T> : Module<T> where T : struct, INumber<T>
```

`y = x @ Wᵀ + b` with shape `[batch, inFeatures] → [batch, outFeatures]`.

- Registers weight (shape `[outFeatures, inFeatures]`) and optional bias (shape `[1, outFeatures]`) as parameters
- Initializes weights with Kaiming-Uniform: `U(-√(6/fanIn), √(6/fanIn))`
- Forward transposes weight, applies MatMul, then broadcasts bias via `ones @ bias`

### Sequential\<T\>

```csharp
public sealed class Sequential<T> : Module<T>
```

Pipes forward pass through an ordered list of modules. Supports `Append()` for dynamic construction.

### Activation\<T\>

Wraps a single activation function as a module: `Relu`, `Sigmoid`, `Tanh`, `LeakyRelu`.

### Dropout\<T\>

Inverted dropout — scales by `1/(1-p)` during training, identity during eval. Training-mode dropout is differentiable: the sampled keep mask is reused during backward so gradients before dropout receive `gradOutput * keepMask * scale`.

### Initializers

| Class | Formula | Use |
|-------|---------|-----|
| `KaimingUniform` | `U(-√(6/fanIn), √(6/fanIn))` | ReLU layers |
| `KaimingNormal` | `N(0, √(2/fanIn))` | ReLU layers |
| `XavierUniform` | `U(-√(6/(fanIn+fanOut)), √(6/(fanIn+fanOut)))` | Tanh/Sigmoid layers |
| `XavierNormal` | `N(0, √(2/(fanIn+fanOut)))` | Tanh/Sigmoid layers |
| `Uniform(bound)` | `U(-bound, bound)` | Generic |
| `Normal(mean, std)` | `N(mean, std)` | Generic |

Call `KaimingUniform.Init(model.Parameters())` or individual parameter init after construction.

### Example

```csharp
class MLP : Module<float>
{
    Linear<float> L1, L2, L3;

    public MLP()
    {
        L1 = new Linear<float>(784, 256);
        L2 = new Linear<float>(256, 64);
        L3 = new Linear<float>(64, 10);
        RegisterModules(L1, L2, L3);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> x)
    {
        var h = GradOperations.Relu(L1.Forward(x));
        h = GradOperations.Relu(L2.Forward(h));
        return L3.Forward(h);
    }
}
```

---

## Loss Functions

All loss functions live in `Nivara.AutoDiff.Nn.Functional`. Each has a `Forward(predictions, targets)` method returning a scalar loss tensor.

| Loss | Forward Formula | Notes |
|------|----------------|-------|
| `MSELoss<T>` | `Σ(pred - target)²` | Mean Squared Error (sum reduction) |
| `L1Loss<T>` | `Σ\|pred - target\|` | Mean Absolute Error (sum reduction) |
| `BCELoss<T>` | `-Σ(y·log(p) + (1-y)·log(1-p))` | Inputs clamped to `[eps, 1-eps]` for numerical stability |
| `BCEWithLogitsLoss<T>` | Fused sigmoid + BCE | Numerically stable — no clamp needed |
| `CrossEntropyLoss<T>` | LogSoftmax + NLL ÷ batchSize | Expects logits + one-hot targets |
| `Softmax<T>` | dim-aware softmax | Wrapper around `GradOperations.Softmax` |
| `LogSoftmax<T>` | dim-aware log-softmax | Wrapper around `GradOperations.LogSoftmax` |

---

## Optimizers

### Optimizer\<T\> (abstract base)

```csharp
public abstract class Optimizer<T> : IDisposable where T : struct, INumber<T>
```

**Parameter groups** — optimizers can manage multiple groups with different learning rates and weight decays:

```csharp
optimizer.AddParameterGroup(parameter);                      // uses optimizer.LearningRate
optimizer.AddParameterGroup(model.GetParameters().Values);   // uses optimizer.LearningRate
optimizer.AddParameterGroup(parameter, learningRate, weightDecay);
optimizer.AddParameterGroup(model.GetParameters().Values, learningRate, weightDecay);
```

| Member | Description |
|--------|-------------|
| `LearningRate` | Default learning rate used by parameter groups when no group override is supplied |
| `Step()` | Abstract — applies updates to all parameters |
| `ZeroGrad()` | Zeros gradients on all managed parameters |
| `AddParameterGroup(...)` | Registers owning `Parameter<T>` objects; use `model.GetParameters().Values` for modules |
| `Dispose()` | Releases rented state buffers |

### SGD\<T\>

```csharp
public sealed class SGD<T> : Optimizer<T>
// new SGD<T>(learningRate, momentum: 0.0)
```

- Optional momentum (`[0, 1)`)
- Optional weight decay per parameter group
- No-null fast path uses `TensorPrimitives.Multiply`/`Subtract/Add`
- Static `SgdUpdate(tensor, lr, wd)` helper for single-parameter updates

### Adam\<T\>

```csharp
public sealed class Adam<T> : Optimizer<T>
// new Adam<T>()                         // learningRate = 0.001
// new Adam<T>(learningRate)
// new Adam<T>(beta1: 0.9, beta2: 0.999, eps: 1e-8)
```

- Default learning rate is `0.001` unless overridden by constructor or parameter group
- Bias-corrected first/second moment estimates
- State buffers rented from `ArrayPool<T>.Shared`
- Null-skip: null positions zero momentum buffers (no update)
- Decoupled weight decay via per-group `weightDecay`

### AdamW\<T\>

```csharp
public sealed class AdamW<T> : Optimizer<T>
// new AdamW<T>()                         // learningRate = 0.001
// new AdamW<T>(learningRate)
// new AdamW<T>(beta1: 0.9, beta2: 0.999, eps: 1e-8)
```

- Default learning rate is `0.001` unless overridden by constructor or parameter group
- Identical to Adam except weight decay is applied directly to weights (not through gradients) — Loshchilov & Hutter 2019 formulation
- Same null-skip semantics and `ArrayPool` buffer management

### SGD\<T\>.SgdUpdate (static helper)

`SGD<T>.SgdUpdate` is available for single-tensor updates outside the module system.

---

## Training

### TensorDataset\<T\>

```csharp
public sealed class TensorDataset<T> where T : struct, INumber<T>
```

Wraps a `NivaraFrame` with named feature and label columns. `GetBatch(indices)` returns a `Batch<T>` with shaped tensors (flat data reshaped to `[batchSize, numCols]`). Uses `ArrayPool<T>.Shared` for batch construction.

### DataLoader\<T\>

```csharp
public sealed class DataLoader<T> : IEnumerable<Batch<T>>
// new DataLoader<T>(dataset, batchSize, shuffle: true, seed: null)
```

Fisher-Yates shuffle (optionally seeded), yields batches of the requested size (final batch may be smaller).

### Batch\<T\>

```csharp
public sealed class Batch<T>
// { Features: ReverseGradTensor<T>, Labels: ReverseGradTensor<T>, Size: int }
```

### TrainingLoop\<T\>

```csharp
public class TrainingLoop<T> : IDisposable where T : struct, INumber<T>
```

Standard epoch-per-batch training loop:

```csharp
var optimizer = new SGD<float>(learningRate: 0.01f);
optimizer.AddParameterGroup(model.GetParameters().Values);

var loop = new TrainingLoop<float>(
    model, loader,
    (pred, target) => new MSELoss<float>().Forward(pred, target),
    optimizer,
    epochs: 20);

var result = loop.Run();
result.PrintSummary();
```

| Feature | Description |
|---------|-------------|
| Virtual callbacks | `OnEpochStart(epoch)`, `OnBatchEnd(epoch, batch, loss)`, `OnEpochEnd(epoch, result)` |
| Checkpointing | `SaveCheckpoint(path, epoch, result)` — writes JSON checkpoint |
| Results | `TrainingResult<T>` with `PrintSummary()`, epoch-level loss/timing/batches |

### DataParallelTrainer\<T\>

```csharp
public class DataParallelTrainer<T> : IDisposable where T : struct, INumber<T>
```

Multi-core training via `Parallel.For` over data chunks:

```
Split rows into chunks (batchSize per chunk)
  ↓
Parallel.ForEach(chunks):
  ├── GetBatch → Forward → loss → Backward
  └── CloneGradients() → snapshot of per-chunk gradients
  ↓
SumAndApplyGradients(allGradients)     ← TensorPrimitives.Add across chunks
  ↓
Optimizer.Step() + ZeroGrad()
```

| Feature | Description |
|---------|-------------|
| Chunk sizing | Uses `ParallelExecutionHelper` for optimal chunk count |
| Gradient merge | `SumAndApplyGradients` sums per-chunk gradients via `TensorPrimitives.Add` |
| Results | `DataParallelTrainingResult<T>` with `PrintSummary()` |
| Virtual callbacks | `OnEpochStart(epoch)`, `OnEpochEnd(epoch, result)` |

```csharp
var optimizer = new Adam<float>(learningRate: 0.001f);
optimizer.AddParameterGroup(model.GetParameters().Values);

var trainer = new DataParallelTrainer<float>(
    model, loader,
    (pred, target) => new MSELoss<float>().Forward(pred, target),
    optimizer,
    epochs: 10);

var result = trainer.Run();
result.PrintSummary();
// Epoch   1 | Loss:   0.542100 | Workers:  8 | Chunks:   32 | Grad Norm:   1.234500 | Time: 0.42s
```

---

## Serialization

### ModelSerializer

Static class for saving/loading model parameter state dicts:

```csharp
// Save
ModelSerializer.Save(model, "model.json");

// Load (mutates model parameters in-place)
ModelSerializer.Load(model, "model.json");

// Checkpoint
ModelSerializer.SaveCheckpoint(model, epochResult, "checkpoint.json");
var checkpoint = ModelSerializer.LoadCheckpoint<float>("checkpoint.json");
```

**Format:** JSON with format marker `"nivara-ss-v1"` / `"nivara-ckpt-v1"`, version field, type name, and parameter entries. Each parameter entry stores:
- `Shape` — `int[]` dimension sizes
- `Values` — base64-encoded binary (via `MemoryMarshal.AsBytes`), length-validated on load
- `HasNulls` / `NullMask` — optional null mask as base64 bool array

**Validation on load:** shape rank, element count, and parameter name matching with descriptive error messages.

### Checkpoint\<T\>

```csharp
public sealed class Checkpoint<T> where T : struct, INumber<T>
{
    public int Epoch { get; init; }
    public double Loss { get; init; }
    public IReadOnlyDictionary<string, ParameterData<T>> Parameters { get; init; }
}
```

### Example

```csharp
// Train
var loop = new TrainingLoop<float>(model, loader, lossFn, optimizer, epochs: 10);
var result = loop.Run();
result.PrintSummary();

// Save model
ModelSerializer.Save(model, "trained_model.json");

// Load into fresh model for inference
var loaded = new MLP(784, 256, 10);
ModelSerializer.Load(loaded, "trained_model.json");
loaded.Eval();
var prediction = loaded.Forward(testInput);
```

---

## Utility Functions (GradientUtils)

### Gradient Management

| Method | Description |
|--------|-------------|
| `ZeroGrad(tensor)` | Clears gradients recursively via `ComputationGraph.ZeroGrad` |
| `ZeroGrad(tensors)` | Batch zero-grad |
| `Detach(tensor)` | Removes from computation graph |
| `Detach(tensors)` | Batch detach |

### Gradient Clipping

| Method | Description |
|--------|-------------|
| `ClipGradValue(tensor, maxValue)` | Clips each gradient element to `[-maxValue, maxValue]` (uses `TensorPrimitives.Clamp`) |
| `ClipGradNorm(tensor, maxNorm)` | Scales gradient if L2 norm exceeds `maxNorm` (uses `TensorPrimitives.SumOfSquares`) |
| `ClipGradNorm(tensors, maxNorm)` | Global norm clipping across multiple tensors |

All clipping preserves null positions (nulls are skipped).

### Constant Tensor Creators

| Method | Description |
|--------|-------------|
| `Constant(data)` | Creates non-gradient tensor from array or column |
| `Zeros(length)` | Filled with `T.Zero` |
| `Ones(length)` | Filled with `T.One` |
| `Full(length, value)` | Filled with specific value |

### Diagnostics

AutoDiff hot paths participate in the shared `DiagnosticsTracker` when
diagnostics are enabled. Recorded operation names include
`AutoDiffBackward`, `AutoDiffMatMul`, `AutoDiffTranspose`, `AutoDiffRelu`,
`AutoDiffSigmoid`, `AutoDiffTanh`, and `AutoDiffSgdUpdate`; each record
captures elapsed time, managed allocation deltas, element type, input length,
null participation, and operation-specific notes such as shape metadata.

| Method | Description |
|--------|-------------|
| `GetGraphInfo(tensor)` | Delegates to `ComputationGraph.GetGraphInfo` |
| `PrintGraphSummary(tensor)` | Human-readable graph summary string |
| `DescribeTensor(tensor)` | Detailed tensor debug info (length, grad norm, operation, etc.) |
| `HasGradient(tensor)` | Whether `Grad != null` |
| `GetGradientNorm(tensor)` | L2 norm of gradient (uses `TensorPrimitives.SumOfSquares`) |
| `GetGlobalGradientNorm(tensors)` | Combined L2 norm across tensors |
| `CanBackward(tensor)` | Whether `Backward()` can be called (scalar + requiresGrad) |

---

## Type System

### Supported Types

Only **float** and **double** are supported for autograd. This is enforced at two levels:

1. **Generic constraint**: `where T : struct, INumber<T>` (wide, accepts int, long, etc.)
2. **Runtime check**: `TypeValidator.ValidateNumericType<T>()` throws `TypeValidationException` for unsupported types

### Type Conversion (TypeConverter)

| Method | Description |
|--------|-------------|
| `Convert<TSource, TTarget>(source, requiresGrad?)` | Converts between supported types |
| `ToFloat(source, requiresGrad?)` | Converts to float |
| `ToDouble(source, requiresGrad?)` | Converts to double |
| `TryConvert<TSource, TTarget>(...)` | Returns null on failure |
| `CanConvert<TSource, TTarget>()` | Checks if both types are supported |

Conversion preserves null masks and optionally overrides `requiresGrad`.

---

## Nivara Frame Integration

The `NivaraAutoGradExtensions` class (in `Nivara.AutoDiff.Extensions`) provides conversion between Nivara types and autograd tensors:

### Column/Series → Tensor

```csharp
column.ToReverseGradTensor(requiresGrad: false)    // NivaraColumn<T> → ReverseGradTensor<T>
series.ToReverseGradTensor(requiresGrad: false)    // NivaraSeries<T> → ReverseGradTensor<T>
```

### Frame → Tensor batch

```csharp
// Specific columns by name
var tensors = frame.ToReverseGradTensors<float>(
    new[] { "Age", "Income" }, requiresGrad: true);

// Auto-detect numeric columns
var tensors = frame.ToReverseGradTensorsAuto(requiresGrad: false);
// Returns Dictionary<string, object> — only float/double columns
```

### Tensor batch → Frame

```csharp
var dataFrame = tensors.ToFrame();           // Values as NivaraFrame
var gradFrame = tensors.ToGradientFrame();   // Gradients as NivaraFrame (null if no grads)
```

### Batch Operations

```csharp
tensors.BatchBackward(loss);    // Calls loss.Backward()
tensors.BatchZeroGrad();        // Calls ZeroGrad() on all tensors
```

### Type Checking

```csharp
NivaraAutoGradExtensions.IsAutoGradSupported<T>();
NivaraAutoGradExtensions.GetSupportedAutoGradTypes();  // [typeof(float), typeof(double)]
```

---

## Exception Types

| Exception | Context | Key Properties |
|-----------|---------|----------------|
| `AutoGradException` | Base class for all autograd errors | `OperationContext`, `InvolvedShapes`, `GetDetailedContext()` |
| `TypeValidationException` | Unsupported numeric type | `ExpectedType`, `ActualType` |
| `ShapeIncompatibilityException` | Shape mismatch in operations | `ExpectedShape`, `ActualShape` |
| `CircularDependencyException` | Cycle detected in computation graph | `CycleOperationNames` |
| `InvalidBackwardCallException` | Backward called on invalid tensor | `TensorShape`, `RequiresGrad` |
| `GradientComputationException` | Backward pass failure | `FailedOperation`, `FailingNodeInputCount` |

---

## Examples

Manual examples that call `Backward()` run inside `GradientUtils.Grad()`.
Plain forward calls outside this scope are inference and do not build a
computation graph.

### 1. Basic scalar gradient

```csharp
using (GradientUtils.Grad())
{
    var a = new ReverseGradTensor<float>(
        NivaraColumn<float>.Create(new float[] { 3.0f }), requiresGrad: true);
    var b = new ReverseGradTensor<float>(
        NivaraColumn<float>.Create(new float[] { 4.0f }), requiresGrad: true);

    var result = GradOperations.Add(a, b);  // 7.0
    result.Backward();

    Console.WriteLine(a.Grad[0]);  // 1.0  (∂result/∂a)
    Console.WriteLine(b.Grad[0]);  // 1.0  (∂result/∂b)
}
```

### 2. Non-scalar backward with explicit gradient

```csharp
var x = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f }), requiresGrad: true);
var relu = GradOperations.Relu(GradOperations.Negate(x));
// relu = max(-x, 0) = [0, 0, 0]

var gradInput = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new float[] { 1.0f, 1.0f, 1.0f }), requiresGrad: false);
relu.Backward(gradInput);

Console.WriteLine(x.Grad[0]);  // 0.0  (∂relu/∂x at index 0: -1 < 0 → 0)
Console.WriteLine(x.Grad[1]);  // 0.0
Console.WriteLine(x.Grad[2]);  // 0.0
```

### 3. Small neural network

```csharp
// y = mean(relu(x * w + b))
var x = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f }), requiresGrad: false);
var w = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new float[] { 0.5f, 0.5f, 0.5f }), requiresGrad: true);
var b = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new float[] { -1.0f, 0.0f, 1.0f }), requiresGrad: true);

var mul = GradOperations.Multiply(x, w);    // [0.5, 1.0, 1.5]
var add = GradOperations.Add(mul, b);       // [-0.5, 1.0, 2.5]
var relu = GradOperations.Relu(add);        // [0.0, 1.0, 2.5]
var mean = GradOperations.Mean(relu);       // 1.1667

mean.Backward();
Console.WriteLine(w.Grad[0]);  // 0.0 (relu blocked gradient at index 0: -0.5 < 0)
Console.WriteLine(w.Grad[1]);  // >0 (∂mean/∂w at index 1 = x[1]/3 = 2/3 ≈ 0.333)
Console.WriteLine(w.Grad[2]);  // >0 (∂mean/∂w at index 2 = x[2]/3 = 3/3 = 1.0)
```

### 4. Matrix multiplication

```csharp
var a = ReverseGradTensor<float>.FromMatrix(
    new float[] { 1, 2, 3, 4 }, rows: 2, cols: 2, requiresGrad: true);
var b = ReverseGradTensor<float>.FromMatrix(
    new float[] { 5, 6, 7, 8 }, rows: 2, cols: 2, requiresGrad: true);

var c = GradOperations.MatMul(a, b);  // 2x2 matrix product
var sum = GradOperations.Sum(c);      // scalar sum
sum.Backward();

// a.Grad = grad @ bᵀ  (grad = [1], so a.Grad = bᵀ)
Console.WriteLine(a.Grad[0]);  // 5
Console.WriteLine(a.Grad[1]);  // 7
Console.WriteLine(a.Grad[2]);  // 6
Console.WriteLine(a.Grad[3]);  // 8
```

### 5. SGD optimizer update

```csharp
var param = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f }), requiresGrad: true);

var loss = GradOperations.Sum(param);  // loss = 6
loss.Backward();                       // grad = [1, 1, 1]

var updated = SGD<float>.SgdUpdate(param, 0.1f);
// updated = param - 0.1 * grad = [0.9, 1.9, 2.9]
// updated.RequiresGrad == false
```

### 6. Nivara Frame integration

```csharp
using Nivara.AutoDiff.Extensions;

var frame = NivaraFrame.Create(
    ("Age", NivaraColumn<float>.Create(new float[] { 25, 30, 35 })),
    ("Income", NivaraColumn<float>.Create(new float[] { 50000, 70000, 90000 }))
);

// Convert to tensors
var tensors = frame.ToReverseGradTensors<float>(
    new[] { "Age", "Income" }, requiresGrad: true);

// Forward: compute a loss
var income = tensors["Income"];
var loss = GradOperations.Sum(income);

// Backward
tensors.BatchBackward(loss);

// Extract gradients back to a frame
var gradFrame = tensors.ToGradientFrame();
// Columns: Age (null if no grad), Income (grad = [1, 1, 1])

// Extract updated parameters
var updatedFrame = tensors.ToFrame();

// Zero gradients for next iteration
tensors.BatchZeroGrad();
```

### 7. Module-based training with TrainingLoop

```csharp
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Training;

// Define model
class LinearModel : Module<float>
{
    Linear<float> L1;
    public LinearModel()
    {
        L1 = new Linear<float>(3, 1);
        RegisterModules(L1);
    }
    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> x)
        => L1.Forward(x);
}

// Data
var frame = NivaraFrame.Create(
    ("x0", NivaraColumn<float>.Create([1.0f, 2.0f, 3.0f, 4.0f])),
    ("x1", NivaraColumn<float>.Create([2.0f, 3.0f, 4.0f, 5.0f])),
    ("x2", NivaraColumn<float>.Create([3.0f, 4.0f, 5.0f, 6.0f])),
    ("y", NivaraColumn<float>.Create([6.0f, 9.0f, 12.0f, 15.0f]))
));

var loader = new DataLoader<float>(
    new TensorDataset<float>(frame, ["x0", "x1", "x2"], "y"),
    batchSize: 2, shuffle: false);

// Train
var model = new LinearModel();
var optimizer = new SGD<float>(learningRate: 0.01f);
optimizer.AddParameterGroup(model.GetParameters().Values);

var loop = new TrainingLoop<float>(
    model, loader,
    (pred, target) => new MSELoss<float>().Forward(pred, target),
    optimizer,
    epochs: 5);

var result = loop.Run();       // TrainingResult<float>
result.PrintSummary();
// Epoch   1 | Loss:  ... | Batches:  2 | Time: 0.02s
// Epoch   2 | Loss:  ... | Batches:  2 | Time: 0.02s
```

### 8. Graph diagnostics

```csharp
var loss = ...; // from a computation graph
var info = ComputationGraph.GetGraphInfo(loss);
// { TotalNodes: 4, IsLeaf: false, RequiresGrad: true,
//   OperationCounts: { Multiply: 1, Add: 1, Relu: 1, Mean: 1 } }

var summary = GradientUtils.PrintGraphSummary(loss);
// Computation Graph Summary:
//   Total Nodes: 4
//   Is Leaf: False
//   Requires Grad: True
//   Operation Counts:
//     Multiply: 1
//     Add: 1
//     Relu: 1
//     Mean: 1

var description = GradientUtils.DescribeTensor(loss);
// ReverseGradTensor<Single>:
//   Length: 1
//   Requires Grad: True
//   Has Gradient: True
//   Is Leaf: False
//   Has Nulls: False
//   Gradient Norm: 1.000000
//   Operation: Mean
```

### 9. Gradient clipping

```csharp
// Per-value clipping
GradientUtils.ClipGradValue(tensor, maxValue: 1.0f);
// Each gradient element clamped to [-1.0, 1.0]

// Norm clipping
GradientUtils.ClipGradNorm(tensor, maxNorm: 5.0);
// Scales gradient if L2 norm > 5.0

// Global norm clipping across all parameters
GradientUtils.ClipGradNorm(new[] { w, b }, maxNorm: 5.0);
// Combines all gradients into one global norm, scales proportionally
```

### 10. Type conversion

```csharp
var floatTensor = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new float[] { 1.5f, 2.5f }), requiresGrad: true);

var doubleTensor = floatTensor.ToDouble();
// ReverseGradTensor<double> with values [1.5, 2.5], requiresGrad: true

var backToFloat = doubleTensor.ToFloat(requiresGrad: false);
// ReverseGradTensor<float> with values [1.5, 2.5], requiresGrad: false
```

### 11. Null propagation in gradients

```csharp
var a = new ReverseGradTensor<float>(
    NivaraColumn<float>.CreateFromNullable(new float?[] { 1.0f, null, 3.0f }),
    requiresGrad: true);
var b = new ReverseGradTensor<float>(
    NivaraColumn<float>.CreateFromNullable(new float?[] { 10.0f, 20.0f, null }),
    requiresGrad: true);

var result = GradOperations.Add(a, b);  // [11, null, null] (mask OR)
var sum = GradOperations.Sum(result);   // 11 (nulls skipped in sum)
sum.Backward();

Console.WriteLine(a.Grad[0]);  // 1.0  (∂sum/∂a₀)
Console.WriteLine(a.Grad[1]);  // null (output at position 1 is null, no gradient flows)
Console.WriteLine(a.Grad[2]);  // 1.0  (but output is null, so this may be masked)
```

### 12. Model serialization

```csharp
// Train
var loop = new TrainingLoop<float>(model, loader, lossFn, optimizer, epochs: 10);
var result = loop.Run();

// Save model
ModelSerializer.Save(model, "model.json");

// Load and run inference
var loaded = new LinearModel();
ModelSerializer.Load(loaded, "model.json");
loaded.Eval();
var prediction = loaded.Forward(testInput);
```

---

## Implementation Map

| Component | File |
|-----------|------|
| `GradTensor<T>` base class | `src/Nivara/AutoDiff/GradTensor.cs` |
| `ReverseGradTensor<T>` | `src/Nivara/AutoDiff/ReverseGradTensor.cs` |
| `OpNode<T>` | `src/Nivara/AutoDiff/OpNode.cs` |
| `ComputationGraph` | `src/Nivara/AutoDiff/ComputationGraph.cs` |
| `GradOperations` (all ops) | `src/Nivara/AutoDiff/Operations/GradOperations.cs` |
| `MatMulHelper` (SIMD MatMul) | `src/Nivara/Tensors/MatMulHelper.cs` |
| `SGD<T>.SgdUpdate` (static) | `src/Nivara/AutoDiff/Optimizer/SGD.cs` |
| `GradientUtils` | `src/Nivara/AutoDiff/Utilities/GradientUtils.cs` |
| `TypeValidator` | `src/Nivara/AutoDiff/Utilities/TypeValidator.cs` |
| `TypeConverter` | `src/Nivara/AutoDiff/Utilities/TypeConverter.cs` |
| `NivaraAutoGradExtensions` | `src/Nivara/AutoDiff/Extensions/NivaraAutoGradExtensions.cs` |
| Exception types | `src/Nivara/AutoDiff/Exceptions/AutoGradExceptions.cs` |
| `Parameter<T>` | `src/Nivara/AutoDiff/Nn/Parameter.cs` |
| `Module<T>` | `src/Nivara/AutoDiff/Nn/Module.cs` |
| `Linear<T>` | `src/Nivara/AutoDiff/Nn/Linear.cs` |
| `Sequential<T>` | `src/Nivara/AutoDiff/Nn/Sequential.cs` |
| `Activation<T>` / `Dropout<T>` | `src/Nivara/AutoDiff/Nn/Activation.cs` / `Dropout.cs` |
| Initializers (6) | `src/Nivara/AutoDiff/Nn/Initializers/*.cs` |
| Loss functions (7) | `src/Nivara/AutoDiff/Nn/Functional/*.cs` |
| `Optimizer<T>` base | `src/Nivara/AutoDiff/Optimizer/Optimizer.cs` |
| `SGD<T>` | `src/Nivara/AutoDiff/Optimizer/SGD.cs` |
| `Adam<T>` | `src/Nivara/AutoDiff/Optimizer/Adam.cs` |
| `AdamW<T>` | `src/Nivara/AutoDiff/Optimizer/AdamW.cs` |
| `TensorDataset<T>` | `src/Nivara/AutoDiff/Training/TensorDataset.cs` |
| `DataLoader<T>` | `src/Nivara/AutoDiff/Training/DataLoader.cs` |
| `Batch<T>` | `src/Nivara/AutoDiff/Training/Batch.cs` |
| `TrainingLoop<T>` | `src/Nivara/AutoDiff/Training/TrainingLoop.cs` |
| `DataParallelTrainer<T>` | `src/Nivara/AutoDiff/Training/DataParallelTrainer.cs` |
| `DataParallelTrainingResult<T>` | `src/Nivara/AutoDiff/Training/DataParallelResult.cs` |
| `ModelSerializer` | `src/Nivara/AutoDiff/Serialization/ModelSerializer.cs` |
| `Checkpoint<T>` | `src/Nivara/AutoDiff/Serialization/Checkpoint.cs` |
| Tests | `tests/Nivara.Tests/AutoDiff/*.cs` (13+ files) |
