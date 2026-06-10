# Nivara Automatic Differentiation (AutoDiff)

Nivara provides a PyTorch-inspired reverse-mode automatic differentiation engine built on top of its columnar DataFrame types. Tensors wrap `NivaraColumn<T>` and the computation graph is a DAG of operation nodes built implicitly during forward operations.

---

## Architecture

```
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
├── Element-wise: Add, Subtract, Multiply, Divide
├── Matrix: MatMul, Transpose
├── Reductions: Sum, Mean
├── Activations: Relu, Sigmoid, Tanh
├── Unary: Negate, Abs
└── ...vectorized via TensorPrimitives where available

SgdOptimizer                          ← SGD update with null-skip semantics

GradientUtils                         ← Utility functions
├── ZeroGrad, Detach (single + batch)
├── ClipGradValue, ClipGradNorm (single + global)
├── Constant/Zeros/Ones/Full creators
├── GetGraphInfo, PrintGraphSummary, DescribeTensor
├── HasGradient, GetGradientNorm, GetGlobalGradientNorm
└── CanBackward

TypeValidator                         ← Runtime type checking (only float/double)
TypeConverter                         ← Cross-type tensor conversion (float ↔ double)

NivaraAutoGradExtensions              ← NivaraColumn/NivaraSeries/NivaraFrame ↔ ReverseGradTensor
```

---

## Key Design Principles

- **Only `float` and `double` are supported** — enforced at runtime by `TypeValidator.ValidateNumericType<T>()`. Other numeric types (int, long, etc.) throw `TypeValidationException`.
- **1D storage, shape metadata** — data is always stored as a flat `NivaraColumn<T>`. Shape is metadata (`int[] shape`) with `Reshape()` validation. Default shape is `[Length]`.
- **Computation graph is built implicitly** — every `GradOperations` call checks if any input `requiresGrad` and attaches an `OpNode` to the result.
- **Gradient accumulation** — `AccumulateGradient()` either sets or adds to `Grad` (supports fan-in from multiple paths).
- **Explicit null-mask propagation** — Nivara's nullable semantics flow through gradients. Nulls propagate via mask OR; null positions in gradients are skipped during accumulation.
- **IDisposable** — `GradTensor<T>` implements `IDisposable` to release underlying column data.

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

### Matrix

| Op | Forward | Backward Rule | Requirements |
|----|---------|---------------|--------------|
| `MatMul(a, b)` | `a @ b` | `∂/∂a = grad @ bᵀ`, `∂/∂b = aᵀ @ grad` | Both tensors rank 2; `a.Cols == b.Rows` |
| `Transpose(a)` | `aᵀ` | `∂/∂a = gradᵀ` | Rank 2 |

MatMul uses a triple-loop kernel (not `TensorPrimitives`). Both tensor shape and explicit-parameter overloads are available (the explicit overload is marked `[Obsolete]`).

### Reductions

| Op | Forward | Backward Rule | Notes |
|----|---------|---------------|-------|
| `Sum(a)` | `∑a` | `broadcast(grad, n)` — fills gradient value to all positions | Expects scalar output |
| `Mean(a)` | `(∑a)/n` | `broadcast(grad/n, n)` — fills gradient/n to all positions | Expects scalar output |

Reductions use `NivaraSeries<T>.Sum()` / `.Average()` for the forward pass.

### Activations

| Op | Forward | Backward Rule | Vectorization |
|----|---------|---------------|---------------|
| `Relu(a)` | `max(a, 0)` | `grad * (1 if a > 0 else 0)` | `TensorPrimitives.Max` for forward; manual loop for grad |
| `Sigmoid(a)` | `σ(a) = 1/(1+e⁻ᵃ)` | `σ(a) * (1-σ(a)) * grad` | Manual loop via `Math.Exp` |
| `Tanh(a)` | `tanh(a)` | `(1 - tanh²(a)) * grad` | Manual loop via `Math.Tanh` |
| `Negate(a)` | `-a` | `-grad` | `TensorPrimitives.Negate` |
| `Abs(a)` | `\|a\|` | `sign(a) * grad` | `TensorPrimitives.Abs` for forward; manual loop for grad |

### Vectorization Strategy

All operations follow a two-path approach:

1. **No-null fast path**: Extract `ReadOnlySpan<T>` via `TryGetSpan()`, call `TensorPrimitives` kernel, return `NivaraColumn<T>.Create(result)`.
2. **Null-aware fallback**: Rent buffers from `ArrayPool<T>.Shared`, call `CopyTo()` to fill buffers with `T.Zero` for null positions, run `TensorPrimitives` kernel, merge null masks via OR, return `NivaraColumn<T>.CreateFromSpans(result, nullMask)`.

Operations that lack `TensorPrimitives` support (e.g., Sigmoid, Tanh, MatMul) use manual loops in both paths.

---

## Null Handling

Nivara's explicit null-mask semantics flow through all gradient operations:

- **Null in input** → propagates to both forward result and gradient masks (mask OR semantics)
- **Null in gradient** → `AccumulateGradient` skips the position entirely (no zeroing)
- **Null in SGD** → if parameter is null → stays null; if gradient is null → no update at that position (parameter unchanged)
- **Null in MatMul** → position-level null: if any contributing element is null, the corresponding output position is null

The `MergeNullMasks(a, b, destination)` helper performs the OR operation, handling cases where one or both inputs lack null masks.

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
| `ClipGradValue(tensor, maxValue)` | Clips each gradient element to `[-maxValue, maxValue]` |
| `ClipGradNorm(tensor, maxNorm)` | Scales gradient if L2 norm exceeds `maxNorm` |
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

| Method | Description |
|--------|-------------|
| `GetGraphInfo(tensor)` | Delegates to `ComputationGraph.GetGraphInfo` |
| `PrintGraphSummary(tensor)` | Human-readable graph summary string |
| `DescribeTensor(tensor)` | Detailed tensor debug info (length, grad norm, operation, etc.) |
| `HasGradient(tensor)` | Whether `Grad != null` |
| `GetGradientNorm(tensor)` | L2 norm of gradient |
| `GetGlobalGradientNorm(tensors)` | Combined L2 norm across tensors |
| `CanBackward(tensor)` | Whether `Backward()` can be called (scalar + requiresGrad) |

---

## SGD Optimizer

```csharp
public static class SgdOptimizer
{
    public static ReverseGradTensor<T> SgdUpdate<T>(ReverseGradTensor<T> parameter, T learningRate)
        where T : struct, INumber<T>
}
```

**Behavior:**
- `parameter = parameter - learningRate * gradient`
- Returns a new `ReverseGradTensor` with `requiresGrad = false` (the update step is not differentiable)
- If parameter is null at a position → stays null
- If gradient is null at a position → parameter unchanged at that position
- Throws if `Grad` is null (call `Backward()` first) or learning rate is non-positive

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

### IAutoGradNumeric\<T\>

Marker interface with static abstract members:

| Member | Description |
|--------|-------------|
| `Zero` | Zero value for the type |
| `One` | One value for the type |
| `FromDouble(double)` | Creates from double |
| `ToDouble(T)` | Converts to double |

Concrete implementations: `Float32` (`IAutoGradNumeric<float>`) and `Float64` (`IAutoGradNumeric<double>`).

### IGradOperation\<T\>

Interface for implementing custom operations:

| Member | Description |
|--------|-------------|
| `Name` | Operation name for debugging |
| `Forward(inputs)` | Forward pass returning result tensor |
| `Backward(gradOutput, inputs, output)` | Backward pass computing gradients |

The built-in `GradOperations` use internal helpers rather than this interface, but it's available for custom operation plugins.

---

## Nivara Frame Integration

The `NivaraAutoGradExtensions` class (in `Nivara.Extensions.AutoDiff.Extensions`) provides conversion between Nivara types and autograd tensors:

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

### 1. Basic scalar gradient

```csharp
var a = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new float[] { 3.0f }), requiresGrad: true);
var b = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new float[] { 4.0f }), requiresGrad: true);

var result = GradOperations.Add(a, b);  // 7.0
result.Backward();

Console.WriteLine(a.Grad[0]);  // 1.0  (∂result/∂a)
Console.WriteLine(b.Grad[0]);  // 1.0  (∂result/∂b)
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

var updated = SgdOptimizer.SgdUpdate(param, 0.1f);
// updated = param - 0.1 * grad = [0.9, 1.9, 2.9]
// updated.RequiresGrad == false
```

### 6. Nivara Frame integration

```csharp
using Nivara.Extensions.AutoDiff.Extensions;

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

### 7. Graph diagnostics

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

### 8. Gradient clipping

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

### 9. Type conversion

```csharp
var floatTensor = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new float[] { 1.5f, 2.5f }), requiresGrad: true);

var doubleTensor = floatTensor.ToDouble();
// ReverseGradTensor<double> with values [1.5, 2.5], requiresGrad: true

var backToFloat = doubleTensor.ToFloat(requiresGrad: false);
// ReverseGradTensor<float> with values [1.5, 2.5], requiresGrad: false
```

### 10. Null propagation in gradients

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

---

## Implementation Map

| Component | File |
|-----------|------|
| `GradTensor<T>` base class | `src/Nivara.Extensions/AutoDiff/GradTensor.cs` |
| `ReverseGradTensor<T>` | `src/Nivara.Extensions/AutoDiff/ReverseGradTensor.cs` |
| `OpNode<T>` | `src/Nivara.Extensions/AutoDiff/OpNode.cs` |
| `ComputationGraph` | `src/Nivara.Extensions/AutoDiff/ComputationGraph.cs` |
| `GradOperations` (all ops) | `src/Nivara.Extensions/AutoDiff/Operations/GradOperations.cs` |
| `IGradOperation<T>` interface | `src/Nivara.Extensions/AutoDiff/IGradOperation.cs` |
| `IAutoGradNumeric<T>` interface | `src/Nivara.Extensions/AutoDiff/IAutoGradNumeric.cs` |
| `Float32` / `Float64` structs | `src/Nivara.Extensions/AutoDiff/AutoGradNumericTypes.cs` |
| `SgdOptimizer` | `src/Nivara.Extensions/AutoDiff/Optimizer/SgdOptimizer.cs` |
| `GradientUtils` | `src/Nivara.Extensions/AutoDiff/Utilities/GradientUtils.cs` |
| `TypeValidator` | `src/Nivara.Extensions/AutoDiff/Utilities/TypeValidator.cs` |
| `TypeConverter` | `src/Nivara.Extensions/AutoDiff/Utilities/TypeConverter.cs` |
| `NivaraAutoGradExtensions` | `src/Nivara.Extensions/AutoDiff/Extensions/NivaraAutoGradExtensions.cs` |
| Exception types | `src/Nivara.Extensions/AutoDiff/Exceptions/AutoGradExceptions.cs` |
| Tests | `tests/Nivara.Tests/AutoDiff/*.cs` (8 files) |
