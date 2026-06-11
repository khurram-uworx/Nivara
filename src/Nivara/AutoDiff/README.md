# Nivara.AutoDiff

`Nivara.AutoDiff` is an experimental reverse-mode automatic
differentiation layer over Nivara columns. It wraps `NivaraColumn<T>` values in
`ReverseGradTensor<T>`, records differentiable operations in a computation graph, and
computes gradients with `Backward()`.

The implementation is intentionally small and explicit. It is useful for
gradient-aware column workflows and ML experiments, but it is not a full
training framework.

## Supported Types

AutoDiff currently supports:

- `float`
- `double`

The public generic constraints use `INumber<T>`, but runtime validation rejects
other numeric types. Integral gradients are not supported.

## Core Concepts

- `ReverseGradTensor<T>` wraps a `NivaraColumn<T>` and carries `RequiresGrad`, `Grad`,
  graph attachment, conversion helpers, `Detach()`, `ZeroGrad()`, and
  `Backward()`.
- `Backward()` performs reverse-mode gradient computation. Without an explicit
  gradient argument, it can only be called on scalar tensors, usually the result
  of a reduction such as `Sum` or `Mean`.
- `OpNode` stores operation metadata, input tensors, and the backward function
  used during graph traversal.
- `ComputationGraph` attaches operation nodes to result tensors, validates graph
  traversal, runs the backward pass, clears gradients, and exposes graph
  inspection metadata.

## Basic Usage

```csharp
using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;

var input = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new[] { 1.0f, 2.0f, 3.0f }),
    requiresGrad: false);

var weight = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new[] { 0.5f, 0.5f, 0.5f }),
    requiresGrad: true);

var bias = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new[] { -1.0f, 0.0f, 1.0f }),
    requiresGrad: true);

var weighted = GradOperations.Multiply(input, weight);
var shifted = GradOperations.Add(weighted, bias);
var activated = GradOperations.Relu(shifted);
var loss = GradOperations.Mean(activated);

loss.Backward();

Console.WriteLine(loss[0]);
Console.WriteLine(weight.Grad![0]);
Console.WriteLine(bias.Grad![0]);
```

For non-scalar outputs, pass an explicit gradient tensor to `Backward()` with
the same length as the output.

## Operations

`Nivara.AutoDiff.Operations.GradOperations` exposes static methods:

- Arithmetic: `Add`, `Subtract`, `Multiply`, `Divide`
- Unary: `Negate`, `Abs`
- Reductions: `Sum`, `Mean`
- Activations: `Relu`, `Sigmoid`, `Tanh`
- Matrix helpers: `MatMul`, `Transpose`

`MatMul` and `Transpose` operate on row-major flattened tensor values with shape
metadata. Use `Reshape(rows, cols)` before calling the shape-aware overloads:

```csharp
var a = ReverseGradTensor<float>.FromArray(new[] { 1f, 2f, 3f, 4f }, requiresGrad: true);
var b = ReverseGradTensor<float>.FromArray(new[] { 5f, 6f, 7f, 8f }, requiresGrad: true);
a.Reshape(2, 2);
b.Reshape(2, 2);

var product = GradOperations.MatMul(a, b);
```

Legacy explicit-dimension overloads still exist for compatibility, but new code
should prefer shape metadata.

## Nivara Integration

`Nivara.AutoDiff.Extensions.NivaraAutoGradExtensions` provides helpers
for converting Nivara data structures:

- `NivaraColumn<T>.ToReverseGradTensor(requiresGrad)`
- `NivaraSeries<T>.ToReverseGradTensor(requiresGrad)`
- `NivaraFrame.ToReverseGradTensors<T>(columnNames, requiresGrad)`
- `NivaraFrame.ToReverseGradTensorsAuto(requiresGrad)`
- `Dictionary<string, ReverseGradTensor<T>>.BatchBackward(loss)`
- `Dictionary<string, ReverseGradTensor<T>>.BatchZeroGrad()`
- `Dictionary<string, ReverseGradTensor<T>>.ToFrame()`
- `Dictionary<string, ReverseGradTensor<T>>.ToGradientFrame()`

`ReverseGradTensor<T>` can also convert back to Nivara types with `ToColumn()` and
`ToSeries()`.

## Optimizer

`Nivara.AutoDiff.Optimizer.SgdOptimizer` provides a minimal SGD update step:

- `SgdUpdate<T>(parameter, learningRate)` — returns a new `ReverseGradTensor<T>` with values updated by `param - lr * grad`.
  - Returns a new tensor (caller owns it, can dispose the old one).
  - Skips positions where the gradient is null.
  - The returned tensor has `requiresGrad: false`.

```csharp
using Nivara.AutoDiff.Optimizer;

var loss = GradOperations.Sum(weight);
loss.Backward();
var updated = SgdOptimizer.SgdUpdate(weight, 0.01f);
```

## Utilities

`Nivara.AutoDiff.Utilities.GradientUtils` contains support helpers:

- Constants: `Constant`, `Zeros`, `Ones`, `Full`
- Gradient management: `ZeroGrad`, `Detach`
- Gradient clipping: `ClipGradValue`, `ClipGradNorm`
- Gradient norms: `GetGradientNorm`, `GetGlobalGradientNorm`
- Graph inspection: `GetGraphInfo`, `PrintGraphSummary`, `DescribeTensor`
- Checks: `HasGradient`, `CanBackward`

## Null Handling

AutoDiff operations preserve Nivara's explicit null-mask behavior. Null masks
flow through operation results, and gradient helpers skip null positions when
computing norms or clipping values. When adding new operations, keep null-mask
tests alongside forward and backward tests.

## Current Limitations

- Operations are static methods. There are no fluent `GradTensor<T>` methods or
  operator overloads yet.
- Matrix helpers use row-major flattened matrix conventions plus explicit shape
  metadata. Legacy explicit-dimension overloads remain for compatibility.
- `Backward()` defaults to a scalar-loss workflow. Non-scalar outputs require an
  explicit matching gradient.
- Optimizer: `Nivara.AutoDiff.Optimizer.SgdOptimizer.SgdUpdate` provides a minimal SGD update with null-skip support.
- There is no layer, model, metric, dataloader, or training-loop API.
- The implementation favors correctness and integration with current Nivara
  types over being a performance-final tensor runtime.

See [`../../../docs/AUTODIFF-GAPS.md`](../../../docs/AUTODIFF-GAPS.md) for
deferred gaps and [`../../../EXAMPLES.md`](../../../EXAMPLES.md) for a concise
user-facing example.

## Directory Structure

```text
AutoDiff/
├── README.md
├── GradTensor.cs
├── OpNode.cs
├── ComputationGraph.cs
├── IGradOperation.cs
├── IAutoGradNumeric.cs
├── AutoGradNumericTypes.cs
├── Exceptions/
│   └── AutoGradExceptions.cs
├── Extensions/
│   └── NivaraAutoGradExtensions.cs
├── Operations/
│   └── GradOperations.cs
├── Optimizer/
│   └── SgdOptimizer.cs
└── Utilities/
    ├── GradientUtils.cs
    ├── TypeConverter.cs
    └── TypeValidator.cs
```

All automatic differentiation functionality lives under the
`Nivara.AutoDiff` namespace and its `Operations`, `Utilities`,
`Exceptions`, and `Extensions` sub-namespaces.
