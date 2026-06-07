# AutoDiff Suggestions

This document replaces the historical design framing in
[`IDEA-AUTODIFF.md`](IDEA-AUTODIFF.md) with a current-state review of the
AutoDiff code that exists in the repo today.

Actionable implementation follow-ups live in
[`AUTODIFF-TASKS.md`](AUTODIFF-TASKS.md).

`IDEA-AUTODIFF.md` is still useful as design context, but it reads like a
pre-implementation sketch. The implementation now lives in
`src/Nivara.Extensions/AutoDiff`, has samples, and has a focused test suite.

## What Exists Today

- `GradTensor<T>` wraps `NivaraColumn<T>` and adds `RequiresGrad`, `Grad`,
  graph attachment, `Backward()`, `Detach()`, `ZeroGrad()`, and conversion
  helpers.
- Reverse-mode AD is implemented through `OpNode` and `ComputationGraph`.
  `Backward()` starts from a scalar result unless the caller provides an
  explicit gradient tensor for non-scalar outputs.
- Supported numeric types are intentionally limited to `float` and `double`
  through `TypeValidator`.
- `GradOperations` currently supports element-wise `Add`, `Subtract`,
  `Multiply`, `Divide`, matrix `MatMul`, `Transpose`, reductions `Sum` and
  `Mean`, and activations `Relu`, `Sigmoid`, and `Tanh`.
- Nivara integration exists through extension methods for columns, series, and
  selected frame workflows: `ToReverseGradTensor`, `ToReverseGradTensors`,
  `ToReverseGradTensorsAuto`, `BatchBackward`, `BatchZeroGrad`, `ToFrame`, and
  `ToGradientFrame`.
- Utility support exists for constants, zeros, ones, full tensors, gradient
  clipping, gradient norms, graph summaries, and backward eligibility checks.
- Null masks are preserved across AutoDiff operations in the same spirit as
  core Nivara column operations.
- `samples/Nivara.SampleApp/AutoDiffExample.cs` demonstrates reductions,
  activations, a small neural-network-like expression, gradient utilities, and
  graph inspection.
- The AutoDiff test subset currently passes: `150` tests under
  `FullyQualifiedName~AutoDiff`.

## What Is Still Rough

- `src/Nivara.Extensions/AutoDiff/README.md` now documents the implemented
  surface, but it should stay in sync as new operations and integration helpers
  are added.
- The user-facing operation style is static and verbose:
  `GradOperations.Multiply(a, b)` rather than `a.Multiply(b)` or `a * b`.
- `MatMul` and `Transpose` operate on flattened `GradTensor<T>` values with
  explicit row and column arguments. That is workable, but it is easy to misuse
  because shape metadata is not carried by `GradTensor<T>`.
- There is no optimizer abstraction yet. Samples show gradient computation and
  mention parameter updates, but users still perform update steps manually.
- There is no layer, parameter, model, loss, metric, dataloader, or `Fit` API.
  That is fine for the current scope, but docs should not imply a Keras-like
  stack exists today.
- Some internal helpers allocate temporary arrays to access column data or pass
  gradients through object columns. That is acceptable for a first functional
  layer, but it should not be presented as a performance-final tensor runtime.
- Type support is `float` and `double` only. This should stay explicit because
  gradients for integral types are not generally meaningful.
- Null semantics are preserved, but the gradient policy around null positions
  should be kept documented and tested for every new operation.

## Suggested Next Steps

1. Keep the AutoDiff README current.
   The README has been refreshed with the current API surface, a small usage
   example, supported types, operation list, null policy, and known gaps.

2. Keep `EXAMPLES.md` current.
   `EXAMPLES.md` includes a concise AutoDiff example that uses the actual
   `Nivara.Extensions.AutoDiff` API and points readers to the sample app for the
   longer walkthrough. Keep aspirational optimizer, layer, and training-loop
   ideas out of examples until those APIs exist.

3. Add a simple scalar-loss documentation example.
   The best first example is a small expression such as
   `mean(relu(x * weight + bias))`, because it exercises a graph, reduction,
   scalar `Backward()`, and parameter gradients without inventing a training
   framework.

4. Improve shape ergonomics before expanding matrix-heavy examples.
   Either add a small shape-aware wrapper for AutoDiff matrix values or provide
   named factory helpers that make flattened matrix conventions explicit.

5. Add optimizer primitives after gradient behavior is fully documented.
   Start with `Sgd` or a minimal `ParameterUpdate` helper. Keep it separate from
   a higher-level model API until parameter ownership and disposal semantics are
   settled.

6. Add performance follow-ups for allocation-heavy paths.
   Review temporary array creation in AutoDiff helpers, especially column data
   extraction and gradient conversion. Prefer existing Nivara zero-copy or
   pooled-buffer patterns where they are safe with null masks.

7. Expand operation coverage only with tests.
   Each new differentiable operation should include forward-value tests,
   backward-gradient tests, null-mask tests, and shape/error tests.

## Not Goals Yet

- GPU or accelerator backends.
- Keras-like `Model.Compile()` or `Fit()`.
- Implicit broadcasting.
- Broad numeric type support beyond `float` and `double`.
- Replacing `System.Numerics.Tensors` or core Nivara storage.

## Recommended Documentation Position

Describe AutoDiff as an experimental but implemented extension layer for
reverse-mode differentiation over Nivara columns. It is useful today for small
gradient-aware column workflows and as a foundation for ML experiments, but it
is not yet a full deep-learning framework.
