# Crazy Roadmap: DataFrames and operators over Nivara's fast tensor layer

This document is deliberately speculative. It captures a long-term direction: keep
Nivara's dense tensor, AutoDiff, and Silk kernels as the primitive execution
layer, while allowing `NivaraFrame`, `QueryFrame`, and `NivaraColumn` to become
the higher-level surface, in the same way ADO.NET's connections, commands,
buffers, and readers remain available beneath higher-level data APIs.

Nothing here replaces existing tensors. The low-level APIs stay fully available and
supported for exact, performance-critical, and cross-backend work.

## ADO.NET versus Entity Framework analogy

- **ADO-like / primitive layer:**
  `NivaraColumn`, `ReverseGradTensor`, `Module`, `KernelSelector`,
  `FusedExpressionEvaluator`, `TensorPrimitives`, Silk launcher methods, and
  OpenCL/Silk kernels.
- **EF-like / high-level layer:**
  `NivaraFrame`, `QueryFrame`, named columns, column operators, execution plans,
  streaming results, windowing, diagnostics, and export.
- **Migration target:**
  Users can start with frames and operators, then drop to tensors/Kernels when
  they need exact control or maximum performance.

## North star

A model or feature pipeline may eventually be expressed as a transformation
over named columns, while execution lowers to the same tensor and Silk machinery
that exists today.

```text
NivaraFrame
  -> encode/tokenize/embed columns
  -> attention or tabular operator
  -> logits/probability column
  -> loss
  -> backward
  -> optimizer update
```

The high-level API is expected to remain opt-in and experimental for a long time.

## Why Nivara is a plausible place for this

Nivara already has both halves, but they are separated.

### Primitive half

- `ReverseGradTensor`, modules, and optimizers.
- Column/tensor conversions and fused expression evaluation.
- `KernelSelector` and diagnostics.
- Silk/OpenCL kernels for BERT-family training and inference.
- Strong parity testing against reverse-mode AutoDiff.

### High-level half

- `NivaraColumn`, `NivaraFrame`, `QueryFrame`.
- Execution strategies: eager, streaming, lazy, and parallel.
- Window functions and fused expressions.
- Arrow, Parquet, CSV, JSON, and streaming interop.
- Diagnostics for plans and execution.

The missing ingredient is a carefully specified operator bridge between named
columns and differentiable tensors.

## Product instinct: start small and pragmatic

`NivaraClassifier` is a good first validation target. It already uses tabular
model flows and can be treated as a realistic "shape" for higher-level data APIs.
Instead of forcing the entire transformer world at once, use it to prove that
columns and tensors can cooperate without becoming a second query-engine fork.

A progressive path:

1. Make column operators excellent for common numeric and tensor work.
2. Add a small differentiable column operator gateway.
3. Use that gateway to expose one or two real model examples.
4. Add Silk-aware operator backends for kernels that are already proven.
5. Decide whether a full model-as-query API is worth its complexity.

## Design rules

1. Never duplicate tensor math just to make a frame convenience layer.
2. Lower column operators to existing tensor/function/kernels.
3. Preserve chunked/streaming semantics for large frames.
4. Resolve null-mask, padding, and missing-value semantics at boundaries.
5. Keep AutoDiff non-nullable internally; convert nullable columns at the domain
   boundary.
6. Keep Silk optional: if no GPU/OpenCL backend is available, fall back to
   TensorPrimitives/SIMD/CPU.
7. Make kernel and diagnostic visibility available through high-level operators.
8. Ensure primitives remain first-class APIs for advanced users.
9. No silent ownership transfer across CPU managed memory and Silk device
   buffers.
10. Prefer additive, opt-in APIs over large breaking changes.

## Three execution layers over time

### Layer A: column math gateway

Expose common column operations that delegate to the primitive layer.

Examples:

- `NivaraColumn.Add`, `Subtract`, `Multiply`, `Divide`
- `NivaraColumn.Exp`, `Gelu`, `Silu`, `Relu`, `Sigmoid`, `Tanh`
- `NivaraColumn.Sum`, `MinMax`, `Quantile`, `TopKDescending`
- elementwise binary operators such as `column * column`

These are useful even when no GPU or AutoDiff is involved. They should reuse
existing vectorized paths and null-mask propagation.

### Layer B: operator plans with backend choice

Wrap operator execution in a small plan/diagnostics layer.

- Each operator has a CPU/SIMD backend and may have a Silk backend.
- `KernelSelector` chooses the backend.
- Diagnostics report backend, shape, null-mask state, and elapsed time.
- Streaming and chunked execution remain explicit.

This layer is where the abstraction becomes genuinely useful: users see columns
and operators while developers keep control of dispatch.

### Layer C: differentiable columns and higher-order operators

Only after Layers A and B are solid:

- Add differentiable column operators for `MatMul`, `Attention`, `LayerNorm`,
  `Gelu`, `Softmax`, and similar model building blocks.
- Keep weight parameters as AutoDiff objects, not as frame rows.
- Produce parameter gradients in the same way existing modules do.
- Add facade kernels as backends, not as public API.

## Progressive edge for inference and tuning

The initial focus is BERT and Laya:

- **Inference:**
  Build column-aware preprocessing plus Silk-backed inference through existing
  runner seam.
- **Tuning:**
  Build a controlled, opt-in training operator for `SilkDistilBertTrainer`-style
  flows.
- **Fallback:**
  When no Silk device exists, use CPU/SIMD, `TensorPrimitives`, and existing
  AutoDiff.

This is the direct continuation of the Silk integration work: proof first, then
careful generalization.

## NivaraClassifier as the seed

NivaraClassifier is an attractive first user-facing example because:

- It already represents a model lifecycle: load, predict, train, benchmark.
- It can expose operator-level transformations without exposing OpenCL details.
- It can prove CPU and Silk fallback behavior from the same high-level API.
- Its use cases are feature-oriented and natural for columns and frames.

Possible first API shape:

```text
features
  -> Normalize(column)
  -> Predict(model, column)
  -> Explain(model, row)
```

Internally this can delegate to `Module`, `ReverseGradTensor`, and existing
model code.

## Data and tensor boundary rules

The most important engineering constraint is boundary clarity.

- Frames carry row-oriented named values, labels, masks, and metadata.
- AutoDiff carries gradient-enabled tensors in the model domain.
- Silk carries device buffers and kernel-specific layouts.
- Conversion must be explicit and diagnosable.

Recommended rules:

1. Columns convert to tensors at a documented method.
2. Tensor-to-frame conversion is explicit.
3. Nulls and masks are normalized before AutoDiff.
4. Silk buffers are never exposed as public column storage.
5. Device buffers use existing disposal patterns.
6. Shape and null-mask assertions are centralized in operators.

## Layer-by-layer implementation milestones

### Milestone 0: document + minimal API issue

- Create and link an umbrella issue.
- Define operator naming and result semantics.
- Decide which existing abstractions are authoritative.

### Milestone 1: column math gateway

- Add basic arithmetic/numeric operator methods.
- Add cleanup around expression engine selection.
- Add performance tests for large and sliced columns.
- Keep all fallback behavior unchanged.

### Milestone 2: bridge and diagnostics

- Add tensor/frame bridge methods.
- Record backend choices and shapes.
- Add cross-backend and null-mask tests.

### Milestone 3: Silk-aware high-level operators

- Add kernels and dispatch through `SilkLauncher`.
- Reuse proven Silk runtime creation and disposal.
- Start with GEMM, LayerNorm, and activation operators.
- Do not add custom unsafe kernel calls without diagnostics.

### Milestone 4: differentiable columns

- Introduce experimental operator graph for tiny models.
- Compare against existing AutoDiff modules.
- Add save/load and serialization contracts.

### Milestone 5: NivaraClassifier path

- Convert one classifier workflow to high-level operators.
- Measure and publish before/after profile.
- Retain primitive escape hatch.

## Measuring success

- No performance regression for existing primitive AutoDiff models.
- High-level operator path earns a measured benefit for target shapes.
- Diagnostics make CPU versus Silk selection explainable.
- Tests cover null masks, slicing, chunking, streaming, large columns, graph
  reuse, and device fallback.
- Expert users retain direct access to kernels and tensor APIs.

## Risks

1. Duplicate tensor math in a second framework instead of reuse.
2. Hidden device ownership and disposal defects.
3. Numerical parity drift between CPU and Silk.
4. Streaming changes producing incorrect gradients.
5. Frames becoming a performance bottleneck.
6. User-facing API complexity before stable contracts.

## What to avoid

- Treating this as a rewrite.
- Exposing OpenCL buffer details in high-level APIs.
- forcing Silk availability.
- Avoiding existing diagnostics.
- shipping model-as-query before column operators are robust.

## Closing thought

The ambition is worthwhile only if these higher-level constructs create less
friction than they cause. A useful way to proceed is to build just enough data
and operator abstractions to teach Nivara's primitives, then let usage determine
whether the larger model-as-query idea deserves full investment.