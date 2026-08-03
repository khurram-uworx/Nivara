# ADR-002: AutoDiff Keeps the NivaraColumn Span Boundary

**Status:** Accepted
**Date:** 2026-08-03

## Context

The original refactoring discussion plan (§2) proposed replacing
`GradTensor<T>.Data` (`NivaraColumn<T>`) with raw `Tensor<T>` storage from
System.Numerics.Tensors, and rewiring every op, module, optimizer, and
serializer onto `Tensor<T>`. Later, the plan's own "Observations" section
re-scoped that: keep `NivaraColumn<T>` as the accepted boundary, implement
internal kernels over spans, and only replace AutoDiff storage with raw
`Tensor<T>` if profiling proves the wrapper is the actual problem.

Since the original plan, the AutoDiff surface grew substantially:
`ReverseGradOperations.cs` is ~2373 lines and `ForwardGradOperations.cs` is
~783 lines, including Conv1d/Conv2d, BatchNorm, LayerNorm, RMSNorm,
MultiheadAttention, TransformerBlock, and VAE — all built on the column
boundary. The test suite is ~1948 tests. The operations already consume spans:
they call `TensorPrimitives` on contiguous column spans, so SIMD vectorization
is already realized regardless of the wrapper type.

Three options were considered (maintainer decision, 2026-08-03):

- **A. Span boundary** — keep `NivaraColumn<T>` as the I/O/ownership boundary,
  add a span-based `GradKernels` layer, wrap op results once.
- **B. Raw `Tensor<T>` backing** — full rewrite per the original §2.
- **C. Internal `T[]` + shape, column only at I/O** — middle ground.

## Decision

**Adopt Option A — span boundary.**

- `GradTensor<T>.Data` remains `NivaraColumn<T>` (public, with `ToColumn()`),
  as the boundary and ownership wrapper.
- New `src/Nivara/AutoDiff/Operations/GradKernels.cs` provides pure
  span-in/span-out kernels over generic `TensorPrimitives`
  (`where T : IFloatingPointIeee754<T>`).
- Operations compute over `T[]` / `ReadOnlySpan<T>` and wrap results once
  (`ResultTensor` pattern).
- `OpNode<T>.BackwardFunction` stays `Action<NivaraColumn<T>>`;
  `ComputationGraph.Backward` stays `NivaraColumn<T>?` — no delegate changes.
- Raw `Tensor<T>` backing (option B) is explicitly declined.
- Internal span access is added on the tensor types; `IsNull(int)` is removed
  from the public surface (meaningless in the non-null domain, ADR-001).

## Consequences

**Positive:**

- SIMD/vectorization is already in place (ops use `TensorPrimitives` on
  contiguous spans); the refactor's goodness comes from lean kernels, dead-code
  removal, and boundary enforcement — not a storage swap.
- Low regression risk: no changes to delegate/graph contracts, modules, or
  optimizer shapes.
- Testable in slices; the kernel layer is independently unit-testable.
- If profiling later shows the ~2 objects/op wrapper cost matters, a follow-up
  issue can swap storage to `Tensor<T>` without touching kernels (they are
  span-based).

**Negative:**

- AutoDiff is not literally "no NivaraColumn"; the column remains the accepted
  boundary. Its null machinery is never exercised inside the domain (ADR-001).
- A small per-op allocation remains (column + storage wrapper around each
  result array).

**Options declined:**

- **B** (raw `Tensor<T>` backing): high churn/risk across ~40 files for a
  purity win; the ops already consume spans. Revisit only if profiling shows
  the wrapper dominates op cost.
- **C** (internal `T[]` + shape): same churn magnitude as B with messier
  `ToColumn()` ownership semantics and less platform-tensor alignment.

## Amendment process

This decision is recorded so agents do not re-open the architecture fork. To
revisit, file an issue referencing ADR-002 with profiling evidence or a concrete
API-pressure case.
