# AutoDiff Gaps — What Remains

This document catalogs future AutoDiff API, performance, testing, and design
work. Current bugs, correctness risks, and immediate technical debt are tracked
in `KNOWN-ISSUES.md`.

---

## API & Ergonomics Gaps

### E1. Matrix values still use row-major flattened storage

**Where:** `GradTensor<T>.Reshape`, `GradOperations.MatMul`, `GradOperations.Transpose`.

**Status:** Shape metadata is now available and shape-aware matrix overloads
exist, but callers still create flat arrays and then call `Reshape`.

**Recommendation:** Add named factory helpers such as
`ReverseGradTensor<T>.FromMatrix(T[] rowMajorData, int rows, int cols,
bool requiresGrad = false)` if matrix examples become common enough to justify
a clearer creation path.

---

### E2. Legacy explicit-dimension matrix overloads remain public

**Where:** `GradOperations.MatMul(a, b, aRows, aCols, bCols)` and
`GradOperations.Transpose(a, rows, cols)`.

**Status:** Shape-aware overloads are preferred, but explicit overloads remain
for existing callers and tests that need direct flattened conventions.

**Recommendation:** Keep the overloads for now. If the API is still pre-1.0
when matrix shape helpers stabilize, consider marking explicit overloads as
legacy or internalizing them.

---

### E3. No forward-mode or mixed-mode AutoDiff flavor yet

**Where:** `GradTensor<T>` / `ReverseGradTensor<T>` type hierarchy.

**Status:** The base `GradTensor<T>` is now intentionally separated from
reverse-mode graph behavior, but no `ForwardGradTensor<T>` or mixed-mode
abstraction exists.

**Recommendation:** Keep reverse-mode APIs explicitly named. Add a new forward
flavor only after tangent storage, operation coverage, and null-gradient policy
are documented.

---

### E4. No AutoGrid integration contract

**Where:** Future AutoGrid work.

**Status:** AutoDiff is explicit about reverse-mode tensors, but there is no
contract that defines how AutoGrid should consume gradients, tensors, parameter
updates, or null masks.

**Recommendation:** Before adding AutoGrid APIs, define ownership boundaries:
who owns tensors, how gradients are zeroed, whether updates mutate or return
new values, and how null positions flow through grid search or optimization.

---

### E5. No optimizer abstraction beyond SGD helper

**Where:** `SgdOptimizer`.

**Status:** `SgdUpdate` is a minimal return-new-tensor primitive. There are no
parameter groups, schedules, Adam/RMSProp variants, or stateful optimizer
objects.

**Recommendation:** Keep this gap open until a real training-loop workflow
requires stateful optimizer semantics.

---

## Performance Gaps

### P1. AutoDiff operations still allocate for null-mask reconstruction

**Where:** `GradOperations`, `GradientUtils`, `TypeConverter`, `SgdOptimizer`.

**Status:** Correctness currently takes priority. Null-preserving helpers often
materialize nullable arrays to rebuild `NivaraColumn<T>` masks.

**Recommendation:** Add internal column creation APIs that accept value spans
and null-mask spans directly, then replace nullable-array reconstruction where
it is hot.

---

### P2. Matrix multiplication is scalar-loop based

**Where:** `GradOperations.MatMulVectorized`.

**Status:** Null-aware matrix multiplication uses nested loops. This preserves
mask semantics but is not a BLAS-level or batched tensor kernel.

**Recommendation:** Benchmark first. For non-null `float`/`double` matrices,
consider a `TensorPrimitives.Dot` row/column path or another optimized kernel.

---

## Testing Gaps

### T1. No allocation benchmarks for AutoDiff hot paths

**Status:** Correctness tests cover null masks and gradients, but allocation
pressure is not measured.

**Recommendation:** Add benchmarks for repeated `Backward`, activation
gradients with nulls, `SgdUpdate`, and matrix operations.

---

### T2. No compatibility tests for legacy matrix overloads vs shape-aware overloads

**Status:** Both API shapes are tested, but there is no direct property-style
test asserting equivalent values, null masks, gradients, and shapes for the two
matrix call styles.

**Recommendation:** Add paired tests before changing or deprecating explicit
dimension overloads.

---

## Deferred Design Items

### D1. Broadcasting

Broadcasting remains out of scope. Add it only with explicit shape algebra,
null-mask semantics, and tests for ambiguous cases.

### D2. Operator overloads

Operators such as `+`, `-`, `*`, and `/` would improve ergonomics but need a
stable policy for shape compatibility, null propagation, and future AutoDiff
flavors.

### D3. Layer/model/training-loop APIs

AutoDiff should remain a low-level gradient layer until parameter ownership,
optimizer state, and AutoGrid integration are designed.

---

## Summary

| Area | Important | Deferred |
|------|-----------|----------|
| API/Ergonomics | E1, E2, E5 | E3, E4 |
| Performance | P1, P2 | |
| Testing | T1, T2 | |
| Design | | D1, D2, D3 |
