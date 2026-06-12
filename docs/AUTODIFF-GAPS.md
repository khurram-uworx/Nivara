# AutoDiff Gaps — What Remains

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
| Optimizer API | `optimizer = Adam(model.parameters())` — unambiguous | Two `AddParameterGroup` overloads with different semantics — `Dictionary` overload creates copies, `IEnumerable<Parameter<T>>` passes references | Cost hours to debug; the dictionary version silently trains nothing |
| Error messages | Polished after a decade | Generic operation-node stack traces | Harder to debug autograd failures |
| Ecosystem | TensorBoard, torchinfo, etc. | `result.PrintSummary()` — minimal | Fine for small models, insufficient at scale |

**Root cause of the optimizer trap:** `AddParameterGroup(Dictionary<string,
ReverseGradTensor<T>>)` creates **new** `Parameter<T>` wrappers internally.
`Step()` then updates those copies, never touching the model's actual
`Parameter<T>` objects. The fix is to always use
`optimizer.AddParameterGroup(model.GetParameters().Values, ...)` (the
`IEnumerable<Parameter<T>>` overload). This is not obvious from the API
surface and the compiler offers no protection.

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

## Current Gaps

This document catalogs future AutoDiff API, performance, testing, and design
work. Current bugs, correctness risks, and immediate technical debt are tracked
in `KNOWN-ISSUES.md`.

---

## API & Ergonomics Gaps

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

## Performance Gaps

### P2. Null-mask propagation in matrix multiplication uses boolean loops

**Where:** `MatMulHelper.Multiply<T>` (null-aware) / `MatMulHelper.MultiplyCore<T>` (dense).

**Status:** The dense (non-null) path now uses `TensorPrimitives.Dot` + `Parallel.For` with
a transposed B buffer — efficient and near BLAS-level. The null-aware path fills nulls with
`T.Zero`, runs the dense kernel, then propagates the result mask via triple-nested boolean
loops. That mask propagation loop is the remaining bottleneck.

**Recommendation:** Profile the null-aware path to see whether the boolean mask
propagation dominates. For large matrices with sparse nulls, consider a column-scan or
bitmask-based propagation. The dense path needs no further work.

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


---

## Summary

| Area | Important | Deferred |
|------|-----------|----------|
| API/Ergonomics | | E3, E4 |
| Performance | P2 | |
| Testing | T1, T2 | |
| Design | | D1, D2 |
