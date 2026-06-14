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

## Current Gaps

This document catalogs future AutoDiff API, performance, testing, and design
work. Fixed issues are reflected in source, tests, and the current API docs.

---

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

## Summary

| Area | Important | Deferred |
|------|-----------|----------|
| Testing | T1, T2 | |
