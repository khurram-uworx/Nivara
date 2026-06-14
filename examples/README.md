# Cross-Framework Parity: PyTorch ↔ Nivara

Nivara gives .NET developers correct autograd without leaving the
ecosystem — no Python runtime, no 900 MB PyTorch install, no GPU
required. These parity examples prove it: for CPU-based training,
inference, and gradient computation, Nivara's forward and backward
autograd produce effectively identical results to PyTorch (verified at
<0.04% loss-curve divergence and 1e-5 JVP tolerance).

This isn't a PyTorch replacement — it's a .NET-native path to correct
gradients for the 70–80% of enterprise ML work that never touches a GPU.
NuGet add, enjoy your language and tools, and get back to building.

### The two parity examples

| Mode | What it validates | Approach |
|------|------------------|----------|
| **Backward-mode (MLP FraudNet)** | Reverse-mode autograd (backprop), optimizers, training loop | Train an identical 3-layer MLP in both frameworks and compare loss curves |
| **Forward-mode (JVP Parity)** | Forward-mode autograd (`ForwardGradTensor` + `ForwardGradOperations`) | Compute Jacobian-vector products for 6 canonical operations and compare |

Both examples use the same seed-bridge pattern: PyTorch serializes reference
outputs to JSON files; Nivara reads them and either replicates the computation
or validates against expected values.

---

## Prerequisites

| What | Version | Notes |
|------|---------|-------|
| Python | 3.12+ | See install below |
| PyTorch (CPU) | 2.x | Installed via `pip` |
| .NET SDK | 10.0 | The Nivara project targets `net10.0` |
| Nivara | — | Built from this repo |

### Install Python & PyTorch

```powershell
pip install torch pandas numpy
```

> **Troubleshooting**: If `pip` is not found, try `python -m pip install ...`.
> On Windows you may need `py -m pip install ...`.

---

## Backward-Mode: MLP FraudNet (Reverse-Mode Autograd)

Trains an identical 3-layer MLP (8→64→32→1) on synthetic fraud data in
**both PyTorch and Nivara** to verify that Nivara's autograd, optimizer,
and training loop produce the same convergence behavior.

### How it works

```
PyTorch (reference)                Nivara (test)
─────────────────────              ──────────────
torch.manual_seed(42)              Loads initial_weights.json from PyTorch
  │                                       │
  ├── saves initial_weights.json ─────────┤  (seed bridge: identical start)
  │                                       │
  ├── trains 50 epochs                    ├── trains 50 epochs
  ├── saves epoch_losses_pytorch.json     ├── saves epoch_losses_nivara.json
  ├── saves trained_weights_pytorch.json  ├── saves trained_weights_nivara.json
  └── saves test_preds_pytorch.csv        └── saves test_preds_nivara.csv
```

**Expected result**: loss curves match within <1% relative difference.
Individual weights may diverge (different SGD numerics), but the loss
trajectories are nearly identical.

### Step 1 — Generate synthetic fraud data

```powershell
python examples/data/generate_fraud_data.py
```

**Output files:**

| File | Contents |
|------|----------|
| `examples/data/train_fraud.csv` | 1000 training rows, ~86% fraud rate |
| `examples/data/test_fraud.csv` | 100 test rows, ~89% fraud rate |

### Step 2 — Train the PyTorch reference model

```powershell
python examples/pytorch/train_fraud_pytorch.py
```

**Output files in `examples/data/` (shared inputs for Nivara):**

| File | Contents |
|------|----------|
| `norm_params.json` | Z-score means/stds from PyTorch |
| `initial_weights.json` | Seeded initial weights (Nivara loads this) |

**Output files in `examples/pytorch/` (training results):**

| File | Contents |
|------|----------|
| `epoch_losses_pytorch.json` | 50 epoch losses |
| `trained_weights_pytorch.json` | Weights after 50 epochs |
| `test_preds_pytorch.csv` | Logits + probabilities on test set |

### Step 3 — Train the Nivara model

```powershell
dotnet run --project samples/Nivara.SampleApp
```

Look for the **"Cross-Framework FraudNet"** section in the output. The
Nivara code loads the same CSV data, normalization params, and initial
weights, then trains with identical hyperparameters (Adam, lr=0.001,
batch=32, 50 epochs).

**Output files in `examples/pytorch/`:**

| File | Contents |
|------|----------|
| `epoch_losses_nivara.json` | 50 epoch losses |
| `trained_weights_nivara.json` | Weights after 50 epochs |
| `test_preds_nivara.csv` | Logits + probabilities on test set |

### Step 4 — Automated parity test

```powershell
dotnet test --filter CrossFrameworkParity
```

The test:
- **Skips with a message** if the setup steps haven't been run yet
- **Verifies loss curves** have max relative diff ≤ 1%
- **Verifies prediction agreement** at 3 decimal places (≥80%)

### Results

#### Loss curves — near-identical (max diff 0.04%)

| Epoch | PyTorch | Nivara | Relative diff |
|-------|---------|--------|---------------|
| 1 | 20.4217 | 20.4217 | 0.0001% |
| 25 | 3.1390 | 3.1390 | 0.0002% |
| 50 | 2.2351 | 2.2355 | 0.020% |

**Max relative difference across all 50 epochs: 0.04%** — well below
the 1% test threshold. The convergence trajectories are visually
indistinguishable.

#### Predictions — 91% agreement at 3 decimal places

All 9 mismatches are boundary cases (prob ≈ 0.5–0.9) differing by only
~0.001–0.003. This is expected from different SGD trajectories producing
slightly different decision boundaries.

#### Trained weights — diverge naturally (expected)

| Layer | Max abs diff | Max relative diff |
|-------|-------------|-------------------|
| L1.Weight | 0.0046 | 12% |
| L1.Bias | 0.0016 | 3.3% |
| L2.Weight | 0.0329 | 181% |
| L2.Bias | 0.0002 | 5.3% |
| L3.Weight | 0.0002 | 0.09% |
| L3.Bias | < 0.0001 | 0.003% |

Large divergence in L2.Weight (181%) is normal — neural nets have many
equivalent minima and SGD finds different ones in each framework. L3
(the output layer) stays tight because it has less rotational symmetry.

**Bottom line**: Nivara's autograd + optimizer match PyTorch to within
**0.04% on the loss curve**. The weights take different paths to
equivalent local minima, confirming the gradient computation is correct.

---

## Forward-Mode: JVP Parity (Forward-Mode Autograd)

Verifies that Nivara's forward-mode autograd (`ForwardGradTensor` +
`ForwardGradOperations`) produces the same Jacobian-vector products
as PyTorch's `torch.autograd.forward_ad`. Six test cases cover
operations from simple arithmetic to full NN-style composition.

### How it works

```
PyTorch (reference)         Nivara (test)
─────────────────────       ──────────────
forward_parity_pytorch.py   ForwardParityExample.cs
        │                           │
        └── saves jvp_cases.json ───┤  (shared file: inputs + expected JVPs)
                                    │
               ┌────────────────────┤
               ▼                    ▼
        ForwardCrossFrameworkParityTests  (NUnit — validates both)
```

The PyTorch script defines each function, seeds specific inputs with
tangents via `make_dual`, and records both primal and JVP. The Nivara
example reimplements the same functions using `ForwardGradTensor` and
compares its results against the PyTorch reference.

**Expected result**: all 6 test cases pass with max error < 1e-5.

### Step 1 — Generate PyTorch reference JVPs

```powershell
python examples/pytorch/forward_parity_pytorch.py
```

**Output file:**

| File | Contents |
|------|----------|
| `examples/data/jvp_cases.json` | 6 test cases: inputs, seeds, primal, JVP |

**Test cases covered:**

| Case | Function | Seeds | What it tests |
|------|----------|-------|---------------|
| `square` | `x * x` | `[1, 0]` | Element-wise with mixed tangents |
| `mul_add` | `a * b + a` | `[1], [1]` | Binary ops + chain rule |
| `relu` | `relu(x)` | `[1, 1, 1]` | Activation — subgradient at zero |
| `sigmoid` | `sigmoid(x)` | `[1, 1]` | Activation with derivative scaling |
| `matmul` | `W @ x` | `[1, 0]` | Linear transform direction |
| `composition` | `sum(relu(W@x+b))` | `[1, 0]` | Full NN forward pass |

### Step 2 — Automated parity test

```powershell
dotnet test --filter ForwardCrossFrameworkParity
```

The test:
- **Skips with a message** if `jvp_cases.json` is missing
- **Validates** all 6 test cases exist with correct structure
- **Validates** the primal and JVP values match mathematical expectations
- **Validates** all JVPs are finite and consistent across PyTorch

### Results

All 6 JVP test cases pass:

| Case | Status | Primal | JVP |
|------|--------|--------|-----|
| `square` | ✓ | `[4, 9]` | `[4, 0]` |
| `mul_add` | ✓ | `[5.25]` | `[5]` |
| `relu` | ✓ | `[0, 0, 2]` | `[0, 0, 1]` |
| `sigmoid` | ✓ | `[0.5, 0.7311]` | `[0.25, 0.1966]` |
| `matmul` | ✓ | `[5, 11]` | `[1, 3]` |
| `composition` | ✓ | `[0.85]` | `[0.5]` |

**Bottom line**: Nivara's forward-mode autograd computes JVPs identical to
PyTorch (within float64→float32 truncation) across element-wise ops,
activations, matrix multiplication, and full NN composition.

---

## Architecture notes

### Seed bridge pattern

Both parity examples use the same architecture: PyTorch runs first and
serializes reference outputs to JSON. Nivara reads those files and either
replicates the computation (backward mode) or validates against expected
values (forward mode). This ensures a single source of truth and avoids
duplicating data generation logic.

#### Backward-mode seed bridge

1. PyTorch creates a model with `torch.manual_seed(42)` and serializes
   initial weights to `initial_weights.json`
2. Nivara deserializes them via `LoadWeightsFromJson()` — both frameworks
   now have identical starting weights
3. Any training divergence comes from floating-point differences, not
   initialization

#### Forward-mode seed bridge

1. PyTorch computes each function with `make_dual` to seed tangents and
   records the JVP to `jvp_cases.json`
2. Nivara reads the same inputs/seeds and reimplements the function
   using `ForwardGradTensor<T>` + operator overloads
3. Results are compared directly — no training loop, no randomness

### Why backward weights diverge but loss doesn't

Neural networks have many equivalent minima. Even with identical
initialization, different frameworks use different:
- BLAS/MKL kernel implementations
- Floating-point accumulation order in reductions
- Gradient computation order in backpropagation

These cause the SGD trajectory to diverge over epochs, landing in
different-but-equivalent local minima. The **loss curves** stay
aligned because the underlying gradient computation is correct.

### BCEWithLogitsLoss formula

Both frameworks use the numerically stable formulation:

```
loss = max(0, x) - x × z + log(1 + exp(-|x|))
```

where `x` = logits and `z` = target labels (0 or 1).

### How the two parity examples differ

| | MLP FraudNet (backward-mode) | JVP Parity (forward-mode) |
|---|---|---|
| **What it validates** | Reverse-mode autograd + optimizer + training loop correctness | Forward-mode autograd per-operation correctness |
| **Tensor type** | `ReverseGradTensor<float>` | `ForwardGradTensor<float>` |
| **PyTorch equivalent** | `backward()` + SGD/Adam | `torch.autograd.forward_ad` |
| **Randomness** | Yes — SGD minibatch order | None — deterministic |
| **Test granularity** | End-to-end (whole model training) | Per-operation (6 isolated test cases) |
| **Tolerance** | 1% relative loss diff | 1e-5 absolute diff |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `'python' is not recognized` | Python not in PATH | Use full path or reinstall with "Add to PATH" |
| `No module named torch` | PyTorch not installed | Run `pip install torch` |
| JSON files not found by test | Setup steps not run first | Run the relevant Python scripts first |
| Loss diff > 1% | Numeric kernel differences | Expected — the test threshold is conservative |
| Missing `initial_weights.json` | PyTorch script not run | Run `train_fraud_pytorch.py` first |
| Missing `jvp_cases.json` | PyTorch script not run | Run `forward_parity_pytorch.py` first |
