# Cross-Framework MLP Parity: PyTorch ↔ Nivara

This example trains an identical 3-layer MLP (8→64→32→1) on synthetic fraud
data in **both PyTorch and Nivara** to verify that Nivara's autograd,
optimizer, and training loop produce the same convergence behavior.

## How it works

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

---

## Prerequisites

| What | Version | Notes |
|------|---------|-------|
| Python | 3.12+ | See install below |
| PyTorch (CPU) | 2.x | Installed via `pip` |
| .NET SDK | 10.0 | The Nivara project targets `net10.0` |
| Nivara | — | Built from this repo |

---

## Step 1 — Install Python & PyTorch

If you don't have Python 3.12+, download it from
[python.org](https://www.python.org/downloads/). During installation,
check **"Add Python to PATH"**.

Open a terminal and verify:

```powershell
python --version
# Should print: Python 3.12.x
```

Install PyTorch (CPU-only, ~800 MB):

```powershell
pip install torch pandas numpy
```

> **.NET users**: `pip` is Python's package manager (like NuGet). The
> command above installs PyTorch (CPU), pandas (DataFrame library),
> and numpy (numerical arrays). We use CPU-only because this example
> is tiny — no GPU needed.

> **Troubleshooting**: If `pip` is not found, try `python -m pip install ...`.
> On Windows, you may need `py -m pip install ...` if both Python 2
> and 3 are installed.

---

## Step 2 — Generate synthetic fraud data

Run the data generator from the repo root:

```powershell
python examples/data/generate_fraud_data.py
```

**Output files:**

| File | Contents |
|------|----------|
| `examples/data/train_fraud.csv` | 1000 training rows, ~86% fraud rate |
| `examples/data/test_fraud.csv` | 100 test rows, ~89% fraud rate |

**What to observe**: The script prints the fraud rate for train and test
sets. The rates are high because this is deliberately easy synthetic data
(clearly separable) to make convergence fast and reliable.

---

## Step 3 — Train the PyTorch reference model

```powershell
python examples/pytorch/train_fraud_pytorch.py
```

This script:
1. **Loads** the CSV data and computes z-score normalization
2. **Saves** `norm_params.json` (means/stds for each feature) to `examples/data/`
3. **Creates** a 3-layer MLP with `torch.manual_seed(42)` and `np.random.seed(42)`
4. **Saves** `initial_weights.json` — the seed-bridge file (Nivara loads this) — to `examples/data/`
5. **Trains** 50 epochs with Adam (lr=0.001, batch=32, BCEWithLogitsLoss)
6. **Saves** trained weights, loss curve, and test predictions

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

**What to observe**: Watch the loss descend — it starts around 20 and
finishes near 2.2. The model converges cleanly.

---

## Step 4 — Train the Nivara model

```powershell
dotnet run --project samples/Nivara.SampleApp
```

This runs all sample app demos. Look for the **"Cross-Framework FraudNet"**
section. The Nivara code:

1. **Loads** the same CSV data + normalization params
2. **Loads** `initial_weights.json` for identical weight initialization
3. **Builds** an identical 3-layer MLP (`FraudNet : Module<float>`)
4. **Trains** with the same hyperparameters (Adam, lr=0.001, batch=32, 50 epochs)
5. **Saves** output files to `examples/pytorch/`

**Output files:**

| File | Contents |
|------|----------|
| `epoch_losses_nivara.json` | 50 epoch losses |
| `trained_weights_nivara.json` | Weights after 50 epochs (same JSON format) |
| `test_preds_nivara.csv` | Logits + probabilities on test set |

**What to observe**: The loss values should closely match PyTorch's.
At epoch 1, both should output `≈20.42`. At epoch 50, both should be
`≈2.2`. Small differences in the 4th-5th decimal place are expected.

---

## Step 5 — Compare results manually

### Loss curves

Save the following as `compare_losses.py` in the repo root and run it:

```python
import json
pyt = json.load(open('examples/pytorch/epoch_losses_pytorch.json'))
niv = json.load(open('examples/pytorch/epoch_losses_nivara.json'))
for i, (p, n) in enumerate(zip(pyt, niv), 1):
    diff = abs(p - n) / max(abs(p), 1e-10) * 100
    print(f'Epoch {i:2d}: Py={p:.4f}  Ni={n:.4f}  rel_diff={diff:.4f}%')
max_rel = max(abs(p-n)/max(abs(p),1e-10)*100 for p,n in zip(pyt,niv))
print(f'\nMax relative difference: {max_rel:.4f}%')
```

```powershell
python compare_losses.py
```

**Expected**: max relative diff < 1%.

### Predictions

Save the following as `compare_preds.py` in the repo root and run it:

```python
import pandas as pd
pyt = pd.read_csv('examples/pytorch/test_preds_pytorch.csv')
niv = pd.read_csv('examples/pytorch/test_preds_nivara.csv')
agreement = (pyt['prob'].round(3) == niv['prob'].round(3)).mean()
print(f'Prediction agreement at 3 decimal places: {agreement:.1%}')
```

```powershell
python compare_preds.py
```

**Expected**: >80% agreement. Boundary cases near prob≈0.5 may round
differently due to small weight differences — this is normal.

---

## Step 6 — Automated parity test

A NUnit test validates the comparison automatically:

```powershell
dotnet test --filter CrossFrameworkParity
```

The test:
- **Skips with a message** if the setup steps (Python + Nivara) haven't
  been run yet (JSON files not found)
- **Verifies loss curves** have max relative diff ≤ 1%
- **Verifies prediction agreement** at 3 decimal places (≥80%)

This test is the canonical comparison — if it passes, the frameworks
produce equivalent training behavior.

---

## Results

The table below shows the actual measured parity on our reference run
(50 epochs, batch 32, Adam lr=0.001, BCEWithLogitsLoss).

### Loss curves — near-identical (max diff 0.04%)

| Epoch | PyTorch | Nivara | Relative diff |
|-------|---------|--------|---------------|
| 1 | 20.4217 | 20.4217 | 0.0001% |
| 25 | 3.1390 | 3.1390 | 0.0002% |
| 50 | 2.2351 | 2.2355 | 0.020% |

**Max relative difference across all 50 epochs: 0.04%** — well below
the 1% test threshold. The convergence trajectories are visually
indistinguishable.

### Predictions — 91% agreement at 3 decimal places

All 9 mismatches are boundary cases (prob ≈ 0.5–0.9) differing by only
~0.001–0.003. This is expected from different SGD trajectories producing
slightly different decision boundaries.

### Trained weights — diverge naturally (expected)

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

## Architecture notes

### Seed bridge (`initial_weights.json`)

Both frameworks start from identical weights so that any divergence
comes from training (not initialization). The flow:

1. PyTorch creates a model with `torch.manual_seed(42)`
2. PyTorch serializes the initial weights to JSON
3. Nivara deserializes them via `LoadWeightsFromJson()`
4. Both frameworks now have identical starting weights

### Why weights diverge but loss doesn't

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

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `'python' is not recognized` | Python not in PATH | Use full path or reinstall with "Add to PATH" |
| `No module named torch` | PyTorch not installed | Run `pip install torch` |
| JSON files not found by test | Setup steps not run first | Run Steps 2, 3, and 4 before Step 6 |
| Loss diff > 1% | Numeric kernel differences | Expected — the test threshold is conservative |
| Missing `initial_weights.json` | PyTorch script not run | Run Step 3 first |
