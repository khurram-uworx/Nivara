# NivaraTorch — Per-Layer PyTorch ↔ Nivara Comparison

## What we're building

1. **`samples/NivaraTorch/gen_reference.py`** — PyTorch reference fixture generator covering ALL Nivara NN layer types
2. **`tests/Nivara.Tests/NivaraTorch/`** — NUnit functional tests with coded inputs validated against PyTorch outputs
3. **This plan** — the source of truth for future agents

## Purpose

Formal A/B validation of every NN layer type: PyTorch generates reference tensors, Nivara's C# implementation reproduces them to machine precision. Two levels of validation:

1. **Per-layer** — functional tests covering every module, activation, loss, and operation used across all samples
2. **Full-model** — MobileNetV2 and ResNet-18 inference logits compared between Python and C# (`compare`/`compare_diag` modes in NivaraInference)

These functional tests establish the correctness baseline before any performance or SIMD work. They use coded inputs (hardcoded float arrays) so they're self-contained and don't depend on fixture files at runtime.

All tests live in the existing `tests/Nivara.Tests/` project so they run alongside core unit tests — nothing gets missed during future performance work.

## Data layout

```
samples/
├── NivaraTorch/
│   └── gen_reference.py                    — generates all torch-comparison/ fixtures
├── data/
│   ├── mobilenet_v2/model.safetensors      — HuggingFace MobileNetV2 weights
│   ├── resnet18/model.safetensors          — HuggingFace ResNet-18 weights
│   ├── compare_input.bin                   — deterministic [1,3,224,224] float32 input (seed=42)
│   └── torch-comparison/                   — per-layer reference fixtures + manifest
│       ├── manifest.json                   — shape metadata for all test cases
│       ├── (existing Conv2d/BN2d/ReLU/Pool/Linear fixtures)
│       ├── conv1d_{k3,k5,k7,s2}_{input,weight,bias,output}.bin
│       ├── bn1d_{2d,3d}_{input,gamma,beta,running_mean,running_var,output}.bin
│       ├── embedding_{single,batch}_{input,weight,output}.bin
│       ├── dropout_eval_{input,output}.bin
│       ├── leaky_relu_{1d,4d}_{input,output}.bin
│       ├── sigmoid_{1d,4d}_{input,output}.bin
│       ├── tanh_{1d,4d}_{input,output}.bin
│       ├── rmsnorm_{input,gamma,output}.bin
│       ├── layernorm_{input,gamma,beta,output}.bin
│       ├── softmax_{input,output}.bin
│       ├── log_softmax_{input,output}.bin
│       ├── matmul_{a,b,output}.bin
│       ├── bce_with_logits_{sum,mean}_{input,target,output}.bin
│       ├── cross_entropy_{input,target,output}.bin
│       ├── mse_loss_{sum,mean}_{pred,target,output}.bin
│       └── l1_loss_{pred,target,output}.bin
└── NivaraInference/
    ├── (existing C# project)
    └── Python/
        └── (existing comparison scripts)

tests/Nivara.Tests/
└── NivaraTorch/
    ├── TestHelpers.cs                      — shared LoadBin, AssertTensorEqual, ExtractOutput
    ├── Conv2dTests.cs                      — existing 5 configs (migrated from PyTorchReferenceTests)
    ├── BatchNorm2dTests.cs                 — existing 3 eval configs
    ├── ReLUActivationTests.cs              — existing ReLU/ReLU6
    ├── MaxPool2dTests.cs                   — existing 2 configs
    ├── AdaptiveAvgPool2dTests.cs           — existing 2 configs
    ├── LinearTests.cs                      — existing 2 configs
    ├── Conv1dTests.cs                      — NEW: kernel 3/5/7, strided
    ├── BatchNorm1dTests.cs                 — NEW: eval 2D and 3D
    ├── EmbeddingTests.cs                   — NEW: vocab lookup
    ├── DropoutTests.cs                     — NEW: eval passthrough
    ├── ActivationTests.cs                  — NEW: LeakyRelu, Sigmoid, Tanh
    ├── NormalizationTests.cs               — NEW: RMSNorm, LayerNorm
    ├── LossTests.cs                        — NEW: BCEWithLogits, CrossEntropy, MSE, L1
    └── OperationTests.cs                   — NEW: Softmax, LogSoftmax, MatMul
```

## Layer coverage matrix

| Layer | Samples using it | Currently tested | Status |
|-------|-----------------|-----------------|--------|
| Conv2d | MobileNetV2, ResNet18 | YES (5 configs) | Existing — migrate to NivaraTorch/ |
| BatchNorm2d | MobileNetV2, ResNet18 | YES (3 eval) | Existing — migrate |
| ReLU/ReLU6 | All CNN samples | YES (1D, 4D) | Existing — migrate |
| MaxPool2d | ResNet18 | YES (2 configs) | Existing — migrate |
| AdaptiveAvgPool2d | MobileNetV2, ResNet18 | YES (2 configs) | Existing — migrate |
| Linear | All samples | YES (2 configs) | Existing — migrate |
| **Conv1d** | NivaraClassifier, NivaraTimeSeries | NO | **NEW — Task 3** |
| **BatchNorm1d** | NivaraTimeSeries | NO | **NEW — Task 4** |
| **Embedding** | NivaraClassifier, MicroGpt | NO | **NEW — Task 5** |
| **Dropout** (eval passthrough) | NivaraVAE, NivaraClassifier | NO | **NEW — Task 10** |
| **LeakyRelu** | NivaraVAE, NivaraTimeSeries | NO | **NEW — Task 6** |
| **Sigmoid** | Activation class | NO | **NEW — Task 6** |
| **Tanh** | Activation class | NO | **NEW — Task 6** |
| **RMSNorm** | MicroGpt | NO | **NEW — Task 7** |
| **LayerNorm** | TransformerBlock | NO | **NEW — Task 7** |
| **Softmax/LogSoftmax** | MicroGpt | NO | **NEW — Task 9** |
| **MatMul** | MicroGpt | NO | **NEW — Task 9** |
| **BCEWithLogitsLoss** | NivaraVAE | NO | **NEW — Task 8** |
| **CrossEntropyLoss** | NivaraClassifier | NO | **NEW — Task 8** |
| **MSELoss** | (exists) | NO | **NEW — Task 8** |
| **L1Loss** | (exists) | NO | **NEW — Task 8** |

## How fixtures were created

### Per-layer fixtures (`torch-comparison/`)

Generated by `samples/NivaraTorch/gen_reference.py`:

```bash
python samples/NivaraTorch/gen_reference.py
```

- Uses PyTorch CPU operations with `torch.manual_seed(42)` for deterministic inputs
- Sets explicit running stats for BatchNorm tests (not random)
- Saves raw float32 little-endian binary for each input, weight, bias, and output
- Writes `manifest.json` with shapes and parameters for each test case

### Full-model input (`compare_input.bin`)

Generated by `samples/NivaraInference/Python/generate_input.py`:

```bash
python samples/NivaraInference/Python/generate_input.py
```

Deterministic `[1,3,224,224]` random input (seed=42). Used by `compare` and `compare_diag` modes in both Python and C#.

### Model weights

Downloaded via HuggingFace CLI:

```bash
hf download google/mobilenet_v2_1.0_224 --local-dir samples/data/mobilenet_v2
hf download timm/resnet18.augreg_in1k --local-dir samples/data/resnet18
```

Both models use only float32 tensors. SafeTensorsLoader rejects BF16.

## Full-model comparison

### Python side

```bash
python samples/NivaraInference/Python/mobilenet_compare.py
python samples/NivaraInference/Python/resnet18_compare.py
python samples/NivaraInference/Python/mobilenet_diag.py
python samples/NivaraInference/Python/resnet18_diag.py
```

### C# side

```bash
dotnet build samples/NivaraInference/NivaraInference.csproj
dotnet run --project samples/NivaraInference -- mobilenet_v2 compare
dotnet run --project samples/NivaraInference -- resnet18 compare
dotnet run --project samples/NivaraInference -- mobilenet_v2 compare_diag
dotnet run --project samples/NivaraInference -- resnet18 compare_diag
```

### Benchmark

```bash
dotnet run --project samples/NivaraInference -- mobilenet_v2 benchmark
dotnet run --project samples/NivaraInference -- resnet18 benchmark
```

## Known results

Both models match Python logits to 6+ decimal places:

| Model | C# top-3 logits | Python top-3 logits |
|---|---|---|
| ResNet-18 | [0.707745, 2.909337, 2.399184] | [0.707746, 2.909338, 2.399184] |
| MobileNetV2 | [0.395821, 0.298235, 0.437236] | [0.395821, 0.298236, 0.437235] |

## Dependencies

### Python

```
torch>=2.0
numpy
```

### C#

Nivara core library only. Zero third-party dependencies.

---

# Task Breakdown

## Execution order

```
Task 1 (gen_reference.py) → Task 2 (test folder + helpers) → Task 3-10 (tests, parallel)
  → Task 11 (dedup, after tests are green)
```

Tasks 3-10 can be parallelized since they're independent test files. Task 11 must come last (needs test safety net).

## Suggested Agent Handout Batches

### Batch A: foundation (sequential)

- Task 1 (gen_reference.py expansion)
- Task 2 (test folder and helpers in tests/Nivara.Tests/NivaraTorch/)

### Batch B: functional tests (parallel)

- Task 3 (Conv1d)
- Task 4 (BatchNorm1d)
- Task 5 (Embedding)
- Task 6 (Activations)
- Task 7 (Normalization)
- Task 8 (Loss functions)
- Task 9 (Operations)
- Task 10 (Dropout)

### Batch C: cleanup (after B)

- Task 11 (Code deduplication)

## Task 1: Expand gen_reference.py to cover all layer types

### Priority

High

### Goal

Extend `samples/NivaraTorch/gen_reference.py` to generate PyTorch reference fixtures for every Nivara NN layer type, not just the 6 currently covered.

### Why this exists

The current script was written for NivaraInference only (Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear). Other samples use Conv1d, BatchNorm1d, Embedding, LeakyRelu, RMSNorm, LayerNorm, Softmax, MatMul, and loss functions — none have PyTorch reference data.

### Scope

- Add Conv1d fixtures: kernel sizes 3, 5, 7 (matching NivaraClassifier usage), with bias
- Add BatchNorm1d fixtures: eval mode, 2D input `[N, C]`, with running stats
- Add BatchNorm1d fixtures: eval mode, 3D input `[B, C, L]` (recently added 3D support)
- Add Embedding fixtures: vocab lookup correctness
- Add Dropout fixtures: eval mode passthrough (deterministic)
- Add LeakyRelu fixtures: slope=0.01 (matching NivaraVAE usage)
- Add Sigmoid fixtures: basic element-wise
- Add Tanh fixtures: basic element-wise
- Add RMSNorm fixtures: normalization correctness
- Add LayerNorm fixtures: normalization over last dim
- Add Softmax fixtures: normalization over dim
- Add LogSoftmax fixtures: log-softmax correctness
- Add MatMul fixtures: 2D matrix multiply
- Add BCEWithLogitsLoss fixtures: forward pass correctness
- Add CrossEntropyLoss fixtures: forward pass with class indices
- Add MSELoss fixtures: forward pass, reduceToMean=true/false
- Add L1Loss fixtures: forward pass
- Update manifest.json with all new test cases

### Constraints

- All inputs use `torch.manual_seed(42)` for determinism
- Save raw float32 binary + JSON manifest (same pattern as existing)
- Each fixture saves input(s), weight(s) where applicable, and output

### Suggested implementation path

- Group fixtures by layer type in the script
- Use small, verifiable shapes (e.g., Conv1d with `[1, 8, 16]` input, 8 output channels)
- For loss functions, save both predictions and targets alongside output
- For Embedding, save indices and weight matrix

### Acceptance criteria

- `python samples/NivaraTorch/gen_reference.py` runs without errors
- `samples/data/torch-comparison/manifest.json` contains entries for all 21+ layer types
- All `.bin` files are valid float32 binary

### Files likely involved

- `samples/NivaraTorch/gen_reference.py`
- `samples/data/torch-comparison/manifest.json` (regenerated)

## Task 2: Create NivaraTorch test folder and helpers

### Priority

High

### Goal

Create `tests/Nivara.Tests/NivaraTorch/` folder with shared test infrastructure and migrate the existing 18 PyTorch reference tests into organized files.

### Why this exists

The existing 18 tests live in `tests/Nivara.Tests/AutoDiff/PyTorchReferenceTests.cs` as a single 458-line file. Organizing them by layer type into a dedicated folder makes it easier to add new tests and keeps the test suite maintainable.

### Scope

- Create `tests/Nivara.Tests/NivaraTorch/` directory
- Create `TestHelpers.cs` with shared `LoadBin`, `AssertTensorEqual`, `ExtractOutput` helpers
- Migrate existing 18 tests from `PyTorchReferenceTests.cs` into:
  - `Conv2dTests.cs` (5 tests)
  - `BatchNorm2dTests.cs` (3 tests)
  - `ReLUActivationTests.cs` (4 tests — ReLU 1D/4D, ReLU6 1D/4D)
  - `MaxPool2dTests.cs` (2 tests)
  - `AdaptiveAvgPool2dTests.cs` (2 tests)
  - `LinearTests.cs` (2 tests)
- Remove the old `PyTorchReferenceTests.cs` (or keep as thin wrapper calling the new files)

### Constraints

- Reuses existing `tests/Nivara.Tests/Nivara.Tests.csproj` — no new project
- NUnit 4.x framework, `[Test]`, `Assert.That(...)` conventions
- Tests reference fixtures via relative path to `samples/data/torch-comparison/`
- ADR-001: `SetUp` creates `GradientUtils.Grad()` scope, `TearDown` disposes it

### Acceptance criteria

- `dotnet test --filter "FullyQualifiedName~NivaraTorch"` runs all 18 migrated tests
- All 18 tests pass
- `TestHelpers.cs` provides reusable infrastructure for Tasks 3-10

### Files likely involved

- `tests/Nivara.Tests/NivaraTorch/TestHelpers.cs` (new)
- `tests/Nivara.Tests/NivaraTorch/Conv2dTests.cs` (new)
- `tests/Nivara.Tests/NivaraTorch/BatchNorm2dTests.cs` (new)
- `tests/Nivara.Tests/NivaraTorch/ReLUActivationTests.cs` (new)
- `tests/Nivara.Tests/NivaraTorch/MaxPool2dTests.cs` (new)
- `tests/Nivara.Tests/NivaraTorch/AdaptiveAvgPool2dTests.cs` (new)
- `tests/Nivara.Tests/NivaraTorch/LinearTests.cs` (new)
- `tests/Nivara.Tests/AutoDiff/PyTorchReferenceTests.cs` (to be removed or thinned)

## Task 3: Conv1d functional tests

### Priority

High

### Goal

Add PyTorch-validated Conv1d tests covering kernel sizes 3, 5, 7 (matching NivaraClassifier usage).

### Why this exists

Conv1d is used by NivaraClassifier and NivaraTimeSeries but has zero PyTorch validation. The Conv1d implementation also lacks SIMD (unlike Conv2d) — these tests establish the correctness baseline before any performance work.

### Scope

- Conv1d kernel=3, stride=1, padding=1, with bias — input `[1, 8, 16]`, 8 output channels
- Conv1d kernel=5, stride=1, padding=2, with bias — input `[1, 8, 16]`, 16 output channels
- Conv1d kernel=7, stride=1, padding=3, with bias — input `[1, 4, 32]`, 8 output channels
- Conv1d kernel=3, stride=2, padding=1, with bias — input `[1, 8, 16]`, 16 output channels (strided)

### Acceptance criteria

- All 4 tests pass comparing C# output against PyTorch fixtures
- Tests use coded inputs (hardcoded float arrays) validated by the fixture generator

### Files likely involved

- `tests/Nivara.Tests/NivaraTorch/Conv1dTests.cs`
- `samples/NivaraTorch/gen_reference.py` (Conv1d section)

## Task 4: BatchNorm1d functional tests

### Priority

High

### Goal

Add PyTorch-validated BatchNorm1d tests for eval mode with running stats.

### Why this exists

BatchNorm1d is used by NivaraTimeSeries. BatchNorm1d and BatchNorm2d share nearly identical code — these tests validate the 1D path specifically, including the recently added 3D input support.

### Scope

- BatchNorm1d eval mode, 2D input `[4, 16]` with explicit running stats
- BatchNorm1d eval mode, 3D input `[2, 8, 20]` with explicit running stats

### Acceptance criteria

- Both tests pass against PyTorch fixtures

### Files likely involved

- `tests/Nivara.Tests/NivaraTorch/BatchNorm1dTests.cs`
- `samples/NivaraTorch/gen_reference.py` (BatchNorm1d section)

## Task 5: Embedding functional tests

### Priority

High

### Goal

Add PyTorch-validated Embedding tests.

### Why this exists

Embedding is used by NivaraClassifier and MicroGpt. Nivara's Embedding uses one-hot + MatMul (inefficient but correct) — these tests validate correctness.

### Scope

- Single token lookup: vocab=100, dim=16, token index=42
- Batch lookup: vocab=100, dim=16, batch of 4 token indices

### Acceptance criteria

- Both tests pass against PyTorch fixtures

### Files likely involved

- `tests/Nivara.Tests/NivaraTorch/EmbeddingTests.cs`
- `samples/NivaraTorch/gen_reference.py` (Embedding section)

## Task 6: Activation function tests (LeakyRelu, Sigmoid, Tanh)

### Priority

Medium

### Goal

Add PyTorch-validated tests for LeakyRelu (slope=0.01), Sigmoid, and Tanh.

### Why this exists

LeakyRelu is used by NivaraVAE and NivaraTimeSeries. Sigmoid and Tanh exist in the Activation class but have no validation.

### Scope

- LeakyRelu slope=0.01, 1D input `[32]` and 4D input `[1, 8, 4, 4]`
- Sigmoid, 1D input `[32]` and 4D input `[1, 8, 4, 4]`
- Tanh, 1D input `[32]` and 4D input `[1, 8, 4, 4]`

### Acceptance criteria

- All 6 tests pass against PyTorch fixtures

### Files likely involved

- `tests/Nivara.Tests/NivaraTorch/ActivationTests.cs`
- `samples/NivaraTorch/gen_reference.py` (activation section)

## Task 7: Normalization tests (RMSNorm, LayerNorm)

### Priority

Medium

### Goal

Add PyTorch-validated tests for RMSNorm and LayerNorm.

### Why this exists

RMSNorm is used by MicroGpt as the primary normalization. LayerNorm exists in the module system. Both have zero PyTorch validation.

### Scope

- RMSNorm: input `[4, 32]`, normalized shape `[32]`, eps=1e-5
- LayerNorm: input `[4, 32]`, normalized shape `[32]`, eps=1e-5

### Acceptance criteria

- Both tests pass against PyTorch fixtures

### Files likely involved

- `tests/Nivara.Tests/NivaraTorch/NormalizationTests.cs`
- `samples/NivaraTorch/gen_reference.py` (normalization section)

## Task 8: Loss function tests (BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, L1Loss)

### Priority

Medium

### Goal

Add PyTorch-validated tests for all loss functions used by samples.

### Why this exists

BCEWithLogitsLoss is used by NivaraVAE, CrossEntropyLoss by NivaraClassifier. MSELoss and L1Loss exist but are unvalidated.

### Scope

- BCEWithLogitsLoss: input `[4, 10]`, target `[4, 10]`, reduceToMean=true and false
- CrossEntropyLoss: logits `[4, 10]`, target indices `[0, 3, 7, 2]`
- MSELoss: pred `[4, 10]`, target `[4, 10]`, reduceToMean=true and false
- L1Loss: pred `[4, 10]`, target `[4, 10]`

### Acceptance criteria

- All 7 tests pass against PyTorch fixtures

### Files likely involved

- `tests/Nivara.Tests/NivaraTorch/LossTests.cs`
- `samples/NivaraTorch/gen_reference.py` (loss function section)

## Task 9: Softmax, LogSoftmax, MatMul tests

### Priority

Medium

### Goal

Add PyTorch-validated tests for Softmax, LogSoftmax, and MatMul operations.

### Why this exists

These are used by MicroGpt for attention computation. Zero PyTorch validation.

### Scope

- Softmax: 2D input `[4, 10]`, over dim=1
- LogSoftmax: 2D input `[4, 10]`, over dim=1
- MatMul: A `[4, 8]` x B `[8, 16]` -> `[4, 16]`

### Acceptance criteria

- All 3 tests pass against PyTorch fixtures

### Files likely involved

- `tests/Nivara.Tests/NivaraTorch/OperationTests.cs`
- `samples/NivaraTorch/gen_reference.py` (operation section)

## Task 10: Dropout eval-mode test

### Priority

Low

### Goal

Verify Dropout in eval mode passes input through unchanged.

### Why this exists

Dropout is used by NivaraVAE and NivaraClassifier. In eval mode, it should be identity.

### Scope

- Dropout eval mode: input `[4, 32]`, verify output == input exactly

### Acceptance criteria

- Test passes with exact equality (no tolerance needed)

### Files likely involved

- `tests/Nivara.Tests/NivaraTorch/DropoutTests.cs`
- `samples/NivaraTorch/gen_reference.py` (Dropout section)

## Task 11: Code deduplication (CopyToTemp, GetInputSpan)

### Priority

Medium

### Goal

Extract duplicated helper methods into shared utilities.

### Why this exists

`CopyToTemp` is duplicated in 5 files (Conv2d, ConvTranspose2d, Conv1d, MaxPool2d, AdaptiveAvgPool2d). `GetInputSpan`/`GetParamSpan` is duplicated in 3 files (BatchNorm1d, BatchNorm2d, LayerNorm).

### Scope

- Extract `CopyToTemp` into a shared internal static helper (e.g., `ModuleHelpers.cs`)
- Extract `GetInputSpan`/`GetParamSpan` into the same shared helper
- Update all call sites
- Verify all existing + new tests still pass

### Constraints

- Do this AFTER the functional tests are in place so we have a safety net
- Keep the helper internal (not public API)

### Acceptance criteria

- No duplicated `CopyToTemp` or `GetInputSpan`/`GetParamSpan` methods
- `dotnet test` passes (all existing + new tests)

### Files likely involved

- `src/Nivara/AutoDiff/Nn/ModuleHelpers.cs` (new)
- `src/Nivara/AutoDiff/Nn/Conv2d.cs`
- `src/Nivara/AutoDiff/Nn/ConvTranspose2d.cs`
- `src/Nivara/AutoDiff/Nn/Conv1d.cs`
- `src/Nivara/AutoDiff/Nn/MaxPool2d.cs`
- `src/Nivara/AutoDiff/Nn/AdaptiveAvgPool2d.cs`
- `src/Nivara/AutoDiff/Nn/BatchNorm.cs`
- `src/Nivara/AutoDiff/Nn/LayerNorm.cs`

---

## Final Checklist

- [ ] Every task has a clear owner-sized scope
- [ ] Every task has acceptance criteria
- [ ] Likely files are listed to reduce agent search time
- [ ] Execution order reflects real dependencies
- [ ] All 21 layer types have corresponding test tasks
- [ ] Code deduplication comes after test safety net is in place
- [ ] All tests in one place (`tests/Nivara.Tests/NivaraTorch/`)
- [ ] Python script at `samples/NivaraTorch/` (no subfolder)
