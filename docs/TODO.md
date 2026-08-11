# Plan: Unify AutoDiff weight access (#177)

**Branch:** `khurram/177` (off `main`)
**Issue:** https://github.com/khurram-uworx/Nivara/issues/177
**Review reference:** `docs/REVIEW.md` finding #6, breaking-change item 9.

## Problem

The AutoDiff NN modules expose the "weight parameter" concept through three
different member names and three different nullability contracts:

| Module | Current |
|---|---|
| `Linear<T>` | `Weight` (tensor) + `WeightParam` (parameter) |
| `Conv1d/2d/Transpose` | `WeightParam` / `BiasParam` only |
| `BatchNorm1d/2d`, `LayerNorm` | `Weight` / `Bias` (`Parameter<T>?`) |
| `Embedding<T>`, `SparseEmbedding<T>` | `Weight` (tensor) + `WeightParam` (parameter) |

Generic introspection, serialization, and optimizer wiring must branch per-module.

## Chosen contract (confirmed with user)

**`Weight` / `Bias` of type `Parameter<T>?` on every leaf module that has them.**
Null = parameter absent (`bias: false` / `affine: false`). Remove `WeightParam` /
`BiasParam` and the tensor-typed `Weight` accessors entirely. Consumers reach the
tensor via `Weight!.Tensor`. Matches PyTorch's `module.weight` / `module.bias`
mental model. Breaking change (2.0 candidate).

## Proposed changes

### 1. Module changes — `src/Nivara/AutoDiff/Nn/`

- `Linear.cs` — replace lines 17–19:
  ```csharp
  public Parameter<T>? Weight => weight;
  public Parameter<T>? Bias => bias;
  ```
  (drop `ReverseGradTensor<T> Weight => weight.Tensor;` and `Parameter<T> WeightParam => weight;`)
- `Embedding.cs` lines 15–16 and `SparseEmbedding.cs` lines 22–23:
  ```csharp
  public Parameter<T>? Weight => weight;
  ```
- `Conv1d.cs` lines 29–30, `Conv2d.cs` lines 37–38 and 660–661:
  ```csharp
  public Parameter<T>? Weight => _weight;
  public Parameter<T>? Bias => _bias;
  ```
- `BatchNorm.cs`, `LayerNorm.cs` — unchanged (already target shape).

No behavioral change: `Forward`, `GetParameters()`, `StateDict()`/`LoadStateDict()`
use private fields / parameter registry. State-dict keys remain `"Weight"`/`"Bias"`.

### 2. Call-site renames (mechanical)

- Tensor usages → `.Weight!.Tensor` / `.Bias!.Tensor`
  - `Linear<float>.Weight` (tensor): `NnTests`, `SerializationTests`,
    `LinearInferenceTests`, `LinearTransposedWeightCacheTests`, `AddBiasTests`,
    `TrainingTests` (`model.Weight[i]` → `model.Weight!.Tensor[i]`)
  - `Embedding<float>.Weight` (tensor): `NnTests` (`emb.Weight.Grad`),
    NivaraTorch `EmbeddingTests`
  - Samples: `MicroGptModel.cs` (`wte.Weight`), `NivaraGptModel.cs`
    (`tokenEmb.Weight`), `BatchedTransformer.cs` (`tokenEmb.Weight`)
- `WeightParam` → `Weight`; `BiasParam` → `Bias`
  - `Linear<float>.WeightParam`: `LinearInferenceTests`,
    `LinearTransposedWeightCacheTests`, `MicroGptModel.cs`
  - `Conv*.WeightParam` / `BiasParam`: `NnTests`, `ConvInferenceTests`,
    NivaraTorch `Conv1dTests`/`Conv2dTests`
  - `Embedding.WeightParam`: `NnTests`, NivaraTorch `EmbeddingTests`
- No change: `BatchNorm`/`LayerNorm` `Weight`/`Bias` (`Parameter<T>?`),
  `CrossFrameworkParityTests` (string state-dict keys),
  NivaraTorch `LinearTests` (LoadStateDict keys),
  `ModuleWithParams` test helper (already `Parameter<float>`).

### 3. New regression test

`tests/Nivara.Tests/AutoDiff/WeightAccessConsistencyTests.cs`:
- Reflection over the 9 leaf modules (closed over `float`): each exposes
  `Weight: Parameter<float>?`; no `WeightParam`/`BiasParam`/tensor-typed `Weight`.
- Instance checks: `bias: false` / `affine: false` ⇒ `Bias`/`Weight` null;
  defaults ⇒ non-null; mandatory-weight modules always non-null `Weight`;
  `Weight!.Tensor` matches `GetParameters()` registry entry.

### 4. Docs & changelog

- `docs/AUTODIFF.md` line 521 → unified contract note; add one-line contract
  note in the NN-modules area.
- `docs/REVIEW.md` — mark finding #6 and breaking-change item 9 done.
- `CHANGELOG.md` — `## [Unreleased]` → `### Changed` breaking entry.

## Blast radius

- **Core:** 9 module files in `src/Nivara/AutoDiff/Nn/` (definitions only; no
  other core file references these accessors).
- **Tests:** `NnTests.cs`, `SerializationTests.cs`, `LinearInferenceTests.cs`,
  `LinearTransposedWeightCacheTests.cs`, `ConvInferenceTests.cs`,
  `AddBiasTests.cs`, `TrainingTests.cs` (AutoDiff); `Conv1dTests.cs`,
  `Conv2dTests.cs`, `EmbeddingTests.cs` (NivaraTorch); new
  `WeightAccessConsistencyTests.cs`.
- **Samples:** `MicroGpt/MicroGptModel.cs`, `NivaraGpt/NivaraGptModel.cs`,
  `NivaraChat/Transformer/BatchedTransformer.cs`.
- **Docs:** `docs/AUTODIFF.md`, `docs/REVIEW.md`, `CHANGELOG.md`.
- **Unaffected:** parameter identity, state-dict keys, optimizer wiring,
  serialization, Python fixture generation.

## Verification

1. `dotnet build Nivara.slnx` after each change unit.
2. `dotnet test` (ask human first per AGENTS.md) on AutoDiff + NivaraTorch suites.
3. `git status` / `git diff` before each commit.

## Planned commits

1. `docs: plan #177 - unify AutoDiff weight access in TODO.md`
2. `refactor: unify Weight/Bias as Parameter<T>? across AutoDiff NN modules` (module files)
3. `refactor: update AutoDiff tests to Weight/Bias Parameter<T>? accessors` (test renames)
4. `refactor: update NivaraTorch suite to Weight/Bias Parameter<T>? accessors`
5. `refactor: update samples to Weight/Bias Parameter<T>? accessors`
6. `test: add WeightAccessConsistencyTests for uniform accessor shape`
7. `docs: document unified Weight/Bias contract (AUTODIFF, REVIEW, CHANGELOG)`

## GitHub issues log

- [ ] (none yet — create issues here as they are discovered during execution)
