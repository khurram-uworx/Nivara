# Plan — issue #583: same flake class as #574 (unseeded Kaiming init in convergence assertions)

> **Reminder (per iterative-work):** as each task executes, if deferred work or a concern
> surfaces that is outside this plan, create a GitHub issue immediately
> (`gh issue create --repo khurram-uworx/Nivara`) and record its number in the
> **GitHub issues log** below. Do not rely on memory or wait until the end of the plan —
> compaction can lose it.

## Problem

Issue #583 is the leftover of #574. Both are the same flake class: a test asserts that
training **reduces loss**, but the model's weights come from
`KaimingUniformInitializer<T>.Instance`, which draws from the unseeded `Random.Shared`.
The starting weight therefore varies per process, and for a shallow model the
"last < first loss" assertion can invert on an unlucky draw.

#574 (commit `a8960112`) fixed five `TrainingTests` sites by injecting the shared test
helper `DeterministicInit<float>` (`tests/Nivara.Tests/AutoDiff/DeterministicInit.cs`)
through `Linear<T>`'s existing `weightInitializer:` argument. Four more sites of the same
defect class remain:

- `tests/Nivara.Tests/AutoDiff/DataParallelTests.cs`
  - `Run_MultipleChunks_LossDecreases` (model line 59, assert line 68)
  - `Run_Convergence_LinearRegression` (model line 76, assert line 85)
  - `Run_Deterministic_SameSeedSameResults` (model line 99, asserts lines 115–116)
- `tests/Nivara.Tests/AutoDiff/NnTests.cs` — `VAE_Training_ReducesLoss`
  (model line 687, assert line 721)

**Issue text correction:** #583 names "VAE ELBO test (line ~648)". Line 648 is
`VAE_ElboLoss_ReturnsScalar`, which has no convergence assertion. The test that asserts
`finalLoss < initialLoss` is `VAE_Training_ReducesLoss` (lines 685–723).

**Additional instance not listed in #583** (confirmed same class, included by decision):
`tests/Nivara.Tests/AutoDiff/BFloat16Tests.cs` — `LinearModule_BFloat16SgdTraining_LossDecreases`
(`new Linear<BFloat16>(1, 1)` line 147, assert `losses[^1] < losses[0]` line 168).

### Root cause mechanism (evidence)

- `src/Nivara/AutoDiff/Nn/Linear.cs:62` — `(weightInitializer ?? KaimingUniformInitializer<T>.Instance).Initialize(weight)`.
- `src/Nivara/AutoDiff/Nn/Initializers/KaimingUniformInitializer.cs:14,25` — singleton draws `Random.Shared.NextDouble()`.

### VAE-specific complications

1. `VAE<T>`'s constructor has **no initializer hook** (`src/Nivara/AutoDiff/Nn/VAE.cs:36-77`);
   it `new`s up six `Linear<T>` layers internally (`VAE.cs:58-65`). Threading
   `DeterministicInit` requires a constructor parameter.
2. Even with deterministic weights there is a **second** randomness source: the test calls
   `vae.Reparameterize(mu, logVar)` with no seed → `ModuleHelpers.Reparameterize`
   (`src/Nivara/AutoDiff/Nn/ModuleHelpers.cs:122-131`) → `ReverseGradOperations.SampleNormal(..., seed: null)`
   (`src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs:2666-2705`) → `GenerateStandardNormal<T>(n, null)`,
   again `Random.Shared`-backed. `initialLoss` and `finalLoss` are measured with independent
   noise draws, so the comparison can invert even for a well-trained model.
   `Reparameterize` already accepts `seed:` and sampling is deterministic per seed
   (pinned by `GradOperationsTests.SampleNormal_SameSeed_Deterministic`).

## Proposed changes

### 1. Production — `src/Nivara/AutoDiff/Nn/VAE.cs`

- Add `using Nivara.AutoDiff.Nn.Initializers;` (for `IInitializer<T>`).
- Append two optional constructor parameters after `conditionDim`, preserving positional
  source-compatibility with all 13 existing call sites:

  ```csharp
  int conditionDim = 0,
  IInitializer<T>? weightInitializer = null,
  IInitializer<T>? biasInitializer = null
  ```

  with matching XML doc entries.
- Forward both into all six `new Linear<T>(...)` calls (`VAE.cs:58-65`), mirroring
  `Linear<T>`'s own parameter names.

### 2. Tests — `tests/Nivara.Tests/AutoDiff/DataParallelTests.cs` (lines 59, 76, 99)

Add `weightInitializer: new DeterministicInit<float>()` to the three `Linear<float>(1, 1)`
sites, with #574's explanatory comment on the first. Leave the other five DataParallel
tests unchanged (they assert `Loss > 0` / `GradientNorm > 0`, independent of the init draw).

### 3. Tests — `tests/Nivara.Tests/AutoDiff/NnTests.cs`, `VAE_Training_ReducesLoss`

- Model (line 687) → `new VAE<float>(4, 2, 8, weightInitializer: new DeterministicInit<float>())`.
- Add `seed: 42` to both `vae.Reparameterize(mu, logVar)` call sites (lines 701 and 713),
  so `initialLoss` and `finalLoss` use identical epsilon. Keep the exact
  `finalLoss < initialLoss` assertion.

### 4. Tests — `tests/Nivara.Tests/AutoDiff/BFloat16Tests.cs` (line 147)

`new Linear<BFloat16>(1, 1, weightInitializer: new DeterministicInit<BFloat16>())`.
`BFloat16` is the .NET 11 framework type (satisfies `IFloatingPointIeee754<T>`), so
`DeterministicInit<BFloat16>` compiles; at the `0.05` start the loss drops sharply on the
first optimizer step.

### 5. Docs — `docs/AUTODIFF.md` (lines 613 and 798)

Update the commented VAE constructor signature to include the new optional initializer
args. These fenced blocks are **not** gated (`<!-- gate -->` absent), so the snippet gate
is unaffected.

## Blast radius

- `VAE<T>` constructor: 13 existing call sites (12 in `NnTests.cs`, 1 in
  `OpNodeInputContractTests.cs`) plus samples. All use positional or named
  (`beta:`/`conditionDim:`) arguments; appending optional parameters is source-compatible.
  No public-API baseline test exists in the repo.
- `Linear<T>` / `KaimingUniformInitializer<T>` are unchanged.
- Test-only changes confined to three AutoDiff test files.
- Non-AutoDiff convergence assertions were checked and are not this class:
  `CrossFrameworkParityTests` reads pre-generated JSON and `Assert.Ignore`s when absent;
  `NnTests.ConvVAE_EndToEnd_ReducesLoss` already uses a tolerance.

## Verification steps

1. Targeted Release runs, repeated (~10–20 iterations) to demonstrate determinism and
   absence of flake (especially the parallel `DataParallelTrainer` path):
   - `dotnet test -c Release --filter "FullyQualifiedName~DataParallelTests"`
   - `dotnet test -c Release --filter "FullyQualifiedName~VAE_Training_ReducesLoss"`
   - `dotnet test -c Release --filter "FullyQualifiedName~LinearModule_BFloat16SgdTraining_LossDecreases"`
2. Confirm the VAE deterministic trajectory satisfies `final < initial`. Fallback if it
   does not: more epochs / slightly higher lr — not loosening the assertion.
3. Full suite only with explicit human confirmation, always with `--filter "Category!=Performance"`.

## Planned commits

1. `feat(autodiff): add optional weight/bias initializers to VAE<T> constructor`
2. `test(autodiff): remove unseeded-init flake from DataParallel/VAE/BFloat16 convergence tests`
3. `docs(autodiff): document VAE<T> initializer constructor parameters`

## GitHub issues log

- [ ] (none yet)
