# Plan: fix #574 TrainingLoop convergence test flake (unseeded Kaiming init)

Base branch: `khurram/571` (branch `khurram/574`)

## Problem

`Nivara.Tests.AutoDiff.TrainingTests.TrainingLoop_Run_CompletesEpochs` flakes in CI.
It asserts the last epoch's loss is strictly lower than the first
(`TrainingTests.cs:331`), but `Linear<float>(1, 1)` initializes its weight from
`KaimingUniformInitializer<T>.Instance`, which draws from the **unseeded**
`Random.Shared` (`src/Nivara/AutoDiff/Nn/Initializers/KaimingUniformInitializer.cs:25`).
The starting weight therefore differs per process, and the assertion can be violated
on an unlucky draw.

Confirmed by simulation of the exact test configuration (20 rows `f1=i`,
`label=2i+1`, batch 5, no shuffle, `MSELoss(Reduction.Sum)`, `SGD(0.001)`, 10 epochs):

- The optimum is w≈2.01 (bias stays 0, since no bias initializer is supplied).
- The last-vs-first comparison **fails whenever w₀ ∈ ≈[2.046, 2.449]** — about
  2–8% of Kaiming draws (bound = √6 ≈ 2.449 for fan-in 1). At those starting points
  the first-epoch loss is already tiny (≈1.6) and 10 epochs of SGD at lr=0.001 settle
  slightly above it.
- CI trace matches: `Expected: less than 1.61182761f` / `But was: 1.6521858f`.

Same defect class exists in the four sibling convergence tests in the same fixture
(all construct default-init `Linear` and assert `Epochs[^1].Loss < Epochs[0].Loss`):
`TrainingLoop_WithAdam_Converges` (:396), `TrainingLoop_MultiLayer_Converges` (:417),
`TrainingLoop_PartialBatch_TrainsToCompletion` (:566),
`TrainingLoop_MultiEpochShuffle_Converges` (:600).

## Root cause

- `Linear<T>.Linear(...)` initializes weight with
  `(weightInitializer ?? KaimingUniformInitializer<T>.Instance).Initialize(weight)`
  (`src/Nivara/AutoDiff/Nn/Linear.cs:62`).
- `KaimingUniformInitializer<T>` uses `Random.Shared.NextDouble()` — no seed.
- The model's `weightInitializer` parameter already exists — the intended extension
  point for a deterministic test.

## Proposed changes (tests only — no `src/Nivara` change)

Decision (user-confirmed): fix all five `TrainingTests` convergence tests; promote
`DeterministicInit<T>` to a shared helper; leave `DataParallelTests` / `NnTests` VAE
assertions for separate issues.

1. **Promote `DeterministicInit<T>`** (currently private in
   `tests/Nivara.Tests/AutoDiff/OptimizerTests.cs:1009`) to a shared internal helper
   `tests/Nivara.Tests/AutoDiff/DeterministicInit.cs`
   (namespace `Nivara.Tests.AutoDiff`). It fills values with `0.05 * (i + 1)`,
   preserves `RequiresGrad`/`Shape`, and replaces the tensor (bumps
   `Parameter.Version`). Update the 2 `OptimizerTests` call sites (:877, :881).

2. **Inject `new DeterministicInit<float>()` as `weightInitializer`** in the five
   `TrainingTests` convergence tests:
   - `TrainingLoop_Run_CompletesEpochs` (:321) — the named flaky test
   - `TrainingLoop_WithAdam_Converges` (:396)
   - `TrainingLoop_MultiLayer_Converges` (:417) — both `Linear`s inside `Sequential`
   - `TrainingLoop_PartialBatch_TrainsToCompletion` (:566)
   - `TrainingLoop_MultiEpochShuffle_Converges` (:600)
   Bias remains uninitialized (0), as today. `DataLoader` shuffle is already
   seed-deterministic (`new Random(seed + epoch)`, `DataLoader.cs:78`), so each test
   becomes fully reproducible.

3. Add a short comment in the named test explaining why the init is fixed
   (non-obvious design decision): w₀ near the optimum (~[2.05, 2.45]) violates the
   first-vs-last assertion, so the route must not depend on `Random.Shared`.

Safety margin at w₀=0.05: first-epoch loss ≈493, last ≈0.66 (~750×), far from the
failure band.

## Verification

- `dotnet test -c Release --filter "FullyQualifiedName~Nivara.Tests.AutoDiff.TrainingTests"`
  (targeted — run without asking).
- Loop `TrainingLoop_Run_CompletesEpochs` ~50× in Release capturing the process exit
  status each iteration — must be green every time, proving no `Random.Shared`
  dependence.
- `dotnet test -c Release --filter "FullyQualifiedName~Nivara.Tests.AutoDiff.OptimizerTests"`
  to confirm the promoted helper's existing callers still pass.
- Full suite only with explicit human go-ahead, carrying
  `--filter "Category!=Performance"`.

## Planned commits

1. `docs: plan #574 deterministic training tests in TODO.md`
2. `test(autodiff): promote DeterministicInit to shared helper for #574`
3. `test(autodiff): make TrainingTests convergence tests deterministic (#574)`

## Blast radius

- `tests/Nivara.Tests/AutoDiff/DeterministicInit.cs` — new shared internal helper.
- `tests/Nivara.Tests/AutoDiff/OptimizerTests.cs` — 2 call sites updated to reference
  the shared helper (same behavior, no assertion change).
- `tests/Nivara.Tests/AutoDiff/TrainingTests.cs` — 5 tests gain a deterministic
  `weightInitializer` argument; assertions unchanged.
- No production code touched. No public API change. No CHANGELOG entry needed.
- Covered by the targeted fixtures above; the promoted helper changes no behavior.

## GitHub issues log

- [ ] #574 — TrainingLoop convergence flake from unseeded Kaiming init (this fix).
- Out of scope, tracked separately by the issue author/repo:
  - `DataParallelTests` (`Run_MultipleChunks_LossDecreases`, `Run_Convergence_LinearRegression`)
    share the default-init + last<first pattern (Adam-based — low risk).
  - `NnTests` VAE `finalLoss < initialLoss` randomness.