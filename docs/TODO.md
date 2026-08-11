# Plan: Dead public surface in AutoDiff (#178)

**Branch:** `khurram/178` (off `main`)
**Issue:** https://github.com/khurram-uworx/Nivara/issues/178
**Review reference:** `docs/REVIEW.md` finding #7, cleanup items 3 & 5.

## Problem

The AutoDiff subsystem exposes dead public surface:

1. `GradKernels` (`AutoDiff/Operations/GradKernels.cs`) is a `public static class`
   of span-level kernels (no AD semantics, no null guards) — internal machinery.
   Contrast: `AttentionKernels`, `BatchNormKernel<T>`, `LayerNormKernel<T>`,
   `RMSNormKernel<T>`, `ModuleHelpers<T>` are correctly internal.
2. `ComputationGraph` (`AutoDiff/ComputationGraph.cs`) is `public sealed class`
   with all members `internal static` — an assembly-internal service that leaked.
   Public graph introspection already lives on `GradientUtils`
   (`ZeroGrad`, `GetGraphInfo`, `PrintGraphSummary`, `DescribeTensor`).
3. 4 of 5 sealed exception types are never thrown:
   `GradientComputationException`, `CircularDependencyException`,
   `InvalidBackwardCallException`, `TypeValidationException` are constructed only
   in `ExceptionTests.cs`; the hot paths throw `InvalidOperationException` /
   `ArgumentException` and re-wrap. `AutoGradException` and
   `ShapeIncompatibilityException` are actually thrown.
4. Two generations of the initializer API coexist. Legacy static classes
   `KaimingNormal`/`KaimingUniform`/`XavierNormal`/`XavierUniform`/`Normal`/`Uniform`
   mutate `Dictionary<string, ReverseGradTensor<T>>` in place (breaking `Parameter<T>`
   identity); modules use the current `IInitializer<T>` instance API. Legacy set is
   used only by 4 tests in `NnTests.cs`.
5. `DefaultInitializers.Bias<T>()` returns null. The whole `DefaultInitializers`
   class is dead (`Weight<T>()` has zero callers too).

**Chosen exception strategy (confirmed with user):** delete the 4 never-thrown
sealed types; keep `AutoGradException` + `ShapeIncompatibilityException` and the
deliberate `InvalidOperationException`/`ArgumentException` hot-path contract.

## Proposed changes

### 1. Internalize `GradKernels` — `src/Nivara/AutoDiff/Operations/GradKernels.cs:16`

`public static class GradKernels` → `internal static class GradKernels`. Method
signatures untouched (minimal diff). Callers: `ReverseGradOperations`,
`ForwardGradOperations`, `AttentionKernels` (same assembly). Tests in `Nivara.Tests`
(`GradKernelsTests.cs`, `PerfTests.cs`) compile via existing `InternalsVisibleTo`
in `src/Nivara/Nivara.csproj`. Samples only mention it in READMEs.

### 2. Internalize `ComputationGraph` — `src/Nivara/AutoDiff/ComputationGraph.cs:7`

`public sealed class` → `internal sealed class`. No test/sample references it
directly (tests use `GradientUtils`).

### 3. Delete never-thrown exception types — `AutoDiff/Exceptions/AutoGradExceptions.cs`

- Delete `GradientComputationException` (97-173), `CircularDependencyException`
  (293-352), `InvalidBackwardCallException` (358-416), `TypeValidationException`
  (422-492). Keep `AutoGradException` (thrown in `TypeValidator`,
  `GradTensor`/`ForwardGradTensor`/`ReverseGradTensor`, `TensorDataset`) and
  `ShapeIncompatibilityException` (thrown in `TypeValidator.ValidateShapeCompatibility`).
  Base members (`OperationContext`, `InvolvedShapes`, `GetDetailedContext()`) stay —
  `ShapeIncompatibilityException` still uses them.
- `tests/Nivara.Tests/AutoDiff/ExceptionTests.cs`: delete tests covering removed
  types (`GradientComputationException_...` ×2, `CircularDependencyException_...` ×2,
  `InvalidBackwardCallException_...` ×2, `TypeValidationException_...` ×2,
  `ExceptionHierarchy_AllExceptionsInheritFromAutoGradException`,
  `ExceptionChaining_InnerExceptionPreserved`). Keep the 3 `AutoGradException` +
  3 `ShapeIncompatibilityException` tests.

### 4. Delete legacy static initializers — `AutoDiff/Nn/Initializers/`

- Delete `KaimingNormal.cs`, `KaimingUniform.cs`, `XavierNormal.cs`,
  `XavierUniform.cs`, `Normal.cs`, `Uniform.cs`.
- `tests/Nivara.Tests/AutoDiff/NnTests.cs`: delete 4 legacy tests
  (`KaimingUniform_InitializesWithCorrectShapes`, `XavierUniform_InitializesWithCorrectShapes`,
  `Normal_Initializer_ProducesNonNanValues`, `Uniform_Initializer_ProducesNonNanValues`).
  Current-API equivalents already exist (lines 387-465).
- Samples (MicroGpt, NivaraGpt) already use the instance API.

### 5. Delete `DefaultInitializers` — `AutoDiff/Nn/Initializers/DefaultInitializers.cs`

Delete the whole file — `Weight<T>()` is dead too (zero code references).
(If the human wants `Weight<T>()` kept, keep the file with only that method.)

### 6. Docs & changelog

- `docs/REVIEW.md`: resolve finding #7 with the "Resolved 2026-08-12 (issue #178)"
  convention; update inventory rows 319/322/325/326 (`ComputationGraph` internal,
  exceptions trimmed, initializers single-generation, `GradKernels` internal); mark
  cleanup items 3 & 5 done.
- `docs/AUTODIFF.md`: initializers table (800-810) → `*Initializer<T>` names + instance
  usage example; note `ComputationGraph` is internal (243-282) with `GradientUtils` as
  public surface; fix `ComputationGraph.GetGraphInfo(loss)` example (1445) →
  `GradientUtils.GetGraphInfo(loss)`; trim Exception Types table (1239-1246); update
  file maps (1549, 1585).
- `CHANGELOG.md`: `## [Unreleased]` entry (Changed) summarizing the surface cleanup.

## Blast radius

- **Core:** `Operations/GradKernels.cs`, `ComputationGraph.cs`,
  `Exceptions/AutoGradExceptions.cs`, 7 files in `Nn/Initializers/`. All other core
  files reference these symbols only from within the assembly.
- **Tests:** `AutoDiff/GradKernelsTests.cs`, `AutoDiff/PerfTests.cs` (internalized —
  IVT covers them), `AutoDiff/ExceptionTests.cs`, `AutoDiff/NnTests.cs`.
- **Samples:** none (code unaffected; READMEs mention `GradKernels`/`ComputationGraph`
  in prose only, not API usage).
- **Docs:** `docs/REVIEW.md`, `docs/AUTODIFF.md`, `CHANGELOG.md`.
- **Unaffected:** `Backward()` exception contract (keeps `InvalidOperationException`),
  public `GradientUtils` graph API, current initializer classes, optimizer/serialization.

## Verification

1. `dotnet build Nivara.slnx` after each change unit.
2. `dotnet test` (ask human first per AGENTS.md).
3. Grep for residuals before committing the final unit:
   `new (GradientComputationException|CircularDependencyException|InvalidBackwardCallException|TypeValidationException)`,
   `\.Init<T>(Dictionary`, `DefaultInitializers`, legacy class names in src/tests/samples.
4. `git status` / `git diff` before each commit.

## Planned commits

1. `docs: plan #178 - dead public surface in AutoDiff (TODO.md)`
2. `refactor: internalize GradKernels in AutoDiff`
3. `refactor: internalize ComputationGraph in AutoDiff`
4. `refactor: delete never-thrown AutoGrad exception types`
5. `test: remove legacy-initializer tests from NnTests`
6. `refactor: delete legacy static initializer classes`
7. `refactor: delete DefaultInitializers`
8. `docs: document #178 surface cleanup (REVIEW, AUTODIFF, CHANGELOG)`

## GitHub issues log

- [ ] (none yet — create issues here as they are discovered during execution)
