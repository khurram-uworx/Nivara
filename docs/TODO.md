# Plan: #753 Phase 1 + 2 — mixed-precision Embedding<TModel,TWeight> + serialization alignment

## Problem
`Linear<TModel,TWeight>` mixed-precision primitive exists, but `Embedding` is still single-typed (`Embedding<T>`), and cannot register narrow weights inside a `Module<TModel>`. Model-class wiring is tracked separately in #758 (Phase 3).

## Proposed changes

### Phase 1 — `Embedding<TModel, TWeight>`
- Convert `src/Nivara/AutoDiff/Nn/Embedding.cs` to `Embedding<TModel, TWeight> : Module<TModel>`:
  - `Parameter<TWeight> weight`, registered via `RegisterParameterObject` (same type) or `RegisterMixedParameters` (mixed).
  - Normal init over `TWeight`.
  - `Forward(ReverseGradTensor<TModel>)`, `Forward(int[])`, `Forward(int)` gather from `TWeight` and widen to `TModel`.
  - Manual autograd scatter-add (TModel accumulator → narrow to TWeight).
  - Mixed-aware `StateDict()` / `LoadStateDict()`.
- Keep `sealed Embedding<T> : Embedding<T, T>` for compatibility.

### Phase 2 — serialization alignment
- Keep `StateDict` shape `ReverseGradTensor<TModel>` for compatibility; Embedding overrides widen/narrow Weight.

## Verification steps
- `dotnet build Nivara.slnx` (Release).
- Targeted tests: MixedPrecisionTrainingTests, EmbeddingBFloat16Tests, NnTests Embedding section.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits
1. docs: plan #753 Phase 1+2 in TODO.md
2. feat: mixed-precision Embedding<TModel,TWeight>
3. test: mixed-precision Embedding training + StateDict round-trip
4. docs: remove TODO.md — plan executed

## Blast radius
- `src/Nivara/AutoDiff/Nn/Embedding.cs`
- `tests/Nivara.Tests/AutoDiff/MixedPrecisionTrainingTests.cs`

## GitHub issues log
- [x] #753 — mixed-precision Embedding (this work)
- [ ] #758 — Phase 3 model-class wiring (deferred)