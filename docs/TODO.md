# Plan: mixed-precision Module/Parameter/StateDict (#746)

## Problem
#387 added mixed-precision matmul kernels (BF16/Half weights x F32 activations) and an
inference path. Core Module/Parameter/StateDict remain single-typed, so trainable BF16/Half
weights cannot be registered, optimized, or serialized inside Module<T>.

## Proposed change (incremental scope)
Phase 1 (this branch) unblocks mixed-precision training for Linear layers:
- Add `Linear<TModel, TWeight>` over `Module<TModel>`.
- Store weight as `Parameter<TWeight>` via a new mixed-parameter registration channel on
  `Module<TModel>`.
- Bias remains `Parameter<TModel>`.
- Forward uses `GradKernels.MatMulTransposedB(float, TWeight)`.
- VJP accumulates activation gradient through mixed `MatMul` and weight gradient through generic
  float matmul, then narrows into `Parameter<TWeight>`.
- Add `Linear<T>` alias as `Linear<T,T>`.
- Update Llama/StateDictLoader optional callers later (deferred).
- Add mixed `Embedding<TModel,TWeight>` in a follow-up if not completed (deferred).
- Update WeightAccessConsistency/OpNode contract tests to be aware.

## Verification plan
- dotnet build Nivara.slnx
- Targeted Nivara.Tests AutoDiff tests for Linear/OpNode contracts
- dotnet format whitespace verification

## Planned commits
1. docs: plan
2. Add mixed-parameter registration to Module<T>
3. Implement Linear<TModel,TWeight>
4. Tests/contract updates
5. docs/pins if needed
6. docs: remove TODO.md

## Blast radius
src/Nivara/AutoDiff/Nn/Module.cs, Linear.cs, Parameter.cs, ReverseGradOperations.cs,
StateDictLoader.cs, mixed registration, tests AutoDiff contract tests.

## GitHub issues log
- [ ] #747 - JVP for mixed kernels (dependency follow-up)
- [ ] #753 - Mixed-precision Embedding and model integration follow-up (created during #746 execution)
- [ ] #748 - Silk/GPU BF16 (dependency follow-up)
- [ ] #749 - memory mapped weights (dependency follow-up)
