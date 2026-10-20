# Plan: Forward-mode AD (JVP) mixed-precision matmul (#747)

## Problem
Issue #747 (follow-up to #387): forward-mode AD assumes a single compute type.
`ForwardGradOperations.MatMulTransposedB<T>` and the JVP fragment fail when weights are
narrow (BF16/Half) and activations are F32, so forward-mode users cannot differentiate
models using raw BF16 weights. Reverse-mode already supports this via Linear<TModel,TWeight>.

## Proposed change
Add fixed (non-generic) overloads to `ForwardGradOperations` for F32 activations x
BF16/Half weights, covering both `MatMul` and `MatMulTransposedB`, each with full JVP:
- primal via the mixed #387 kernels
- tangent terms via the same mixed kernels (a-tangent, b-tangent, or sum)
Add a missing scalar Half plain mixed kernel to `NarrowFloatKernels` + `GradKernels`
wrapper for parity with the BF16 plain path.

## Planned commits
1. docs: plan
2. Add Half plain mixed kernel (NarrowFloatKernels + GradKernels)
3. Add mixed transposedB overloads to ForwardGradOperations
4. Add mixed plain MatMul overloads
5. Tests
6. docs pins (if new test file)
7. docs: remove TODO.md (at G2 completion)

## Blast radius
src/Nivara/Primitives/NarrowFloatKernels.cs, src/Nivara/AutoDiff/Operations/GradKernels.cs,
src/Nivara/AutoDiff/Operations/ForwardGradOperations.cs, tests.

## GitHub issues log
- [ ] #747 - JVP for mixed kernels (this branch)
- [ ] #748 - Silk/GPU BF16
- [ ] #749 - memory mapped weights