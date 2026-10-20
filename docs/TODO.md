# Plan: Issue #387 — Keep BF16 weights in memory as ushort (skip load-time F32 widen)

## Problem

Qwen2.5-0.5B ships BF16 on disk (989 MB). Current `qwen` path widens to F32 at load (~2.2 s, ~2 GB in memory). Proposal: keep weights as raw BF16 bits (`BFloat16[]`, 2 bytes/param), widen inside the matmul kernel, F32 activations/compute. Memory ~1 GB, no load-time widen.

## Scope (#387)

In scope: core mixed-precision span kernels (forward + VJP for activations/weights), parity tests (kernel level), docs.

The end-to-end qwen raw-BF16 inference path (keeping BF16 weights in a model, the opt-in
`--qwen-weights` flag, the ~1 GB measurement and A/B decode timing) requires mixed-precision
weights inside `Module<T>`/`Parameter<T>`, which is out of scope for this issue and tracked by
#746. #387 unblocks #746 by providing the kernel family; #746 will add the flag and the
InferenceCliArgumentTests coverage that a new flag requires.

Out of scope (follow-ups created):
- #746 Mixed-precision Module/Parameter/StateDict (trainable BF16 weights) + VJP wiring into modules + qwen raw path + flag.
- #747 JVP / forward-mode AD for mixed precision.
- #748 Silk/GPU device-side BF16 weights.
- #749 Memory-mapped / paged weights for very large models.

## Design

- `BFloat16`/`Half` are 2-byte structs wrapping raw ushort bits; `NarrowFloatKernels.WidenBf16` already does `(uint)w << 16`. Storage as `BFloat16[]`/`Half[]` == raw ushort memory.
- Mixed-precision matmul: activations `float`, weights `BFloat16`/`Half`, output `float`. Widen weight per lane, accumulate in `Vector128<float>`, scalar tail. Bit-exact SIMD vs scalar.
- qwen CPU: inference-only path keeping `BFloat16[]` weights, computing in float, using fused kernels. Does not allocate float weights. Opt-in flag `--qwen-weights {f32|bf16-raw}` (default f32). Silk stays F32.

## Blast radius

- `src/Nivara/AutoDiff/Operations/GradKernels.cs` — add mixed-precision matmul span kernels.
- `src/Nivara/Primitives/NarrowFloatKernels.cs` — add `MatMulTransposedB` mixed variants (SIMD + scalar).
- `src/Nivara/AutoDiff/Nn/LlamaFusedKernels.cs` — expose mixed-precision `MatMulTransposedB`.
- `src/Nivara/KernelSelector.cs` — dispatch for mixed precision; tests.
- `samples/NivaraInference/Qwen.cs` — raw-BF16 inference path + flag handling (CPU modes).
- `samples/NivaraInference/Program.cs` — argument parsing.
- Tests: parity (reuse `LlamaCausalLMBf16ParityTests` pattern), kernel SIMD-vs-scalar.
- Docs: `samples/NivaraInference` README benchmark row.

## Planned commits

1. docs: plan #387 in TODO.md
2. feat(primitives): mixed-precision MatMulTransposedB (BF16/Half weights x float) SIMD + scalar
3. feat(gradkernels): expose mixed-precision matmul span kernels
4. feat(fused): LlamaFusedKernels mixed-precision MatMulTransposedB
5. test: mixed-precision matmul parity (SIMD vs scalar, vs float-widened reference)
6. docs: remove TODO.md

## GitHub issues log

- [x] #387 — this issue
- [x] #746 — mixed-precision Module/Parameter/StateDict (follow-up)
- [x] #747 — JVP for mixed precision (follow-up)
- [x] #748 — Silk/GPU device-side BF16 (follow-up)
- [x] #749 — memory-mapped/paged weights (follow-up)

## Verification

- Build: `dotnet build Nivara.slnx`
- Targeted tests: docs + new parity/kernel tests
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`