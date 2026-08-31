# TODO — SmolLM BFloat16 Widen: Phase 1 (core widen-compute-narrow)

**Branch:** `khurram/smollm-1`
**Source direction:** `docs/BFLOAT16-TRANSFORMER.md` (§6 Phase 1, §3.1–3.3)
**Companion planning note:** `C:\Users\khurram\.opencode\plan\bf16-widen-phase1.md`
**Phase 0 (merged, PR #369):** `WidenPrimitives` dispatch contract + `KernelType.WidenToFloatSimd` + `NivaraPrimitives.UseWidenSimd` toggle (off) + `KernelSelector` wiring. Stubs gated behind `ShouldWiden` (off → scalar, zero behavior change).
**Kernel reference:** `tests/Nivara.SimdProbe/NarrowSimdKernels.cs` (validated standalone: BF16 dot ~12–21×, Half ~6–8× vs scalar).

## Problem

Phase 0 built the dispatch contract but the widen branches are **stubbed** — with the toggle on, `WidenPrimitives.Dot/Add/Multiply` still fall through to the scalar `TensorPrimitives<T>` backend, so there is no speedup yet. Phase 1 fills the stubs with the productionized widen-compute-narrow kernels so that element-wise column ops and matmul (and — via the shared matmul — AutoDiff) actually accelerate for `BFloat16`/`Half`.

## Design decisions (confirmed with human)

| Decision | Choice |
|---|---|
| **Constraint bridging** | Relax `WidenPrimitives` element-wise + Dot methods from `IFloatingPointIeee754<T>` → **`INumber<T>`** so both `NumericTensorKernels<T>` (INumber) and AutoDiff (IFloatingPointIeee754 ⊂ INumber) consume the **same** shared surface. |
| **Kernel strategy** | **Fused per-vector, no temp buffer** (promote SimdProbe kernels). BF16 = pure 16-bit bit-shift widen/narrow intra-vector (lossless, no float[] temp). Half = portable element-wise conversion while accumulating in float SIMD. `ConvertChecked` stays as documented fallback/reference. |
| **Branching** | New branch `khurram/smollm-1` off merged `main` (Phase 0 landed). |

## Proposed changes (this branch)

1. **New `src/Nivara/Primitives/NarrowFloatKernels.cs`** — `internal static class NarrowFloatKernels`, concrete typed `BFloat16`/`Half` kernels promoted + productionized from `NarrowSimdKernels`:
   - Widen/narrow primitives: `WidenBf16/NarrowBf16` (bit-shift), `WidenHalf/NarrowHalf` (portable `BitConverter`), plus scalar variants. `[MethodImpl(AggressiveInlining)]`.
   - Typed ops: `Dot`, `Add`, `Subtract`, `Multiply`, `Divide` for each of `BFloat16`/`Half` (Subtract/Divide are new vs probe).
   - **No delegate-in-loop** — production inlines the widen/narrow/op into per-type loops (delegate hoisting is the main productionization delta).
   - Keep `Vector128.IsHardwareAccelerated` runtime guard + scalar tail/fallback.
   - Reinterpret via `MemoryMarshal.Cast<T, ushort>` (sanctioned BCL pattern).
2. **Fill `WidenPrimitives`** (existing file) — relax constraint to `INumber<T>`; fill `Dot/Add/Multiply` stubs and add `Subtract/Divide`. Widen branch dispatches by `typeof(T)` to `NarrowFloatKernels` (mirrors `RMSNormKernel` narrow-type dispatch). Scalar fallback (toggle off / below threshold / no HW) stays exactly the `TensorPrimitives.X(x,y,dst)` call → bit-identical.
3. **Route `NumericTensorKernels<T>` span-span element-wise** through `WidenPrimitives` (Add/Subtract/Multiply/Divide). `NumericTensorKernels<T>` is `INumber<T>`-constrained → direct call (this is why the relaxation matters). Scalar-operand overloads (`Add(x, T y, …)`, `SubtractFrom`, `DivideBy`) left unchanged (Phase 1.5 follow-up if needed).
4. **Route `TensorsHelper.MultiplyCore` generic path** — `MultiplyRowScalar` inner `TensorPrimitives.Dot<T>` → `WidenPrimitives.Dot<T>`. Lifts AutoDiff matmul (Linear/Attention/Conv) for free (F6). Length gate granularity = per-row (`aCols`); no temp buffer per row.
5. **Unit tests** (`tests/Nivara.Tests/Primitives/WidenPrimitivesPhase1Tests.cs`):
   - Correctness (toggle on): widen vs scalar reference for Dot/Add/Sub/Mul/Div × {BFloat16, Half}, lengths above+below threshold, tolerance-based (F32 accumulate → narrow result), incl. known-sign + zero/negative.
   - Regression guard (toggle off): identical to pure `TensorPrimitives<T>` path.
   - `NumericTensorKernels<T>` end-to-end: column element-wise op vs scalar, toggle on/off.
   - `TensorsHelper.MultiplyCore<BFloat16/Half>` matmul vs widened reference (reuse `CheckMatMulShapes` pattern).
6. (Optional) extend `KernelSelectorWidenTests`: assert `DetermineKernelType<BFloat16>` → `WidenToFloatSimd` on + long, `Scalar` otherwise.

## Verification

- `dotnet build Nivara.slnx` (Release) — must compile clean, 0 warnings.
- Target the new tests via `--filter` (default verification path).
- Ensure Phase 0 `KernelSelectorWidenTests` still pass (signature relaxation must not break them).
- Ask human before running `dotnet test` (full suite).

## Blast radius

- `src/Nivara/Primitives/NarrowFloatKernels.cs` — new, internal, additive.
- `src/Nivara/Primitives/WidenPrimitives.cs` — public API constraint relaxation `IFloatingPointIeee754<T>` → `INumber<T>` (widening of accepted types, additive-compatible; no internal callers yet outside tests). Flag in PR.
- `src/Nivara/Helpers/NumericTensorKernels.cs` — span-span element-wise now routes via `WidenPrimitives`; `INumber<T>` constraint unchanged; scalar-operand overloads untouched.
- `src/Nivara/Tensors/TensorsHelper.cs` — `MultiplyRowScalar` inner dot via `WidenPrimitives.Dot<T>`; float/double typed kernels untouched; int/other types fall through to `TensorPrimitives.Dot<T>`.
- Tests: new `WidenPrimitivesPhase1Tests.cs`; existing `KernelSelectorWidenTests` must stay green.
- **Toggle-off → zero behavior change**: all narrower types (Half/BF16) fall through to the exact scalar `TensorPrimitives<T>` call when `ShouldWiden` is false; int/etc. unaffected. AutoDiff float/double paths unaffected.

## Planned commits

1. ✅ `feat: promote NarrowFloatKernels (BFloat16/Half widen-compute-narrow) into src/Nivara` — 8c4e794
2. ✅ `feat: fill WidenPrimitives Dot/Add/Sub/Mul/Div, relax constraint to INumber<T>` — 4feb4ed
3. ✅ `feat: route NumericTensorKernels span-span element-wise through WidenPrimitives` — 3e9c663
4. ✅ `feat: route TensorsHelper matmul row-dot through WidenPrimitives (lifts AutoDiff)` — f07eba3
5. ✅ `test: widen vs scalar-reference correctness + regression toggles` — 789b356
6. ⬜ `docs: verify BFLOAT16.md / BFLOAT16-TRANSFORMER.md Phase 1 notes`

## Deviations from plan (noted for PR)

- **`struct` constraint ripple (commit 3):** the plan assumed routing `NumericTensorKernels<T>` through `WidenPrimitives` was a "direct call, no constraint gymnastics." In practice `WidenPrimitives` needs `T : struct` (for `MemoryMarshal.Cast<T, ...>` dispatch), while `NumericTensorKernels<T>` declared only `INumber<T>`. Fixed by adding `struct` to `NumericTensorKernels<T>` and to the two private delegate-creating methods in `NumericKernelDispatcher` (`createArithmetic<U>`, `createComparison<U>`), which reference `NumericTensorKernels<U>` directly and are invoked via `MakeGenericMethod` (so the constraint is compile-time only, no public-API or call-site change).
- **`NarrowFloatKernels.Dot` return type (commit 4):** the promoted Dot returned a raw `float`, but `WidenPrimitives.Dot<T>` must return `T` (matmul assigns to `T[]`). Fixed by having `NarrowFloatKernels.Dot` return the narrowed `BFloat16`/`Half`. This also makes the `(T)(object)` box-unbox cast valid.

## Out of scope (deferred → later phases)

- AutoDiff element-wise VJP rules + optimizers (Sigmoid/Tanh/GELU/ReLU, Adam/AdamW/SGD, GradientUtils) — route through shared wrapper in Phase 2/3.
- `RMSNormKernel<T>` Half/BF16 scalar fallback — Phase 2.
- RoPE / GQA / SiLU / causal-LM / generation loop — **Phase 2** (issues #367, #368).
- Retrofitting existing 4 models / flipping global switch — **Phase 3** (regression A/B).

## GitHub issues log

- [ ] #367 — GQA (grouped-query attention: 9 Q heads / 3 KV heads) support for the SmolLM causal-LM driver (Phase 2, created during Phase 0 planning)
- [ ] #368 — Causal-LM ops for SmolLM: RoPE, gated SiLU FFN, tied-embedding LM head, greedy generation loop (Phase 2, created during Phase 0 planning)
