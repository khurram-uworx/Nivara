# BFloat16 Transformer Acceleration — Plan & Research

**Status:** Planning / agreed direction.
**Goal:** Give Nivara SIMD-accelerated `BFloat16` / `Half` numeric math — something the BCL
(`TensorPrimitives`) cannot do on .NET 11 — by widening narrow floats to `float` and running
the genuinely-SIMD `TensorPrimitives<float>` kernels, then narrowing back. The work is driven
by a **5th HuggingFace model** (a BF16-native causal LM, TinyLlama / SmolLM class) so the shared
primitive layer grows organically and is validated end-to-end against a HuggingFace reference.

This document is the companion to [`BFLOAT16.md`](BFLOAT16.md) (engine-level BFloat16 support)
and the `samples/NivaraInference` README (existing inference models).

---

## 1. Research findings

### F1 — `TensorPrimitives` *encapsulates* dispatch; "support" ≠ "SIMD"
`TensorPrimitives` selects the best path for the element type `T`: it uses the SIMD kernel when
`Vector<T>.IsSupported` is `true`, otherwise a scalar fallback. The dispatch is per-`T` and
internal — callers do not choose. So "`TensorPrimitives` supports `BFloat16`" is true at the
*type* level (the generic overloads compile and are correct) but **false at the acceleration
level**. See the "Hints for .NET Numerics engineers" box in `BFLOAT16.md`.

### F2 — `Vector<BFloat16>` / `Vector<Half>` are unsupported on .NET 11
Empirically (`BFLOAT16.md`, *SIMD / Vector Lane Support*): `Vector<BFloat16>.IsSupported ==
false` at every width (64/128/256), and `Vector.Create<BFloat16>()` throws
`NotSupportedException`. Microsoft's SIMD docs state acceleration applies only to types
`Vector<T>` supports. Therefore the BCL silently runs its **scalar** fallback for BF16/Half —
correct, but not vectorized. `Half` is in the identical situation.

### F3 — `BFloat16` widens to `float` losslessly
`BFloat16` is the top 16 bits of `float32`. Reinterpreting the 16-bit value into the high half
of a `uint32` and bit-casting to `float` is **lossless**; `Half` widens via the standard
conversion. This is the key to SIMD: we can turn a BF16/Half buffer into a `float` buffer, run
the SIMD `TensorPrimitives<float>` kernels, and narrow back — the BCL will not do this for us
(see F4).

### F4 — the BCL does **not** auto-widen narrow floats for SIMD
`TensorPrimitives.Add<BFloat16>` operates on `BFloat16` *as* `BFloat16` and drops to scalar.
Only the explicit `ConvertToSingle` / `ConvertChecked<TFrom,TTo>` helpers widen, and they are
not invoked by the arithmetic overloads. So vectorization for BF16/Half must be implemented by
the library (Nivara), not relied upon from the BCL.

### F5 — Nivara already owns its numeric dispatch and knows the data length
Relevant seams (all in `src/Nivara`):
- `KernelSelector.DetermineKernelType<T>` (`KernelSelector.cs`) — returns `Scalar` today
  because `ColumnStorageFactory.IsVectorizable<Half/BFloat16>()` is `false`
  (`Storage/ColumnStorageFactory.cs:98`). This is the natural place to add a `WidenToFloatSimd`
  branch.
- `Helpers/NumericTensorKernels.cs` — column element-wise `Add/Mul/Div/...` route through
  `TensorPrimitives<T>`; the widen path goes here.
- `Tensors/TensorsHelper.cs` (`MultiplyCore<T>`) — **shared** matmul kernel, also used by
  AutoDiff (see F6). The widen branch here benefits both layers.

### F6 — AutoDiff partially shares the kernel path
- **Matmul / Linear / Attention / Conv** funnel through the shared
  `TensorsHelper.MultiplyCore<T>` (`AutoDiff/Operations/GradKernels.cs:633,637`,
  `AutoDiff/Operations/AttentionKernels.cs:11`). A widen branch here speeds up AutoDiff
  **automatically**.
- **Element-wise VJP rules** (`GradKernels.cs` Sigmoid/Tanh/GELU/ReLU) and the **optimizers**
  (`Optimizer/Adam.cs`, `AdamW.cs`, `SGD.cs`) and `GradientUtils` call `TensorPrimitives`
  **directly**. They would only benefit if we route them through the same shared wrapper.

### F7 — narrow-precision BF16/Half is ~26× slower than F32 today (issue #363)
`samples/NivaraInference/README.md` ("Speed") shows BF16/Half inference runs through non-SIMD
fallbacks and is dramatically slower per pass than F32 (fp16 measured ~26× slower, MiniLM).
The *win* of narrow precision is halved weight memory (2 B/param), not speed. The widen-SIMD
work converts that speed penalty into a speed *gain* for the math while keeping the memory win.

### F8 — token-ID correctness is already solved
`Embedding<T>`, `BertEncoder<T>`, `MiniLMDistilled<T>`, `DistilBertForSequenceClassification<T>`
expose `Forward(int[] tokenIds, ...)` overloads so vocab indices survive narrow dtypes exactly
(`BFLOAT16.md`). A causal LM generates token IDs as exact `int` anyway, so this is not a new
risk, merely a constraint to preserve in the generation loop.

---

## 2. The opportunity

Nivara is structurally positioned to do what the BCL will not:
- It owns `KernelSelector` and knows the tensor length, so it can decide *when* widening pays
  off (length-gated) and *how* (widen → SIMD `float` → narrow).
- A **single shared primitive layer** (`WidenPrimitives`) consumed by `NivaraColumn`,
  `NivaraSeries`, `NivaraFrame`, **and** AutoDiff turns a column-only speedup into a
  library-wide one from one implementation + one test surface.
- This is a credible **differentiator**: "the only .NET dataframe + AutoDiff library with
  SIMD-accelerated BFloat16/Half math," since the BCL cannot provide it.

---

## 3. Design

### 3.1 Shared primitive layer — `WidenPrimitives`
A single dispatch surface replacing the scattered `TensorPrimitives.X<T>` calls:
- `float` / `double` → transparent forward to `TensorPrimitives` (already SIMD; **no behavior
  change**).
- `Half` / `BFloat16`, `Vector.IsHardwareAccelerated`, `length ≥ threshold`, and the widen
  toggle on → **widen-to-float SIMD**: `TensorPrimitives.ConvertChecked<Half/BFloat16,float>`
  (rented buffer) → `TensorPrimitives<float>` op → `ConvertChecked<float,Half/BFloat16>`.
- otherwise → scalar (today's path; no regression).

For matmul this lives in `TensorsHelper.MultiplyCore<T>` (benefits AutoDiff + columns). For
element-wise/reductions it lives in `NumericTensorKernels` and is mirrored into the AutoDiff
element-wise/optimizer calls.

### 3.2 KernelSelector toggle + A/B switch
- Add `KernelType.WidenToFloatSimd`. `KernelSelector.DetermineKernelType<T>` returns it for
  `Half`/`BFloat16` when the toggle is on, hardware is accelerated, and `length ≥ threshold`.
- Toggle: static `NivaraPrimitives.UseWidenSimd` (or `AppContext` switch
  `Nivara.Primitives.WidenSimd`), **default off**. This keeps existing BF16 behavior and
  references intact and enables a clean **A/B** on the same model.
- `samples/NivaraInference` exposes `--simd-widen`; run the 5th model twice (off = scalar,
  on = widen) and diff (a) correctness vs the HuggingFace reference and (b) timings.

### 3.3 Widen mechanics & correctness
- **Widening:** `ConvertChecked<Half/BFloat16,float>` (BF16 is a shift-or; `Half` a standard
  conversion). A custom SIMD BF16→F32 (reinterpret `ushort`, shift-left-16, bit-cast) keeps the
  conversion itself vectorized if the BCL `ConvertChecked` proves scalar-bound.
- **Null masks:** unchanged — the column layer ORs masks around the value kernel; widening
  operates on raw backing values (null slots hold `default(T)=0`) and masked positions are
  overwritten by the caller's mask.
- **Rounding fidelity:** F32-compute-then-narrow-to-BF16 rounds once from a more accurate
  intermediate. It is faithful and matches real hardware BF16 (FMA uses an F32 accumulator);
  documented as "F32 intermediate, BF16 result," not "native BF16 accumulation."
- **Memory:** widening needs a temp `float` buffer (2–4×). Rent + chunk for very large tensors,
  consistent with the existing streaming/memory-budget patterns.
- **Threshold:** tune `k` (≥ `vectorSize * 4`, likely higher) via benchmark so tiny columns stay
  scalar — the conversion overhead must be amortized.
- **Forward-compat:** if a future .NET flips `Vector<Half/BFloat16>.IsSupported`, update
  `IsVectorizable` and those types take the native SIMD path; the widen branch remains a valid
  fallback. No Nivara change needed when the BCL catches up.

---

## 4. Op coverage

### 4.1 Existing 4 models (regression harness, not retrofitted yet)
`MobileNetV2` (vision CNN), `ResNet-18` (vision CNN), `MiniLM` (transformer embedding),
`DistilBERT` / `DistilBERT SST-2` (transformer classification) already cover:

| Capability | Where exercised |
|---|---|
| `Conv2d<T>` (incl. grouped / depthwise), `BatchNorm2d<T>` | MobileNetV2, ResNet-18 |
| `Linear<T>` / MatMul, bias broadcast | all classifiers |
| `MaxPool2d<T>`, `AdaptiveAvgPool2d<T>` | ResNet-18 head |
| `Embedding<T>` (gather) | MiniLM, DistilBERT |
| `LayerNorm<T>`, `GeluExact`, `Softmax`, residual `Add` | MiniLM, DistilBERT |
| `MultiheadAttention<T>` (bidirectional) | MiniLM, DistilBERT |

These remain the **regression harness**: once the global switch flips later, their existing
HuggingFace references (e.g. DistilBERT SST-2 `8/8` argmax) confirm widen didn't change
results.

### 4.2 5th model (TinyLlama / SmolLM causal LM) — new ops to implement
A decoder-only causal LM adds the genuinely **new** primitive coverage the shared layer lacks
today:

| New op | Why needed | Notes |
|---|---|---|
| **Causal self-attention mask** | autoregressive attention | new mask shape vs bidirectional |
| **RoPE (RotaryEmbedding)** | Llama positional encoding | new op: rotate Q/K by position |
| **GPT-2 `Gelu` (tanh approx)** | Llama FFN activation | `ReverseGradOperations.Gelu` path (already exists, wire it) |
| **Greedy generation loop** | token-by-token decode | reuses `int[]` token-ID path (F8) |
| (optional) **Conv1d** | only if a Whisper-style model is chosen instead | not needed for TinyLlama |

Core ops (matmul/Linear, LayerNorm, Softmax, embeddings, residual `Add`) are already covered by
the existing 4 models, so the shared layer's *core* is exercised by all five; the 5th model's
job is to pull in the *new* ops above.

---

## 5. The 5th model: TinyLlama / SmolLM-class causal LM

**Why this model:**
- **Highest "wow":** real generative text inference end-to-end in pure managed BF16 .NET, no
  Python/CUDA — a compelling demo.
- **Edge AI fit:** TinyLlama / SmolLM-class models are explicitly designed for edge and
  on-device inference, where halved weight memory (BF16) + SIMD math is exactly the sweet spot
  Nivara can own. This aligns the feature with a concrete, marketable use case.
- **New-op coverage:** adds causal masking, RoPE, GPT-2 Gelu, and a generation loop — the
  missing pieces in the shared layer.

**Architecture mapping (target Nivara modules):**
- Token + position embeddings → `Embedding<T>` (+ RoPE applied to Q/K before attention).
- Layers: `LayerNorm<T>` → causal `MultiheadAttention<T>` (masked) → residual → `LayerNorm<T>`
  → `Linear<T>` → `Gelu` (tanh) → `Linear<T>` → residual.
- LM head: `Linear<T>` → logits; greedy decode via `Softmax` + argmax, feeding `int[]` token
  IDs back (F8).

**Loading:** BF16-native checkpoint via `SafeTensorsLoader.Read<BFloat16>` + `Module<BFloat16>`
(already supported). The model is the *driver*, not a new loading feature.

---

## 6. Implementation plan (incremental, switch-gated)

- **Phase 0 — skeleton (model-agnostic, low risk):** create `WidenPrimitives` with the dispatch
  contract; add `KernelType.WidenToFloatSimd` + `NivaraPrimitives.UseWidenSimd` toggle (default
  off); wire `KernelSelector`. No call sites changed yet → zero behavior change.
- **Phase 1 — core widen (benefits all):** implement widen for element-wise
  `Add/Mul/Div/Sub` and `MatMul`/`Dot` in `NumericTensorKernels` + `TensorsHelper.MultiplyCore`
  (the latter also lifts AutoDiff matmul). Unit tests: scalar BF16/Half reference vs widen,
  per op.
- **Phase 2 — 5th model + its ops:** add the TinyLlama/SmolLM sample; implement RoPE, causal
  attention mask, GPT-2 Gelu wiring, generation loop; route its numeric ops through
  `WidenPrimitives`. Only the ops this model needs are implemented first.
- **Phase 3 — A/B + correctness + docs:** `--simd-widen` in `NivaraInference`; benchmark scalar
  vs widen; add a Python reference generator for the 5th model; verify argmax/logit diff vs
  HuggingFace; update `BFLOAT16.md` and this doc with results. Optionally flip the global switch
  and re-run the 4 existing models as a regression check.

---

## 7. Risks / open questions

- **Conversion cost:** if `ConvertChecked<BFloat16,float>` is scalar-bound on .NET 11, the net
  win shrinks; mitigate with a custom SIMD BF16→F32 widen. Must be confirmed by benchmark.
- **Scope:** start with matmul + element-wise + the 5th model's ops only; do not retrofit the 4
  existing models until the global switch is proven.
- **Memory:** temp `float` buffers for very large tensors — rent + chunk.
- **Precision messaging:** document "F32-intermediate, BF16 result" so users don't expect
  bit-identical native-BF16 accumulation (acceptable; matches hardware).
- **Future .NET:** `Vector<Half/BFloat16>` support may arrive later and obsolete the widen
  branch — design so the native path is preferred automatically.

---

## 8. References

- `docs/BFLOAT16.md` — engine-level BFloat16 support, vectorization note, Numerics hints.
- `src/Nivara/KernelSelector.cs` — kernel dispatch (add `WidenToFloatSimd` here).
- `src/Nivara/Helpers/NumericTensorKernels.cs` — column element-wise kernels.
- `src/Nivara/Storage/ColumnStorageFactory.cs` — `IsVectorizable` (BF16/Half currently `false`).
- `src/Nivara/Tensors/TensorsHelper.cs` — shared `MultiplyCore<T>` (matmul; lifts AutoDiff too).
- `src/Nivara/AutoDiff/Operations/GradKernels.cs`, `Optimizer/{Adam,AdamW,SGD}.cs` — direct
  `TensorPrimitives` calls to route through the shared wrapper.
- `samples/NivaraInference/README.md`, `samples/NivaraInference/Program.cs` — existing 4 models,
  `--precision bf16/fp16`, A/B `--simd-widen` target; issue #363 (narrow ~26× slower).
