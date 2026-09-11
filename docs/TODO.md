# Plan: P1 — On-the-fly BF16 weights with F32 compute (Qwen inference)

Branch: `khurram/qwen-perf` (off `main` @ 83ef104). Preceding P0s are merged
(#398 single-row GEMV, #400 fused GQA decode-attention, #401 batched prefill)
and recorded in `docs/QWEN-PERF.md`.

## Problem

Per the `docs/QWEN-PERF.md` review, the largest remaining decode win is
halving weight memory traffic: every Qwen pipeline today upcasts the
BF16-on-disk checkpoint to F32 once at load, giving a ~2 GB model and
~1.97 GB read/token. The widen-compute-narrow SIMD dot kernel
(`NarrowFloatKernels.DotBf16Core`) and the P0-2 single-row GEMV fast path
(`DotFor` → `WidenPrimitives.Dot`) **already support BFloat16** — what is
missing is wiring + measurement + parity:

1. **NivaraInference qwen benchmark rejects `--precision bf16`**
   (`Program.cs:154-158`) and `Qwen.cs` is hard-typed `<float>` end-to-end
   (model, cache, `ArgMax`) → there is no way to measure the BF16 path.
2. **NivaraChat `--qwen --precision bf16` never enables
   `NivaraPrimitives.UseWidenSimd`** → it runs the scalar
   `TensorPrimitives.Dot<BFloat16>` BCL fallback (~26× slower than F32 SIMD).
   Same for smollm chat. This is a correctness-of-performance bug, not wiring.
3. **No model-level bf16-vs-f32 parity test** (kernel-level parity exists:
   `WidenPrimitivesPhase*Tests`).

Measurement prerequisites discovered during review:

4. **Different machine.** Ledger numbers were measured on an Intel Core Ultra 7
   255H (16 logical processors); this machine is an 11th Gen Intel Core
   i5-1135G7 @ 2.40 GHz (8 logical processors, lower bandwidth ceiling).
   Every before/after must be re-established apples-to-apples **on this
   machine**, and must be self-attesting (harness currently records only
   `.NET version`).
5. **No qwen checkpoint on this machine** (`samples/data/qwen2.5-0.5b-instruct`
   absent) → synthetic-weights measurement only for this plan; real-checkpoint
   work is a later follow-up (user will recheck on the same machine later).

## Proposed changes

### Step 1 — harness machine identity (small, do first)

- `tests/Nivara.PerformanceTests`: record machine identity into JSON scenario
  reports + console headers — CPU name (`PROCESSOR_IDENTIFIER` on Windows,
  empty fallback elsewhere), `Environment.ProcessorCount`,
  `RuntimeInformation.ProcessArchitecture`, `Environment.Version`. Additive
  only; gate logic unchanged.
- `samples/NivaraInference` qwen benchmark: same identity in headers.

Enables "same machine" to be a checkable field instead of prose.

### Step 2 — BF16 wiring, NivaraInference (qwen benchmark)

- `Program.cs`: lift the qwen `bf16` rejection (keep `fp16` rejected). For
  qwen mode, `--precision bf16` reads `SafeTensorsLoader.Read<BFloat16>` →
  `LlamaLoader.Load<BFloat16, BFloat16>`.
- Genericize the qwen **benchmark path** over `T` (`RunBenchmark`,
  `RunSyntheticBenchmark`, `RunDecodeBenchmark`, `TimeGeneration`,
  `GenerateCore`, `ArgMax` last-row, `LlamaKVCache<T>`,
  `ReverseGradTensor<T>`). Keep `LoadModel` / tools / distill paths float
  (unchanged surface).
- Auto-enable `NivaraPrimitives.UseWidenSimd = true` when `T` is
  `BFloat16`/`Half` in the qwen mode (mirror the SmolLM pattern in
  `Program.cs` ~line 1436), restoring the previous value afterwards.

### Step 3 — BF16 wiring, NivaraChat (bug fix)

- Auto-enable `UseWidenSimd` when `T` is `BFloat16`/`Half` at the start of the
  chat host load path (qwen `QwenMode.Execute<T>`; apply the same guard to the
  smollm chat path). F32/Half paths unaffected (toggle is a no-op for `float`
  per existing tests).

### Step 4 — model-level bf16-vs-f32 parity test

- New `Nivara.Tests` fixture: run the **same synthetic Qwen-shaped weights**
  through `LlamaForCausalLM<float>` and `LlamaForCausalLM<BFloat16>` (widen
  SIMD on), assert greedy decode equality over a seeded prompt and logits
  within a documented tolerance (widen is lossless → differences are only dot
  accumulation order + per-layer BF16 activation rounding). Follows
  `LlamaForCausalLMPrefillTests` conventions (synthetic configs, no
  checkpoint).

### Step 5 — measurement + ledger (apples-to-apples on this machine)

1. **Baseline (before BF16 wiring):** `Nivara.PerformanceTests --json
   i5-baseline.json --runs 3` (resets minOps/alloc/gen0 gates for this
   machine) and synthetic qwen E2E medians via `NivaraInference qwen
   benchmark --synthetic-weights` (prefill + decode ms/token, KV-cached vs
   full-forward).
2. **After:** same two surfaces, `--compare i5-baseline.json`, record
   before/after in `docs/QWEN-PERF.md` as a new P1 ledger entry including
   machine identity; note the user will recheck on the same machine later.

## Verification steps

- `dotnet build Nivara.slnx` clean after every step.
- Targeted fixture: new bf16-vs-f32 parity test + existing
  `LlamaForCausalLMPrefillTests` / KV-cache fixtures.
- Full `dotnet test` (human-confirmed before running).
- `PerformanceTests --json i5-baseline.json --runs 3` (before) and
  `--compare i5-baseline.json --runs 3` (after), same machine.
- Synthetic qwen E2E before/after medians, same machine.

## Planned commits

1. `docs: plan P1 BF16 wiring in TODO.md` (+ carry the QWEN-PERF.md ledger
   header tidy from the local tree)
2. `perf: record machine identity in PerformanceTests reports and qwen benchmark`
3. `infer: genericize qwen benchmark over compute type and enable bf16 precision`
4. `chat: enable widen SIMD for narrow-precision chat hosts`
5. `tests: qwen bf16-vs-f32 model parity over synthetic weights`
6. `perf: record i5 baseline and postfix measurements in QWEN-PERF ledger`

## Blast radius

- `tests/Nivara.PerformanceTests/` — JSON report schema (additive fields only;
  no gate-logic change).
- `samples/NivaraInference/Program.cs` + `Qwen.cs` — qwen mode only; other
  model modes untouched; qwen `fp16` still rejected.
- `samples/NivaraChat/Qwen/QwenMode.cs` (+ smollm chat path if applicable) —
  process-global `NivaraPrimitives.UseWidenSimd` toggle; bf16 numerics
  unchanged (lossless widen); f32 unaffected.
- `src/Nivara/` — **no changes**; kernel, GEMV fast path, and toggle already
  exist.
- Tests: one new parity fixture; existing suites as guardrails.

## GitHub issues log

- Pre-existing tracked items referenced by this plan: #387/#391 (BF16 SIMD
  research), #402 (sampling path), #403 (GQA-aware batched prefill follow-up),
  #404 (per-token fused decoder-block), #390 (GGUF / quantized weights).
- [ ] #406 — QwenInstructParityTests fail (FileNotFound) when the checkpoint is
  present but the Torch reference `.bin` fixtures are absent — tests should
  `Assert.Ignore` until `samples/NivaraTorch/gen_reference.py` runs (created
  while executing P1; surfaced by the checkpoint download activating them).
- New issues created during execution, if any, are recorded here as they land.