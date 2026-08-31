# TODO — SmolLM BFloat16 Widen: Phase 0 + model download

**Branch:** `khurram/smollm-0`
**Source direction:** `docs/BFLOAT16-TRANSFORMER.md` (§6 Phase 0, §3.1–3.3)
**Companion planning note:** `C:\Users\khurram\.opencode\plan\bf16-widen-phase0.md`

## Progress (updated during execution)

- ✅ Download: SmolLM-135M-Instruct into `samples/data/smollm-135m` (hf CLI, 8 files; 272 tensors all BF16, 269 MB). Model files gitignored.
- ✅ PyTorch reference generator: `samples/NivaraInference/Python/smollm_generate_reference.py` (saves token-id stream + final-position logits).
- ✅ Phase 0 skeleton: `KernelType.WidenToFloatSimd`, `NivaraPrimitives.UseWidenSimd` (off), `WidenPrimitives` dispatch contract (stubbed widen, length gate), `KernelSelector` wiring. No call sites changed.
- ✅ Unit tests: `KernelSelectorWidenTests` (7 pass, dispatch contract only).
- ✅ README: SmolLM download + BF16-native + config + Phase 2 gaps + reference-gen recorded in `samples/NivaraInference/README.md`.
- ✅ Issues created for Phase 2 deferrals: #367 (GQA), #368 (causal-LM ops).

## Problem

Nivara has no SIMD-accelerated `BFloat16`/`Half` math because `Vector<BFloat16>.IsSupported`
and `Vector<Half>.IsSupported` are `false` on .NET 11 — the BCL `TensorPrimitives` runs **scalar
fallback loops** for these narrow types (issue #363: MiniLM FP16 inference ~26× slower than F32).
The narrow-precision win is halved weight memory, but CPU compute currently negates it.

The agreed direction (BFLOAT16-TRANSFORMER.md): a shared `WidenPrimitives` layer that widens
narrow floats to `float`, runs the genuinely-SIMD `TensorPrimitives<float>` kernels, and narrows
back. It is driven by a **5th HuggingFace model** (causal LM) so the layer grows organically and is
validated end-to-end against a HuggingFace reference. The probe in `tests/Nivara.SimdProbe`
validated the widen-compute-narrow kernels in isolation (BF16 dot ~12–21×, Half ~6–8×, small
`n<128` slower). This plan executes **Phase 0** (the model-agnostic, zero-behavior-change skeleton)
plus the model download as infra and records the model decision.

## Model decision (agreed with human)

**`HuggingFaceTB/SmolLM-135M-Instruct`** — a Llama-family BF16-native causal LM.

- `safetensors.parameters = {"BF16": 134515008}` → **100% BF16-native** checkpoint; exercises the
  native `SafeTensorsLoader.Read<BFloat16>` zero-hop path (`ConvertBF16ToBFloat16`).
- Config: `hidden_size=576`, `intermediate_size=1536`, 30 layers, `num_attention_heads=9`,
  `num_key_value_heads=3` (**GQA**), `hidden_act=silu` (gated FFN), RMSNorm, RoPE (`θ=10000`),
  `max_position_embeddings=2048`, `vocab_size=49152`, `tie_word_embeddings=true`,
  `rms_norm_eps=1e-5`.
- Tokenizer: chat variant (`<|im_start|>`/`<|im_end|>` template, bos `<|im_start|>`).
- **New ops Phase 2 must add (revised for SmolLM vs the doc's tanh-GELU assumption):** RoPE,
  **GQA attention (9↔3 heads)**, **gated SiLU FFN**, causal mask (exists), greedy generation
  loop, tied-embedding LM head. These are **Phase 2**, not this plan.

## Proposed changes (this branch)

### A. Model download (infra) + fixtures
- `hf download HuggingFaceTB/SmolLM-135M-Instruct --local-dir samples/data/smollm-135m \
  config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt \
  generation_config.json special_tokens_map.json` (with `HF_TOKEN` set).
- Add `.gitignore` entry `samples/data/smollm-135m/`.
- Add a PyTorch reference-generator script (`samples/NivaraInference/Python/`) that runs
  `generate(..., max_new_tokens=N)` on a fixed prompt and saves `samples/data/compare_smollm_py.bin`
  for the Phase 3 correctness diff. (Script only; not run unless Python present.)

### B. Phase 0 skeleton (strict, zero behavior change)
1. **`KernelType`** — add `WidenToFloatSimd` member. Do **not** add `Half/BFloat16` to
   `ColumnStorageFactory.IsVectorizable` yet (that flip is Phase 1).
2. **`NivaraPrimitives`** (new, `src/Nivara/Primitives/NivaraPrimitives.cs`) — static
   `bool UseWidenSimd { get; set; }`, **default `false`**; optional `AppContext` switch
   `Nivara.Primitives.WidenSimd`.
3. **`WidenPrimitives`** (new, `src/Nivara/Primitives/WidenPrimitives.cs`) — the dispatch
   **contract**: `float`/`double` → forward to `TensorPrimitives` (transparent); `Half`/`BFloat16`
   widen branches **stubbed** (documented placeholder that falls back to scalar
   `TensorPrimitives<T>`); `ShouldWiden<T>(int length)` length gate (`>= vectorSize * 4`).
4. **`KernelSelector`** — new decision path returning `WidenToFloatSimd` for `Half`/`BFloat16`
   **only when** `UseWidenSimd` is on + `Vector.IsHardwareAccelerated` + `length >= threshold`.
   Default (off) => `Scalar`, unchanged.
5. **No call sites changed** — `NivaraColumn`, `NumericTensorKernels`, `TensorsHelper.MultiplyCore`,
   AutoDiff ops all still call `TensorPrimitives<T>` directly; toggle off means nothing routes through
   `WidenPrimitives` at runtime.

### C. Unit tests (dispatch contract — `tests/Nivara.Tests/`)
- `DetermineKernelType<Half/BFloat16>` returns `Scalar` with toggle off (regression guard).
- Toggle on + hardware + length ≥ threshold → `WidenToFloatSimd`.
- Toggle on + length < threshold → `Scalar`.
- `float`/`double` never select `WidenToFloatSimd`.

## Verification

- `dotnet build Nivara.slnx` (release) — must compile clean.
- Defendant unit tests (the new `WidenPrimitives`/`KernelSelector` dispatch tests).
- Ask the human before running `dotnet test` (full suite) — per AGENTS.md.

## Blast radius

- **Phase 0 is additive and switch-gated.** Touches:
  - `src/Nivara/KernelType.cs` (enum: +1 member — additive, no existing case exhausts it fatally).
  - `src/Nivara/KernelSelector.cs` (add a branch behind a default-off toggle).
  - New files `src/Nivara/Primitives/{NivaraPrimitives,WidenPrimitives}.cs`.
  - New test file(s) in `tests/Nivara.Tests/`.
  - `.gitignore`, model data dir, Python reference script.
- **No runtime behavior change**: existing `DetermineKernelType` callers get `Scalar` as before;
  no existing test should need modification (any regression is a red flag).
- Downstream callers of `DetermineKernelType` / `IsVectorizable` are unaffected because the toggle
  is off and the widen branch is inert.

## Planned commits (one per logical unit)

1. `chore: download SmolLM-135M-Instruct into samples/data/smollm-135m` (files + .gitignore)
2. `feat: add PyTorch reference generator for SmolLM phase-3 diff`
3. `feat: add WidenToFloatSimd kernel type and UseWidenSimd toggle`
4. `feat: add WidenPrimitives dispatch contract and KernelSelector wiring`
5. `test: cover WidenPrimitives/KernelSelector dispatch contract`

## GitHub issues log

- [ ] #367 — GQA (grouped-query attention: 9 Q heads / 3 KV heads) support for the SmolLM causal-LM driver (created while planning the Phase 2 model ops)
- [ ] #368 — Causal-LM ops for SmolLM: RoPE, gated SiLU FFN, tied-embedding LM head, greedy generation loop (created while planning the Phase 2 model ops)

---

**Resistance G1 (grounding) note:** run immediately after the plan commit, before implementation:
- Ground the .NET tensor/widen strategy + `TensorPrimitives.ConvertChecked` / `Vector128.Widen` /
  `Vector<T>.IsSupported` behavior via microsoft-learn MCP.
- Navigate the codebase (code-memory MCP) to confirm exact `KernelType` declaration site, all
  `IsVectorizable` callers, and that no caller toggles widen implicitly.
- Escalate any finding to the human rather than assume.
