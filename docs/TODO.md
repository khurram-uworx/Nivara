# TODO — Issue #1394: mixed-dtype LoRALinear<TModel, TWeight> for frozen Llama base (BF16 storage / F32 adapters)

- Work branch: `khurram/1394`
- Tracking issue: #1394 (open, enhancement, opened 2026-11-21)

## Problem

`LoRALinear<T>` (src/Nivara/AutoDiff/Nn/LoRALinear.cs) is single-dtype: it wraps a
`Linear<T>` (i.e. `Linear<T, T>`) base and keeps adapter matrices `A`/`B` in the same type `T`.
The causal LMs the LoRA work targets (`LlamaForCausalLM<TModel, TWeight>`, samples/Nivara.Samples/LlamaForCausalLM.cs)
use mixed-dtype projections `Linear<TModel, TWeight>` — BF16 storage / F32 compute — because
`LlamaLoader.Load<float, BFloat16>` loads BF16 checkpoints into F32-compute modules.

Because of that, the minimal LoRA module from #1380 cannot wrap a frozen
`Linear<TModel, TWeight>` base, and the full-stack mixed-dtype training guard
(`LlamaForCausalLM.ThrowIfMixedDtypeTraining`) currently rejects mixed compute/weight dtypes
inside a `GradientUtils.Grad()` scope as inference-only.

Issue #1394 asks for a mixed-dtype `LoRALinear<TModel, TWeight>` (or a generic overload) that
wraps a frozen `Linear<TModel, TWeight>` base with adapter matrices `A`/`B` in compute type
`TModel` (F32), preserving step-0 parity and adapter-only gradients.

## Proposed changes

1. Add mixed-dtype LoRA module (core) that wraps a frozen `Linear<TModel, TWeight>`:
   - adapter `A: [in, r]`, `B: [r, out]` in `TModel` (F32); `B` zero-init, `A` Kaiming; scale = `alpha / r`.
   - base frozen via `Module<T>.Freeze()`; base forward goes through the mixed kernel
     (`MixedMatMulTransposedB`), adapter math stays same-type F32.
   - step-0 parity: adapted == base (exact); `Merge()` folds adapter delta into base weight
     (needs BF16 narrowing decision).
2. Wire adapter injection into the causal-LM stack (targets: attention q/v default, optional k/o/ffn).
3. Reconcile the mixed-dtype training guard so a frozen mixed base + F32 adapter graph is allowed.
4. Tests per acceptance criteria:
   - step-0 parity adapted == base (exact) on a tiny Llama-shaped block;
   - frozen base produces no base gradients after backward;
   - adapter gradients match finite differences;
   - works with `LlamaLoader.Load<float, BFloat16>` frozen base; adapters stay F32.

## Verification steps

- Build (`dotnet build Nivara.slnx`).
- Targeted tests: `tests/Nivara.Tests` LoRA / mixed-precision / Llama parity fixtures.
- Docs snippet gate if `docs/` changes; `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits

- `docs: plan mixed-dtype LoRA for issue #1394 in TODO.md` (this file, own logical unit).

## Blast radius

- `src/Nivara/AutoDiff/Nn/LoRALinear.cs`, `src/Nivara/AutoDiff/Nn/Linear.cs`, `src/Nivara/AutoDiff/Nn/Module.cs`
- `samples/Nivara.Samples/LlamaForCausalLM.cs` (+ `LlamaCausalAttention` projection types)
- `tests/Nivara.Tests/AutoDiff/*` (LoRALinear, MixedPrecision, Llama parity, LlamaLoader)
- `docs/plan/LORA.md` (design note §3/§9 may need a status/plan update)

## GitHub issues log

- [ ] #1394 — LoRA: mixed-dtype LoRALinear<TModel,TWeight> for frozen Llama base (BF16 storage / F32 adapters) (in scope)
- [ ] #1407 — Mixed-dtype LoRALinear adapter save/load round trip is uncovered (created while working on #1394)
