# Qwen synthetic weights honesty — implementation plan

**Branch:** `khurram/qwen-synth`
**Status:** Plan (2026-11-17)

## Problem

`--synthetic-weights` fabricates a full F32 pool (`SynthesizeTensors` returns
`float[]`) and narrows it to the storage dtype afterwards (`NarrowTo<TWeight>`),
so a BF16 synthetic run still materialises the widened duplicate that
MODEL-LOADING-HONESTY forbids. The Silk/OpenCL synthetic path ignores
`--precision` entirely and always opens the F32 device session, even on
`--precision native`/`bf16` where inference runs the BF16 device path.

## Proposed changes

1. `Qwen.SynthesizeTensors<T>` generic overload fabricating directly in `T`;
   keep `SynthesizeTensors(LlamaConfig)` as the float overload.
2. `RunSyntheticBenchmark<TModel,TWeight>` uses `SynthesizeTensors<TWeight>` and
   reports storage bytes.
3. `RunSyntheticBenchmarkSilk(plain, bf16)` opens the BF16 device session when
   `bf16`; Program dispatch passes `bf16 || native`.
4. Tidy the `--synthetic-weights` help line.
5. Update SmolLM synthetic helpers to use the generic synthesiser.

## Verification

- `dotnet build Nivara.slnx`
- targeted `InferenceCliArgumentTests` + `SynthesizeTensorsBiasGatingTests`
- `dotnet format ... whitespace --verify-no-changes`
- smoke runs (human-confirmed)

## GitHub issues log

(none yet)
