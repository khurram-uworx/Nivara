# Plan: Synthetic SmolLM benchmark (#718)

Follow-up to #426. Add synthetic SmolLM decode / A/B / GPU-vs-CPU benchmark
analogous to Qwen's RunSyntheticBenchmark / RunSyntheticBenchmarkSilk.

## Problem

SmolLM benchmark modes require the ~269 MB checkpoint. There is no synthetic
(fabricated weights) path, so timing work on SmolLM shapes needs the model file.
#426 made SynthesizeTensors bias-gated, so a SmolLM-shaped config (AttentionBias=false)
now synthesizes a loadable, bias-free checkpoint.

## Proposed changes

1. Add `SmolLMSyntheticConfig` LlamaConfig constant (in Program.cs) mirroring
   samples/data/smollm-135m/config.json with AttentionBias=false.
2. Add synthetic CPU decode benchmark method `RunSmolLMSyntheticBenchmark<T>()` that
   fabricates tensors via Qwen.SynthesizeTensors, narrows to T, loads model, builds
   a random 64-token prompt, runs median-of-3 decode via RunSmolLMGeneration.
3. Add synthetic A/B method `RunSmolLMSyntheticAb<T>()` mirroring SmolLMAb but
   using fabricated weights.
4. Add synthetic Silk method `RunSmolLMSyntheticBenchmarkSilk()` mirroring
   RunSyntheticBenchmarkSilk.
5. Wire `--synthetic-weights` for smollm in Program.Main (guard + dispatch), analogous
   to qwen, including ConsumesNarrow synthetic handling.
6. Update InferenceCliArgumentTests accepted cases for smollm synthetic.

## GitHub issues log

- (none yet)
