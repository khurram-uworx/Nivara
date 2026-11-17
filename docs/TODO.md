# Plan: Laya synthetic weights support (#1255)

**Branch:** khurram/1255
**Issue:** https://github.com/khurram-uworx/Nivara/issues/1255
**Related:** docs/plan/MODEL-LOADING-HONESTY.md §8.3

## Problem

`--synthetic-weights` is rejected for every model except qwen/smollm. Laya (ModernBERT
encoder + typed decision head) has no synthetic tensor generator and no self-contained
config/tokenizer path, so it cannot be exercised without a multi-GB checkpoint. Native
Laya = `Laya.Run<float, Half>` (F16 storage / F32 compute); the single F32 `temperature`
scalar must be narrowed to Half and documented.

## Proposed changes

1. `Laya.SynthesizeTensors<T>(ModernBertConfig config)` + overload taking `AgentConfig`,
   fabricating the exact keys `ModernBertEncoder.LoadWeights(prefix="encoder")` and
   `LayaDecisionHead.LoadWeights` expect, directly in storage dtype T. Local `AddSynthetic<T>`.
2. `Laya.CreateSyntheticModelDir(config, agent)` writes a temp dir with encoder/config.json,
   rl_agent_config.json, and a minimal tokenizer.json so existing Run/RunGpu paths work
   unchanged.
3. `Program.cs`: embedded `LayaSyntheticConfig` / `LayaSyntheticAgent`; allow
   `--synthetic-weights` for laya in guards; dispatch block mirroring SmolLM; Silk stays
   F32/bf16 widened + native fp16 device path.
4. Tests: update reject message expectation and add laya synthetic cases.
5. Docs note on temperature narrowing.

## Blast radius

- samples/NivaraInference/Laya.cs, Laya.Narrow.cs (none needed), Program.cs
- tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs
- docs/LAYA.md, docs/plan/MODEL-LOADING-HONESTY.md

## Verification

- dotnet build Nivara.slnx
- dotnet test InferenceCliArgumentTests (ask first)
- manual: laya --synthetic-weights (cpu + --silk) runs end-to-end

## Planned commits

1. docs: plan Laya synthetic weights in TODO.md
2. feat: Laya.SynthesizeTensors + CreateSyntheticModelDir
3. feat: wire --synthetic-weights for laya (guards + dispatch + runners)
4. test: CLI coverage for laya synthetic
5. docs: Laya synthetic weights note

## GitHub issues log

- (none yet)
