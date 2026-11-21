# Plan: Extract SmolLM/Qwen causal-LM loading helpers (issue #1373)

## Problem
`samples/NivaraInference` owns the SmolLM/Qwen config + tokenizer loading helpers:
- `Qwen.LoadModel<TModel,TWeight>` and `Qwen.LoadConfigAndTokenizer` (Qwen.cs)
- inline `LlamaConfig.FromJson` + `new Gpt2BpeTokenizer(...)` in Program.cs (SmolLM paths)

`LlamaLoader.Load<TModel,TWeight>` already lives in `samples/Nivara.Samples`. Duplicating
these helpers in the upcoming `samples/NivaraFineTuning` (LoRA pre-req) would drift.

## Goal
Promote the helpers into `samples/Nivara.Samples` so both `NivaraInference` and
`NivaraFineTuning` share one implementation. A shared helper loads a model directory into
`(LlamaForCausalLM<TModel,TWeight> Model, LlamaConfig Config, Gpt2BpeTokenizer Tokenizer)`
for both SmolLM and Qwen (Qwen keeps qkv-bias / RoPE-theta differences, which live in
config.json, plus its tokenizer.json added-token merge).

## Proposed changes
1. `samples/Nivara.Samples/LlamaForCausalLM.cs` — extend `LlamaLoader`:
   - `LoadConfigAndTokenizer(modelDir, qwen?)` → `(LlamaConfig, Gpt2BpeTokenizer)`
   - `LoadSmolLMConfigAndTokenizer`, `LoadQwenConfigAndTokenizer`
   - `LoadModel<TModel,TWeight>(modelDir, qwen?)` (reads `model.safetensors`)
   - `LoadModel<TModel,TWeight>(tensors, modelDir, qwen?)` compat overload
   - `LoadSmolLM<TModel,TWeight>`, `LoadQwen<TModel,TWeight>`, `Load<TModel,TWeight>(string, ...)`
   - `IsQwenModel` auto-detect (dir name contains "qwen", or RopeTheta>=100000, or VocabSize>=100000)
   - Thin `LlamaModelLoader` alias class delegating to `LlamaLoader`.
2. `samples/NivaraInference/Qwen.cs` — remove local helpers; call sites use `LlamaLoader`.
3. `samples/NivaraInference/Program.cs` — SmolLM Silk/CPU paths use `LlamaLoader.LoadConfigAndTokenizer`.

## Blast radius
- `samples/Nivara.Samples` (LlamaLoader) — additive only; existing `Load(config, tensors)` unchanged.
- `samples/NivaraInference` — call-site edits only; no behaviour change intended.
- Tests referencing `LlamaLoader` unaffected (existing 2-arg `Load` unchanged).
- `NivaraChat` also uses LlamaLoader.Load; unaffected.

## Verification steps
- Build `Nivara.Samples` and `NivaraInference`.
- Run `smollm generate` and `qwen tools` modes; compare outputs to baseline.
- Targeted tests: `QwenInstructParityTests`, `SmolLMNativeCheckpointTests`, `QwenNativeCheckpointTests`.

## Planned commits
1. docs: plan #1373 in TODO.md
2. feat(samples): add SmolLM/Qwen loading helpers to LlamaLoader
3. refactor(inference): route Qwen loading through shared helper
4. refactor(inference): route SmolLM loading through shared helper
5. docs: remove TODO.md — plan executed

## GitHub issues log
- (none yet)
