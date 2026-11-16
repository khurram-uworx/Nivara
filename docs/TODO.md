# Plan: NivaraChat SmolLM/Qwen drop --precision, load native weights only (issues #1216, #1221)

## Problem
`NivaraChat --smollm` and `--qwen` still expose `--precision` and do not default to native (BF16 storage / F32 compute) weights, unlike `NivaraInference`. Both options are wrong-shaped for a chat host: `bf16` is BF16 compute (scalar fallback), default `f32` widens the whole checkpoint at load. Goal: exactly one weight-loading path (native) and no `--precision` flag.

## Proposed changes
See plan summary. Files: SmollmMode.cs, SmolLMChatClient.cs, QwenMode.cs, QwenChatClient.cs, Program.cs, docs/SMOLLM.md, docs/QWEN.md.

## Verification
- `dotnet build samples/NivaraChat`
- single-turn plain runs on local snapshots
- `--precision bf16`/`--precision f32` fall through as unknown (not honoured)

## Planned commits
1. docs: plan
2. SmolLM: drop --precision, native load, SmolLMChatClient<TModel,TWeight>
3. Qwen: drop --precision, native load, QwenChatClient<TModel,TWeight>
4. Program.cs help line
5. docs updates + DocsCoveragePins if needed

## Blast radius
samples/NivaraChat only; no core library. Call sites compile with <float, BFloat16>. No LlamaForCausalLM<T,T> remaining in sample.

## GitHub issues log

- [ ] #1216 — NivaraChat SmolLM drop --precision
- [ ] #1221 — NivaraChat Qwen drop --precision
