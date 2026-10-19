# Plan: LlamaLoader honor explicit AttentionBias flag (#717)

## Problem

`LlamaLoader.Load` infers `qkvBias` solely from tensor presence:

```csharp
bool qkvBias = tensors.ContainsKey("model.layers.0.self_attn.q_proj.bias");
```

A config that declares `AttentionBias = true` but ships bias tensors under alternate naming
therefore loads as bias-free. Honor the explicit config flag in addition, per issue #717.

## Proposed changes

1. `samples/Nivara.Samples/LlamaForCausalLM.cs` — `LLamaConfigLike`: add `bool AttentionBias { get; }`.
2. `samples/Nivara.Samples/LlamaForCausalLM.cs` — `LlamaLoader.Load`: change to
   `bool qkvBias = config.AttentionBias || tensors.ContainsKey("model.layers.0.self_attn.q_proj.bias");`.

`LlamaConfig` already exposes `AttentionBias`, so no change needed there.

## Blast radius

- Symbols: `LLamaConfigLike`, `LlamaLoader.Load`.
- Implementers of `LLamaConfigLike`: only `LlamaConfig` (grep confirmed).
- Callers of `LlamaLoader.Load`: tests and sample/inference code passing `LlamaConfig`.
- Tests: `SynthesizeTensorsBiasGatingTests`, Qwen parity tests.

## Verification steps

- `dotnet build Nivara.slnx`
- Targeted tests: `SynthesizeTensorsBiasGatingTests` and Qwen parity tests.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits

1. docs: plan #717 in TODO.md
2. fix: honor AttentionBias flag in LlamaLoader (interface + loader)
3. docs: remove TODO.md — plan executed

## GitHub issues log

- (none created yet)
