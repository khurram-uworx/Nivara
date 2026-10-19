# Plan: Fix #426 — SynthesizeTensors q/k/v bias gating

## Problem

`samples/NivaraInference/Qwen.cs` `SynthesizeTensors(LlamaConfig)` unconditionally emits
`q_proj`/`k_proj`/`v_proj` bias tensors. `LlamaLoader.Load` infers `qkvBias` from tensor
presence, so a synthesized SmolLM checkpoint (real checkpoint has no q/k/v biases,
`attention_bias: false`) would load with `qkvBias == true`, corrupting synthetic SmolLM numerics.

## Proposed changes

1. `samples/Nivara.Samples/LlamaForCausalLM.cs` — `LlamaConfig`:
   - Add `public bool AttentionBias { get; init; } = true;`
   - Add `SnakeKey` mapping `AttentionBias -> attention_bias`.
2. `samples/NivaraInference/Qwen.cs` — `SynthesizeTensors`:
   - Make `public static`.
   - Gate q/k/v bias emission on `config.AttentionBias`.

## Verification

- Build solution.
- Targeted tests in Nivara.Tests (Qwen namespace) if available.
- Manual check: synthesized SmolLM has no bias keys; Qwen has them.

## Planned commits

1. `docs: plan #426 in TODO.md`
2. `Fix #426: gate q/k/v bias synthesis on AttentionBias`
3. `docs: remove TODO.md — plan executed`

## Blast radius

- `LlamaConfig` (samples) — new property, parsed from config.json.
- `Qwen.SynthesizeTensors` — visibility + conditional emission.
- No core library changes.

## GitHub issues log

- (none yet)
