# Plan: Expose qkvBias on LlamaForCausalLM<T> (#716)

## Problem
`LlamaForCausalLM<T>` accepts `qkvBias` in its constructor and forwards it to every decoder
block, but the value is not exposed publicly. Tests/diagnostics must reflect into `layers`
to assert the loaded model's bias mode. Follow-up to #426.

## Proposed changes
- `samples/Nivara.Samples/LlamaForCausalLM.cs`: store `qkvBias` field; add public read-only
  `QkvBias` property.
- `src/Nivara/AutoDiff/Nn/LlamaDecoderBlock.cs`: add public read-only `QkvBias` property
  delegating to `Attention.QkvBias`.
- `src/Nivara/AutoDiff/Nn/LlamaCausalAttention.cs`: store `qkvBias` field; add public
  read-only `QkvBias` property.

## Verification
- `dotnet build Nivara.slnx`
- Targeted tests: `SynthesizeTensorsBiasGatingTests`, `QwenInstructParityTests` (if model data present).

## Planned commits
1. docs: plan #716 in TODO.md
2. feat: expose QkvBias on LlamaForCausalLM, LlamaDecoderBlock, LlamaCausalAttention
3. docs: remove TODO.md — plan executed

## GitHub issues log
- (none yet)

## Blast radius
- Public API additions only; no behavior change. Downstream: samples, tests referencing model bias mode.