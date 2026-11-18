# Plan: NivaraChat CPU-vs-Silk parity test for SmolLM (#1326)

## Problem

SmolLM has only BF16-vs-BF16 parity coverage (`SilkSmolLMBf16ParityTests`). Qwen has
`SilkQwenParityTests` (CPU `LlamaForCausalLM` vs `SilkQwenSession`, prefill + cached step), and
TinyShakespeare has `SilkTinyShakespeareParityTests`. SmolLM lacks the CPU-vs-Silk parity gate, so a
divergence between `LlamaLoader`/CPU forward and the Silk runner could ship silently.

## Proposed changes

Add `tests/Nivara.Tests/SmolLM/SilkSmolLMParityTests.cs` mirroring `SilkQwenParityTests`:

- `ForwardPrefill_SyntheticWeights_SilkMatchesCpuArgmaxAndLogits`
- `ForwardCached_SyntheticWeights_SilkMatchesCpuArgmaxAndLogits`

Use `SmolLMTestWeights.Synthesize()` + `SilkSmolLMSession.Open(config, tensors)` (F32), same prompt
`[1, 2, 3, 4]`, argmax + logits closeness within `1e-3`.

`SmolLMTestWeights.Synthesize()` returns a `SmolLMConfig` (Silk) while the CPU path needs a
`LlamaConfig` (`LLamaConfigLike`). Build a `LlamaConfig` in-test by copying fields, with
`TieWordEmbeddings = true` and `HiddenAct = "silu"` (matches the synthetic weight layout, which has
only `model.embed_tokens.weight` and no separate LM head).

`[Category("Gpu")]`, ignore when no OpenCL device (standard probe pattern).

No production code changes.

## Verification steps

- `dotnet build tests/Nivara.Tests/Nivara.Tests.csproj -c Release`
- Targeted: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkSmolLMParityTests"` (ignored without OpenCL)
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits

1. `docs: plan SmolLM CPU-vs-Silk parity test in TODO.md`
2. `test(smollm): add CPU-vs-Silk parity tests (#1326)`

## Blast radius

- New test file only. No production code. No doc pins affected (no docs/*.md changes).

## GitHub issues log

- (none yet)
