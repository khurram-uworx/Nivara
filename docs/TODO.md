# Plan for issue #715 — Tests: pin SynthesizeTensors q/k/v bias gating (#426)

## Problem

Issue #426 fixed `Qwen.SynthesizeTensors` so q/k/v bias tensors are emitted only when
`LlamaConfig.AttentionBias` is true (Qwen-style), and `LlamaLoader.Load` infers `qkvBias`
from tensor presence. The acceptance criteria were only verified manually; #715 asks for
regression tests pinning:

- `Qwen.SynthesizeTensors(smollmConfig)` emits no `model.layers.*.self_attn.{q,k,v}_proj.bias` keys.
- `Qwen.SynthesizeTensors(qwenConfig)` still emits them.
- Both synthesized checkpoints load via `LlamaLoader.Load<...>` with the correct
  `qkvBias` (false for SmolLM, true for Qwen).

## Proposed changes

Add `tests/Nivara.Tests/Qwen/SynthesizeTensorsBiasGatingTests.cs` with three tests using
small, self-contained `LlamaConfig` instances (`AttentionBias = false` for SmolLM,
`true` for Qwen) so the suite does not depend on the (gitignored) checkpoint files.
The load test checks `layers[0].Attention.QProj.Bias` nullness, which is the observable
signal for `qkvBias`.

No production code changes are required; the #426 fix is already in place.

## Verification steps

- `dotnet build Nivara.slnx`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SynthesizeTensorsBiasGatingTests"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits

1. `docs: plan #715 in TODO.md`
2. `test: pin SynthesizeTensors q/k/v bias gating (#715)`
3. `docs: remove TODO.md — plan executed`

## GitHub issues log

- #715 — Tests: pin SynthesizeTensors q/k/v bias gating (#426) (this issue)
