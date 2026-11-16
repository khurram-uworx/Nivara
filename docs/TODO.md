# Plan: Issue #1241 — StateDictLoader silently skips missing keys on partial checkpoint load

## Problem

`StateDictLoader` helpers that build a partial state dict from `TryGetValue` guards and only
call `LoadStateDict` when the dict is non-empty silently leave randomly-initialized (but
trainable) weights/biases in place when a checkpoint key is missing. A half-loaded model can
still train and look fine, masking the missing data.

Silent-skip sites (verified against current source):
- `LoadLinear<TModel>` (single-type): dict built from `{prefix}.weight` / `{prefix}.bias`;
  `LoadStateDict` only if `dict.Count > 0`.
- `LoadLinear<TModel,TWeight>` (mixed): `BindWeight` only if weight present; bias
  `LoadStateDict` only if present.
- `LoadLinearWidened`: same pattern as `LoadLinear` (explicitly named in the issue).
- `LoadLayerNorm`: same pattern; leaves default gamma=1 when key missing (test comment
  in `ModernBertWeightLoadingTests` calls this out).
- `LayaDecisionHead.LoadWeights`: optional `temperature` buffer — legitimately optional
  (reference registers `torch.ones(3)`; checkpoint overrides), keep lenient.

Already strict (throw `KeyNotFoundException` naming the key): `LoadEmbed`, `LoadEmbed` (mixed),
`LoadEmbedWidened`, `LoadLinearSlice` (both), `LoadFusedLinearSlice` (both), `LoadRMSNorm`.
`DistilBertLoader.LoadEncoderWeights` is named in the issue but no such type exists in this
repo; the DistilBERT path here is `SilkDistilBertTrainer`, which does not use
`StateDictLoader`. Treat as out of scope / not present.

## Proposed changes

Add a `bool strict = false` parameter (default = current lenient behavior, for backward
compat) to the genuinely silent helpers:
- `LoadLinear<TModel>` (single)
- `LoadLinear<TModel,TWeight>` (mixed)
- `LoadLinearWidened`
- `LoadLayerNorm`

In strict mode, throw `KeyNotFoundException` naming the missing tensor (`{prefix}.weight`,
`{prefix}.bias`, `{prefix}.weight`, etc.) when a key the model expects is absent:
- weight is always expected (Linear always needs a weight; Forward throws if null).
- bias is expected only when the target module actually has a `Bias` parameter
  (`linear.Bias != null` / `ln.Bias != null`), mirroring `Module.LoadStateDict(strict)`
  which checks the model's own parameter keys.
- LayerNorm weight expected only when `ln.Weight != null` (affine); bias only when
  `ln.Bias != null`.

Propagate `strict` through the `LoadWeights` entry points so callers can opt in end-to-end:
- `ModernBertEncoder<TModel,TWeight>.LoadWeights(..., strict = false)` — threads strict
  into every `StateDictLoader` call it makes.
- `LayaDecisionHead<TModel,TWeight>.LoadWeights(..., strict = false)` — threads strict.
- `MiniLMDistilled<TModel,TWeight>.LoadWeights(..., strict = false)` (BertModel.cs) —
  threads strict (note: `tokenTypeEmbed` load stays guarded by the existing null check).

Also update XML docs on touched methods to describe strict behavior.

## Verification steps

- Build: `dotnet build Nivara.slnx` (Extensions + tests).
- Targeted tests: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~ModernBertWeightLoading|StateDict|LayaDecisionHead|MiniLM|BertModel"`.
- Add tests: lenient default unchanged (missing keys → no throw, weights stay default);
  strict → `KeyNotFoundException` naming the missing key, for each helper.
- Docs snippet gate: `EveryGatedBlock_CompilesWithoutErrors` (no doc changes expected, but verify).
- Docs coverage pins: re-check `DocsCoveragePins` after changes (no new docs/citations expected).

## Planned commits

1. `docs: plan issue #1241 in TODO.md`
2. `feat(AutoDiff): add strict param to StateDictLoader.LoadLinear/LoadLinearWidened/LoadLayerNorm`
3. `feat(AI): thread strict through ModernBertEncoder/LayaDecisionHead/MiniLM LoadWeights`
4. `test(AutoDiff): cover strict vs lenient missing-key behavior in StateDictLoader`
5. `docs: remove TODO.md — plan executed` (after G2)

## Blast radius

- `src/Nivara.Extensions/AI/StateDictLoader.cs` — signature additions (default params) to 4 methods; behavior unchanged when strict=false.
- `src/Nivara.Extensions/AI/ModernBertModel.cs` — `LoadWeights` signature + threading.
- `src/Nivara.Extensions/AI/LayaHeadModel.cs` — `LoadWeights` signature + threading.
- `src/Nivara.Extensions/AI/BertModel.cs` — `MiniLMDistilled.LoadWeights` signature + threading.
- Callers of these `LoadWeights` use positional args with defaults → still compile.
- Tests: existing `ModernBertWeightLoadingTests` etc. use `StateDictLoader` directly with old signatures → still compile (defaults).
- No core (`src/Nivara`) changes.

## GitHub issues log

- (none created yet; create if deferred work surfaces during execution)
