# Plan — #446: bias-free mode for `LayerNorm<T>`

## Problem

`LayerNorm<T>` (`src/Nivara/AutoDiff/Nn/LayerNorm.cs:37`) has `affine` as an
all-or-nothing flag: `affine: true` creates both `Weight` and `Bias`, `affine: false`
creates neither. There is no `affine: true, bias: false` mode.

A model whose config says `norm_bias: false` therefore cannot be represented exactly.
The only way to get one today is to construct a normal `LayerNorm<T>` and never load
`Bias`, leaving it at its zero initialization. That is numerically identical for
inference but semantically wrong for training: the optimizer still sees `Bias` as a
real parameter with a gradient, `Bias` drifts off zero on the first step, and the model
silently becomes a *different architecture* from the one the checkpoint defines. There
is no compile-time or runtime signal that this happened.

`ModernBertLayer` / `ModernBertEncoder` (`samples/Nivara.Samples/ModernBertModel.cs:358,360,399,403`)
do exactly this today, documented in the XML comment at line 344-346. Bias-free
pre-norm LayerNorm is common across modern architectures, not just ModernBERT.

## Proposed changes

### 1. `src/Nivara/AutoDiff/Nn/LayerNorm.cs` — the `bias` flag

Add `bool bias = true` after `affine` in the constructor. All 27 existing call sites
pass at most `(normalizedShape, eps)` or `(normalizedShape, eps, affine)`, so this is
source-compatible.

```csharp
public LayerNorm(int normalizedShape, float eps = 1e-5f, bool affine = true, bool bias = true)
{
    if (normalizedShape <= 0) throw new ArgumentOutOfRangeException(nameof(normalizedShape));
    if (eps <= 0) throw new ArgumentOutOfRangeException(nameof(eps));
    if (!affine && bias) throw new ArgumentException(
        "affine: false removes both gamma and beta; bias: true contradicts it.", nameof(bias));
    ...
    if (affine) { /* weight always */ }
    if (affine && bias) { /* bias */ }
}
```

Three edits in `Forward`:
- gamma/beta spans derive from `weight != null` / `bias != null` (drop the `affine &&`
  prefix). `affine` survives only to back the public `Affine` property, which now
  correctly means "gamma is applied".
- `useAffine` in the `OpNode` closure becomes the span-derived flag.
- `BackwardBias` is called only when `bias != null` — today `gradBetaData` is allocated
  unconditionally inside `if (useAffine)` at line 124 even when there is no bias.

### 2. `Module<T>` — deliberately unchanged

`Module.StateDict()` (`Module.cs:133`) is a pure projection of `Parameters()` →
`GetParameters()` → the registered-parameter list, and `LoadStateDict` (`Module.cs:150`)
keys off the same. So *not registering* the bias automatically delivers:

- `StateDict()` emits `Weight` only (acceptance 1a)
- `LoadStateDict()` on a `weight`-only checkpoint succeeds **in strict mode too**,
  because there is no `Bias` model parameter to be reported missing (acceptance 1b)
- the optimizer's parameter list has no `Bias`, so nothing can drift (acceptance 2)

This answers the issue's open question: make `beta` optional and do not register it.
No `HasBias` flag is needed for `StateDict`/`LoadStateDict` to honour, matching
`Linear<T>` (`Linear.cs:38`), which already has a `bias:` parameter and no `HasBias`
property. `Bias != null` is the signal.

### 3. `src/Nivara/AutoDiff/Nn/LayerNormKernel.cs` — three states, not two

`bool affine` cannot express `gamma without beta`. Replace it with span-length-derived
flags (`applyGamma = gamma.Length > 0`, `applyBeta = beta.Length > 0`) in `Forward`,
`ForwardInference`, and `BackwardInput`.

Bit-exactness: the affine branch today computes `diff*inv` into the pooled `diff`
buffer, multiplies by gamma into `output`, adds beta, then copies `diff` to `xHat`.
Writing `diff*inv` straight into `xHat` is the same operands in the same order, so
the results are identical.

Callers to drop the now-redundant `affine:` argument — all already pass empty spans,
so the argument was already redundant before this change:
- `TransformerBlock.cs:157, 169, 184` (`PerRowLayerNorm`)
- `tests/Nivara.Tests/AutoDiff/NnTests.cs:3597, 3708` (direct kernel tests)

### 4. `samples/Nivara.Samples/ModernBertModel.cs`

The four norm constructions become `bias: false`. XML docs at 344-346 and 381-382 lose
the "emulated by leaving the Beta at its zero initialization" hedge.
`StateDictLoader.LoadLayerNorm` is **unchanged** — it already `TryGetValue`s
`$"{prefix}.bias"`.

### 5. Tests

New `tests/Nivara.Tests/AutoDiff/LayerNormTests.cs`:

| Test | Assertion |
| --- | --- |
| `LayerNorm_BiasFalse_ExposesWeightAndNoBias` | `Weight` non-null, `Bias` null, `Affine` true |
| `LayerNorm_AffineFalseWithBias_Throws` | `ArgumentException` |
| `LayerNorm_BiasFalse_StateDict_EmitsWeightOnly` | acceptance 1a |
| `LayerNorm_BiasFalse_LoadStateDict_WithoutBias_SucceedsWhenStrict` | acceptance 1b |
| `LayerNorm_BiasFalse_LoadStateDict_WithBias_Throws` | loud, not silent |
| `LayerNorm_BiasFalse_Backward_GradientsWeightAndRegistersNoBias` | acceptance 2 |
| `LayerNorm_BiasFalse_Forward_MatchesZeroBetaEmulation` | **exact** equality vs. same-weight, zero-init-bias norm |
| `LayerNorm_BiasFalse_BackwardInput_MatchesZeroBetaEmulation` | **exact** equality on the input gradient |
| `LayerNorm_BiasFalse_Dispose_DisposesOnlyWeight` | no leak of the absent bias |

The two emulation tests assert **exact** equality, not a tolerance band
(`GUIDELINES.md`, "exactness over tolerance"): the bias-free path performs strictly
fewer operations, so the only possible divergence is signed zero, and NUnit's `==`
treats `-0.0f == 0.0f`.

Plus: one focused test in `WeightAccessConsistencyTests` for the weight-only accessor
contract (the shared `ModuleSpec` is keyed on `Name` with a name-based `affine:false`
special case, so a second LayerNorm row cannot be added without refactoring that
test's logic — out of scope here); one assertion in `ModernBertWeightLoadingTests`
that loaded norms have `Bias == null`; one case in `InferenceFastPathTests` for the
weight-only inference path, which is a separate kernel branch currently only covered
for `affine: false`.

### 6. Docs

`docs/AUTODIFF.md` §LayerNorm (signature comment + kernel tree),
`docs/MODERNBERT.md:13`, `docs/LAYA.md:207`, the stale "matching the CPU's
zero-initialised Beta" comment at `ModernBertGpuRunner.cs:32`, `CHANGELOG.md`.

## Blast radius

- **Library (`src/Nivara/AutoDiff/Nn/`)**: `LayerNorm.cs`, `LayerNormKernel.cs`.
  `Module.cs`, `Parameter.cs`, `Optimizer.cs` untouched.
- **Kernel signature change** — 5 call sites: `LayerNorm.cs` ×3, `TransformerBlock.cs`
  ×3 (both in `PerRowLayerNorm`), `NnTests.cs` ×2. All are `internal`, so no public
  API breaks.
- **Public API addition**: one optional trailing constructor parameter. Binary- and
  source-compatible for all 27 existing `new LayerNorm<...>` call sites.
- **Behavior change**: `LayerNorm` with `bias: false` has no `Bias` key in
  `StateDict()`/`GetParameters()`. Anyone who round-tripped a *biased* norm through
  `StateDict()`→`LoadStateDict(strict: true)` is unaffected (they keep the default
  `bias: true`). `Affine` now returns `true` for `affine: true, bias: false` — correct
  under its documented wording ("the affine gamma/beta transform is applied" — gamma
  *is* applied), but a consumer that tested `Affine` to mean "has parameters" would
  need `Weight != null`. No such consumer exists in-repo; it is public API, so noted
  in the CHANGELOG.
- **Sample**: `ModernBertModel.cs` ×4 constructions; `StateDictLoader.cs` unchanged.
- **GPU sample**: `ModernBertGpuRunner.cs` is out of scope — it is a hand-rolled
  ILGPU inference path with its own `layerNorm` kernel and a shared `zeroBeta` buffer
  (lines 32, 59, 131, 296-304), it has no optimizer, and it does not use
  `LayerNorm<T>`. Comment at line 32 goes stale and is corrected.
- **Tests**: new `LayerNormTests.cs`; additions to `NnTests.cs` (drop `affine:`),
  `WeightAccessConsistencyTests.cs`, `ModernBertWeightLoadingTests.cs`,
  `InferenceFastPathTests.cs`.

## Verification

1. `dotnet build Nivara.slnx`
2. `dotnet test` — full NUnit suite. The two exactness tests are load-bearing.
3. `dotnet run --project samples/NivaraInference -c Release -- modernbert compare`
   — CPU-only, fixtures are local in `samples/data/modernbert/`
   (`model.safetensors`, `last_hidden_state_py.bin`, `input_ids_py.bin`,
   `compare_meta.json`). Must reproduce the published baseline in
   `docs/MODERNBERT.md:52`: `maxAbs 1.62e-5`, `meanAbs 9.22e-7`, `maxRel 6.09e-4`,
   `cosine 1.0000000000`, under the existing gate bound `|diff| <= 1e-3·(1 + |ref|)`.
   **If it does not reproduce, the zero-Beta emulation was not exact — that is a
   finding to report, not to paper over.**

## Planned commits

1. `feat(autodiff): add a bias-free mode to LayerNorm` — kernel + module + tests
2. `fix(samples): ModernBERT encoder uses the real bias-free norm` — sample switch +
   parity-gate result
3. `docs: record the LayerNorm bias-free mode` — docs + CHANGELOG

## Reminder

As each task executes, if deferred work or a concern surfaces, create a GitHub issue
immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number in
the log below — do not hold it in memory, since compaction during execution can lose
it.

## GitHub issues log

- [ ] #446 — LayerNorm<T> has no bias-free mode (the work of this plan)
