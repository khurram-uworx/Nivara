# TODO — #596: injectable initialization for `Conv1d<T>` / `DepthwiseSeparableConv2d<T>`

## Problem

#594 gave `Conv2d<T>` (both ctors), `ConvTranspose2d<T>`, and `ConvVAE<T>` an optional
`IInitializer<T>? weightInitializer`, mirroring `Linear<T>`/`VAE<T>`. Two convolution layers were
left out:

1. `src/Nivara/AutoDiff/Nn/Conv1d.cs:78` initializes its weight inline from a hard-coded
   `new Random(42)` and exposes no hook — neither a seed nor an `IInitializer<T>`.
2. `src/Nivara/AutoDiff/Nn/DepthwiseSeparableConv2d.cs` constructs its `Conv2d` depthwise/pointwise
   sublayers directly, so it cannot forward an initializer even though `Conv2d` now accepts one.

Folded in (per the "long term / do it correctly" decision): the default draw itself is duplicated
verbatim in three places — `Conv1d.cs:77-80`, `Conv2d.cs:137-140`, `ConvTranspose2d.cs:770-773` —
and is neither named nor pinned by a test. `docs/AUTODIFF.md:668` also documents it **incorrectly**
(claims `U(-√(6/fanIn), √(6/fanIn))`; the code uses `√(2/fanIn)`).

## Decisions (raised at G1)

- **D1 — single shared initializer for `DepthwiseSeparableConv2d`.** One `weightInitializer`,
  forwarded to both sublayers, matching `ConvVAE` and the issue's own wording. Per-sublayer
  parameters rejected (no precedent, leaks the internal split). Escape hatch: `DepthwiseConv`/
  `PointwiseConv` are public, so a caller can re-init a sublayer afterwards.
- **D2 — how to consolidate the default draw.** The three inline draws are *not* semantically
  identical: `Conv1d`/`Conv2d` use `fanIn == shape[1]`, but `ConvTranspose2d` uses
  `fanIn = inChannels·k²` while its weight is shaped `[inChannels, outChannels·k²]`, so
  `fanIn != shape[1]` in general. A shape-derived `IInitializer<T>` therefore **cannot** reproduce
  `ConvTranspose2d`'s default bit-for-bit. **Recommendation:** an internal helper that fills a span
  given an explicit `fanIn`, used by all three ctors. Keeps every default byte-identical, no extra
  allocation, and one definition. Deviates from the earlier "named `IInitializer`" phrasing for this
  reason.
- **D3 — default bound is non-standard.** `√(2/fanIn)` uniform has variance `2/(3·fanIn)`; a true
  He-uniform is `√(6/fanIn)` (what `KaimingUniformInitializer<T>` does). Defaults must stay
  unchanged, so this change only *documents* the real formula and files an issue; it does not
  change the draw.
- **D4 — pin test form.** Exact equality against an independent re-derivation of the documented rule
  (`seed 42`, `bound = √(2/fanIn)`, `(u·2−1)·bound`) — not a golden float snapshot (breaks on .NET
  version bumps per the official `Random` docs) and not a tolerance (turns a structural change into a
  judgement call). `Conv1d` only: `DSC`'s default is a `null` pass-through into the untouched
  `Conv2d` path.

## Proposed changes

1. **New internal helper** `src/Nivara/AutoDiff/Nn/DefaultConvInitialization.cs` (namespace
   `Nivara.AutoDiff.Nn`) — the single definition of the built-in draw:

   ```csharp
   internal static class DefaultConvInitialization<T> where T : struct, IFloatingPointIeee754<T>
   {
       public static void Fill(Span<T> destination, int fanIn)
       {
           var bound = T.CreateChecked(Math.Sqrt(2.0 / fanIn));
           var rng = new Random(42);
           for (int i = 0; i < destination.Length; i++)
               destination[i] = T.CreateChecked((rng.NextDouble() * 2.0 - 1.0) * double.CreateChecked(bound));
       }
   }
   ```

2. **`Conv1d.cs`** — add `using Nivara.AutoDiff.Nn.Initializers;`; add
   `IInitializer<T>? weightInitializer = null` after `bool bias = true`; guard the default fill with
   `if (weightInitializer is null)` calling the helper; after `RegisterParameters(weight)` call
   `if (weightInitializer is not null) weightInitializer.Initialize(weight);`. Add `<param>` doc.

3. **`Conv2d.cs`** (`Conv2d` ctor) and **`ConvTranspose2d`** — replace the inline draw with
   `DefaultConvInitialization<T>.Fill(weightData, fanIn)`, each passing its own `fanIn`. Values
   unchanged.

4. **`DepthwiseSeparableConv2d.cs`** — add the `using`, the optional parameter + doc, and
   `weightInitializer: weightInitializer` to both `Conv2d` sublayer constructions.

5. **Tests** (`tests/Nivara.Tests/AutoDiff/NnTests.cs`):
   - `Conv1d_CustomWeightInit_UsesInitializer` — length `outChannels·inChannels·kernelSize` and the
     `DeterministicInit` ramp.
   - `DepthwiseSeparableConv2d_CustomWeightInit_ForwardsToEveryLayer` — both sublayer weights carry
     the ramp; pointwise bias stays zero.
   - `Conv1d_DefaultInit_KeepsFixedSeedDraw` — exact equality against the re-derived rule (D4).

6. **Docs** — `docs/AUTODIFF.md`: add the hook bullet + `weightInitializer?: null` to the `Conv1d`
   (line 651) and `DepthwiseSeparableConv2d` (line 763) sections; correct the Conv1d bound to
   `√(2/fanIn)` (line 668).

7. **`CHANGELOG.md`** — `[Unreleased] → Added` entry mirroring the #594 wording and the
   "omitted ⇒ unchanged" guarantee.

## Blast radius

- **Changed:** `Conv1d.cs`, `Conv2d.cs`, `DepthwiseSeparableConv2d.cs`, new
  `DefaultConvInitialization.cs`, `NnTests.cs`, `docs/AUTODIFF.md`, `CHANGELOG.md`.
- **Call sites of `Conv1d<T>` (all positional ⇒ source-compatible with a trailing optional param):**
  `samples/NivaraClassifier/ConvTextClassifierModel.cs`, `samples/NivaraTimeSeries/TimeSeriesModel.cs`,
  `WeightAccessConsistencyTests.cs:26-28`, `OpNodeInputContractTests.cs:597`, 4 PyTorch parity tests
  (which overwrite weights from `.bin`), `NnTests`, `ConvInferenceTests`. No public-API baseline file
  exists; `Nivara.Tests` has `InternalsVisibleTo`.
- **Touched by the default-draw swap:** `Conv2d` (default consumed by `ConvVAE`'s convergence
  trajectory, which relies on the fixed-seed draw) and `ConvTranspose2d`. Values are unchanged by
  construction, but these two ctors now route through the helper — the convergence test is the
  downstream guard.
- **Not affected:** forward/backward kernels, serialization, safetensors.

## Verification

- `dotnet build Nivara.slnx`
- Targeted (no-ask): `dotnet test -c Release --filter "FullyQualifiedName~AutoDiff.NnTests"`
  and `~NivaraTorch` (Release per the optimization/threshold rule). Full suite only with human OK.

## Planned commits

1. `docs: plan #596 in TODO.md`
2. `refactor(autodiff): centralize the convolution default weight draw`
3. `feat(autodiff): add weightInitializer to Conv1d`
4. `feat(autodiff): forward weightInitializer through DepthwiseSeparableConv2d`
5. `test(autodiff): pin Conv1d default draw and cover the new hooks`
6. `docs: document conv initializer hooks and correct the Conv1d bound`
7. `chore: changelog for #596`
8. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] (pending) — `ConvTranspose2d` default fanIn uses `inChannels·k²` (weight is
  `[inChannels, outChannels·k²]`); PyTorch uses `shape[1]·k²`. Verify intent; file during execution.
- [ ] (pending) — conv default uses `√(2/fanIn)` uniform (non-standard); consider standardizing to
  He-uniform `√(6/fanIn)`. File during execution; do not change here.
