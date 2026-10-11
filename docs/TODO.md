# Plan — issue #594: `ConvVAE<T>` (and `Conv2d`/`ConvTranspose2d`) weight-initializer hook

> **Reminder (per iterative-work):** as each task executes, if deferred work or a concern
> surfaces that is outside this plan, create a GitHub issue immediately
> (`gh issue create --repo khurram-uworx/Nivara`) and record its number in the
> **GitHub issues log** below. Do not rely on memory or wait until the end of the plan —
> compaction can lose it.
>
> **Landed on the same branch/PR as #583** at the human's request (`khurram/583`, PR #595).

## Problem

`VAE<T>` gained optional `weightInitializer`/`biasInitializer` constructor parameters in #583.
`ConvVAE<T>` has no equivalent hook, so callers cannot inject a custom (e.g. seeded) initializer —
an API asymmetry with `VAE<T>`.

### Corrected finding (the filed issue's premise was partly wrong)

`Conv2d<T>` and `ConvTranspose2d<T>` do **not** use `KaimingUniformInitializer<T>`/`Random.Shared`;
they initialize weights inline from a hard-coded `new Random(42)`
(`src/Nivara/AutoDiff/Nn/Conv2d.cs:131` and `:756`, He bound `sqrt(2/fanIn)`; bias stays zeros).
So conv initialization is **already deterministic**, and it was never the source of the loose
`finalLoss < firstLoss * 2.0f + 0.5f` guard in `ConvVAE_EndToEnd_ReducesLoss`.

The test's only nondeterminism is the **two unseeded `cvae.Reparameterize(mu, logVar)` calls**
(`tests/Nivara.Tests/AutoDiff/NnTests.cs:3375` and `:3396`), which drain `Random.Shared` through
`ReverseGradOperations.SampleNormal(..., seed: null)`. The hook #594 asks for is therefore an
**API-symmetry/feature** addition, not a flake fix. This plan corrects #594's body accordingly.

## Decisions (confirmed by the human)

- **Scope:** `Conv2d<T>` (both ctors), `ConvTranspose2d<T>`, and `ConvVAE<T>` only.
  `Conv1d<T>` and `DepthwiseSeparableConv2d<T>` also hard-code `new Random(42)` and are left as a
  tracked follow-up (see issues log).
- **Default:** when no initializer is supplied, keep the current fixed-seed-42 He draw
  **byte-for-byte**. The change is purely additive; no existing caller's weights change.
- `weightInitializer` only (no `biasInitializer`); conv bias stays zero-initialized as today.

## Proposed changes

### 1. `src/Nivara/AutoDiff/Nn/Conv2d.cs`

- Add `using Nivara.AutoDiff.Nn.Initializers;`.
- `Conv2d<T>` symmetric-padding ctor: append `IInitializer<T>? weightInitializer = null` and
  forward it to the per-side-padding ctor.
- `Conv2d<T>` per-side-padding ctor: append the same parameter; generate `weightData` from the
  fixed-42 draw only when `weightInitializer is null`, then
  `weightInitializer.Initialize(weight)` after `RegisterParameters(weight)` (mirroring
  `Linear<T>`).
- `ConvTranspose2d<T>` ctor: same shape of change.
- XML docs for the new parameter on each ctor.

### 2. `src/Nivara/AutoDiff/Nn/ConvVAE.cs`

- Add `using Nivara.AutoDiff.Nn.Initializers;`.
- Append `IInitializer<T>? weightInitializer = null` to the constructor (after `padding`) with an
  XML doc entry.
- Forward `weightInitializer: weightInitializer` to all five `new Conv2d<T>` / `new ConvTranspose2d<T>`
  sites (lines 57, 61, 62, 73, 76).

### 3. Tests — `tests/Nivara.Tests/AutoDiff/NnTests.cs`

- `ConvVAE_EndToEnd_ReducesLoss` (3341–3401): remove the dead `var rng = new Random(42);` (3356);
  pass `seed: 42` to both `Reparameterize` calls (3375, 3396); tighten the assertion to a strict
  `finalLoss[0] < firstLoss` **if** a Release run confirms it holds over 5 epochs (else raise the
  epoch count rather than keep the 2×+0.5 slack).
- New `Conv2d_CustomWeightInit_UsesInitializer`: construct with `DeterministicInit<float>()` and
  assert the weight data equals `0.05*(i+1)` (the hook is actually applied).
- New `ConvTranspose2d_CustomWeightInit_UsesInitializer`: same for the transposed conv.
- New `ConvVAE_CustomWeightInit_ForwardsToEveryLayer`: construct `ConvVAE<float>` with
  `DeterministicInit<float>()`; iterate `GetParameters()` and assert every key ending in `Weight`
  matches `0.05*(i+1)` and every `Bias` is zero — proving the hook reaches all conv layers.

### 4. Docs

- `docs/AUTODIFF.md`: add the new optional `weightInitializer` argument to the `ConvVAE<T>` /
  `Conv2d<T>` / `ConvTranspose2d<T>` signature comments.
- `CHANGELOG.md` (`[Unreleased] → Added`): one entry for the conv initializer hook (#594).

## Blast radius

- `Conv2d<T>`: two public ctors; callers include `ConvVAE<T>` and `DepthwiseSeparableConv2d<T>`.
  Appending an optional parameter is source-compatible; the default path is unchanged.
- `ConvTranspose2d<T>`: one public ctor; callers include `ConvVAE<T>`.
- `ConvVAE<T>`: constructor call sites are `NnTests.cs` (`ConvVAE_*`) — all use named or positional
  args within the existing parameters, so appending is source-compatible.
- No behavior change when the new argument is omitted (weights remain the fixed-42 He draw).
- Covered by the existing `ConvInferenceTests` / `NnTests` conv suites plus the new tests.

## Verification steps

1. Targeted Release runs (repeat to show stability):
   - `dotnet test -c Release --filter "FullyQualifiedName~Conv"` (Conv2d/ConvTranspose2d/ConvVAE)
   - the #583 filter (`~DataParallelTests|~VAE_Training_ReducesLoss|~LinearModule_BFloat16SgdTraining_LossDecreases`)
   - `~VAE|~OpNodeInputContractTests` (VAE ctor regression from #583)
2. Confirm the tightened `ConvVAE_EndToEnd_ReducesLoss` assertion holds; adjust epochs if not
   (never loosen past determinism).
3. Full suite only with explicit human confirmation, always with `--filter "Category!=Performance"`.

## Planned commits

1. `feat(autodiff): add optional weight initializer to Conv2d/ConvTranspose2d/ConvVAE`
2. `test(autodiff): make ConvVAE convergence test deterministic and cover the conv hook`
3. `docs(autodiff): document the conv initializer hook`

## GitHub issues log

- [ ] #594 — this work (`ConvVAE<T>` initializer hook); body to be corrected re: the conv-init
      premise.
- [ ] (to file) follow-up — `Conv1d<T>` / `DepthwiseSeparableConv2d<T>` still hard-code
      `new Random(42)` with no seed/initializer hook.
- [ ] #595 — PR carrying both #583 and #594; body to be updated and linked to #594.
