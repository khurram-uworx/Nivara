# TODO — Issue #181: Optimizer API consistency

Branch: `khurram/181` (off `main`). Tracks GitHub issue
[#181](https://github.com/khurram-uworx/Nivara/issues/181) (code-quality, medium;
REVIEW.md finding #10).

## Problem

The optimizer API presents two conflicting mental models for the same knob, and
one optimizer exposes a functional entry point the others don't:

1. `Optimizer<T>.LearningRate` is get-only, but nested `ParameterGroup.LearningRate`
   is settable and `SetGroupLearningRate` mutates a specific group.
2. `SGD<T>.SgdUpdate` is a public static functional entry point; Adam/AdamW have
   no comparable functional `Update`.
3. `Step()` documents accumulate-style semantics, but nothing pins it with a test.
4. `ParameterGroup` is public with mutable `LearningRate`/`WeightDecay`, inviting
   direct state corruption by consumers.

## Decisions (confirmed by human)

1. **LR mutation:** make `Optimizer<T>.LearningRate` settable; it validates and
   forwards to all groups created WITHOUT an explicit override (tracked via an
   internal `UsesDefaultLearningRate` flag). Explicit per-group overrides stay
   intact. `SetGroupLearningRate` clears the flag on the target group so an
   explicitly managed group is no longer driven by the base setter.
2. **Functional entry point:** keep `SgdUpdate` and ADD `AdamUpdate`/`AdamWUpdate`
   static functional entries that take explicit state (expAvg, expAvgSq buffers,
   step) and return a new `requiresGrad:false` tensor.
3. **Step-accumulation:** pin with a test and align the `Step()` doc contract:
   `Step()` leaves `Grad` intact; accumulation happens at `Backward()` time via
   `AccumulateGradient` (adds into the existing slot) — step-without-zero-grad
   accumulates stale gradients (PyTorch semantics).
4. **ParameterGroup hardening:** `LearningRate`/`WeightDecay` become public-read,
   internal-write; add `SetGroupWeightDecay` for parity with `SetGroupLearningRate`.

## Proposed changes

### `src/Nivara/AutoDiff/Optimizer/Optimizer.cs`

- `public T LearningRate { get; }` → settable property with private backing field.
  Setter validates positivity then forwards to every group with
  `UsesDefaultLearningRate == true`.
- `ParameterGroup`: add `internal bool UsesDefaultLearningRate`;
  `LearningRate`/`WeightDecay` become `{ get; internal set; }`.
- `AddParameterGroup(...)` overloads: the no-lr overloads create groups with
  `UsesDefaultLearningRate = true`; the explicit-lr overloads use `false`.
- `SetGroupLearningRate`: after mutating, clear the flag on the target group.
- New `SetGroupWeightDecay(int groupIndex, T weightDecay)` (bounds-check; no
  positivity validation — matches current `AddParameterGroup` weight-decay
  behavior).
- Align `Step()` XML doc with the accumulation contract.

### `src/Nivara/AutoDiff/Optimizer/Adam.cs` / `AdamW.cs`

- Refactor private `applyAdam`/`applyAdamW` to accept the destination writable
  span instead of always writing to `data.AsWritableSpan()`. Instance `Step()`
  passes the in-place span; functional path passes a fresh result array (kernels
  read `dataSpan` before writing `writable`, so aliasing stays safe).
- Add functional entries:

```csharp
public static ReverseGradTensor<T> AdamUpdate(
    ReverseGradTensor<T> tensor, T learningRate, T[] expAvg, T[] expAvgSq, int step,
    double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, T weightDecay = default)
```

(and the same for `AdamWUpdate`). Returns a new `requiresGrad:false` tensor,
mutates caller-owned buffers in place. Throws `ArgumentNullException` on null
tensor/buffers, `InvalidOperationException` on `Grad == null`,
`ArgumentException` on non-positive lr / step < 1. Wrapped in
`AutoDiffDiagnostics.Measure` with `AutoDiffAdamUpdate`/`AutoDiffAdamWUpdate`.

### `tests/Nivara.Tests/AutoDiff/OptimizerTests.cs`

- `LearningRate_Set_UpdatesDefaultGroups` (default group follows new base LR;
  explicit-override group does not).
- `LearningRate_Set_RejectsNonPositive`.
- `LearningRate_Set_MatchesSetGroupLearningRate` (identical stepped values).
- `LearningRate_Set_DoesNotOverrideExplicitlyManagedGroup` (after
  `SetGroupLearningRate`, base setter leaves it alone).
- `SetGroupWeightDecay_UpdatesGroup`.
- `ParameterGroup_MutableMembers_NotPublicSettable` (reflection contract pin —
  no `InternalsVisibleTo` needed).
- `AdamUpdate_*` / `AdamWUpdate_*`: simple case matches existing hand-computed
  instance-step references (`[0.99, 1.99, 2.99]`); multi-step matches instance
  `Step()`; `NoGradient_Throws`; `NonPositiveLr_Throws`.
- `Step_WithoutZeroGrad_AccumulatesGradients` (backward → step → backward without
  `ZeroGrad` → step, pinned against hand-computed double accumulation).

### Docs

- `docs/AUTODIFF.md` — Optimizer section: settable `LearningRate`, internal-mutation
  `ParameterGroup`, `SetGroupWeightDecay`, functional `AdamUpdate`/`AdamWUpdate`,
  Step/ZeroGrad accumulation contract.
- `docs/REVIEW.md` — mark finding #10 resolved.
- `CHANGELOG.md` — entry.
- `AGENTS.md` — add `AdamUpdate`/`AdamWUpdate` to "Useful helpers".
- `EXAMPLES.md` — unchanged (`SgdUpdate` kept); no doc change required.

## Blast radius

- **`Optimizer.cs`** — affects all three optimizers (SGD/Adam/AdamW derive from it)
  and every caller of `AddParameterGroup`/`SetGroupLearningRate`/`.LearningRate`.
  `LearningRate` setter is additive (was get-only), so no existing call sites
  break. Group creation semantics unchanged unless the base LR is later mutated.
- **`Adam.cs`/`AdamW.cs`** — the `applyAdam`/`applyAdamW` refactor is internal;
  `Step()` behavior must remain bit-identical (guarded by existing
  `OptimizerTests` hand-computed references).
- **`SGD.cs`** — untouched.
- **Downstream:** `TrainingLoop`/`DataParallelTrainer` only call
  `Step`/`ZeroGrad`/`StateDict`/`LoadStateDict` — unaffected. Samples using
  `SetGroupLearningRate` (MicroGpt, NivaraGpt) unaffected.
- **Tests:** `OptimizerTests.cs` (extended), `AutoDiffDiagnosticsTests.cs`
  (SgdUpdate keeps its op name). Full suite is the regression guardrail.

## Verification

- `dotnet build Nivara.slnx` after each step.
- Run `OptimizerTests` + `AutoDiffDiagnosticsTests` (ask before any
  long-running `dotnet test` per AGENTS.md).

## Planned commits

1. `docs: plan #181 optimizer API consistency in TODO.md`
2. `feat(autodiff): make Optimizer.LearningRate settable with default-group forwarding`
3. `feat(autodiff): add functional AdamUpdate/AdamWUpdate entries`
4. `test(autodiff): cover LR mutation, functional updates, and step accumulation`
5. `docs: document optimizer API consistency (#181)`
6. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] (none yet — create via `gh issue create --repo khurram-uworx/Nivara` as
      deferred work is found during execution, then record the number here)
