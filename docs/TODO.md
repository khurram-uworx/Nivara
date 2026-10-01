# #494 — BatchNorm eval mode produces no weight/bias gradient

Branch: `khurram/494`. Issue: https://github.com/khurram-uworx/Nivara/issues/494

> This file previously held the merged-and-closed #487 plan, which was never
> cleaned up (PR #495 merged with it still present). Its content was discarded
> with the human's approval — #487's outcome lives in `CHANGELOG.md` and its
> commit messages. This plan supersedes it here.

## Problem

`BatchNorm1dTrain` and `BatchNorm2d` accumulate gradients into `weight.Tensor` and
`bias.Tensor`. Their eval-path twins — `BatchNorm1dEval` and `BatchNorm2dEval` in
`src/Nivara/AutoDiff/Nn/BatchNorm.cs` — do not: those closures call
`AccumulateGradient` on `input` only, and list only `[input]` in `OpNode.Inputs`.

The eval *forward* still depends on gamma and beta:
`BatchNormKernel<T>.ForwardEval` is passed `gamma` and `beta` and computes
`y = gamma * xhat + beta`. So the parameters are inside the differentiable path,
but backward silently produces nothing for them.

Failure shape (no error, no warning — a clean backward with an absent gradient):

```csharp
using (GradientUtils.Grad())
{
    bn.Eval();
    var y = bn.Forward(input);   // input.RequiresGrad == true
    loss.Backward();
    // bn.Weight!.Tensor.Grad == null
}
```

### Relationship to #487

#487 (merged as PR #495) deliberately left the eval path as `[input]`: at that time
its backward accumulated into nothing else, so `[input]` was contract-correct.
Its commit message explicitly deferred the PyTorch divergence to this issue. The
resolution here therefore **amends** that decision — and the test that pinned it
(`BatchNorm_EvalPath_ListsOnlyTheInput`, added by 4d2fb0b) — deliberately.

This is not a `LayerNorm`/`RMSNorm`/`Conv` concern: those have no eval/train split
in this codebase, so they have a single backward that already covers parameters.

## Decision (Option A — PyTorch parity)

Chosen over (B) throw-on-eval-backward and (C) document-as-intentional because the
eval forward genuinely differentiates gamma/beta, and the repo carries a PyTorch
parity suite (`tests/Nivara.Tests/NivaraTorch/`). Divergence is the bug.

### What Torch does (verified empirically, torch 2.13.0+cpu)

PyTorch's eval-mode BN backward computes grad_weight/grad_bias and lets autograd
attach them per-leaf. The gate is each leaf's own `requires_grad`, decided when the
graph is built — **never** a runtime training-flag branch. Probe results:

| eval scenario                                | `weight.grad` | `bias.grad` |
| -------------------------------------------- | ------------- | ----------- |
| affine, all require grad                     | populated     | populated   |
| affine, **input** not require grad           | populated     | populated   |
| affine, **weight** not require grad          | `None`        | populated   |
| affine, **bias** not require grad            | populated     | `None`      |
| affine, `track_running_stats=False`          | populated     | populated   |

Row 2 is the load-bearing one: an input without a gradient does **not** suppress
the parameter gradients. That is precisely the case Nivara gets wrong today.

## Proposed changes

### 1. `src/Nivara/AutoDiff/Nn/BatchNorm.cs` — `BatchNorm1d.Forward` eval branch (~line 141)

```csharp
var gradFn = new OpNode<T>("BatchNorm1dEval",
    affine
        ? ModuleHelpers<T>.NodeInputs(input, weight?.Tensor, bias?.Tensor)
        : ModuleHelpers<T>.NodeInputs(input),
    (typedGradOutput) =>
{
    /* existing: BackwardInput -> AccumulateGradient(input) */

    if (affine)
    {
        var gradGammaData = BatchNormKernel<T>.BackwardWeight(
            gradOutData, savedXHat, savedN, savedC, savedPlaneSize);
        var gradBetaData = BatchNormKernel<T>.BackwardBias(
            gradOutData, savedN, savedC, savedPlaneSize);

        if (weight != null)
            ReverseGradOperations.AccumulateGradient(weight.Tensor, NivaraColumn<T>.Create(gradGammaData));
        if (bias != null)
            ReverseGradOperations.AccumulateGradient(bias.Tensor, NivaraColumn<T>.Create(gradBetaData));
    }
});
```

### 2. `BatchNorm2d.Forward` eval branch (~line 396)

Same, with `"BatchNorm2dEval"` and `hw` as plane size.

No new kernel math: `ForwardEval` populates `xHat` on **both** affine branches
(BatchNormKernel.cs:217-220), so `BackwardWeight`/`BackwardBias` are valid as-is.
`gradGamma = Σ(gradOut · xHat)` and `gradBeta = Σ gradOut` are identical in form
to the train path — eval's `xHat` is simply computed from running stats.

### 3. `tests/Nivara.Tests/AutoDiff/OpNodeInputContractTests.cs:222`

Rewrite `BatchNorm_EvalPath_ListsOnlyTheInput` ->
`BatchNorm_EvalPath_ListsAffineParameters`:

- affine eval: `Inputs` contains input + weight + bias
- `affine: false` eval: `Inputs == [input]`

Rename rather than delete, so eval-path coverage is preserved. The comment must
record that the old assertion pinned a since-resolved divergence, so a future
reader does not "restore" it.

### 4. Parity tests — `tests/Nivara.Tests/AutoDiff/NnTests.cs`

- `BatchNorm1d_EvalMode_ParameterGradientsFlow` / `BatchNorm2d_...`
- `BatchNorm*_EvalMode_AffineFalse_NoParameterGradients`

Expected values derived from the PyTorch reference and asserted **exactly** (the
repo's exactness rule: a tolerance turns a structural defect into a judgement call).

### 5. Docs

CHANGELOG entry — this is a public behaviour change (PyTorch parity).

## Blast radius

- **Modified:** `src/Nivara/AutoDiff/Nn/BatchNorm.cs` (2 closures, +Inputs each).
- **Modified:** `tests/.../OpNodeInputContractTests.cs` (1 test rewritten), `tests/.../NnTests.cs` (new tests).
- **Downstream of the behaviour change:** any caller that runs `Backward()` while in
  `Eval()` now also receives parameter gradients. Validation loops that call
  `Backward()` purely for input gradients must be reviewed. `Optimizer.ZeroGrad()`
  walks `Module.Parameters()` directly, and `TrainingLoop` enters `Grad()` internally —
  the standard train path is unaffected (it never takes the eval branch).
- **Not affected:** no forward-mode BatchNorm exists (BatchNorm lives only in `Nn`,
  reverse-mode only — confirmed absent from `ForwardGradOperations.cs`).
- **Not affected:** `ZeroGrad` reachability, since module parameters are graph leaves;
  `BuildBackwardPlan`/`GetGraphInfo` skip them as before. Listing them in `Inputs`
  only widens what `GradientUtils.ZeroGrad` clears — which is #487's stated intent.

## G1 grounding outcome

**Empirical (torch 2.13.0+cpu, run directly).** Five eval-mode probes established
that PyTorch's BN backward computes grad_weight/grad_bias and lets autograd attach
them per-leaf, gated on each leaf's own `requires_grad` — never on the training
flag. Decisive case: an input without a gradient does **not** suppress the
parameter gradients. Full table above.

**Formula cross-checked against Microsoft's DirectML BN grad operator** (the
inference-mode sibling of the training-grad operator):
`OutputScaleGradient = sum(InputGradient * (Input - Mean) / sqrt(Variance + Epsilon))`
and `OutputBiasGradient = sum(InputGradient)` — identical in form to Nivara's
`BackwardWeight`/`BackwardBias`. Two independent references agree, so the eval
path reuses the existing kernels unchanged.

**`ForwardEval` populates `xHat` on both affine branches** (BatchNormKernel.cs:217-220),
so the saved value needed by `BackwardWeight` is already there — no kernel change.

**No forward-mode BatchNorm exists.** BatchNorm lives only in `Nn/`, reverse-mode
only; `ForwardGradOperations.cs` contains no BatchNorm reference. Single-mode fix.

### Regression check: no caller runs `Backward()` while in `Eval()`

Every `Backward()` in the tree was audited:

| Site | Eval nearby? |
|---|---|
| `Training/TrainingLoop.cs:155` | no — loop never touches `Eval`/`IsTraining` |
| `Training/DataParallelTrainer.cs:124` | no |
| `samples/NivaraFineTuning/Program.cs:208` | no — that `Backward` is in the train loop; the three `Eval()` calls (249, 277, 318) are in later evaluate/predict modes |
| `samples/NivaraChat/Training/IntentTrainer.cs:162` | no — `Eval()` is the final statement, after `loop.Run()` returns |
| all `NivaraChat` / `NivaraInference` sites | inference only, no backward |

So the new parameter gradients cannot reach an optimizer unexpectedly. The one
intended behaviour change is the documented one: `Backward()` in eval now yields
correct PyTorch-matching parameter gradients instead of `null`.

## Verification steps

1. `dotnet build Nivara.slnx` — Debug, compile check only.
2. `dotnet test -c Release --filter "FullyQualifiedName~BatchNorm|FullyQualifiedName~OpNodeInputContract"`
   — **ask the human before running.** Release is mandatory (Debug runs are false
   regressions; AGENTS.md).
3. Full `dotnet test -c Release --filter "Category!=Performance"` if the targeted run is green.

## Planned commits

1. `docs: plan #494 BatchNorm eval-mode parameter gradients in TODO.md`
2. `fix(autodiff): accumulate weight/bias gradients on the BatchNorm eval path`
3. `test: assert BatchNorm eval path lists its affine parameters`
4. `test: cover BatchNorm eval-mode parameter gradients against PyTorch`
5. `docs: record the #494 PyTorch-parity behaviour change in CHANGELOG`
6. `docs: remove TODO.md — #494 plan executed`

## GitHub issues log

- [ ] #494 — BatchNorm eval mode produces no weight/bias gradient (this work)

> As each task executes, if you find deferred work or a concern, create a tracked
> issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its
> number here — do not rely on memory, compaction can lose it.