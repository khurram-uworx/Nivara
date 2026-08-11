# Plan: Issue #180 — Loss API is a grab bag (common base + `Reduction` enum)

Branch: `khurram/180` (off `main`). No package version bump this session — `src/Nivara/Nivara.csproj` left at 1.2.0.

## Problem

The functional loss classes (`src/Nivara/AutoDiff/Nn/Functional/`) have no common base, no
`Reduction` enum, and three different reduction/ctor styles:

| Loss | Today | Reduction surface | Ctor style |
|---|---|---|---|
| `BCELoss<T>` | always sum | none | `(double eps = 1e-7)` stateful |
| `BCEWithLogitsLoss<T>` | sum by default | `Forward(..., bool reduceToMean)` | stateless |
| `CrossEntropyLoss<T>` | always mean | none | stateless |
| `MSELoss<T>` | sum by default | `Forward(..., bool reduceToMean)` | stateless |
| `L1Loss<T>` | always sum | none | stateless |
| `Softmax<T>` / `LogSoftmax<T>` | activations, misgrouped with losses | unused `dim` resolved in #179 | stateful |

`TrainingLoop<T>` / `DataParallelTrainer<T>` work around this by taking `Func<tensor,tensor,tensor>`.
`CrossEntropyLoss` mean divides by batch; MSE/BCEWithLogits mean divides by element count — three
divergent mean implementations. See `docs/REVIEW.md` item 9/10.

## Design decisions (confirmed with human)

1. **Constructor-based, nn-style** reduction: `Loss<T>` base stores a `Reduction` set at
   construction; `Forward(p, t)` uses it; a `Forward(p, t, Reduction)` overload allows per-call
   override (mirrors `nn.Loss` + `F.*` dual shape).
2. **Default reduction = `Mean` everywhere** (PyTorch parity). Callers that relied on the old
   implicit sum must pass `Reduction.Sum` explicitly.
3. **`Reduction.None` implemented now** — each loss returns the elementwise loss tensor.
4. `BCELoss.eps` stays a ctor argument.
5. `Softmax`/`LogSoftmax` move to `Activation` static wrappers; the two classes are deleted
   (verified: no production/sample usage, tests only).
6. `TrainingLoop`/`DataParallelTrainer` signatures stay `Func<...>` — `Loss<T>` composes via
   method group (`loss.Forward`). No change to the training-loop classes.
7. **No version bump** in this session.

## New core surface

```csharp
// src/Nivara/AutoDiff/Nn/Functional/Reduction.cs
namespace Nivara.AutoDiff.Nn.Functional;
public enum Reduction { Sum, Mean, None }
```

```csharp
// src/Nivara/AutoDiff/Nn/Functional/Loss.cs
public abstract class Loss<T> where T : struct, IFloatingPointIeee754<T>
{
    public Reduction Reduction { get; }
    protected Loss(Reduction reduction);                       // validates defined value

    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> predictions, ReverseGradTensor<T> targets)
        => Forward(predictions, targets, Reduction);

    public abstract ReverseGradTensor<T> Forward(
        ReverseGradTensor<T> predictions, ReverseGradTensor<T> targets, Reduction reduction);

    // single shared reduction path; divisor overridden to batch size by CrossEntropyLoss
    protected static ReverseGradTensor<T> Reduce(
        ReverseGradTensor<T> elementwiseLoss, Reduction reduction, int? divisor = null);
    // None -> elementwise; Sum -> ReverseGradOperations.Sum; Mean -> Divide(Sum, Full(1, T.CreateChecked(divisor ?? length)))
}
```

Loss rewrites (each keeps per-loss config in the ctor; `bool reduceToMean` overloads removed):

| Class | Ctor | Elementwise core | Mean divisor |
|---|---|---|---|
| `MSELoss<T>` | `(Reduction = Mean)` | `(p-t)²` | length |
| `L1Loss<T>` | `(Reduction = Mean)` | `\|p-t\|` | length |
| `BCELoss<T>` | `(Reduction = Mean, double eps = 1e-7)` | clamped BCE | length |
| `BCEWithLogitsLoss<T>` | `(Reduction = Mean)` | fused per-element loop + custom OpNode backward (unchanged); None returns elementwise tensor | length |
| `CrossEntropyLoss<T>` | `(Reduction = Mean)` | `-logSoftmax · targets`; keep `Forward(logits, int[] targets)` (delegates with ctor reduction) | batch (`shape[0]`) |

`Activation` gains `Softmax<T>(input, int dim = -1)` and `LogSoftmax<T>(input, int dim = -1)`;
`Softmax.cs`/`LogSoftmax.cs` deleted.

## Blast radius

**Core:** `src/Nivara/AutoDiff/Nn/Functional/` (2 new files, 5 rewritten, 2 deleted) +
`src/Nivara/AutoDiff/Nn/Activation.cs` (2 added wrappers). No other core files change.

**Downstream consumers (must compile after change):**
- Tests: `tests/Nivara.Tests/AutoDiff/{LossTests,NnTests,TrainingTests,DataParallelTests,SerializationTests,DistilBertSequenceClassificationTests}.cs`; `tests/Nivara.Tests/NivaraTorch/LossTests.cs`.
- Samples: `samples/NivaraChess`, `NivaraTimeSeries`, `NivaraVAE`, `NivaraGpt`,
  `NivaraClassifier`, `NivaraFineTuning`, `Nivara.SampleApp/CrossFrameworkFraudNet.cs`,
  `NivaraChat` trainers (TransformerMode, ValidatorTrainer, SentimentTrainer, IntentTrainer,
  EntityTrainer, AgentsValidatorTrainer). `EXAMPLES.md` loss code snippets.
- `TrainingLoop<T>`/`DataParallelTrainer<T>` take `Func<...>` — unaffected.
- VAE `ElboLoss` uses `ReverseGradOperations` directly — unaffected.

**Covering tests:** `LossTests.cs` (unit, incl. backward grads), `NnTests.cs` MSE
reduceToMean sites, `TrainingTests.cs`/`DataParallelTests.cs`/`SerializationTests.cs`
(training loops with `new MSELoss<float>()`), `NivaraTorch/LossTests.cs` (PyTorch parity:
MSE sum/mean, BCEWithLogits sum/mean, CrossEntropy mean, L1 sum).

**Behavioral change to watch:** default `Mean` flips gradient scale at training sites that used
the implicit-sum no-arg `Forward` (TrainingTests, DataParallelTests, SerializationTests,
NivaraChess). These sites get an explicit `Reduction.Sum` to preserve current numerics.

## Execution steps (one commit per logical unit)

1. `docs: plan #180 loss API unification in TODO.md` — this file.
2. `feat(autodiff): add Reduction enum and Loss<T> base` — `Reduction.cs`, `Loss.cs`.
3. `feat(autodiff): rewrite losses onto Loss<T> base with Reduction support` — 5 loss files.
4. `feat(autodiff): add Activation.Softmax/LogSoftmax wrappers; remove Functional Softmax classes`.
5. `test(autodiff): update loss tests to Reduction enum` — LossTests/NnTests/TrainingTests/
   DataParallelTests/SerializationTests/DistilBert tests; add `Reduction.None` + polymorphic
   `Loss<T>` tests.
6. `test(autodiff): NivaraTorch loss parity with Reduction.None` — gen_reference.py fixtures
   (reduction='none' for MSE/L1/BCEWithLogits/CrossEntropy) + NivaraTorch/LossTests.cs.
7. `refactor: update samples to Reduction-based loss API` — all sample call sites + EXAMPLES.md.
8. `docs: update loss documentation for Reduction API` — AUTODIFF.md, REVIEW.md, CHANGELOG,
   AGENTS.md, sample READMEs, GETTING-STARTED stale `CrossEntropyLoss<float>.Compute`.
9. Verify: `dotnet build Nivara.slnx` (no test run without asking). Then, with human
   confirmation, run AutoDiff + NivaraTorch test suites.
10. Review `docs/TODO.md`, remove it, offer push + PR.

## Verification

- `dotnet build Nivara.slnx` after core change (step 2–3) and again after all source edits.
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj` filtered to `AutoDiff` + `NivaraTorch`
  (ask first — AGENTS.md). NivaraTorch `Reduction.None` fixtures need Python + torch; if the
  environment lacks them, cover `None` numerically in `LossTests.cs` and file a follow-up issue.
- NivaraTorch sum/mean fixtures already exist (`*_sum_output.bin`, `*_mean_output.bin`,
  `cross_entropy_output.bin`, `l1_loss_output.bin`) — those tests just switch `bool` → enum.

## GitHub issues log

- [ ] (none yet — created during execution if deferred work surfaces)

**Reminder:** as each task executes, if you find deferred work or a concern (known limitations,
follow-ups, refactors) outside this plan, create a GitHub issue immediately
(`gh issue create --repo khurram-uworx/Nivara`) and record its number here — don't rely on
memory or wait until the plan finishes, as compaction can lose it.
