# TODO: Issue #1397 — SFT prompt-masking loss (next-token CE over answer tokens only)

> Iterative-work plan. Execute one logical change at a time, commit after each, never push.
> As each task executes, if deferred work or a concern is found outside this plan, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number in the GitHub issues log — do not rely on memory or wait until plan end.

## Problem

Issue [#1397](https://github.com/khurram-uworx/Nivara/issues/1397) (milestone 1.5.4, enhancement): SFT examples are system + question + answer; loss is next-token cross-entropy over answer tokens only (prompt masked). `docs/plan/LORA.md` §4/§10 calls this out as missing.

Acceptance criteria:
- Loss and gradients are zero for prompt positions.
- Gradients flow for answer positions.
- Overfit sanity: a 10-example set drops loss near zero within a few epochs.
- Unit tests for masking correctness.

## Grounding / current state

- `Nivara.AutoDiff.Nn.Functional.CrossEntropyLoss<T>` (`src/Nivara/AutoDiff/Nn/Functional/CrossEntropyLoss.cs:12-99`): sealed `Loss<T>` where `T : IFloatingPointIeee754<T>`. 
  - `Forward(logits, targets, reduction)` builds log-softmax NLL; `Reduction.None` returns per-row NLL `[N]`; `Mean` divides by batch size (`logits.shape[0]`).
  - `Forward(logits, int[] targets)` validates targets in `[0, numClasses)`, builds one-hot, delegates to tensor Forward. Negative targets throw — no ignore-index/mask.
  - `PerRowNll` sums class-weighted NLL per row via `MatMul(nll, ones)`.
- `Loss<T>` base (`src/Nivara/AutoDiff/Nn/Functional/Loss.cs`): 2-arg Forward non-virtual delegates to abstract 3-arg `Forward(pred, targets, reduction)`. `Reduce` supports optional divisor. Subclasses should not override 2-arg form.
- Tests: `tests/Nivara.Tests/AutoDiff/LossTests.cs` covers CE paths; no masking.
- Usage: NivaraGpt flattens causal LM to `[N,V]` logits and `[N]` int targets (`samples/NivaraGpt/Program.cs:196,235`). TrainingLoop takes `Func<ReverseGradTensor<T>, ReverseGradTensor<T>, ReverseGradTensor<T>> lossFn`.

## Blast radius

- **Direct code**: `src/Nivara/AutoDiff/Nn/Functional/CrossEntropyLoss.cs` (add overload/behavior). No change to `Loss<T>` abstract signature, no change to existing CE overloads' behavior.
- **Tests**: `tests/Nivara.Tests/AutoDiff/LossTests.cs` (extend) or new focused test file.
- **Docs**: `docs/AUTODIFF.md` loss table row for `CrossEntropyLoss<T>` (mention prompt-masking/ignore-index). Possibly `docs/plan/LORA.md` status note only if desired (not required by issue).
- **Downstream callers**: existing `new CrossEntropyLoss<T>().Forward(logits, targets)` and `Forward(logits, int[] targets)` remain source/binary compatible. TrainingLoop `lossFn` signature unchanged; masked path can be passed as a closure.
- **Risk**: low. Additive overload; existing tests must stay green. Main correctness risk is gradient leakage through masked positions.

## Proposed changes

### A. API (reuse existing CrossEntropyLoss<T>)

Add int-label masked overload on `CrossEntropyLoss<T>` (consistent with existing int[] overload being CE-specific):

```csharp
public ReverseGradTensor<T> Forward(ReverseGradTensor<T> logits, int[] targets, int ignoreIndex)
```

Behavior:
- `targets.Length == batchSize` required (as today).
- For each target: if `target == ignoreIndex`, position is masked (prompt); otherwise validate `0 <= target < numClasses` as today.
- Build one-hot only for active positions; masked positions contribute no one-hot row.
- Compute per-row NLL `[N]` as today. Multiply per-row NLL by a non-grad 0/1 mask tensor so masked positions become 0 loss and **zero gradient** (mask blocks backward). Do not rely on all-zero target alone — that would leak `softmax(x)` gradient.
- `Reduction.None`: return masked per-row NLL `[N]` (masked entries 0).
- `Reduction.Sum`: sum masked per-row NLL.
- `Reduction.Mean`: divide by `activeCount` (number of non-masked positions), not batch size / element count.
- `activeCount == 0`: **decision needed** (see G1). Recommend throwing `ArgumentException` with clear message, or return scalar 0 — PyTorch mean-ignore-all is historically awkward; explicit is safer. Plan default: throw `ArgumentException` unless human prefers 0.
- Existing `Forward(logits, int[] targets)` remains unchanged.

Optional convenience: `Forward(logits, int[] targets, bool[] answerMask)` or `ReadOnlySpan<bool>` (true = include in loss). Only add if it clearly helps; ignoreIndex covers the SFT case described in issue/LORA plan.

### B. Implementation sketch

```csharp
public ReverseGradTensor<T> Forward(ReverseGradTensor<T> logits, int[] targets, int ignoreIndex)
{
    // validate args, batchSize, numClasses
    // validate active targets; allow ignoreIndex
    int activeCount = 0;
    var oneHotData = new T[batchSize * numClasses];
    var maskData = new T[batchSize];
    for (int i = 0; i < batchSize; i++)
    {
        int t = targets[i];
        if (t == ignoreIndex) { maskData[i] = T.Zero; continue; }
        // range check
        oneHotData[i * numClasses + t] = T.One;
        maskData[i] = T.One;
        activeCount++;
    }
    if (activeCount == 0) throw new ArgumentException(...); // confirm

    var oneHot = new ReverseGradTensor<T>(NivaraColumn<T>.Create(oneHotData), requiresGrad: false);
    oneHot.Reshape(batchSize, numClasses);
    var logSoftmax = ReverseGradOperations.LogSoftmax(logits);
    var nll = ReverseGradOperations.Negate(ReverseGradOperations.Multiply(logSoftmax, oneHot));
    var perRow = PerRowNll(nll, logits); // [N]

    var mask = new ReverseGradTensor<T>(NivaraColumn<T>.Create(maskData), requiresGrad: false);
    mask.Reshape(batchSize);
    var maskedPerRow = ReverseGradOperations.Multiply(perRow, mask); // blocks grad at masked positions

    if (Reduction == Reduction.None) return maskedPerRow;
    return Reduce(maskedPerRow, Reduction, Reduction == Reduction.Mean ? activeCount : null);
}
```

Note: existing `Forward(logits, targets, reduction)` 3-arg uses batchSize divisor for Mean. Masked path should divide by `activeCount` for Mean; Sum/None unaffected. The overload may take `Reduction?`/explicit reduction or use stored `Reduction` like int[] overload does today — follow existing style (int[] overload delegates to `Forward(logits, oneHot)` which uses stored Reduction). To support per-call reduction, either add 4-arg or route through a private helper.

### C. Tests

Add tests (in `LossTests.cs` or a new `PromptMaskedCrossEntropyLossTests.cs`; prefer extending existing loss test suite unless it gets unwieldy):

1. `CrossEntropyLoss_IgnoreIndex_MaskedPositions_ZeroLossAndZeroGradients` — prompt positions masked; assert their per-sample loss is 0 and logits grad rows are all zero.
2. `CrossEntropyLoss_IgnoreIndex_AnswerPositions_GradientsFlow` — unmasked positions match unmasked-only CE loss/grad.
3. `CrossEntropyLoss_IgnoreIndex_Mean_DividesByActiveCount` — Mean uses active count, not batch size.
4. `CrossEntropyLoss_IgnoreIndex_None_ReturnsMaskedPerRowNll` — shape `[N]`, masked entries 0.
5. `CrossEntropyLoss_IgnoreIndex_MatchesFilteredCrossEntropy` — compare against CE computed on filtered active rows.
6. `CrossEntropyLoss_IgnoreIndex_AllMasked_ThrowsOrDefined` — per decision.
7. `CrossEntropyLoss_IgnoreIndex_InvalidActiveLabel_Throws` — out-of-range active labels still throw; ignoreIndex allowed.
8. `CrossEntropyLoss_SftOverfit_10Examples_LossDropsNearZero` — small overfit sanity (few epochs, tiny model/tensor params, fixed seed) to satisfy acceptance criteria. Keep <2s; if not, mark `[Category("Performance")]` and get confirmation before running full suite.

### D. Docs

- Update `docs/AUTODIFF.md` loss table row for `CrossEntropyLoss<T>` to document prompt-masking/ignoreIndex support and Mean divisor semantics.
- Verify snippet gate impact (no new gated snippets planned; docs table edit only). Run DocsCoveragePins / docs tests if citations/counts affected.

## Verification

- Targeted: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~LossTests|FullyQualifiedName~CrossEntropy"`.
- If docs changed: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`.
- Format before PR: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- Ask before running full suite / long-running commands.

## Planned commits

1. `docs: plan issue 1397 SFT prompt-masking loss in TODO.md` (this file)
2. `Add prompt-masked CrossEntropyLoss ignoreIndex support` (impl + unit tests)
3. `docs: document prompt-masked cross-entropy loss` (AUTODIFF.md, if needed)
4. `docs: remove TODO.md — plan executed` (after G2 clears)

## GitHub issues log

- [ ] #1397 — SFT prompt-masking loss (next-token CE over answer tokens only) (this plan)

## Decisions / red flags for G1

1. **Mask API**: `ignoreIndex` int sentinel (PyTorch-style, ergonomic for token labels) vs `bool[]` answer mask vs both. Recommend `ignoreIndex` on int[] overload first; it matches issue/LORA plan and SFT practice.
2. **All-masked behavior**: throw `ArgumentException` (recommended, explicit) vs return 0. Need human call.
3. **Tensor-target overload**: support masking for one-hot tensor targets too, or int[] only? SFT uses int labels; int[] is sufficient for acceptance criteria.
4. **Per-call reduction on masked overload**: add `Forward(logits, targets, ignoreIndex, reduction)` or use stored Reduction only? Existing int[] overload uses stored Reduction.
5. **Default ignoreIndex value**: none (explicit parameter, no default) is safest for compatibility/clarity.

## G1 grounding checklist (before implementation)

- [ ] Confirm .NET/AutoDiff tensor op behavior for mask multiply blocking gradients (ReverseGradOperations.Multiply with requiresGrad:false mask).
- [ ] Confirm existing CrossEntropyLoss Mean divisor is batchSize (it is: `CrossEntropyLoss.cs:45-46`) and decide masked Mean divisor = activeCount.
- [ ] Navigate callers/relations via code-memory MCP; confirm no `Loss<T>` base signature change needed.
- [ ] Confirm docs coverage pins impact if AUTODIFF.md table grows (citation count unchanged unless adding file:line citations).
- [ ] Escalate decisions above to human before implementing.
