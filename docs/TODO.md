# Plan: #481 — attention ops declare the mask as an `OpNode` input but never differentiate it

Tracking issue: [#481](https://github.com/khurram-uworx/Nivara/issues/481)

## Problem

`ReverseGradOperations.MultiHeadAttention<T>` and
`ReverseGradOperations.BatchedMultiHeadAttention<T>` both pass the optional
attention mask into the `OpNode` input list, but neither backward function ever
accumulates a gradient into it.

- `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs:587-589` builds
  `inputs = [query, key, value, mask]`; `:648-653` accumulates only dQ, dK, dV.
- Same shape at `:832-834` (inputs) and `:851-856` (accumulation).

So `mask.Grad` stays `null` after `Backward`. The op declares a dependency it
does not honour, and a caller who passes `requiresGrad: true` gets silence
rather than an error.

### Two corrections to the issue's framing

Grounding the issue against the tree produced two findings that change the fix
from the one the issue proposes.

**1. The mask is not an additive bias, so "propagate `dmask = dscore`" is not
available as described.** Issue #448 (closed) changed `ApplyMask` from a
sum into a hybrid select/add:

```csharp
// src/Nivara/AutoDiff/Operations/AttentionKernels.cs:121
scores[i] = m == T.NegativeInfinity ? T.NegativeInfinity : scores[i] + m;
```

The `-inf` branch **assigns** `-inf`, discarding the score. So
`d(scores)/d(mask)` is `0` on suppressed cells and `1` on kept cells. The issue
describes the gradient of a pure-add op that no longer exists. The two halves
of the mask have different derivative structure, which is exactly why
"differentiate the mask" is not a small change.

**2. Mask inclusion in `Inputs` is not anomalous, so "the op declares a
dependency it does not honour" is weaker than stated.** `OpNode.Inputs` is not
an exhaustive declaration of gradient-carrying dependencies in this codebase —
it is the topological-traversal set consumed by `ComputationGraph`:

- `src/Nivara/AutoDiff/ComputationGraph.cs:91-92` and `:106-110` — node
  ordering and cycle detection.
- `src/Nivara/AutoDiff/ComputationGraph.cs:155-157` — `ZeroGrad` walk.
- `src/Nivara/AutoDiff/ComputationGraph.cs:171-174` — `GetGraphInfo` counts.

Ops routinely accumulate into tensors they do **not** list:

| Op | `Inputs` | Also accumulates into |
|---|---|---|
| `Conv1d` (`Nn/Conv1d.cs:153`) | `[input, weight.Tensor]` | `bias` (`:174-179`) |
| `ConvTranspose2d` (`Nn/Conv2d.cs:841`) | `[input, weight.Tensor]` | `bias` |
| `BatchNorm1dTrain` (`Nn/BatchNorm.cs:177`) | `[input]` | `weight`, `bias` (`:195-198`) |
| `SparseEmbeddingBag` (`ReverseGradOperations.cs:2441`) | `[weight]` | — (`indices` excluded) |

Those are `Parameter`s reached via `Module.Parameters()`, so their absence is
correct. The mask is nonetheless misleading, because unlike a bias it is a plain
tensor argument a caller could plausibly set `requiresGrad: true` on.

### The load-bearing failure is the silent one

`shouldTrack` is computed as `GradientUtils.ShouldTrackGrad(query, key, value)`
at `:529` and `:739` — **the mask is excluded**. So a mask-only-`requiresGrad`
call builds *no node at all*, and the output tensor is constructed with
`requiresGrad: shouldTrack == false`. `Backward` then throws the generic
`"Cannot perform backward pass on tensor that doesn't require gradients"`
(`ComputationGraph.cs:33`), which points at the wrong tensor. This is strictly
worse than the reported "no gradient" symptom, and the issue does not name it.

### Why it has not bitten anyone

Every in-tree mask builder passes `requiresGrad: false`, so the missing branch
is unreachable: `ModuleHelpers.cs:62`, `:81`; `MultiheadAttention.cs:190`;
`TransformerBlock.cs:94`; `LlamaCausalAttention.cs:274`;
`BatchedTransformer.cs:107`; `BertModel.cs:135`; `ModernBertModel.cs:260`.
`ReverseGradTensor.FromArray`/`FromMatrix` default to `requiresGrad = false`
(`ReverseGradTensor.cs:103`, `:125`). **No in-tree caller can trip a throw.**

## Proposed changes

Adopt **Option B from the issue — stop declaring the mask** — plus the throw.
Rationale:

1. A select cannot carry a meaningful gradient. Optimising a mask whose entries
   are `0`/`-inf` is a category error, and nothing in-tree wants it.
2. The forward-mode twin already documents the mask as "a non-differentiable
   constant" (`ForwardGradOperations.cs:1542`, `:1727`;
   `docs/AUTODIFF.md:448`) and excludes it from `trackTangent` at `:1579`.
   Option B reconciles the two halves; Option A widens the split.
3. Option B is a strict subset of Option A's work. If mask gradients are ever
   wanted, that is a feature with its own parity fixtures, not a bug fix.

In **both** `MultiHeadAttention<T>` and `BatchedMultiHeadAttention<T>`:

1. Drop `mask` from the `OpNode` input array. The `mask != null ? { q, k, v, mask }
   : { q, k, v }` ternary collapses to `[query, key, value]`, which can then use
   collection-expression syntax per `.editorconfig`.
2. Add a guard beside the existing mask shape validation (`:526-527`, `:736-737`):

```csharp
if (mask is { RequiresGrad: true })
    throw new ArgumentException(
        "Mask is a non-differentiable constant; it cannot require gradients.", nameof(mask));
```

3. Update the XML doc on both ops to state the mask is a non-differentiable
   constant, matching the forward-mode wording.

The GPU/sample `BatchedAttention` and `DecodeAttention` build no graph nodes, so
they are out of scope.

### G1 grounding outcome — decisions confirmed

**Guard scope: unconditional.** Confirmed by the human. `RequiresGrad` is a
property of the tensor, and a caller who set the flag wants a gradient
regardless of ambient scope. Gating on `Grad()` would leave the silent path
open for anyone who sets the flag outside a scope.

**Exception type: `ArgumentException`.** Confirmed by the human. Matches the
mask shape validation a few lines above it and the convention in the
neighbouring attention ops.

**PyTorch parity (the deciding evidence for Option B).** PyTorch's documented
SDPA reference is a pure add — `attn_bias = attn_mask + attn_bias` — with no
documented gradient contract for the mask. PyTorch also has an open bug for
exactly this scenario: [pytorch#148476](https://github.com/pytorch/pytorch/issues/148476)
crashes with *"Illegal memory access in `scaled_dot_product_attention` ... when
using a float attention mask that requires grad while q, k and v do not require
grad."* So the reference implementation neither supports a grad-requiring mask
nor degrades gracefully — it faults. This validates the `shouldTrack`
mask-exclusion at `:529`/`:739` as the correct shape, and makes Option B the
parity-matching choice rather than merely the conservative one.

**BCL limitation noted.** `TensorPrimitives` has no select/blend primitive (it
offers `IsNegativeInfinity`, generic over `INumberBase<T>`, but no blend),
matching the existing note at `AttentionKernels.cs:104` (tracked as #480). Not
needed for Option B; recorded because it would constrain any future
mask-gradient work to a scalar loop.

## Verification steps

- `dotnet build Nivara.slnx` (solution is `.slnx`, not `.sln`).
- New tests, always in `Release` (`AGENTS.md`: a Debug run compares an
  unoptimized handwritten kernel against ReadyToRun BCL and reports false
  regressions). **Ask the human before running `dotnet test`.**
- `dotnet test -c Release --filter "FullyQualifiedName~AttentionMask"`.

## Planned commits

1. `docs: plan #481 attention mask differentiability in TODO.md`
2. `fix: stop declaring the attention mask as an OpNode input`
3. `test: cover the attention mask non-differentiable contract`
4. `docs: record #481 in CHANGELOG`
5. `docs: remove TODO.md — plan executed` (only after G2 clears)

## Blast radius

**Directly changed**

- `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` — two methods
  (`:496-685`, `:703-953`), the `inputs` array and one added guard each.

**Behaviour change**

- `ReverseGradOperations.MultiHeadAttention<T>` and
  `BatchedMultiHeadAttention<T>` now throw `ArgumentException` when
  `mask.RequiresGrad`. This is the public contract change.

**Callers verified unaffected** (all pass `requiresGrad: false`)

- `Nn/MultiheadAttention.cs:173` via `ComputeAttention`
- `Nn/TransformerBlock.cs:140` via `CausalMaskSlice`
- `Nn/LlamaCausalAttention.cs:274`
- `samples/NivaraChat/Transformer/BatchedTransformer.cs:168`
- `samples/Nivara.Samples/BertModel.cs` (via `BuildBatchedPaddingMask`)
- `samples/Nivara.Samples/DistilBertModel.cs`, `LayaHeadModel.cs`
- `samples/Nivara.Samples/ModernBertModel.cs:260`

**Tests that cover the touched ops** (must stay green)

- `tests/Nivara.Tests/AutoDiff/BatchedMultiHeadAttentionTests.cs`
- `tests/Nivara.Tests/AutoDiff/ForwardGradOperationsTests.cs` (forward twin)
- `tests/Nivara.Tests/AutoDiff/ForwardParityTests.cs` (finite-difference JVP)
- `tests/Nivara.Tests/AutoDiff/AttentionKernelsTests.cs` (13 `ApplyMask` tests)
- `tests/Nivara.Tests/AutoDiff/GradKernelsTests.cs`
- `tests/Nivara.Tests/AutoDiff/NnTests.cs` (padding-mask backward)
- `tests/Nivara.Tests/NivaraTorch/MultiHeadAttentionTests.cs` (Torch parity)
- `tests/Nivara.Tests/NivaraTorch/BatchedAttentionTests.cs` (Torch parity)
- `tests/Nivara.Tests/NivaraTorch/BandedAttentionTests.cs` (Torch parity)
- `tests/Nivara.Tests/AutoDiff/ModernBertMaskTests.cs`

**Not at risk**: gradients of q/k/v are untouched. The backward bodies are not
modified — only the declared input list and the argument guard.

**Low risk overall.** The throw is unreachable from every in-tree call site; the
`OpNode.Inputs` change removes a leaf that carries no gradient. The
`OpNode.Inputs` edit is directly observable from `Nivara.Tests` because
`InternalsVisibleTo` is set (`Nivara.csproj:18-20`) and `GradFn` is `internal`.

## GitHub issues log

As each task executes, if deferred work or a concern is found that is outside
this plan, create it immediately with
`gh issue create --repo khurram-uworx/Nivara` and record the number here. Do
not rely on memory — compaction during execution can lose items.

- [ ] #481 — the issue this plan closes (mask declared but never differentiated)
