# #487 — make the `OpNode.Inputs` contract explicit, enforced, and tested

Branch: `khurram/487`. Issue: https://github.com/khurram-uworx/Nivara/issues/487

## Problem

`OpNode<T>.Inputs` is the traversal set consumed by `ComputationGraph`
(`BuildBackwardPlan`, `ZeroGrad`, `GetGraphInfo`). Nothing in the type or its
constructor records whether an omitted input is deliberately
non-differentiable or accidentally unwired, so a caller who passed a
`RequiresGrad == true` input that an op silently excluded got an output with
`RequiresGrad == false` and only learned about it when `Backward` threw its
generic `"Cannot perform backward pass on tensor that doesn't require gradients"`
— blaming the *output* rather than the input.

#481 fixed the reverse-mode attention mask per-op (throw when `mask.RequiresGrad`).
The *pattern* is still unenforced and nothing would catch a new occurrence.

## Audit (complete — 58 `new OpNode<` sites across 11 files, both AD modes)

Reverse- and forward-mode ops whose signature takes ≥1 tensor argument. A tensor
argument is *tracked* when it appears in `OpNode.Inputs` (reverse) / in the
`RequiresTangent` predicate (forward).

| Op | Args | Tracked | Excluded arg | Enforced by |
|---|---|---|---|---|
| `Add`, `AddBias`, `Subtract`, `Multiply`, `Divide`, `MatMul`, `MatMulTransposedB` | 2 tensors | all | — | n/a |
| `Concat` (`:1876` rev) | tensor[] | all | — | n/a |
| `KlDivergence`, `SampleNormal`, `BroadcastMultiply`, `BroadcastAdd` | 2 tensors | all | — | n/a |
| `MultiHeadAttention` rev `:503` / fwd `:1546` | q,k,v,mask | q,k,v | `mask` | rev: throws (#481). **fwd: silent** |
| `BatchedMultiHeadAttention` rev `:723` / fwd `:1738` | q,k,v,mask | q,k,v | `mask` | rev: throws (#481). **fwd: silent** |
| `SparseEmbeddingBag` rev `:2412` / fwd `:653` | weight,indices | weight | `indices` | **none** |
| `Gather` rev `:2508` / fwd `:548` | source,`int[]` | source | `int[] indices` | structural (not a tensor) |
| `DropoutWithMask` | input,`ReadOnlySpan<bool>` | input | `keepMask` | structural (not a tensor) |

So exactly **three** unguarded exclusions remain: forward-mode `mask` ×2 and
`indices` ×2 (both modes).

### The `Parameter` carve-out in the issue text does not match the tree

The issue states module `Parameter`s are "deliberately excluded". They are not
uniformly excluded, and the inconsistency has an observable consequence:

| Site | `OpNode.Inputs` today | Accumulated into but absent |
|---|---|---|
| `Conv1d` `Nn/Conv1d.cs:153` | `[input, weight.Tensor]` | `bias` |
| `Conv2d` `Nn/Conv2d.cs:253` | `[input, weight.Tensor]` | `bias` |
| `ConvTranspose2d` `Nn/Conv2d.cs:841` | `[input, weight.Tensor]` | `bias` |
| `BatchNorm1dEval` `Nn/BatchNorm.cs:141` | `[input]` | `weight`, `bias` |
| `BatchNorm1dTrain` `Nn/BatchNorm.cs:177` | `[input]` | `weight`, `bias` |
| `BatchNorm2dEval` `Nn/BatchNorm.cs:392` | `[input]` | `weight`, `bias` |
| `BatchNorm2d` `Nn/BatchNorm.cs:429` | `[input]` | `weight`, `bias` |
| `LayerNorm` `Nn/LayerNorm.cs:124` | `[input]` | `weight`, `bias` |
| `RMSNorm` `Nn/RMSNorm.cs:97` | `[input]` | `weight` |

`ComputationGraph.ZeroGrad` walks `GradFn.Inputs` (`ComputationGraph.cs:155`), so
the public `GradientUtils.ZeroGrad(tensor)` today clears `input` **and** `weight`
for Conv, but clears **neither** weight nor bias for LayerNorm/RMSNorm/BatchNorm,
and never `bias` for Conv. That contradicts its own doc ("Clears all gradients in
the computation graph reachable from the specified tensor",
`GradientUtils.cs:66`). Training loops are unaffected because
`Optimizer.ZeroGrad()` walks `Module.Parameters()` directly — which is why this
was never noticed.

### Stale documentation

`docs/AUTODIFF.md:59-65` and `:249-265` describe an `OpNode` that does not exist:
`IReadOnlyList<object> Inputs`, `Action<NivaraColumn<T>, bool> BackwardFunction`,
`ShouldSaveForBackward`, `SavedValues`. `:283` names a `BuildNodeToOutputMap` step
that is not there either (the real code has `BuildBackwardPlan`). Per the repo's
claim-discipline rules this is worse than no documentation.

## Decisions taken (human-confirmed)

1. **Forward-mode attention throws** on `mask.RequiresTangent`, matching the
   reverse guard added in #481. Rationale: the same piecewise argument applies
   (`ApplyMask` *assigns* `-inf`, it does not add), PyTorch faults on a mask that
   requires grad (pytorch#148476, already cited in `CHANGELOG.md` for #481), and
   forward mode has *no* graph and *no* `Backward` to fail later — a mask-only
   tangent yields `RequiresTangent == false` with no error at all, which is
   strictly worse than the reverse-mode shape. No in-tree caller passes a tangent
   mask: `FromMatrix`/`FromArray` default `tangent: null`, and every in-tree mask
   builder is reverse-mode only (`ModuleHelpers.CreateCausalMask` hardcodes
   `requiresGrad: false`).
2. **`SparseEmbeddingBag.indices` gets the same guard in both modes.** Behaviour
   change, called out deliberately: `SparseEmbedding<T>.Forward` forwards the
   caller's tensor straight through as `indices`, so any caller passing a
   requires-grad indices tensor now gets an `ArgumentException`. Verified no
   in-tree caller does (`NnTests`/`SparseEmbeddingTests` build indices via
   `FromArray` with the `requiresGrad: false` default; `NivaraChess` likewise).
3. **`Parameter`s are brought into `Inputs`** so the contract reads uniformly,
   rather than documenting a carve-out that the tree does not honour.
4. **Enforcement is a table-driven contract test plus a reflection completeness
   guard**, so a *new* op fails the build until it gets a contract row.

## The contract, stated once

> `OpNode<T>.Inputs` is every tensor whose gradient this node's backward function
> is responsible for accumulating into. An argument absent from it is structurally
> non-differentiable — an index/discrete argument, or a mask whose kernel assigns
> rather than adds — and the op **must** reject it loudly (via `RequireConstant`)
> when the caller asks for a gradient or tangent on it. A tensor absent from it is
> never silently ignored.

Consequences:
- `ZeroGrad` reachability becomes a consequence of the contract instead of an
  accident of which module happened to list its `weight`.
- `Parameter` tensors are leaves (`GradFn == null`), so adding them to `Inputs` is
  a no-op for `BuildBackwardPlan` (`ComputationGraph.cs:108` already skips them)
  and for `GetGraphInfo`; the only behaviour change is that `ZeroGrad` clears
  *more*, which is what its doc promises.

## Proposed changes

### 1. `RequireConstant` — one uniform choke point

New `internal static void RequireConstant<T>(string name, ReverseGradTensor<T>? tensor)`
next to the existing `internal static bool ShouldTrackGrad` in
`src/Nivara/AutoDiff/Utilities/GradientUtils.cs:32`, plus a `ForwardGradTensor<T>`
overload checking `RequiresTangent`. Internal (not public) — `ShouldTrackGrad` is
internal too and this is an implementation detail.

```csharp
internal static void RequireConstant<T>(string name, ReverseGradTensor<T>? tensor)
    where T : struct, IFloatingPointIeee754<T>
    => RequireConstant(name, tensor is { RequiresGrad: true }, "gradients");

internal static void RequireConstant<T>(string name, ForwardGradTensor<T>? tensor)
    where T : struct, IFloatingPointIeee754<T>
    => RequireConstant(name, tensor is { RequiresTangent: true }, "tangents");

static void RequireConstant(string name, bool tracks, string noun)
{
    if (tracks)
        throw new ArgumentException($"{name} is a non-differentiable constant and cannot require {noun}.", name);
}
```

Message keeps the `"non-differentiable"` phrase and `ParamName` keeps the C#
parameter name, so the existing `tests/Nivara.Tests/AutoDiff/AttentionMaskTests.cs`
assertions (`Does.Contain("non-differentiable")`, `ParamName == "mask"`) keep
passing unchanged.

### 2. Wire the four attention sites through the helper

- `ReverseGradOperations.MultiHeadAttention:537` and
  `BatchedMultiHeadAttention:760` — replace the two ad-hoc `throw` statements with
  `RequireConstant(nameof(mask), mask);` (no behaviour change).
- `ForwardGradOperations.MultiHeadAttention` (~`:1565`) and
  `BatchedMultiHeadAttention` (~`:1774`) — **new** `RequireConstant` call. Needs
  `using Nivara.AutoDiff.Utilities;` added to that file.

### 3. Guard `SparseEmbeddingBag.indices` in both modes

`ReverseGradOperations.cs:2412` and `ForwardGradOperations.cs:653` get
`RequireConstant(nameof(indices), indices);` next to the null checks, plus the
"structurally non-differentiable because it carries integer row selectors" note in
their XML docs.

### 4. Bring module `Parameter`s into `Inputs`

Nine sites in `Nn/` gain the tensors their backward function already accumulates
into, conditionally on the same `affine` / `useBias` predicates that gate the
accumulation. Conditional collection expressions, e.g. for Conv:

```csharp
ReverseGradTensor<T>[] nodeInputs = useBias && bias != null
    ? [input, weight.Tensor, bias.Tensor]
    : [input, weight.Tensor];

var gradFn = new OpNode<T>("Conv1d", nodeInputs, (typedGradOutput) => { ... });
```

The `weight`/`bias` `Parameter` nullability must follow the same predicate that
already guards the accumulation in the closure — no new null paths.

### 5. Tests

New `tests/Nivara.Tests/AutoDiff/OpNodeInputContractTests.cs`:

- **Per-op rows** for every public op taking a tensor argument. Each row calls the
  op inside a `Grad()` scope with **exactly one** tensor argument requiring a
  gradient and asserts one of three outcomes: the argument is in
  `result.GradFn.Inputs` (tracked), or the op throws `ArgumentException` naming
  that argument (guarded), or the argument is excluded by type (`int[]`,
  `ReadOnlySpan<bool>` — listed explicitly, no invocation needed).
  The isolate-one-argument shape is what catches the #481 defect: setting *all*
  arguments to requires-grad would pass even when one is excluded.
- **A reflection completeness guard**: enumerate the public static methods of
  `ReverseGradOperations` and `ForwardGradOperations` whose signature contains a
  `ReverseGradTensor<T>`/`ForwardGradTensor<T>` parameter and assert each method
  name is covered by a row. This is what catches a genuinely new occurrence.
- **Module `Parameter` rows**: for each of the nine module sites, assert the
  parameter tensors appear in `output.GradFn.Inputs`, and that
  `GradientUtils.ZeroGrad(output)` clears a parameter gradient seeded before the
  call. This pins the `ZeroGrad` behaviour the contract now implies.

Extend `tests/Nivara.Tests/AutoDiff/AttentionMaskTests.cs` with the forward-mode
`mask.RequiresTangent` throws cases (currently only reverse is covered).

### 6. Documentation

- `src/Nivara/AutoDiff/OpNode.cs` — XML doc on the type and on `Inputs` stating
  the contract verbatim.
- `docs/AUTODIFF.md` — correct the stale `OpNode` signature block (`:59-65`,
  `:249-265`) to the real members, correct `:283`'s `BuildNodeToOutputMap`
  reference, and add the `OpNode.Inputs` contract section.
- `CHANGELOG.md` — record the forward-mode guard, the `indices` guard, and the
  `ZeroGrad` reachability widening.

## Verification

- `dotnet build Nivara.slnx` clean.
- `dotnet test -c Release` (ask first) — full suite. Note AGENTS.md: never measure
  or gate timing in Debug; this change adds no timing assertions, but the suite
  must be run in Release so unrelated `[Category("Performance")]` gates behave.
- Targeted: `OpNodeInputContractTests`, `AttentionMaskTests`,
  `ForwardGradOperationsTests`, `NnTests`, `GradientUtilsTests`,
  `BackwardPassTests`, `SparseEmbedding`/`SparseEmbeddingTests`.
- Release-config per AGENTS.md.

## Blast radius

- **Public API**: no signature changes. Two behaviour changes, both deliberate and
  both loud throws rather than silent wrongness: forward-mode attention with a
  tangent-carrying mask now throws; `SparseEmbeddingBag` with a requires-grad
  `indices` tensor now throws (reachable through `SparseEmbedding<T>.Forward`).
- **Graph semantics**: `BuildBackwardPlan` and `GetGraphInfo` are provably
  unaffected (they already skip `GradFn == null` inputs, and every added entry is
  a leaf `Parameter`).
- `ZeroGrad` reachability widens for 9 module ops — the intended fix.
- **Downstream callers** of the 9 module ops: `Linear`, `Conv1d/2d`,
  `BatchNorm1d/2d`, `LayerNorm`, `RMSNorm`, `TransformerBlock`, `PerRowLayerNorm`,
  `PerRowRMSNorm`, plus every model built from them (`TransformerBlock`,
  `LlamaCausalAttention`, `MultiheadAttention`, BERT/DistilBERT/ModernBERT
  samples). All covered by the existing suite.
- **Touches**: `src/Nivara/AutoDiff/{OpNode.cs, ComputationGraph.cs (no change),
  Utilities/GradientUtils.cs, Operations/ReverseGradOperations.cs,
  Operations/ForwardGradOperations.cs, Nn/{Conv1d,Conv2d,BatchNorm,LayerNorm,RMSNorm}.cs}`,
  `docs/{AUTODIFF.md, TODO.md}`, `CHANGELOG.md`,
  `tests/Nivara.Tests/AutoDiff/{OpNodeInputContractTests, AttentionMaskTests}.cs`.
- **No probe/harness needed** — this is a contract and test-coverage change with no
  performance question. Per AGENTS.md I checked `Nivara.SimdProbe` and
  `Nivara.PerformanceTests` mode lists; neither has a mode that answers "is this
  argument wired into the graph", so no probe is warranted and none is created.

## Planned commits

1. `docs: plan #487 OpNode.Inputs contract in TODO.md`
2. `fix(autodiff): reject a gradient-carrying constant argument in every mode`
   (`RequireConstant` + the 4 attention sites + the 2 `indices` sites)
3. `fix(autodiff): list module parameters in OpNode.Inputs` (the 9 `Nn/` sites)
4. `test: enforce the OpNode.Inputs contract across both AD modes` (new contract
   tests + reflection completeness guard + forward-mode mask tests)
5. `docs: state the OpNode.Inputs contract and correct the stale OpNode reference`
   (`OpNode.cs` XML doc, `docs/AUTODIFF.md`, `CHANGELOG.md`)

## GitHub issues log

- [ ] (none yet — create at discovery time as execution proceeds; do not defer to
      the end of the plan, compaction can lose it)