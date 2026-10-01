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
uniformly excluded, and the inconsistency has an observable consequence.
**Corrected at G1** — the first draft of this plan said all nine sites gain
tensors; reading every closure shows **seven** do, and two are already correct:

| Site | `OpNode.Inputs` today | Accumulates into | Action |
|---|---|---|---|
| `Conv1d` `Nn/Conv1d.cs:153` | `[input, weight.Tensor]` | input, weight, bias | add `bias` |
| `Conv2d` `Nn/Conv2d.cs:253` | `[input, weight.Tensor]` | input, weight, bias | add `bias` |
| `ConvTranspose2d` `Nn/Conv2d.cs:841` | `[input, weight.Tensor]` | input, weight, bias | add `bias` |
| `BatchNorm1dTrain` `Nn/BatchNorm.cs:177` | `[input]` | input, weight, bias | add `weight`, `bias` |
| `BatchNorm2d` `Nn/BatchNorm.cs:429` | `[input]` | input, weight, bias | add `weight`, `bias` |
| `LayerNorm` `Nn/LayerNorm.cs:124` | `[input]` | input, weight, bias | add `weight`, `bias` |
| `RMSNorm` `Nn/RMSNorm.cs:97` | `[input]` | input, weight | add `weight` |
| `BatchNorm1dEval` `Nn/BatchNorm.cs:141` | `[input]` | input only | **none — correct** |
| `BatchNorm2dEval` `Nn/BatchNorm.cs:392` | `[input]` | input only | **none — correct** |

The two eval-path nodes accumulate into `input` only, so `[input]` is already
the contract-correct answer for them. (That they produce *no* parameter gradient
in eval mode is a separate PyTorch divergence — tracked as **#494**, out of
scope here.)

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
3. **`Parameter`s are brought into `Inputs`** at the **seven** sites whose backward
   already accumulates into them, so the contract reads uniformly rather than
   documenting a carve-out the tree does not honour. The two BatchNorm eval-path
   nodes keep `[input]` — they accumulate into nothing else.
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

### 4. Bring module `Parameter`s into `Inputs` (seven sites)

**Corrected at G1.** The first draft said nine sites and sketched a single
`useBias`/`affine` predicate. Reading every closure shows neither is safe:

| Site | Gate the `Inputs` list must mirror |
|---|---|
| `Conv1d` / `Conv2d` / `ConvTranspose2d` | weight always (as today); bias only under `useBias && bias != null` |
| `LayerNorm` | `weight != null` and `bias != null` **independently** — *not* on `affine`, so all four combinations occur |
| `BatchNorm1dTrain` / `BatchNorm2d` | `useAffine` **plus** independent `weight != null` / `bias != null` |
| `RMSNorm` | `weight.Tensor` unconditionally — the closure accumulates into it with no guard, so the list must too |

A single shared predicate would silently diverge from the closure at three of the
seven. So the conditional lives in **one** authoritative helper, per AGENTS.md
rule 8, rather than being re-derived per site:

```csharp
// src/Nivara/AutoDiff/Nn/ModuleHelpers.cs
internal static ReverseGradTensor<T>[] NodeInputs<T>(
    ReverseGradTensor<T> input,
    ReverseGradTensor<T>? second = null,
    ReverseGradTensor<T>? third = null)
    where T : struct, IFloatingPointIeee754<T>
{
    if (second == null) return [input];
    if (third == null) return [input, second];
    return [input, second, third];
}
```

Each site becomes one expression that cannot drift from its own closure:

```csharp
var gradFn = new OpNode<T>("Conv1d",
    useBias && bias != null
        ? ModuleHelpers<T>.NodeInputs(input, weight.Tensor, bias.Tensor)
        : ModuleHelpers<T>.NodeInputs(input, weight.Tensor),
    (gradOutput) => { /* unchanged closure */ });

var gradFn = new OpNode<T>("LayerNorm",
    ModuleHelpers<T>.NodeInputs(input, weight?.Tensor, bias?.Tensor),
    (typedGradOutput) => { /* unchanged closure */ });
```

One 1–3 element array per module forward, allocated only on the
`RequiresGrad` path. Cost is negligible against the per-call `T[]` output and
gradient buffers these forwards already allocate.

### 5. Tests

New `tests/Nivara.Tests/AutoDiff/OpNodeInputContractTests.cs`:

- **Per-op rows** for every public op taking **two or more** tensor arguments, plus
  `Concat` (one argument that is a tensor array). Each row calls the op inside a
  `Grad()` scope with **exactly one** tensor argument requiring a gradient and
  asserts one of three outcomes: the argument is in `result.GradFn.Inputs`
  (tracked), or the op throws `ArgumentException` naming that argument (guarded),
  or the argument is excluded by type (`int[]`, `ReadOnlySpan<bool>` — pinned
  explicitly, no invocation needed).
  The isolate-one-argument shape is what catches the #481 defect: setting *all*
  arguments to requires-grad would pass even when one is excluded.
  Single-tensor-argument ops are out of scope — see the deviations section for why,
  and note the reduced coverage is deliberate and recorded rather than accidental.
- **A reflection completeness guard**: enumerate the public static methods of
  `ReverseGradOperations` and `ForwardGradOperations` whose signature contains a
  `ReverseGradTensor<T>`/`ForwardGradTensor<T>` parameter and assert each method
  name is covered by a row. This is what catches a genuinely new occurrence.
- **A second reflection guard over modules** (human-confirmed at G1): the static
  guard cannot see module sites, which are exactly the seven that change here.
  Reflect over public `Module<T>` subclasses that own a `Parameter<T>` field and
  require each in the module row table. Targeting only `Parameter`-owning modules
  keeps it low-noise — modules without parameters need no row, and this is
  precisely the category the contract change makes load-bearing.
- **Module `Parameter` rows**: assert each parameter tensor is **reachable** from
  `output.GradFn.Inputs` (walking the graph), and that `GradientUtils.ZeroGrad(output)`
  clears a parameter gradient seeded before the call. Reachability rather than direct
  membership, because three Parameter-owning modules reach their parameter through a
  delegated op rather than their own node — see the deviations section. This pins the
  `ZeroGrad` behaviour the contract now implies.
- The two BatchNorm eval-path nodes get explicit rows asserting `[input]` only,
  so the *correct* exclusion is pinned rather than left to drift into looking
  anomalous — which is exactly what happened to the attention mask.

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
  Operations/ForwardGradOperations.cs, Nn/{ModuleHelpers,Conv1d,Conv2d,BatchNorm,LayerNorm,RMSNorm}.cs}`,
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
3. `fix(autodiff): list module parameters in OpNode.Inputs` (the 7 `Nn/` sites that
   accumulate into parameters — see the G1 correction)
4. `test: enforce the OpNode.Inputs contract across both AD modes` (new contract
   tests + reflection completeness guard + forward-mode mask tests)
5. `docs: state the OpNode.Inputs contract and correct the stale OpNode reference`
   (`OpNode.cs` XML doc, `docs/AUTODIFF.md`, `CHANGELOG.md`)

## G1 grounding record (complete)

Grounded before any implementation.

**Microsoft Learn**
- *System.Span\<T\> struct* — `Span<T>` is a `ref struct` and "can't be
  boxed". Confirms a reflection-driven **invocation** harness is impossible for
  Span-taking methods, so per-op contract rows must be hand-written lambdas and
  the reflection guard is restricted to **signature scanning**. Verified no
  *public* op in either class takes a `Span` (`DropoutWithMask` is `internal`),
  so scanning public methods is safe.
- *CA2208: Instantiate argument exceptions correctly* — an `ArgumentException`
  from a method with a parameter must be constructed with the correct
  `paramName`. Confirms `RequireConstant`'s `new ArgumentException(message,
  nameof(param))` shape.
- *Friend assemblies* — `InternalsVisibleTo` grants the test assembly access to
  `internal OpNode<T>.Inputs`, which the contract tests assert on.

**code-memory** — confirmed the exact `Forward` ranges holding all nine sites:
`BatchNorm.cs:102-206` (1dEval/1dTrain), `BatchNorm.cs:351-458`
(2dEval/2d), `Conv1d.cs:101-186`, `Conv2d.cs:154-368`,
`Conv2d.cs:774-888` (ConvTranspose2d), `LayerNorm.cs:85-154`,
`RMSNorm.cs:56-134`.

**Red flags raised at G1 and resolved:**
1. *The plan was wrong about two of nine sites* — `BatchNorm1dEval`/
   `BatchNorm2dEval` accumulate into `input` only and are already
   contract-correct. Corrected to seven sites above.
2. *The gating predicates are not uniform* — a single `useBias`/`affine`
   predicate would have diverged from three of the seven closures. Resolved with
   one authoritative `ModuleHelpers.NodeInputs` helper (proposed change 4).
3. *Out-of-scope PyTorch divergence* — BatchNorm eval mode produces no
   `weight`/`bias` gradient while the eval forward does use them. Captured as
   **#494**, not fixed here.

Blast radius unchanged from the pre-G1 draft except: seven rather than nine
module sites touched, and `Nn/ModuleHelpers.cs` added to the touched set.

## GitHub issues log

- [#494](https://github.com/khurram-uworx/Nivara/issues/494) — BatchNorm eval mode
  produces no `weight`/`bias` gradient, diverging from PyTorch (created at G1
  while auditing the nine `OpNode` sites)

## Deviations from the plan (recorded before G2)

### Commits

Six landed, not five. The plan's numbering put the docs last and had no slot for
the TODO.md update, so the numbering drifted: actual order is `e51f0c3`, `4ce43fe`,
`a959358`, `d2401bb`, `ad8f42b`, `4d2fb0b`, `0d5a478`. Content matches the intent of
each planned commit; only the count and the placement of the G1-record commit differ.

### Module rows: ten, not seven

The plan's module-row scope was the seven sites that change. Writing the reflection
guard showed that scope was the wrong cut: the guard is over *every* public
`Module<T>` owning a `Parameter<T>`, which is ten. Three of them — `Linear`,
`Embedding`, `SparseEmbedding` — build no `OpNode` of their own and reach their
parameter through a delegated op, and the tenth (`VAE`) turned out to reach its
`beta` only through `ElboLoss`, never through `Forward`.

Keeping the guard over ten while writing rows for seven would have made it fail on
every run, so the rows were widened to match the guard. Three consequences worth
recording:

1. **Module rows assert graph reachability, not direct `Inputs` membership.** For
   `Linear`/`Embedding`/`SparseEmbedding`/`VAE` the parameter is not an input of the
   output's own node, so direct membership is the wrong assertion — it would encode
   a structural accident as if it were the contract. Reachability is what the
   contract actually requires and what `ZeroGrad` needs, since `ZeroGrad` walks the
   same edge set.
2. **`Embedding` and `SparseEmbedding` consume their input as integer selectors**
   (`int.CreateChecked`), so the input is a non-differentiable constant that
   correctly never enters the graph. Their rows assert parameter reachability only,
   and `ZeroGrad` is asserted *not* to touch the input. This was a genuine finding
   from a failing row, not an assumption: the first version asserted the input was
   cleared and failed.
3. **`VAE.beta` is registered `requiresGrad: false`** and consumed as a scalar
   multiplier on the KL term, so it is reachable only via
   `ElboLoss(recon, original, mu, logVar)`. That is consistent with the contract —
   beta's backward *is* `Multiply`'s backward, and `Multiply` already lists every
   tensor argument — but it means the row's graph root is the loss, not `Forward`.

### Test scope: single-tensor-argument ops excluded, with the reason recorded

The plan said "every public op taking a tensor argument". That is 84 rows and adds
no defect-catching power: with one tensor argument there is nothing for a node to
omit. The guard's threshold is now *two or more* tensor arguments, plus `Concat`
whose single argument is a tensor array (so a caller can pass a mixed-gradient
array). This is a deliberate coverage reduction from the plan's wording and is
stated in the test fixture's own XML doc so it cannot be mistaken for full coverage.
The guard itself enforces the threshold, so if a single-tensor op ever grows a
second tensor argument it is picked up automatically.

### Verification actually performed

- `dotnet build Nivara.slnx -c Release` clean, 0 warnings, 0 errors.
- `dotnet test -c Release --no-build --filter "FullyQualifiedName~AutoDiff|Nn|Training|Backward"`:
  **1307 passed, 0 failed**, 11 skipped (the skips are the pre-existing
  unhostable-model ones, not new).
- New suite alone: **93 passed, 0 failed**.
- **Mutation-checked, because a green suite proves nothing until you know it can go
  red.** Removing the forward-mode mask guards fails 5 tests; reverting
  `LayerNorm`'s `NodeInputs` call fails 2. Both fixes were then restored and the
  suite returned to green. Neither fix is passing vacuously.

  One process note: the first mutation check appeared to still fail after restoring
  the source, because an incremental build skipped the restore and the test ran
  against the mutated DLL. `--no-incremental` plus a full rebuild confirmed green.
  Had I not re-run after the rebuild I would have reported a passing suite against
  code that was not the committed code.
