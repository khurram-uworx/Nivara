# Plan: Resolve #202 — remove the throwing multi-input `Forward` from `Module<T>`

## Problem

`Module<T>.Forward(input1, input2)` (`src/Nivara/AutoDiff/Nn/Module.cs:15-18`) is a
virtual method on the base class that **always throws** `NotSupportedException`. Only
`MultiheadAttention<T>` legitimately overrides it (fixed for virtual dispatch in
#203). `VAE<T>.Forward(x, condition)` (`src/Nivara/AutoDiff/Nn/VAE.cs:71`) also
declares `new` on the same erased signature (the `?` on a reference type does not
change the signature), so it hides the base member too. Every other `Module<T>`
subclass (~16 in `src`, ~18 in `samples`) advertises a two-input forward it rejects
at runtime — a "silent API lie".

## Decision

There are exactly two classes with multi-input forward capability, so we do not
generalize the API on the base class. The capability becomes opt-in via a new
`IMultipleInputModule<T>` interface implemented only by `MultiheadAttention<T>` and
`VAE<T>`. Consumers holding a `Module<T>` reference dispatch via pattern matching:

```csharp
if (module is IMultipleInputModule<T> multiInput)
{
    var output = multiInput.Forward(a, b);
}
```

## Changes

### 1. New `src/Nivara/AutoDiff/Nn/IMultipleInputModule.cs`

```csharp
namespace Nivara.AutoDiff.Nn;

public interface IMultipleInputModule<T> where T : struct, IFloatingPointIeee754<T>
{
    ReverseGradTensor<T> Forward(ReverseGradTensor<T> input1, ReverseGradTensor<T> input2);
}
```

### 2. `src/Nivara/AutoDiff/Nn/Module.cs` — delete the virtual

Remove `Forward(input1, input2)` (lines 15-18). `Module<T>` keeps only the abstract
single-input `Forward(input)`. Intentional breaking change: the removed member only
ever worked through MHA's concrete type.

### 3. `src/Nivara/AutoDiff/Nn/MultiheadAttention.cs`

- `public sealed class MultiheadAttention<T> : Module<T>, IMultipleInputModule<T>`
- The padding-mask overload drops `override` → plain public method (implicit interface
  implementation; params are non-nullable, so no nullability warnings). The 3-arg
  `Forward(query, key, value, causal, paddingMask)` is untouched.

### 4. `src/Nivara/AutoDiff/Nn/VAE.cs`

- `public sealed class VAE<T> : Module<T>, IMultipleInputModule<T>`
- Remove the `new` keyword from `Forward(x, condition?)` (otherwise CS0109 once the
  base virtual is gone). The public nullable overload stays the idiomatic entry point.
- Add an explicit interface implementation delegating to the public overload (avoids
  CS8767 nullability-mismatch on implicit implementation):

```csharp
ReverseGradTensor<T> IMultipleInputModule<T>.Forward(ReverseGradTensor<T> input1, ReverseGradTensor<T> input2)
    => Forward(input1, input2);
```

### 5. `tests/Nivara.Tests/AutoDiff/NnTests.cs`

- Re-point the #203 regression test `MultiheadAttention_PaddingMask_DispatchesThroughModuleReference`
  (line 2798) at `IMultipleInputModule<float>`; rename to
  `..._DispatchesThroughMultipleInputInterface`.
- Add a VAE interface-dispatch test (`VAE_Conditional_DispatchesThroughMultipleInputInterface`).

### 6. Docs

- `docs/AUTODIFF.md:481` — replace the "Virtual — multi-input forward (throws by
  default)" row with a note that multi-input forward is opt-in via
  `IMultipleInputModule<T>` (implemented by `MultiheadAttention<T>` and `VAE<T>`).
- `CHANGELOG.md` — add a breaking-change entry under `[Unreleased]`.

## Verification

- `dotnet build Nivara.slnx` (asks human first per AGENTS.md if longer verification needed).
- Run `tests/Nivara.Tests` AutoDiff `NnTests` (MHA padding-mask + interface dispatch,
  VAE conditional + interface dispatch). **Ask before running `dotnet test`.**

## Planned commits

1. `docs: plan #202 (IMultipleInputModule) in TODO.md`
2. `refactor(autodiff): introduce IMultipleInputModule<T>, drop throwing multi-input virtual from Module<T> (#202)` — new interface + Module.cs + MHA + VAE (one cohesive change).
3. `test(autodiff): dispatch multi-input forward through IMultipleInputModule<T> (#202)`
4. `docs: document IMultipleInputModule<T>, add CHANGELOG breaking-change entry (#202)`
5. `docs: remove TODO.md — plan executed`

## Blast radius

| File | Change | Downstream impact |
|------|--------|-------------------|
| `src/Nivara/AutoDiff/Nn/Module.cs` | remove virtual 2-arg forward | Only the #203 test calls it via `Module<float>` ref; no production callers. `Sequential<T>` chains single-input only. |
| `src/Nivara/AutoDiff/Nn/MultiheadAttention.cs` | implement interface, drop `override` | Direct calls `mha.Forward(input, paddingMask)` unchanged. |
| `src/Nivara/AutoDiff/Nn/VAE.cs` | implement interface, drop `new`, explicit impl | Direct calls `vae.Forward(x, condition)` unchanged. |
| `src/Nivara/AutoDiff/Nn/IMultipleInputModule.cs` (new) | — | Additive. |
| `tests/Nivara.Tests/AutoDiff/NnTests.cs` | MHA + VAE interface-dispatch tests | None outside tests. |
| `docs/AUTODIFF.md`, `CHANGELOG.md` | docs | None. |

Tests covering this surface: `NnTests.MultiheadAttention_PaddingMask_*`
(ShapeCorrect, DispatchesThroughModuleReference, BackwardFlows, #203 regression) and
`NnTests.VAE_Conditional_*`.

## GitHub issues log

- [ ] none so far — deferred work found during execution is filed here immediately.
