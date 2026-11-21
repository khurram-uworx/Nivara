# Plan: Freeze mechanism for base weights (#1374)

## Problem

LoRA needs a frozen base model: only adapter params (A, B) train. Today `Linear`/`Embedding` create weights with `requiresGrad: true`, and BF16 mixed-typed params are registered via `RegisterMixedParameters`. Passing only adapter params to the optimizer is insufficient — AutoDiff still computes/holds gradients for base weights.

## Proposed changes

1. `Module<T>.Freeze()` (plus `Unfreeze()` / `SetRequiresGrad(bool)`) that rebuilds every leaf parameter tensor (same-type and mixed-type) with the requested `requiresGrad` flag, clearing accumulated gradients, then recurses into child modules.
2. Llama loader option `requiresGrad` (default `true`) on `LlamaLoader` and aliases; when false, call `model.Freeze()` after binding weights, before `Eval()`.

## Blast radius

- `src/Nivara/AutoDiff/Nn/Module.cs` — new public methods; affects all modules.
- `samples/Nivara.Samples/LlamaForCausalLM.cs` — loader signatures; callers use optional trailing params, backward compatible.

## Verification

- Build `Nivara.slnx`.
- Targeted tests: LinearLoadPathTests, RMSNormTests, Module-related.
- Frozen base produces null base gradients after backward.

## Planned commits

1. docs: plan #1374 in TODO.md
2. feat(autodiff): add Module<T>.Freeze/SetRequiresGrad
3. feat(samples): add requiresGrad option to LlamaLoader
4. (optional) tests
5. docs: remove TODO.md — plan executed

## GitHub issues log

- (none yet)