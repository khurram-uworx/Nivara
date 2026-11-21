# Plan: issue #1380 — minimal `LoRALinear<T>` module + grad-check test

## Problem

Add a reusable low-rank adapter module `LoRALinear<T>` (per `docs/plan/LORA.md` §3) that wraps a
frozen base `Linear` and adds `(alpha / r) * ((x @ A) @ B)`. Prerequisite freeze mechanism
(#1374) has landed (`Module<T>.Freeze()` / `SetRequiresGrad`).

## Proposed changes

- `src/Nivara/AutoDiff/Nn/LoRALinear.cs` (new): `sealed class LoRALinear<T> : Module<T>`.
  - Frozen base `Linear<T>` (Freeze on construction).
  - `A` (Kaiming, shape `[in, r]`), `B` (zeros, shape `[r, out]`).
  - Default `r = 8`, `alpha = 16`.
  - `Forward` composes base + scaled adapter via `MatMul`/`Multiply`/`Add`.
  - `Merge()` folds adapter into base weight and sets a `merged` flag.
  - `Base`/`BaseLinear`, `A`/`B` accessors (with aliases).
- `tests/Nivara.Tests/AutoDiff/LoRALinearTests.cs` (new): step-0 parity (exact), frozen-base has
  no grad, adapter grad vs finite differences, optional merge/round-trip.

## Verification

- `dotnet build Nivara.slnx`
- targeted `LoRALinearTests`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits

1. docs: plan #1380 in TODO.md
2. feat: add LoRALinear<T> module
3. test: add LoRALinearTests
4. docs: remove TODO.md — plan executed

## GitHub issues log

(none yet)
