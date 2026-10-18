# Plan: issue #414 — P2: Fuse final Norm + head for prefill (pool `[L, hidden]` workspaces)

## Problem

`LlamaForCausalLM<T>.ForwardPrefill` fused branch allocates the model-level
ping-pong `[L, hidden]` workspaces `h0`/`h1` as fresh `new T[L * hidden]`
arrays on every call, although every decoder block already rents its own
`[L,·]` arrays from `ArrayPool<T>.Shared`. The final RMSNorm + tied LM head
itself is already fused (in-place norm, last-row span slice, GEMV into logits),
but the outer `[L, hidden]` workspaces still cost B/op on the prefill seed
rows tracked in `qwen-decode-block-baseline.json`.

## Blast radius

- `samples/Nivara.Samples/LlamaForCausalLM.cs` — `ForwardPrefillFusedCore` only.
  The non-fused (Grad-enabled / toggle-off) branch keeps its `ReverseGradTensor`
  path untouched (it must build a graph). Decoder blocks, RMSNorm, and
  `LlamaFusedKernels` are not touched.
- Tests: `tests/Nivara.Tests/AutoDiff/LlamaForCausalLMPrefillTests.cs`
  (parity + bit-exact final stage) unchanged bodies.
- Docs: one ledger row in `docs/QWEN.md` (no citation pin changes).

## Proposed changes

Pool `h0`/`h1` via `ArrayPool<T>.Shared.Rent(Math.Max(L*hidden, 1))` and
return them in a `finally`. No other semantic change.

## Verification

- Targeted: `LlamaForCausalLMPrefillTests` (all green, no diff in logits).
- Gate (human-run): `dotnet run --project tests/Nivara.PerformanceTests -c Release -- --compare ...qwen-decode-block-baseline.json --only Qwen --runs 3`.

## GitHub issues log

(none new — closes #414)
