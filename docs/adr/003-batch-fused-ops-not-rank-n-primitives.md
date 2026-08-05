# ADR-003: Batch is a First-Class Dimension Handled Inside Fused Ops, Not via Rank-N Primitives

**Status:** Accepted
**Date:** 2026-08-05

## Context

The NivaraChat "Idea A" workstream (batched causal transformer sample) surfaced a
recurring design fork: to support batched (rank-3) tensors such as `[B, L, D]`,
should the AutoDiff core add a general-purpose rank-3 `MatMul` primitive with a
new backward pass, and then compose batched attention from it?

A maintainer review grounded the premise against the actual core and found it
stale:

- Q/K/V projections and MLP use a **shared** weight: `[B, L, D] @ [D, D]` is a
  rank-2 `[B·L, D] @ [D, D]` after a reshape — the batched `Embedding<T>` already
  does this flatten, and `Linear<T>` covers the rank-2 multiply.
- Attention scores (the only true batched-B case) are already handled by the
  fused `ReverseGradOperations.MultiHeadAttention` op, which loops heads
  internally over rank-2 `TensorsHelper.MultiplyCore(..., bTransposed: true)`.
- Rank-3 MatMul alone would not make batched attention composable anyway: there
  is no batch-aware softmax-with-dim and no rank-4 permute (`TransposeAxes` caps
  at rank 3).
- Every existing batched construct in the library handles the batch **inside the
  op/module with an internal loop** — `BatchNorm1d` over L, `Embedding` via
  reshape, the fused MHA over heads. There is no consumer-driven need for a new
  rank-N idiom.
- `TensorsHelper.cs` is explicitly a thin kernel layer with BCL swap targets
  ("Swap target when `Tensor.MatrixMultiply` ships"; ".NET 11: `Tensor.Transpose`").

## Decision

**The AutoDiff core treats the batch dimension as a first-class dimension that is
handled inside fused ops and modules via internal loops, not by adding general
rank-N primitive ops.**

1. **Extend, don't re-compose.** When a batch dimension is needed, extend the
   relevant fused op/module to accept the batched input and loop the batch
   internally (reusing its already-correct backward), rather than building the
   behavior from new rank-N primitives.

2. **Batch idiom is internal-loop.** New batched capabilities follow the existing
   pattern: batch is consumed by an op/module and iterated (e.g. `Parallel.For`
   over B when large enough), not surfaced as new primitive ops the caller must
   orchestrate.

3. **Rank-N primitives are opt-in, evidence-gated.** A new rank-N primitive op
   (e.g. batched `MatMul`) may be added only when at least one holds:
   - a real consumer outside the fused-op surface needs it and cannot express it
     through existing ops, or
   - the BCL ships a durable kernel to wrap (e.g. `Tensor.MatrixMultiply`), so
     the primitive is not a stopgap kernel wearing a permanent public API.

4. **Consistency obligations.** Any new primitive op ships with forward-mode
   symmetry (`ForwardGradOperations`), shape validation with clear exceptions,
   and PyTorch-parity fixture coverage, per the NivaraTorch correctness gate.

### Non-goals this ADR does NOT change

- The temporary deferral of batched MatMul and its revisit triggers are a
  tactical matter, tracked in the GitHub backlog — not in this ADR.
- Rank-2 `MatMul` stays the general multiply primitive; this ADR does not alter
  its API.

## Consequences

**Positive:**

- Smallest correct surface for batched transformers (the real gap was the batch
  dimension, not rank-2 MatMul); the fused MHA backward is reused, not re-derived.
- One established batching idiom across the library instead of two competing ones.
- No permanent public API added on top of a stopgap kernel while the BCL matmul
  story is still in flux (`TensorsHelper` documents swap targets for `.NET 11`).
- Agents and the sample team get a stable answer to the "add batched MatMul?"
  fork without re-litigating it.

**Negative:**

- Batched-B attention cannot be expressed as a user-composed rank-3 matmul chain
  until (if ever) the conditions in the Decision are met; it is expressed through
  the fused op instead.
- If a real consumer later appears, a small amount of design work was deferred
  (recorded in the backlog issue).

## Amendment process

This decision is recorded so agents and the sample team do not re-open the
rank-N primitive fork. To revisit, file an issue referencing ADR-003 with either
a real consumer that cannot be expressed through the fused-op surface, or
evidence that the BCL now provides a durable kernel (e.g. `Tensor.MatrixMultiply`)
worth wrapping. A revisit must also add the forward-mode symmetry and parity
fixtures described above.
