# NivaraChat — Living Decision Log

This file records decisions made around the NivaraChat sample and its linked
workstreams (Idea A — batched transformer chat sample; MatMul / batched-attention
library work). It is a durable reference for future contributors: each entry
captures the context, the reasoning, the decision, and any follow-ups. Append
new decisions in date order; do not rewrite history.

Related planning documents:
- `docs/SHAKESPEARE.md` — earlier plan (batched rank-3 MatMul core work + sample)
- `docs/SHAKESPEARE2.md` — revised plan (grounded audit + batched attention via fused op)
- `samples/NivaraChat/NEXT.md` — original roadmap for Idea A

---

## Decision D-001 — Do not add rank-3 (batched) MatMul to Nivara core now

**Date:** 2026-08-05
**Status:** Principle accepted (ADR-003). Tactical deferral recorded below.
The durable principle behind this entry is formalized and accepted in **ADR-003**
([`docs/adr/003-batch-fused-ops-not-rank-n-primitives.md`](../../docs/adr/003-batch-fused-ops-not-rank-n-primitives.md)).
This entry keeps the tactical deferral, the code evidence, and the revisit
triggers; the ADR is the decision that prevents re-opening the fork.

### Context

Two competing sample plans exist:
- `docs/SHAKESPEARE.md` assumes the core gap is "MatMul is rank-2 only" and
  proposes adding a general-purpose batched (rank-3) MatMul with a new backward
  pass, then building the sample on top of it.
- `docs/SHAKESPEARE2.md` grounds itself in an audit of `src/Nivara/` and
  concludes the real gap is the **batch dimension**, recommending extending the
  existing fused `MultiHeadAttention` op to accept `[B, L, D]`.

This entry assesses the first plan from the library maintainer's perspective:
should core add rank-3 MatMul with a new backward pass, independent of the
sample's needs?

### Assessment (library maintainer's lens: needed, consistent, tasteful, durable)

**1. Rank-3 MatMul is ergonomics, not capability — a batched transformer is
already expressible.**

- Q/K/V projections and MLP use a *shared* weight: `[B, L, D] @ [D, D]` is a
  rank-2 `[B·L, D] @ [D, D]` after a reshape. The batched `Embedding<T>` already
  does exactly this flatten (`AutoDiff/Nn/Embedding.cs`), and `Linear<T>` covers
  the rank-2 multiply. No new capability.
- Attention scores (the only true batched-B case) are already handled by the
  fused `ReverseGradOperations.MultiHeadAttention` (`ReverseGradOperations.cs:362`),
  which loops heads internally over rank-2
  `TensorsHelper.MultiplyCore(..., bTransposed: true)`. No FLOPs are wasted in
  the current design.
- A rank-3 MatMul alone would not unlock composable batched attention anyway:
  there is no batch-aware softmax-with-dim and no rank-4 permute
  (`TransposeAxes` caps at rank 3). Real batched attention ergonomics would
  require a whole "batched layer" surface, far larger than one op.

Net: rank-3 MatMul buys API shape and a batch-parallel window — not capability.

**2. It violates the library's established batching idiom.**

Every batched construct in `src/Nivara/AutoDiff` handles the batch inside the
op/module with an internal loop — `BatchNorm1d` over L, `Embedding` via reshape,
the fused MHA over heads. Introducing a standalone rank-3 matmul primitive would
create a new idiom with no consumer-driven need. The change consistent with the
codebase is SHAKESPEARE2's Route B: extend the fused `MultiHeadAttention` op +
`AttentionKernels` to `[B, L, D]` and loop the batch internally, reusing the
already-correct fused backward. That is the real library gap (batch) and the
smallest correct surface.

**3. It would be a stopgap kernel wearing a permanent public API.**

`TensorsHelper.cs` is explicitly a thin kernel layer with BCL swap targets
("Swap target when `Tensor.MatrixMultiply` ships"; ".NET 11: `Tensor.Transpose`",
lines 17–22, 78, 95, 119). The VJP backward (grads w.r.t. `a`, and shared-`b`
grad = sum over batch) is genuinely Nivara's own value — the BCL will never
provide autograd — but the kernel a hand-rolled batched op would wrap is exactly
the kind of code this file treats as temporary. Adding a public, forever-tested
op on top of a stopgap kernel for a consumer that does not strictly need it is
not tasteful. It would also need forward-mode symmetry
(`ForwardGradOperations.MatMul` is rank-2 only) and PyTorch parity fixtures to
land consistently.

**4. The legitimate rank-2 MatMul improvement is smaller than SHAKESPEARE.md claims.**

SHAKESPEARE.md Task 3 (pool the transient transpose arrays in rank-2 MatMul) is
partially already done: the `TensorsHelper.MultiplyCore*` kernels already rent
their `aCopy`/`bT` buffers from `ArrayPool` (lines 167–168, 196–197, 225–226).
What remains is the op-level backward closure, which allocates `bTArr` and
`aTArr` as `new T[]` (`ReverseGradOperations.cs:277, 287`) — ~2 arrays per
backward, rentable in a `finally`. That is a real, sample-independent, low-risk
improvement, but smaller than the original plan advertised.

### Decision

- **Do not add rank-3 (batched) MatMul with a new backward pass to Nivara core now.**
- The core work that IS justified, independently of the sample:
  1. Extend the fused `MultiHeadAttention` op + `AttentionKernels` to accept
     `[B, L, D]` with an internal batch loop (e.g. `Parallel.For` when B is
     large), keeping the single-sequence path bit-identical. This is the genuine
     capability gap (batch).
  2. Reduce the remaining rank-2 MatMul backward allocations by renting
     `bTArr`/`aTArr` from `ArrayPool<T>.Shared` (return in `finally`).
- **Backlog candidate:** rank-3 `BatchedMatMul` (shared-B + batched-B overloads,
  with VJP backward). Revisit only when (a) a real consumer outside attention
  needs it, or (b) BCL `Tensor.MatrixMultiply` lands and provides a durable
  kernel to wrap — at which point it should ship with forward-mode symmetry and
  PyTorch parity fixtures.

### Follow-ups

- File a GitHub issue for the batched-MatMul backlog candidate
  (`gh issue create --repo khurram-uworx/Nivara`, `--body-file`).
- File an issue for the rank-2 MatMul backward allocation reduction if the team
  does not pick it up with the batched-attention work.
- Record the sample team's route choice (Route A / B / C) here once made.
- Batched-MatMul revisit conditions are also captured in ADR-003's amendment
  process — a revisit requires a real consumer or a durable BCL kernel.

---

## Open Questions

- Which route does the sample team choose (SHAKESPEARE2 Route A / B / C)?
- Should the batched attention work (D-001 item 1) ship before or with the
  sample, or is a per-sequence loop acceptable for the first cut?
