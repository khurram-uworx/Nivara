# NivaraChat — Living Decision Log

This file records decisions made around the NivaraChat sample and its linked
workstreams (Idea A — batched transformer chat sample; MatMul / batched-attention
library work). It is a durable reference for future contributors: each entry
captures the context, the reasoning, the decision, and any follow-ups. Append
new decisions in date order; do not rewrite history.

Related planning documents:
- `docs/SHAKESPEARE.md` — earlier plan (batched rank-3 MatMul core work + sample); superseded by the decisions below
- `samples/NivaraChat/README.md` — durable NEXT.md content (Idea A description, architecture, CLI spec, gaps, not-doing list); companion-sample section covers `samples/NivaraChatClient/`
- `docs/SHAKESPEARE2.md` — revised plan (grounded audit + batched attention via fused op); **retired 2026-08-05** after all six tasks completed — decisions folded into this log (see D-011)
- `samples/NivaraChat/NEXT.md` — original roadmap for Idea A; **deleted 2026-08-05** after salvage into the README

---

## Decision D-001 — Do not add rank-3 (batched) MatMul to Nivara core now

**Date:** 2026-08-05
**Status:** Principle accepted (ADR-003). Tactical deferral recorded below.
**Resolution (2026-08-05):** sample team chose **Route B** (D-002). Follow-up
issues filed: **#118** (batched-MatMul backlog) and **#119** (rank-2 MatMul
backward allocation reduction).
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

## Decision D-002 — Batched attention API shape: separate `BatchedMultiHeadAttention` op

**Date:** 2026-08-05
**Status:** Accepted. Implemented (`e451f80`), parity-tested (`d2ecc46`).

### Context

Task 1 of SHAKESPEARE2 left the batched API shape open (batch overload of the
existing fused `MultiHeadAttention` op vs. a separate op) and selected Route B
(extend the fused op to `[B, L, D]`, loop the batch internally). D-001 item 1
is the library-level justification for this route.

### Decision

- New public op **`ReverseGradOperations.BatchedMultiHeadAttention<T>(query, key, value, nHead, attnScale, mask?)`**
  (ReverseGradOperations.cs:569) accepting rank-3 `[B, L, D]` Q/K/V and an optional
  additive `[B, qLen, kvLen]` mask. A separate op (not an overload) keeps the
  single-sequence `MultiHeadAttention` op untouched and bit-identical — the
  regression surface is zero.
- Batch parallelization via `Parallel.For` only when `ShouldParallelizeBatch`
  holds: B ≥ 4 **and** B·total-work ≥ 2^21. Below that, sequential is faster.
- All transient forward/backward buffers are pooled via `ArrayPool<T>.Shared`.
- Type constraint `IFloatingPointIeee754<T>`; graph nodes only inside
  `GradientUtils.Grad()` (inference-default).

### Follow-ups

- Backlog issue **#118** — batched (rank-3) MatMul primitives, still open.
- Parity coverage added in Task 3 (D-003); perf verdict in D-004.

---

## Decision D-003 — Parity fixture RNG strategy for batched attention

**Date:** 2026-08-05
**Status:** Accepted. Implemented (`d2ecc46`); NivaraTorch 71/71 pass.

### Context

`gen_reference.py` seeds a generator with 42, but conv/linear/embedding weights
consume *global* torch RNG (documented convention in the file), so inserting new
cases mid-stream would perturb every later fixture. The NivaraTorch parity gate
must stay deterministic and reproducible.

### Decision

- Append batched cases **at the very end of `run()`** only (never mid-stream).
- Batched cases consume a dedicated `attn_rng` (seed 303) drawn **after** the
  existing attention cases, so the seed-42 attention fixtures stay bit-identical.
- The shared-stream conv/linear fixtures were regenerated (expected per the
  documented convention); the full `samples/data/torch-comparison/` tree is
  committed as one unit. Manifest grew to 56 cases; 3 new tests in
  `tests/Nivara.Tests/NivaraTorch/BatchedAttentionTests.cs` cover causal
  forward, causal backward, and cross-attention forward+backward.

### Follow-ups

None.

---

## Decision D-004 — Performance gate: no further kernel changes this cycle

**Date:** 2026-08-05
**Status:** Accepted. Recorded with measurements in SHAKESPEARE2 Task 4.

### Context

"Implement → measure → improve → measure." The honest baseline for batched
`[B, L, D]` is running the existing single-sequence MHA B times. Shapes = Idea A
defaults (`B=16, L=128, D=64, H=4`), causal mask, Release, 16 cores, medians of
3+ runs × 12 iterations.

### Decision

| Scenario | Per-seq loop (baseline) | Batched op | Improvement |
|---|---|---|---|
| forward — time/op | ~20.5 ms | ~7.5 ms | **~2.7x** |
| forward — B/op | ~2.14 MB | ~0.53 MB | **~4.0x** |
| forward+backward — time/op | ~72 ms | ~23 ms | **~3.1x** |
| forward+backward — gen0/op | ~1.1 | ~0.33 | **~3.3x** |

- **No further kernel changes this cycle.** Route A (rank-3 MatMul primitives) is
  not justified by these numbers — it stays a backlog idea (**#118**).
- `ShouldParallelizeBatch` (B≥4 and B·work ≥ 2^21) held at the measured shape;
  cache blocking / parallel-threshold tuning not worth perturbing.
- Caveat documented: the batched path runs `Parallel.For` over B, so a share of
  its allocations occurs on thread-pool workers and is **not** attributed to the
  calling thread by `GC.GetAllocatedBytesForCurrentThread`; gen0/op is the
  reliable signal.

### Follow-ups

- **#119** (filed 2026-08-05): pool the rank-2 `MatMul` backward transients
  (`bTArr`/`aTArr`, ReverseGradOperations.cs:277/287) — the batched op pools its
  buffers but the plain MatMul backward closure still `new T[]`s them every
  backward; shows up on the sample's tied-LM-head hot path.

---

## Decision D-005 — Sample CLI deltas from the NEXT.md spec

**Date:** 2026-08-05
**Status:** Accepted. Implemented (`2a9334b`).

### Context

NEXT.md specified a `--train` / `--interactive` / `--seq-len` CLI. Task 5 landed
with a NivaraGpt-style CLI instead; the delta needed recording so the spec
stops being treated as authoritative.

### Decision

- **Dropped** `--train` and `--interactive` REPL. Default behavior is
  train-when-not-loaded, then generate `--samples` replies and run a DI demo;
  `--load <path>` skips training. Mirrors the NivaraGpt sample's interaction
  model.
- `--seq-len` renamed **`--block-size`** (matches NivaraGpt's `TransformerBlock`
  terminology and the model's `maxSeqLen`).
- Added `--max-new-tokens`, `--vocab-size`, `--data`, `--no-di-demo`, `--beta1`,
  `--beta2`. CLI table + delta note live in the NivaraChat README companion
  section.

### Follow-ups

None.

---

## Decision D-006 — `IChatClient` does not own the model

**Date:** 2026-08-05
**Status:** Accepted. Fixed during Task 5 smoke testing.

### Context

The first version of `BatchedChatClient.Dispose()` disposed the model it was
handed. After sample generation, the DI demo then re-used the same model and
threw `ObjectDisposedException` on `Parameter.get_Tensor` (Embedding forward).

### Decision

- The chat client is a **view** over a model it does not own; `Dispose()` is a
  no-op. Ownership lives with the caller / DI container.
- `Module<T>.Dispose()` is idempotent, so container + caller double-dispose is
  safe when both dispose the model.

### Follow-ups

None.

---

## Decision D-007 — Word-vocab LM-head cost is accepted for the sample

**Date:** 2026-08-05
**Status:** Accepted.

### Context

`TensorsHelper.MultiplyCore` emits one `TensorPrimitives.Dot` per output
element, so the tied LM head `[B*L, D] @ [D, V]` over an ~8k word vocab is ~1M
short dot-calls per training batch (~730 tok/s at `D=32, B=8`).

### Decision

- No core kernel change (consistent with D-004). The cost is a property of the
  existing MatMul; a word-level LM head is inherently wide in V.
- Default `--vocab-size 8000` stays; **smoke/CI runs use `--vocab-size 1200`**
  (~3x faster). Documented in the README companion section.

### Follow-ups

- **#119** — the MatMul backward allocation reduction (see D-004) is the only
  MatMul-side change considered worth doing.

---

## Decision D-008 — Tokenizer persistence convention

**Date:** 2026-08-05
**Status:** Accepted. Implemented (`2a9334b`).

### Context

`ModelSerializer` persists only model state; the word tokenizer (vocab) must be
reconstructed identically on load or the model is unusable.

### Decision

- Tokenizer is saved/loaded via **`<model>.tokenizer.json`** next to the model
  JSON (`--save` writes it; `--load` auto-restores it).
- Load requires re-passing the same architecture flags (`--n-embd`,
  `--n-layer`, `--block-size`, `--n-head`, `--vocab-size`); `LoadStateDict`'s
  strict shape validation fails loudly on a mismatch rather than silently
  corrupting.

### Follow-ups

None.

---

## Decision D-009 — DI wiring via `AddChatClient(factory)`

**Date:** 2026-08-05
**Status:** Accepted. Implemented (`2a9334b`).

### Context

NEXT.md sketched `services.AddChatClient<NivaraChatClient>()`. Microsoft.Extensions.AI
**10.8.3** has no generic `AddChatClient<TClient>()` overload — only
`AddChatClient(IServiceCollection, IChatClient, ServiceLifetime)` and the
`Func<IServiceProvider, IChatClient>` factory overload.

### Decision

- Use the factory overload:
  `services.AddChatClient(sp => new BatchedChatClient(sp.GetRequiredService<BatchedTransformer<float>>(), sp.GetRequiredService<TextTokenizer>(), ...))`
  with the model and tokenizer registered as singletons. The DI demo resolves
  `IChatClient` from the container and runs a chat round-trip.

### Follow-ups

None.

---

## Decision D-010 — `Softmax(dim)` gap closed without a new public API

**Date:** 2026-08-05
**Status:** Accepted.

### Context

NEXT.md Gap 6 claimed core `Softmax` ignores `dim`. In fact core
`Softmax<T>(a)` softmaxes over `shape[1]` for rank ≥ 2 (ReverseGradOperations.cs:1836)
— not a generic `dim` parameter. An early README draft overclaimed that
`Softmax(dim)` was "already fixed in core."

### Decision

- **No public `Softmax(dim)` is needed.** Attention softmax is applied over the
  last dimension *inside* the MHA / `BatchedMultiHeadAttention` op kernels
  (`AttentionKernels.SoftmaxRows`), which is exactly what attention requires.
- README corrected: the "already resolved" list omits `Softmax(dim)`; the sample
  never needed it.

### Follow-ups

None.

---

## Decision D-011 — Retire `docs/SHAKESPEARE2.md`

**Date:** 2026-08-05
**Status:** Accepted (this change).

### Context

All six SHAKESPEARE2 tasks completed and committed (`7b44c30` → `9f9f1cb`):
grounded audit verified, Route B chosen and shipped, parity fixtures, perf gate,
sample, docs finale. The plan's task handouts (acceptance criteria, standing
instructions) have served their purpose; its durable decisions are now recorded
here (D-001–D-010) and its audit facts live in the NivaraChat README "Gaps
resolved along the way" section.

### Decision

- Delete `docs/SHAKESPEARE2.md`. Nothing remains to salvage: the decision log
  (this file) holds the decisions, the README holds the durable NEXT.md content,
  and outstanding work is tracked as issues **#118** and **#119**.
- `docs/SHAKESPEARE.md` (the *earlier* plan) is left in place — out of scope for
  this decision, referenced only as history.

### Follow-ups

None.

---

## Resolved Open Questions

- **Which route does the sample team choose (SHAKESPEARE2 Route A / B / C)?**
  → **Route B** (D-002). Route A deferred as **#118**; Route C rejected (no batch
  parallelism).
- **Should the batched attention work (D-001 item 1) ship before or with the
  sample?** → **Before.** Core + parity + perf (Tasks 2–4) landed before the
  sample (Task 5), so the sample dogfooded a correctness-gated op.
