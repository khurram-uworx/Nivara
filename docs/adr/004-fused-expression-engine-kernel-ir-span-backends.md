# ADR-004: Fused Expression Engine Uses One Kernel IR and Three Backends, Executed Over Chunkable Spans

**Status:** Accepted
**Date:** 2026-08-15

## Context

The fused expression engine (Phase 2 of the POLARS roadmap, issue #153/#157)
already eliminated the boxed interpreter and fused expression trees into a single
pass via a compiled-first `Expression.Compile` target over `T[]` arrays with a
generic sealed node-tree fallback. Two follow-up requirements remained (issue
#167):

1. **Span-capable execution.** The compiled `Expression.Compile` delegate target
   runs over `T[]` arrays; ref-structs cannot appear in expression trees, so the
   fused target could not consume `ReadOnlySpan<T>` leaves directly. Query
   operators materialize whole-column arrays regardless of the requested batch
   size.
2. **Chunked execution.** The Phase 4 async streaming bridge (#171) needs
   memory-budgeted, chunk-at-a-time evaluation so a fused operator can stream
   rows without ever allocating the full result. The existing compiled target
   was all-or-nothing.

Three candidate designs were considered:

- **A — Rewrite storage into chunked/segmented buffers.** Rejected by scoping
  (too invasive); chunked execution should work over the existing contiguous
  `ColumnStorage<T>` buffers, which already slice zero-copy.
- **B — Keep the compiled delegate as the single engine and add chunking to it.**
  The `Expression.Compile` target works, but adding offset args and a chunk loop
  to the tree-compiled delegate gives no natural home for a SIMD
  (`TensorPrimitives`) single-op fast path, and it does not serve the flat,
  null-bearing numeric case (which cannot use `Expression.Compile` generic-math
  kernels over spans).
- **C — Lower every tree to a single post-order Kernel IR and route to three
  backends.** The IR is the sole planning representation; backends consume it
  directly, so every plan type is chunk-capable by construction.

## Decision

**The fused engine lowers every expression tree to a single post-order kernel IR
(`KernelPlan`) and routes it to one of three backends; all backends execute over
zero-copy span/`ReadOnlyMemory` slices with an optional chunk size.**

1. **One IR.** `KernelLowerer` produces a `KernelPlan` (`KernelOp` post-order
   nodes, `MaxStackDepth`, classification flags) from any supported expression
   tree. The evaluator plans once, then the node-tree gate and every backend
   consume the same `KernelPlan`. No parallel hand-written tree walkers for the
   "real" path.

2. **Three backends, chosen by plan classification:**
   - **Flat IR span interpreter** (`FusedKernel`) for null-bearing uniform
     numeric plans: per-element `ReadOnlyMemory<T>` leaf access, hoisted literal
     values, inline OR null-mask propagation. This is the generic
     `INumber<T>`/`IFloatingPointIeee754<T>` SAIS path made span-native.
   - **TensorPrimitives SIMD backend** (`TensorPrimitivesKernel`) for null-free
     uniform single-op plans on Add/Subtract/Multiply/Divide: one BCL SIMD call
     with no per-element loop.
   - **Compiled offset delegate** for everything else (null-free chains,
     bool/And/Or/Not, heterogeneous plans): the existing
     `Expression.Compile` target, generalized to `(leafArrays, dest, start,
     count, destStart)` so it chunks without recompile.

3. **Chunked execution is an axis, not a mode.** `EvaluateChunked(expression,
   input, chunkSize)` slices the existing contiguous leaf storage per chunk
   (zero-copy `ReadOnlyMemory<T>` slices and mask slices) and writes each chunk
   into one shared output array at `destStart = start`. The output is
   bit-identical to whole-column evaluation, enforced by unit tests across chunk
   sizes `{1, 2, 3, 511, 512, 1024, len-1, len}` for all three backends.
   `chunkSize >= length` degrades to a single whole pass.

4. **C# ref-struct limits are honored, not fought.** No `ReadOnlySpan<T>[]`,
   `Nullable<Span<bool>>`, or `Span<ReadOnlySpan<T>>` in generic signatures
   (CS0611/CS9244). Backends take `ReadOnlyMemory<T>[]` inputs and
   `ReadOnlyMemory<bool>[]` masks; an empty output-mask span is the "no mask"
   sentinel. The value stack is a `T[]` rented per execution, not `stackalloc`
   (T is not provably unmanaged).

5. **Modulo stays off the SIMD backend.** `System.Numerics.Tensors` has no
   generic `Modulo<T>` (and `Ieee754Remainder` is not C# `%`), so a null-free
   single-op Modulo routes to the compiled backend. Routing guardrails in tests
   pin which backend each plan shape must take.

## Consequences

**Positive:**

- Every plan type is chunk-capable by construction; the Phase 4 streaming bridge
  (#171) can run fused operators at a memory budget without any storage rewrite.
- The flat IR interpreter finally consumes spans/`ReadOnlyMemory` directly,
  matching the BCL tensor pattern (spans are the currency) and removing the last
  array-only fused target.
- One IR gives a single place to reason about semantics (null short-circuit,
  mask OR, compute type promotion) and a natural future home for
  constant-folding and target lowering.
- The TensorPrimitives single-op backend gives the common `col * scalar` /
  `col + scalar` shapes a one-call SIMD path.
- Correctness is pinned: chunked vs whole bit-identity, null-mask propagation,
  and backend routing are all guarded by unit tests.

**Negative:**

- Three backends to maintain; routing heuristics must stay explicit (guarded by
  the routing tests) or plans can silently shift backend and change
  performance/behavior characteristics.
- The compiled `Expression.Compile` delegate still runs over `T[]` arrays
  internally (ref-struct exclusions); only the IR interpreter and chunk slicing
  are span-native. Full span-nativeness of the compiled target is tracked as
  #155.
- The shared output array (one allocation per evaluation) is the invariant that
  keeps the 1.01 allocation guardrail; callers must not request per-chunk
  results.

## Amendment process

To add a fourth backend or change routing, file an issue referencing ADR-004 and
update the routing guardrail tests (which assert the backend each plan shape
must take) together with the code. Any change must keep chunked output
bit-identical to whole-column evaluation for all backends.
