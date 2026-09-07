# Plan: Qwen-fast P0-2 — eliminate the redundant weight copy in single-row matmul

Branch: `khurram/qwen-perf` (off `main`). Feeds the O(qwen-fast) effort; executes
item **P0-2** from `docs/QWEN-PERF.md`. Harness-first: measure before and after so
progress is objective (README-methodology: idle machine, `--runs 3` medians).

**Target runtime: .NET 11** (preview as of today, stable within weeks; the next
Nivara release targets it). The repo already builds/runs on .NET 11
(`Runtime: 11.0.0`, `System.Numerics.Tensors 11.0.0-preview.7.26381.103`). All
grounding below cites the net-11.0 moniker / mainline runtime source — the
ArrayPool behavior verified here is the .NET 11 implementation.

## Problem

Every decode matmul in the Qwen path is `aRows == 1` against weights stored
row-major `[out, in]` (`bTransposed: true`). `TensorsHelper.MultiplyCore*` rents
a buffer and **copies the entire weight matrix** (`b.CopyTo(bT)`,
`src/Nivara/Tensors/TensorsHelper.cs:150-151`) before dotting. For `aRows == 1`
and `bTransposed` the copy is a pure identity copy — the weight rows *already are*
the rows we dot against.

**Corrected premise (empirically measured + source-grounded, 2026-09-07):** the
original QWEN-PERF.md claim of a "fresh ~544 MB rent per token" is **wrong** on
the current runtime. `ArrayPool<T>.Shared` (TlsOverPerCoreLockedStacksArrayPool)
has 27 buckets pooling arrays up to ~2³⁰ elements (verified against the
dotnet/runtime source pinned by the MS Learn `ArrayPool<T>.Rent` page), so the
136,134,656-float workspace is pooled (~1 GB bucket) and reused across calls —
steady-state `B/op ≈ 0`. The actual per-token waste is **memory traffic**:

- `b.CopyTo(bT)`: read 545 MB + write 545 MB (pure identity copy)
- dot accumulation: read 545 MB (necessary work)
- `ArrayPool.Return(bT, clearArray: true)`: `Array.Clear` of the ~1 GB bucket
  (TensorsHelper.cs:173) — unnecessary zeroing on every return

≈ 2.7 GB redundant traffic per LM-head token (measured pre-fix: **188 ms/op**,
which matches this model). FFN adds ~3 × 17 MB matmuls with the same copy+clear
pattern. The fix removes the copy and the clear: each weight is read **once**
(~545 MB/token LM head + per-layer reads), a **~4-6× traffic cut** and a
**~1 GB working-set reduction** (the giant bucket is never rented again).

Root cause: the transposed-B preparation in `MultiplyCoreFloat` /
`MultiplyCoreDouble` / `MultiplyCoreGeneric` never considers that
`aRows == 1 && bTransposed` needs no preparation at all.

## Proposed changes

### A. Kernel fast path — `src/Nivara/Tensors/TensorsHelper.cs`

In `MultiplyCore<T>`, after the existing argument validation and **before** the
`typeof(T)` dispatch, add:

```csharp
if (bTransposed && aRows == 1)
{
    // Single-row mat-vec (BLAS2 GEMV): weights are row-major [out, in], so each
    // output column is a dot against one weight row. No rent/copy — the copy
    // was a pure identity at aRows == 1 and each return zeroed the ~1 GB pool
    // bucket (clearArray: true), so the LM-head alone moved ~2.7 GB/token
    // (copy read+write, dot read, bucket clear) when one 545 MB read suffices.
    // Reuses the same per-type Dot as the row kernels, so numerics stay
    // bit-identical. Deliberately TensorPrimitives-based, not hand-rolled
    // intrinsics — learn.microsoft.com/dotnet/standard/simd: "reach for the
    // existing higher-level APIs first". Swap target for this whole kernel
    // remains Tensor.MatrixMultiply once the BCL ships it (dotnet/runtime#95863,
    // BLAS epic dotnet/runtime#93286).
    var aRow = a.Slice(0, aCols);
    Span<T> res = result.AsSpan(0, bCols);
    for (int j = 0; j < bCols; j++)
        res[j] = DotFor<T>(aRow, b.Slice(j * aCols, aCols));
    return;
}
```

with a private helper mirroring the existing row-kernel dispatch:

```csharp
static T DotFor<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y) where T : struct, INumber<T>
    => typeof(T) == typeof(float) || typeof(T) == typeof(double)
        ? TensorPrimitives.Dot(x, y)   // matches MultiplyRowFloat / MultiplyRowDouble
        : WidenPrimitives.Dot(x, y);   // matches MultiplyRowScalar (widens Half/BF16 → float)
```

Bit-identical by construction: same `Dot` implementation, same elements, same
accumulation order as the multi-row paths. `ShouldParallelize` already gates at
`aRows >= 4`, so single-row never parallelized — no behavior change there.
Every caller passes a fresh result array, so reading `b` in place is safe
(document the invariant in the comment). All `GradKernels.MatMulTransposedB`
callers (reverse-mode `MatMulTransposedB` op, forward-mode ops, `Linear<T>`,
LM head, FFN) inherit the fast path automatically.

### B. Harness scenarios — `tests/Nivara.PerformanceTests/Program.cs`

Add `RunQwenDecodeMatMulScenarios()` registered from `RegisterScenarios()`.
Qwen2.5-0.5B shapes (hidden 896, vocab 151,936, FFN 4864, GQA heads 14/2):

| Scenario | Shape (aRows=1, aCols @ bCols) | What it isolates |
|---|---|---|
| `Qwen LM head matmul [1x896 @ 151936x896]` | 896 @ 151,936 | Raw `GradKernels.MatMulTransposedB`, preallocated result. **The P0-2 signal is ops/s**: pre-fix ≈ 5 ops/s / ~188 ms/op (copy+clear+dot traffic), post-fix ≈ 20-28 ops/s / ~35-50 ms/op. B/op is ~0 on both sides (pool reuse) — unchanged, so it is NOT the gate. |
| `Qwen FFN gate/up/down x3` | 896 @ 4864 (+ 4864 @ 896) | Three FFN matmuls per token |
| `Qwen attn Q/K/V/O proj` | 896 @ 896 × 2 + 896 @ 128 × 2 | Q/K/V/O projections per layer |
| `Qwen Linear fwd [1x896 -> 2688]` | op-level | `Linear<T>.Forward` single-row — captures per-op boxing allocs (neighbors P1-#4) |
| `Qwen LM head fwd [1x896 -> 151936]` | op-level | End-to-end per-token LM head through the op incl. result alloc |

Preallocated result buffers (fixes the alignment-noise caveat from the SIMD
guidance); small iteration counts for the LM-head row. Follow the `Run(...)`
pattern. Record and commit `qwen-fast-baseline.json` (see §Verification).

### C. E2E synthetic weights — `samples/NivaraInference`

Add `--synthetic-weights` for qwen: fabricate a tensor dict keyed by the same
names `LlamaForCausalLM.LoadModel` expects, with Qwen shapes from `LlamaConfig`
and deterministic pseudo-random values — so `qwen benchmark: KV-cache decode`
(and `RunBenchmark`) runs **without the model file** (timing is shape-driven).
Correctness stays gated on the real fixtures (`qwen_tool_logits_py.bin`, etc.).
Implemented via `Qwen.RunSyntheticBenchmark()` sharing one benchmark body
(`RunDecodeBenchmark`) with the real path. Design details: a random-weight
argmax can never hit a stop id, so the synthetic decode is capped at 24 tokens
per path (vs 160 real) to keep the full-forward path practical (~30 min
otherwise) — same timing regime as the real ~19-token tool turn. Flag must be
rejected for non-qwen models and must not affect real-weight loads.

### D. Unit tests — `tests/Nivara.Tests`

- `TensorsHelperTests`:
  - Extend `CheckMatMulShapes<T>` with a `bTransposed: true` variant against a
    new `ReferenceMatMulTransposedB`, including single-row shapes `(1,1,1)`,
    `(1,5,1)`, `(1,64,48)`, `(1,128,256)`.
  - **Bit-exact** parity (float/double, 0.0 tolerance): fast path vs. explicit
    `TensorPrimitives.Dot` row loop (same implementation — required because Dot
    may call arch-specific code; tolerance tests cover cross-arch safety).
  - Cross-path consistency: `aRows == 2` transposed run, row 0 must bit-equal
    the `aRows == 1` fast path.
  - **Alloc regression guard**: `float`, preallocated result,
    `GC.GetAllocatedBytesForCurrentThread` delta ≈ 0 across the fast-path call.
    This is a **regression lock, not a discriminator** — steady-state B/op is
    ~0 on both sides (pool reuse), so it only pins that the fast path never
    introduces a per-op rent (e.g. a future change going back to a per-call
    copy/rent would light it up).
- `WidenPrimitivesPhase1Tests`: transposed single-row `Half`/`BFloat16` parity
  (guards the `WidenPrimitives.Dot` dispatch).
- AutoDiff level: `MatMulTransposedB` op `aRows == 1` vs. row 0 of `aRows == 2`
  (tolerance).

### E. Docs

- `CHANGELOG.md`: P0-2 entry — **done** (`d6a32df`).
- `tests/Nivara.PerformanceTests/README.md`: Qwen rows in the Results table
  (Prev/Current/Δ%) + notes — **done** (`d6a32df`).
- `docs/QWEN-PERF.md`: becomes the improvement ledger — see §F (final step).

### F. Improvement ledger — `docs/QWEN-PERF.md` (final plan step)

`docs/QWEN-PERF.md` carries an **Improvement ledger** section appended after
the original review prose (which stays untouched as the pre-execution research
record): one entry per executed plan item with status, premise corrections,
and **measured before/after**. The P0-2 entry is written as the **last step of
the plan** — only after the kernel, parity/alloc tests, harness gate, full
test suite, and E2E synthetic run have all passed. Rationale: nothing is
marked done until the full execution evidence exists; the ledger is then the
single source of truth for what changed and by how much. The P0-2 entry also
carries the premise correction (original §2 "544 MB fresh alloc/token" claim
is refuted, superseded by the ledger's measured record). Future items
(batched prefill, fused GQA decode attention, BF16-on-the-fly, fused
decoder block, sampling, INT8/GGUF) get entries the same way as they land.

## Verification

1. `dotnet build Nivara.slnx` after each commit (ask before running).
2. **Baseline** (before the kernel change, harness committed):
   `dotnet run --project tests/Nivara.PerformanceTests -c Release -- --json qwen-fast-baseline.json --runs 3`
   → expect LM-head ≈ 5 ops/s / ~188 ms/op, B/op ≈ 0 (single-run sanity
   already observed). Commit the JSON beside `baseline-v140.json`. **Done**
   (`43346fb`): LM head 5 ops/s / 200 ms/op, B/op 53; FFN 61; attn 640;
   Linear fwd 362; LM head fwd 5.
3. **Post-fix**: `--compare qwen-fast-baseline.json --runs 3` → LM-head ops/s
   ≈ 20-28 (ns/op ~35-50 ms), all other rows unchanged or faster; B/op ≈ 0 both
   sides; keep the no-regression gate green (B/op ≤ 1.01×, ≥ 90% ops/s — drops
   and speedups pass; the gate only flags regressions). **Done** (`d6a32df`):
   LM head matmul 36 ops/s / ~28 ms (+620%, B/op 53→5), FFN 529, attn 6,658,
   Linear fwd 5,088, LM head fwd 42. Qwen rows all PASS; the gate's 3 FAIL rows
   (Linear forward 32x256, Frame Slice, Attn batched fwd+bwd) are pre-existing
   throughput/gen0 noise on rows this change does not touch (byte-identical
   B/op; Frame Slice is issue #354) — machine under load, so a full-table
   refresh was deferred.
4. **E2E**: `dotnet run --project samples/NivaraInference -c Release -- qwen --synthetic-weights benchmark`
   → KV-cached vs full-forward ms/token before/after. **Done** (2026-09-07):
   synthetic Qwen2.5-0.5B F32, 64-token prompt + 24-token decode, median of 3:
   KV-cached 335 ms/token (3.0 tok/s) vs full-forward 2,074 ms/token — **6.2×**.
   No pre-fix E2E number exists (synthetic mode was added with the harness);
   the harness rows carry the kernel-level before/after.
5. `dotnet test` — only after human confirmation (AGENTS.md: ask first).
   **Done** — full suite **3449 passed, 0 failed** (5 m 31 s, net11.0.0).

## Planned commits

1. `docs: plan Qwen-fast P0-2 + measurement harness in TODO.md` — done (`f544fbd`)
2. `perf: add Qwen decode single-row matmul scenarios to PerformanceTests` — done (`4cda229`)
3. `perf: add NivaraInference qwen --synthetic-weights decode benchmark mode` — done (`83cb0a6`)
4. `perf: record qwen-fast baseline (tests/Nivara.PerformanceTests/qwen-fast-baseline.json)` — done (`43346fb`)
5. `core: kill redundant weight copy in single-row transposed-B matmul` — done (`58b721e`, incl. unit tests)
6. `test: pin single-row transposed-B matmul parity + alloc-free fast path` — done (same commit `58b721e`)
7. `docs: record qwen-fast P0-2 results (perf README, CHANGELOG)` — done (`d6a32df`, incl. qwen-fast-postfix.json)
8. `docs: add P0-2 improvement-ledger entry to QWEN-PERF.md` — **remaining (final step, §F)**

## Blast radius

- `TensorsHelper.MultiplyCore<T>` — internal; reachable only via
  `GradKernels.MatMulTransposedB` / `MatMul` (reverse + forward AutoDiff ops),
  the null-aware `TensorsHelper.Multiply`, and tests/harness. All call sites
  allocate a fresh result array (no aliasing). Numerics must be bit-identical
  for float/double; `WidenPrimitives.Dot` dispatch preserves Half/BF16. No
  public API change. Downstream consumers of the single-row path: Qwen,
  SmolLM, MiniLM, DistilBERT samples + `Nivara.Samples` LLM pipelines.
- `tests/Nivara.PerformanceTests` — additive scenario rows only; gate behavior
  unchanged until baselines are re-recorded.
- `samples/NivaraInference` — additive `--synthetic-weights` flag only.
- Unit tests — additive.

## MS Learn verification (G1, folded in)

Performed during planning and updated 2026-09-07 after the single-run sanity
pass surfaced a wrong premise (see Problem). Grounding sources:

- **`ArrayPool<T>.Rent` / `Return` contract** — abstract doc:
  "returns an array at least the requested length", "may not be zero-initialized"
  (net-11.0 moniker included):
  https://learn.microsoft.com/dotnet/api/system.buffers.arraypool-1.rent?view=net-11.0
  The page's Source links pin the dotnet/runtime implementation
  (`TlsOverPerCoreLockedStacksArrayPool.cs`), which is the ground truth the
  premise correction rests on:
  - `NumBuckets = 27; // SelectBucketIndex(1024*1024*1024 + 1)` → the Shared
    pool buckets array lengths up to **~2³⁰ elements** (comment: "a max size of
    1B elements") — NOT ~1M as the original plan assumed.
  - Pooled `Rent` allocates via `GC.AllocateUninitializedArray` (no zeroing) and
    caches in TLS + per-core stacks; a 136M-float workspace is pooled (~1 GB
    bucket) and reused → **steady-state B/op ≈ 0** both before and after P0-2.
  - `Return(clearArray: true)` runs `Array.Clear` over the **whole bucket** on
    every return — the second removable cost (with the identity `CopyTo`) that
    makes the per-token LM-head traffic ≈ 2.7 GB today.
- **`TensorPrimitives.Dot<T>`** is the documented BLAS1 `dot` equivalent —
  generic over numeric types, computes with **no temporary storage**, SIMD /
  architecture-accelerated ("may call into the underlying C runtime or employ
  architecture-specific instructions; exact results may differ between
  OS/arch"). -> fast path must reuse it (bit-exact tests compare Dot-vs-Dot;
  cross-arch safety via tolerance tests, which the plan does).
  https://learn.microsoft.com/dotnet/api/system.numerics.tensors.tensorprimitives.dot?view=net-11.0
- **No `Tensor.MatrixMultiply` shipped.** The `System.Numerics.Tensors` API
  catalog at the repo's pinned version (11.0.0-preview.7.26381.103) exposes only
  *element-wise* `Tensor.Multiply`; dense matmul is still tracking
  dotnet/runtime#95863 (BLAS epic #93286). The code's "swap target" note stays
  accurate — no BCL matmul to switch to.
- **"Reach for existing higher-level APIs first… don't hand-roll what's already
  optimized and tested"** (Use SIMD and hardware intrinsics in .NET) — the fast
  path deliberately routes through the same `TensorPrimitives.Dot` /
  `WidenPrimitives.Dot` the row kernels use; no hand-rolled intrinsics.
- **"Benchmark to confirm the win… benchmark the input sizes your callers
  actually use"** (same guide) — the harness measures at real Qwen shapes;
  preallocated result buffers neutralize the documented randomized-alignment
  noise; `--runs 3` medians on an idle machine are the documented control.
- Single-row mat-vec is BLAS2 **GEMV**, memory-bound: the optimum reads each
  weight exactly once; the fast path achieves it (previously ~5× traffic due to
  copy + bucket clear).

## GitHub issues log

- (none at plan time; created during execution as deferred work is found)
- Related tracked work referenced: #384 (qkvBias), #387/#391 (BF16 SIMD),
  #388 (fused BF16→F32 read), #390 (GGUF backend).

## Open items

- [x] Confirm the revised P0-2 acceptance signal (ops/s, not B/op) — approved
      2026-09-07; baseline + fast path + compare executed.
- [x] Confirm `docs/QWEN-PERF.md` becomes the improvement ledger (final plan
      step, §F) — approved 2026-09-07.
- [x] Run full `dotnet test` suite — approved + executed: 3449 passed.
- [ ] **Write the P0-2 improvement-ledger entry in `docs/QWEN-PERF.md` (final
      step, §F), then G2 review the branch as a whole + against this plan,
      delete this file only when both gates clear, then offer push + PR
      (human-confirmed only).**
- [ ] Ask before any future `dotnet test` / long-running verification.