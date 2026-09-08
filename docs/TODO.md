# Plan: Qwen-fast P0 — Fused GQA decode-attention

Branch: `khurram/qwen-perf` (off `main`; P0-2 for this effort already landed and merged
via #398). Feeds the O(qwen-fast) effort; executes item **P0** "Fused GQA decode-attention
(no BlockCopy / GqaRepeatKV)" from `docs/QWEN-PERF.md` improvement ledger.
**Harness-first** — mirrors the P0-2 empirical workflow exactly: plan → harness scenarios →
baseline JSON → kernel → parity/alloc tests → results JSON → ledger entry → branch-complete
removal. README-methodology: idle machine, `--runs 3` medians.

## Problem

Every per-token decode step in `LlamaCausalAttention<T>.ForwardCached`
(`src/Nivara/AutoDiff/Nn/LlamaCausalAttention.cs:131`) re-runs the *whole* attention
post-projection path unnecessarily:

1. **`Buffer.BlockCopy` of the full cached prefix** (`LlamaCausalAttention.cs:162-163`) —
   copies all `newLen` positions of K and V out of the cache into fresh `T[]` arrays per
   layer per token (`O(newLen·kvWidth·2)` bytes), only to feed the generic kernels.
2. **`GqaRepeatKV` ×2 over the full prefix** (`:173-174`) — materializes `[newLen, 896]`
   K and V per layer (×7 data blowup for Qwen's 14→2 heads) every token.
3. **`MultiHeadAttention` re-packs the full KV** (`AttentionKernels.PackHeads` on the whole
   `[newLen, D]` K and V in `ReverseGradOperations.cs:552-554`) into head-major
   `[numHeads, newLen, headDim]` layout — another full-prefix re-layout each token.
4. **All-zeros open mask allocated every token** (`LlamaCausalAttention.cs:177-180`).

Net: per generated token, per layer, the code copies/moves `O(newLen·numHeads·headDim)` ≈
the whole context, multiple times over, purely to reuse the batched `MultiHeadAttention`
kernels on a single query row. At Qwen shapes this is ≈ **22+ MB/token of copies at
128-token context**, growing linearly with context.

The compute is *already* correct and matches full prefill (cache-vs-full parity test exits).
This is pure redundant memory traffic and allocation. A **single-query decode** kernel reads
cached K/V rows directly with zero copies, no GQA head expansion, no head repacking, no mask
allocation — computing scores **per KV head once** and reusing them across the 7-query-head
group.

### Why a dedicated kernel (and why it de-risks the batched-prefill P0)

A `qLen == 1` decode attention is a BLAS2-style problem: one query row vs. `kvLen` cached
positions. The generic `MultiHeadAttention` is built for `qLen == newLen` prefill (batch) and
pays `O(kvLen·numHeads·headDim)` in copies regardless of `qLen`. The fused kernel is
self-contained, additive, and unit-testable. It also establishes the **virtual GQA head
mapping** (`kvHead = qHead / repeat`, strided cache reads) that the subsequent batched-prefill
P0 will reuse when it captures K/V into the same row-major cache layout — so this P0 is the
correct foundation for that later P0.

## Proposed changes

### A. New fused decode-attention kernel — `src/Nivara/AutoDiff/Nn/LlamaCausalAttention.cs`

Add a fused single-query attention fast path to `ForwardCached` (private helper
`ForwardCachedFused` plus a span-level kernel, or a static `AttentionKernels<T>` helper). It
replaces steps 1–4 and:

- **Reads the cache in place, zero-copy.** `kCache`/`vCache` are already row-major
  `[newLen, numKeyValueHeads*headDim]` — pass `ReadOnlySpan<T>` slices over
  `[0, newLen·kvWidth]` directly. **No `BlockCopy`, no `new T[]`.**
- **Virtual GQA mapping, no repeat.** For each query head `qh` in `[0, numHeads)`,
  `kvHead = qh / (numHeads / numKeyValueHeads)`. Read that KV head's `headDim` columns
  strided from the cache row.
- **Per-KV-head scores, reused across the query group.** All query heads in the same group
  `[g·repeat, (g+1)·repeat)` share the *same* K/V head, hence the *same* score row
  `scores[kvLen]`. Computing scores once **per KV head** (not per query head) and reusing
  across the group is the core GQA win.
- **Folded scale + single-row softmax.** `scores[j] = scale · dot(q_head, k_row[kvHead])`,
  then one maximal softmax over `kvLen` (mask is fully open for decode). Reuse
  `GradKernels.SoftmaxSingle` / `AttentionKernels<T>.SoftmaxRows` so numerics match the
  existing softmax path.
- **Weighted sum over V, direct scatter.** `out[qh·headDim ..] = Σ_j softmax[j]·v_row[kvHead]`,
  written straight into a `[1, numHeads·headDim]` output buffer. No `ScatterHead` needed — the
  fused kernel owns the full output span.

**Numerics contract:** single-query fast path must produce the same last-row attention output
as the existing slow `ForwardCached` on an identical cache (the mandatory gate; see §D).

**Inference-only / grad-safe:** the fused kernel runs with `GradientUtils.Grad()` off (the
product default). It must **not** build graph nodes and must **not** add an `OpNode`. Guard the
fast path behind `!GradientUtils.IsGradEnabled` so the fused kernel is never frozen into a
backward graph — inside `Grad()`, fall back to the existing slow path to keep backward correct.

**Dispatch/call sites:** wire it into `LlamaCausalAttention.ForwardCached` so every model
smoke (`LlamaForCausalLM.ForwardCached` → `LlamaDecoderBlock.ForwardCached`) inherits it
automatically.

### B. Harness scenarios — `tests/Nivara.PerformanceTests/Program.cs`

Add `RunQwenDecodeAttentionScenarios()` registered from `RegisterScenarios()`, mirroring
`RunQwenDecodeMatMulScenarios()`. Qwen2.5-0.5B shapes (hidden 896, heads 14, kvHeads 2,
headDim 64). The **signal** is B/op: the slow path copies the full cache + materializes
expanded K/V + packs heads; the fused path should allocate ≈ 0 B/op with a preallocated output.

| Scenario | What it isolates |
|---|---|
| `Qwen decode-attn full [1x896 @ kvLen=128]` | `ForwardCached` at Qwen shapes, kvLen=128 — B/op pre-fix ≈ copies (~22 MB) → post-fix ≈ 0. **The P0 signal.** |
| `Qwen decode-attn full [1x896 @ kvLen=64]` | Same at shorter context (kvLen growth = increasing per-token cost) |
| `Qwen decode-attn full [1x896 @ kvLen=256]` | Longer-context growth gate |
| `Qwen fused decode-kernel raw [kvLen=128]` | The fused kernel alone (op-level), preallocated output — isolates kernel B/op ≈ 0 |

Use preallocated result buffers (per the SIMD guidance + P0-2 alignment-noise note),
`--runs 3` medians. Record and commit `qwen-gqa-baseline.json` before the kernel change.

### C. E2E synthetic weights — `samples/NivaraInference`

P0-2 already shipped `--synthetic-weights` + `RunDecodeBenchmark` for qwen. The fused
decode-attention change is inside `ForwardCached`, so the existing synthetic decode benchmark
(KV-cached vs full-forward ms/token) captures the E2E effect directly — **no new harness mode
needed**. Record before/after KV-cached `ms/token`; compare against P0-2-postfix
(335 ms/token @ 3.0 tok/s) as the decode-attention delta baseline.

### D. Unit tests — `tests/Nivara.Tests`

- **Fused-vs-slow cached parity (the gate):** seed a cache with `kvLen` positions via the
  slow `ForwardCached`, then for the next token call both the slow path and the fused path on
  identical input + cache, assert outputs match to tolerance (start 1e-5 like the existing
  parity test, tighten if bit-exact holds).
- **Cache-vs-full parity still green:** existing `ForwardCached_QkvBiasTrue_MatchesFullForward`
  must pass with the fused path enabled (proves full-forward equivalence is preserved).
- **GQA grouping correctness:** for models with different `numHeads`/`numKvHeads` ratios
  (14/2, 8/4, 12/4), assert the fused path matches the slow path — pins the virtual KV-head
  mapping `kvHead = qh / repeat` and score reuse across the group.
- **Alloc regression guard:** `float`, preallocated output,
  `GC.GetAllocatedBytesForCurrentThread` delta ≈ 0 across the fused decode call (locks "no
  per-token cache copy" against regressions).
- **Grad-inference guard:** outside `GradientUtils.Grad()`, fused path builds no graph node;
  inside `Grad()`, path falls back to slow and backward still flows (finite gradients).

### E. Docs

- `CHANGELOG.md`: P0 (fused GQA decode-attention) entry with before/after.
- `tests/Nivara.PerformanceTests/README.md`: Results table (Prev/Current/Δ%, B/op, gen0/op)
  + decode-attention scenario rows.
- `docs/QWEN-PERF.md`: append the **P0 (fused GQA decode-attention)** improvement-ledger entry
  (mirror P0-2's format: status, branch/commit, item, numerics, measured before/after table,
  premise corrections, gate note). **Confirm with human before editing the review prose** — P0-2
  appended only, did not touch §2B/§3 prose.

## Verification (harness-first, mirrors P0-2)

1. `dotnet build Nivara.slnx` after each commit (ask before running).
2. **Baseline** (harness committed, kernel unchanged):
   `dotnet run --project tests/Nivara.PerformanceTests -c Release -- --json qwen-gqa-baseline.json --runs 3`
   → expect decode-attn rows show large B/op (full-cache copy + GqaRepeatKV + PackHeads).
   Commit the JSON beside `qwen-fast-baseline.json`.
3. **Post-fix**: `--compare qwen-gqa-baseline.json --runs 3` → fused rows B/op ≈ 0, ops/s up;
   keep the no-regression gate green on untouched rows (watch for documented #354 Frame Slice
   flake + current-machine-load noise, as P0-2 noted).
4. **E2E**: `dotnet run --project samples/NivaraInference -c Release -- qwen --synthetic-weights benchmark`
   → KV-cached ms/token before (P0-2-postfix 335 ms/token) vs after.
5. `dotnet test` — only after human confirmation (AGENTS.md: ask first).

## Planned commits (mirrors P0-2 sequence)

1. `docs: plan Qwen-fast P0 GQA decode-attention + measurement harness in TODO.md`
2. `perf: add Qwen decode-attention scenarios to PerformanceTests`
3. `perf: record qwen-fast GQA decode-attention baseline (qwen-gqa-baseline.json)`
4. `perf: fused GQA single-query decode-attention (no cache copy / GqaRepeatKV / PackHeads)`
5. `test: pin fused GQA decode-attention parity + alloc-free path`
6. `docs: record qwen-fast P0 GQA decode-attention results (perf README, CHANGELOG)`
7. `docs: add P0 GQA decode-attention improvement-ledger entry to QWEN-PERF.md`
8. `docs: remove executed plan (Qwen-fast P0 complete)` → PR

## Blast radius

- `src/Nivara/AutoDiff/Nn/LlamaCausalAttention.cs` — one method (`ForwardCached`) whose body
  gains a fused fast path; the slow path is retained behind the grad/invalid guard. Internal
  inference surface. No public API signature change.
- `tests/Nivara.PerformanceTests` — additive scenario rows; gate unchanged until baselines
  re-recorded.
- `tests/Nivara.Tests` — additive parity/alloc/grad tests in `LlamaCausalAttentionTests` (+
  `GqaKvRepeatTests` if virtual-mapping parity belongs there).
- `samples/NivaraInference` — no changes (P0-2 harness already covers decode).
- Downstream consumers of `ForwardCached`: Qwen, SmolLM, MiniLM, DistilBERT samples + all
  `Nivara.Samples` LLM pipelines — all exercise the fused path, so the cache-vs-full parity
  suite is the safety net.

## MS Learn verification (G1, folded in)

Performed during planning; no decisions surfaced, no plan changes needed:

- **Scaled dot-product attention decomposition** (DirectML MHA): `GEMM(Q,Kᵀ)·Scale → [mask]
  → Softmax → GEMM(P,V)`; `Scale = 1/sqrt(headSize)`; mask filter = large negative. The fused
  decode kernel implements exactly this with `qLen == 1`, no mask (fully open), GQA virtual
  head mapping.
  https://learn.microsoft.com/windows/ai/directml/api/ns-directml-dml_multihead_attention_operator_desc
- **`TensorPrimitives` generic + SIMD**: dot/softmax are architecture-accelerated; reuse the
  existing `TensorPrimitives`/`GradKernels.SoftmaxSingle` paths (as P0-2 did with `Dot`) so
  numerics stay consistent and we don't hand-roll intrinsics
  (learn.microsoft.com/dotnet/standard/simd: "reach for existing higher-level APIs first").
- **"Benchmark the input sizes your callers actually use"** (same guide): harness at Qwen
  decode shapes; B/op is the allocation-driven signal; preallocated outputs neutralize
  alignment noise; `--runs 3` medians on idle machine = documented control.

## GitHub issues log

- [x] #399 — DecodeAttention per-dot score loop may regress at large kvLen (created while
      executing P0 fused GQA decode-attention — follow-up on the per-call
      `TensorPrimitives.Dot` overhead as context grows beyond the measured 64–256).
- Related tracked work referenced: #384 (qkvBias, landed), #387/#391 (BF16 SIMD — orthogonal
  to this memory-traffic P0; composes unchanged), #388 (fused BF16→F32 read), #390 (GGUF
  backend). The batched-prefill P0 remains open in the ledger as the next item after this.

## Open items

- [x] Confirm whether `docs/QWEN-PERF.md` review prose gets edited vs. only a ledger entry
      appended — resolved at execution: **ledger entry only** (user-confirmed, mirrors P0-2).
- [x] Ask before running `dotnet test` / `--runs 3` harness — harness approvals given;
      full `dotnet test` gated separately.
- [x] Lock the fused-vs-slow parity tolerance (bit-exact vs 1e-5) — **1e-5 held** across GQA
      ratios 14/2, 8/4, 12/4, 8/8 (12/12 fixture pass); matches the existing cache-vs-full
      parity convention.
- [x] E2E apples-to-apples (main vs branch) resolved at execution: harness is authoritative;
      E2E A/B documented as inconclusive due to machine-load noise (identical main binaries
      swung 22%; change's max effect ~1% of E2E).
