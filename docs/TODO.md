# #404 — P1: Per-token fused decoder-block kernel (kill ~200 allocs/token during decode)

Branch: `khurram/qwen-perf` · Issue: https://github.com/khurram-uworx/Nivara/issues/404
Ledger row: `docs/QWEN-PERF.md` (P1 #404 entry + future-entries table) · Gate tooling: `docs/QWEN-PERF.md` "per-item gate workflow".

## Problem

Each cached-decode token rebuilds the whole decoder block as a chain of boxed tensor ops
(`LlamaDecoderBlock<T>.ForwardCached`): per layer per token, InputNorm → QKV + biases → RoPE →
KV write → `DecodeAttention` → OProj → residual add → PostNorm → SiLU-FFN (gate/up/down) →
residual add. Every intermediate is a freshly allocated `ReverseGradTensor<T>` + `NivaraColumn<T>`
(~26 KB/op measured floor on the decode-attn row). With 24 layers × ~7–8 boxes per layer, decode
allocates on the order of ~4–6 MB/token. Issue #404 targets exactly this: fuse the per-token
decoder block into a span-in/span-out kernel with zero per-token heap allocations.

## Proposed changes

All within the issue's stated scope (`LlamaDecoderBlock.cs`, the fused-kernel surface,
`tests/Nivara.PerformanceTests/`), plus the model-level wiring in `samples/Nivara.Samples`
required for the E2E/chat/benchmark callers to benefit (zero caller changes).

### Design (locked)

1. **Model-level orchestration, span-in/span-out blocks.** `LlamaForCausalLM<T>.ForwardCached` /
   `ForwardPrefill` gain a fused branch gated on `!GradientUtils.IsGradEnabled &&
   LlamaFusedKernels.DecoderBlockFused`. The model gathers the embed row directly from
   `Embed.Weight!.Tensor.Data` into ping-pong `T[]` scratch, runs 24 `ForwardCachedFused` /
   `ForwardPrefillFused` layers, then the existing `finalNorm.Forward` + tied-head
   `MatMulTransposedB` (samples cannot reach internal `RMSNormKernel`; the ~608 KB `[1, vocab]`
   logits box stays the documented floor). Per-op loops stay byte-identical under Grad /
   toggle-off.
2. **New public block methods** (issue scope, span-based):
   - `ForwardCachedFused(ReadOnlySpan<T> input, Span<T> output, int positionOffset, T[] kCache, T[] vCache, int cacheLen)`
   - `ForwardPrefillFused(ReadOnlySpan<T> input, Span<T> output, int positionOffset, T[] kCache, T[] vCache)`
   Both guard `GradientUtils.IsGradEnabled` (inference-only → clear `InvalidOperationException`).
   Decode uses per-block tiny `T[]` scratch fields (hot, ~20 KB/layer). Prefill (L > 1) rents one
   L-scaled workspace from `ArrayPool<T>.Shared` and returns it — matching the `BatchedAttention`
   precedent, zero gen0.
3. **Reuse existing kernels only:** `GradKernels.MatMulTransposedB` (⇒ P0-2 GEMV fast path at
   `aRows == 1 && bTransposed`, `WidenPrimitives.Dot` bf16 widening for free),
   `AttentionKernels<T>.DecodeAttention` / `.BatchedAttention`,
   `RMSNormKernel<T>.PerRowRMSNormForwardKernel` (+ gamma multiply — the `RMSNorm.ForwardInference`
   pattern), `GradKernels.Silu` (NOT in-place safe — distinct `gateOut` scratch),
   `GradKernels.RotaryForward` (in-place safe — reads both halves before writing).
4. **New internal surface (src/Nivara only):**
   - `RotaryEmbedding<T>.GetPositionTables(int start, int count, out ReadOnlySpan<T> cos, out ReadOnlySpan<T> sin)`
     (internal; ensures lazy cache exactly like `Forward`; never precomputes full maxPos).
   - `LlamaCausalAttention<T>.Rotary` internal property (the `rotary` field is private).
5. **Public toggle** (no `InternalsVisibleTo` in src/Nivara — follow the `NivaraPrimitives` precedent):
   `public static class LlamaFusedKernels { public static bool DecoderBlockFused { get; set; } = true; }`
   Default true (inference-default). Tests reset in `[SetUp]`/`[TearDown]`. Model-level branch AND
   tests A/B fused vs per-op with it.
6. **Cache layout unchanged** (row-major per-KV-head, post-RoPE, pre-repeat) so decode and prefill
   compose; layer-0 K/V stays bit-equal across fused/per-op paths; deeper layers within the
   existing 1e-5 cache-vs-full convention.
7. **bf16 free** — kernels are type-generic; `LlamaCausalLMBf16ParityTests` becomes an automatic
   regression guard on the fused routing.

### File-by-file

| File | Change |
|---|---|
| `src/Nivara/AutoDiff/Nn/RotaryEmbedding.cs` | internal `GetPositionTables(int, int, out ReadOnlySpan<T>, out ReadOnlySpan<T>)` |
| `src/Nivara/AutoDiff/Nn/LlamaCausalAttention.cs` | internal `Rotary` accessor |
| `src/Nivara/AutoDiff/Nn/LlamaFusedKernels.cs` (new) | public toggle (NivaraPrimitives pattern) |
| `src/Nivara/AutoDiff/Nn/LlamaDecoderBlock.cs` | `ForwardCachedFused` + `ForwardPrefillFused` + scratch fields (plain, NOT registered → invisible to StateDict) |
| `samples/Nivara.Samples/LlamaForCausalLM.cs` | fused branch in `ForwardCached`/`ForwardPrefill` (embed-row gather from `Embed.Weight!.Tensor.Data`; ping-pong; existing finalNorm+head unchanged) |
| `tests/Nivara.PerformanceTests/Program.cs` | harness-first: `Qwen decode block [1x896 @ kvLen=64]` (block row; body calls per-op `ForwardCached` at baseline, swaps to `ForwardCachedFused` after) + `Qwen decode fwd [1 step]` (model row, pre-seeded 64-token cache; same body both branches) |
| `tests/Nivara.Tests/AutoDiff/LlamaDecoderBlockTests.cs` (new) | block-level parity + alloc guard |
| `tests/Nivara.Tests/AutoDiff/LlamaForCausalLMPrefillTests.cs` | model-level fused-vs-per-op parity + toggle A/B + scratch-aliasing guard |
| `docs/QWEN-PERF.md` | ledger entry (P1 row l.609) |
| `docs/TODO.md` | this plan (deleted at G2) |

### Fused decode-step sketch (block level)

```csharp
public void ForwardCachedFused(ReadOnlySpan<T> input, Span<T> output, int positionOffset,
    T[] kCache, T[] vCache, int cacheLen)
{
    if (GradientUtils.IsGradEnabled)
        throw new InvalidOperationException("ForwardCachedFused is inference-only; do not call inside GradientUtils.Grad().");
    // scratch fields: norm, q, k, v, attn, o, h, ffn, gateRaw, gateOut, up, gated, mlp, all T[]
    // 1. InputNorm: PerRowRMSNormForwardKernel(input→norm, rows:1) ; norm *= gamma
    // 2. QKV GEMV: MatMulTransposedB(norm, QW, q, 1,h,qw); q += qBias (Qwen qkvBias=true)
    //              MatMulTransposedB(norm, KW, k, 1,h,kw); k += kBias
    //              MatMulTransposedB(norm, VW, v, 1,h,kw); v += vBias
    // 3. RoPE in place: rotary.GetPositionTables(positionOffset, 1, out cos, out sin);
    //                   per head: RotaryForward(q.Slice(h*hd,hd), cos, sin, same)
    //                   same for k
    // 4. KV write: k.CopyTo(kCache.AsSpan(cacheLen*kvWidth, kvWidth)); same v
    // 5. DecodeAttention(q, kCache[0..newLen*kvWidth], vCache[...], attn, newLen, heads, kvHeads, hd, scale)
    // 6. OProj GEMV: MatMulTransposedB(attn, OW, o, 1,h,h)  → o
    // 7. Residual:   TensorPrimitives.Add(input, o, h)
    // 8. PostNorm:   PerRowRMSNormForwardKernel(h→ffn, rows:1); ffn *= gamma
    // 9. FFN:        gateRaw = ffn @ GW^T; GradKernels.Silu(gateRaw, gateOut)
    //                up = ffn @ UW^T; TensorPrimitives.Multiply(gateOut, up, gated)
    //                mlp = gated @ DW^T
    // 10. Residual:  TensorPrimitives.Add(h, mlp, output)
}
```

## Verification

- **Targeted tests** (NUnit, whole suite by human decision per P0 precedent):
  - `LlamaDecoderBlockTests`: `ForwardCachedFused_MatchesPerOpForwardCached_AcrossGqaRatios`
    (4/2, 8/2; L ≥ 36 for RoPE), `ForwardCachedFused_MatchesPerOp_WithQkvBias`,
    `ForwardCachedFused_SteadyState_AllocatesNothing` (< ~2 KB after warmup, clone of
    `DecodeAttention_SteadyState_AllocatesNothing`), `ForwardPrefillFused_MatchesPerOp_AcrossGqaRatios`,
    `ForwardCachedFused_TwoConsecutiveSteps_NoScratchAliasing`.
  - `LlamaForCausalLMPrefillTests` extension: `ForwardCached_ToggleA_B_FusedVsPerOp_Match`,
    `ForwardPrefill_ToggleA_B_FusedVsPerOp_Match` (1e-5; layer-0 K/V bit-equal where the
    convention holds), graph guards (fused outside Grad builds no node; inside Grad per-op path
    unchanged).
  - Existing suite stays green: `LlamaForCausalLMPrefillTests`, `LlamaCausalAttentionTests`,
    `LlamaCausalLMBf16ParityTests`, `QwenInstructParityTests` (skips when fixtures absent, #406).
- **Harness gates** (idle machine, same machine, human-confirmed runs):
  1. Baseline BEFORE kernels: `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --json qwen-decode-block-baseline.json --only Qwen --runs 3`
  2. Gate AFTER: `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --compare qwen-decode-block-baseline.json --only Qwen --runs 3`
  - Gate defaults: minOps 90%, alloc ≤ baseline ×1.01, gen0 ≤ baseline +0.05. All pre-existing
    Qwen rows must PASS; new decode rows are A/B-gated by the same baseline (identical row names).
  - Expected: decode block B/op ~tens of KB → ~0–1 KB; decode fwd → ~608 KB floor (logits box);
    seed rows drain toward one-forward level.
- **E2E** (median of 3, realistic expectation — fused block removes GC churn, not weight traffic;
  wall-clock gain may be modest, possibly inside the instrument's ±22% noise band per P0-3):
  `dotnet run -c Release --project samples/NivaraInference -- qwen benchmark --synthetic-weights`
- **Build** before each commit: `dotnet build Nivara.slnx`.

## Planned commits (one logical change each, verify + human-confirm long-running steps)

1. ✅ `docs: plan #404 fused decoder kernel in TODO.md` — `112d8ed` (committed at planning).
2. ✅ `perf: add Qwen decode block/fwd harness rows (harness-first)` — `3f42c4d`
   (rows only); harness-first baseline captured at commit `5076989`
   (`qwen-decode-block-baseline.json`).
3. ✅ `feat: fused decoder-block kernel surface (RoPE accessor, toggle, span block methods)` —
   `614f2cd` (src/Nivara + harness row bodies swap to fused; same row names ⇒ A/B gate).
4. ✅ `feat: wire fused decode path into LlamaForCausalLM` — included in `614f2cd`
   (samples model-level routing, `fusedH0`/`fusedH1` ping-pong + toggle).
5. ✅ `test: fused-block parity, toggle A/B, alloc guard` — `775a621` (new tests) +
   `6b64eec` (original 7-test fixture restored byte-for-byte). Targeted run **42/42**.
6. ✅ `perf: gate after fused decode block` — `--compare` run **16/16 PASS**; results in
   Execution log + ledger.
7. ✅ `e2e: qwen synthetic benchmark median-of-3` — Gate 3 green (Execution log).
8. ✅ `docs: ledger entry #404 fused decoder block` — QWEN-PERF.md P1 entry +
   future-entries row, `docs/CHANGELOG.md` "Added" note (public surface warrants it),
   this file's Execution log.
9. ⏳ G2 reviews (branch as a whole + against this file) → `docs: remove TODO.md —
   #404 plan executed`, then offer push + PR (human-confirmed).

## GitHub issues log

- No new issues were created during planning. Follow-ups created at discovery
  (during execution, per the "create at discovery, record here" rule):
  - [x] [#413](https://github.com/khurram-uworx/Nivara/issues/413) — fuse the
        final RMSNorm + tied LM head into the fused decode step (~608 KB/token
        logits box + residual ~11 KB boxing) — residual floor of #404.
  - [x] [#414](https://github.com/khurram-uworx/Nivara/issues/414) — fuse the
        final Norm + head for prefill too (the `[L, vocab]`-last-row head
        currently boxes the last single row).
  - [#399](https://github.com/khurram-uworx/Nivara/issues/399) (open, orthogonal): DecodeAttention
        per-dot score loop at large kvLen — the fused block calls the kernel and composes with any
        later fix.

## Execution log

- **Harness-first (before):** `--json qwen-decode-block-baseline.json --only Qwen
  --runs 3` captured at commit `5076989` (human-confirmed). Decode block
  131,985 B/op; decode fwd 3,620,919 B/op; seed rows 23.9M / 46.8M / 184.3M /
  735.2M B/op.
- **Fused surface + routing:** commits `614f2cd` (src/Nivara surface + harness row
  swap), `775a621`/`6b64eec` (tests: new fused parity/A/B/alloc guards + original
  7-test fixture restored byte-for-byte), `77b3df8` (residual fix, see below).
- **Targeted tests (human-confirmed run):** `--filter` over
  `LlamaDecoderBlockTests|LlamaForCausalLMPrefillTests|LlamaCausalLMBf16ParityTests|
  LlamaCausalAttentionTests|InferenceGraphTests` — **42/42 pass**. First run failed
  10/12 parity tests: both fused paths applied PostNorm in place over the
  attention-residual buffer (`ffnIn + mlp`, not `residual + mlp`). Fixed in
  `77b3df8` by preserving the residual in its own per-layer scratch / pool-rented
  prefill buffer. After the fix, fused decode/prefill outputs are bit-identical to
  the per-op chain.
- **Gate 2 (perf `--compare`, human-confirmed):** `--compare
  qwen-decode-block-baseline.json --only Qwen --runs 3` — **16/16 Qwen rows PASS**
  (one-sided no-regression): decode block → 1 B/op, decode fwd → 619,631 B/op,
  seed rows −96…−99%, gen0 → 0 on touched rows.
- **Gate 3 (E2E, human-confirmed):** `qwen benchmark --synthetic-weights`
  (benchmark's built-in median-of-3, 64-token prompt + 24-token decode): decode
  **116.5 ms/token (6.2 tok/s)**, prefill 941 ms, cache total 3,737 ms; full
  re-forward 2,074 ms/token → **13.4×**. Same-machine pre-#404 decode (P0-3 A/B)
  was ~524–571 ms/token → ≈4.5–4.9× decode wall-clock, outside the ±22% noise
  band (the plan's "may be modest" expectation proved conservative).
- **Docs:** `docs/QWEN-PERF.md` P1 #404 ledger entry + future-entries row DONE;
  `docs/CHANGELOG.md` "Added" note; this file updated for the records.
- **Remaining:** step 9 G2 review (below), human-confirmed push + PR.

## Blast radius

- `src/Nivara/AutoDiff/Nn/LlamaDecoderBlock.cs` — adds two public methods + scratch fields; the
  existing `Forward`/`ForwardCached`/`ForwardPrefill` are untouched (all existing callers keep
  their exact behavior).
- `src/Nivara/AutoDiff/Nn/RotaryEmbedding.cs`, `LlamaCausalAttention.cs` — additive internal
  members only (no public API change).
- `src/Nivara/AutoDiff/Nn/LlamaFusedKernels.cs` — new public opt-out toggle, default `true`
  (inference-default; mirrors `NivaraPrimitives.UseWidenSimd`); no behavior change unless a user
  opts out.
- `samples/Nivara.Samples/LlamaForCausalLM.cs` — `ForwardCached`/`ForwardPrefill` route through
  the fused path only outside Grad with the toggle on; Grad / toggle-off paths are byte-identical.
  Chat clients, benchmark, all tests benefit automatically; no caller changes.
- `tests/Nivara.PerformanceTests/Program.cs` — two new Qwen rows (names identical across
  baseline/compare); no existing row behavior changes.
- Defense-in-depth: `ComputationGraph.AddNode` Debug.Assert (graph never built outside Grad);
  existing inference-graph tests already cover the op-level guarantee; the new fused methods
  extend it to the block level.