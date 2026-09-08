# Plan: Qwen-fast P0 — Batched prompt prefill

Branch: `khurram/qwen-perf` (off `main`; P0-2 merged via #398, fused GQA decode-attention
merged via #400 on the same branch name, now re-created fresh). Executes the last open **P0**
"Batched prompt prefill" row from the `docs/QWEN-PERF.md` improvement ledger.
**Harness-first** — mirrors the executed P0-2 and P0 empirical workflows exactly: plan →
harness scenarios → baseline JSON → implementation → parity/alloc tests → results JSON → E2E
→ docs + ledger → branch-complete removal of this plan. README-methodology: idle machine,
`--runs 3` medians, `--compare` no-regression gate.

## Problem

Prompt prefill (`SeedCache`) processes the prompt **token-by-token**:

- `QwenChatClient.SeedCache` (`samples/NivaraChat/Qwen/QwenChatClient.cs:159-165`)
- `Qwen.Generate` (`samples/NivaraInference/Qwen.cs:308-309`)
- `SmolLMChatClient.SeedCache` (`samples/NivaraChat/SmolLM/SmolLMChatClient.cs:179`)

each run `model.ForwardCached(ids[p], p, cache)` — **L full-model walks** (embed + 24 blocks
+ final norm + the 151,936-row LM head). Each walk re-reads **every weight (~2 GB F32 at
Qwen2.5-0.5B)** from DRAM. A ~60-token tool prompt therefore moves **≥120 GB of DRAM
traffic before the first generated token** (docs/QWEN-PERF.md §2A) — matched by the observed
transcript (`docs/QWEN.md`: 3 turns in 342 s). Per-token cost is ~330–500 ms on this machine
(the decode E2E at ~524 ms/token including the same full-model walk).

Meanwhile the **batched full-sequence forward already exists and is PyTorch-pinned**:
`LlamaForCausalLM.Forward(int[])` embeds `[L, hidden]`, runs every block's `Forward`
(`LlamaCausalAttention.Forward` over `[L, D]` with a causal `CreateCausalMask(L,L)` and
`GqaRepeatKV`), final norm, LM head → `[L, vocab]`. It just (a) doesn't write the per-layer
K/V into a cache and (b) computes all L rows of logits when only the last row is needed.

**The win:** one batched forward reads the weights **once** (matmuls with `aRows = L` reuse
the weight rows across all L positions), turning prefill from `O(L)·2 GB` into `2 GB`
traffic — the 10–50× column in §4, and the observed minutes → low seconds.

## Proposed changes

### A. Batched prefill API (captures K/V into the same cache the fused decode reads)

The GQA decode kernel from the previous P0 reads the cache as row-major
`[kvLen, numKeyValueHeads * headDim]` — the prefill must write exactly that layout so decode
after prefill is unchanged (no decode changes in this P0).

1. **`LlamaCausalAttention<T>`** (`src/Nivara/AutoDiff/Nn/LlamaCausalAttention.cs`):
   - Refactor the body of `Forward(ReverseGradTensor<T>)` into a private core
     `ForwardCore(input, positionOffset, kCache, vCache)` that runs QKV projections →
     RoPE at `positionOffset..positionOffset+L-1` → **capture the per-KV-head (pre-repeat)
     RoPE'd K and V into cache slices `[offset·kvWidth, (offset+L)·kvWidth)`** →
     `GqaRepeatKV` → causal `MultiHeadAttention` → `OProj.Forward`. Public `Forward`
     keeps its exact current numerics and shape (core with no cache); only code shape
     changes.
   - New public **`ForwardPrefill(ReverseGradTensor<T> input, int positionOffset, T[] kCache, T[] vCache)`**
     → `[L, hidden]`, validating cache capacity for `(offset + L) * kvWidth` and reusing
     `rotary.Forward(x, positionOffset)` (batched-with-offset exists). The captured rows
     land at absolute cache positions `[offset, offset+L)` — always `0` for full-prompt
     prefill but the offset keeps the API consistent with `ForwardCached`.
   - Capture point note: K/V must be copied **after** RoPE and **before** `GqaRepeatKV`
     (they are `[L, kvWidth]` there — exactly the cache's per-KV-head rows).

2. **`LlamaDecoderBlock<T>`** (`src/Nivara/AutoDiff/Nn/LlamaDecoderBlock.cs`):
   - New **`ForwardPrefill(input, positionOffset, kCache, vCache)`** → `[L, hidden]` — the
     `Forward` structure (pre-norm attn + residual + gated-SiLU FFN + residual) with the
     attention routed through `Attention.ForwardPrefill`.

3. **`LlamaForCausalLM<T>`** (`samples/Nivara.Samples/LlamaForCausalLM.cs`):
   - New **`ForwardPrefill(int[] inputIds, LlamaKVCache<T> cache)`** → **`[1, vocab]`
     last-row logits** (mirrors `ForwardCached`'s return shape so `Generate`/`Select`
     reuse unchanged):
     `cache.Ensure(inputIds.Length)` → `Embed.Forward(int[])` → per-layer
     `block.ForwardPrefill(h, 0, cache.keys[i], cache.values[i])` → `finalNorm.Forward(h)`
     → slice the **last hidden row** and run the tied LM head as a **single-row**
     `MatMulTransposedB` (`[1, hidden] @ [vocab, hidden]ᵀ`) → `[1, vocab]`.
     The last row of the full-L head matmul is bit-identical to the single-row one (P0-2:
     single-row == row-of-batch, same kernel dispatch), and it avoids allocating the
     `L × vocab` logits block (608 KB × L).

4. **Sample call-site swaps** (the only behavior change customers see):
   - `QwenChatClient.SeedCache` → `return model.ForwardPrefill(ids.ToArray(), cache);`
   - `Qwen.Generate` KV branch → `logits = model.ForwardPrefill(promptIds.ToArray(), cache);`
   - `SmolLMChatClient.SeedCache` → same one-liner (shared `LlamaForCausalLM<T>`).
   - Non-cached paths (`model.Forward`) unchanged.

**No new kernels.** The batched attention reuses `MultiHeadAttention` + `GqaRepeatKV`
exactly as `Forward` does today (for L ≤ 256 the `[L, kvWidth] → [L, numHeads·headDim]`
repeat is negligible vs the 2 GB weight traffic). The inference graph guard behavior is
inherited from `Forward`/`ForwardCached` (no `GradientUtils` change). A GQA-aware *batched*
attention (skip the repeat) is a documented follow-up, not P0 scope.

### B. Performance harness (authoritative)

New `RunQwenPrefillScenarios()` in `tests/Nivara.PerformanceTests/Program.cs` (already
references `samples/Nivara.Samples`). Real Qwen2.5-0.5B shapes via a synthetic
`LlamaForCausalLM<float>` (vocab 151,936; hidden 896; 24 layers; 14/2 heads; headDim 64;
intermediate 4864 — ~2 GB resident, one construction per row, outside timing).

Design — one named row whose **internals change** between branches (the P0/P0-2 pattern:
same row name, kernel/prefill implementation swapped):

```csharp
static void RunQwenPrefillScenarios()
{
    // "Qwen prefill seed [L]": on main = the token-by-token SeedCache loop (the current
    // prefill); on this branch = model.ForwardPrefill (one batched forward + K/V capture).
    Run("Qwen prefill seed [8 tok]",   1, 6, () => CreateQwenPrefillScenario(8));
    Run("Qwen prefill seed [16 tok]",  1, 6, () => CreateQwenPrefillScenario(16));
    Run("Qwen prefill seed [64 tok]",  1, 6, () => CreateQwenPrefillScenario(64));
    Run("Qwen prefill seed [256 tok]", 1, 6, () => CreateQwenPrefillScenario(256));
    // Unchanged no-regression siblings: model.Forward(ids) exists on both branches.
    Run("Qwen full fwd [64 tok]", 1, 6, () => CreateQwenFullForwardScenario(64));
    Run("Qwen full fwd [256 tok]", 1, 6, () => CreateQwenFullForwardScenario(256));
}
static Action CreateQwenPrefillScenario(int L)
{
    var model = new LlamaForCausalLM<float>(151936, 896, 24, 14, 2, 4864, ...);
    var cache = new LlamaKVCache<float>(24, 2 * 64);
    var ids = ... L deterministic token ids (seeded) ...
    return () => PrefillInto(model, ids, cache);   // loop on main; ForwardPrefill on branch
}
```

- `PrefillInto(model, ids, cache)` starts as the **loop** (baseline must run on main before
  the implementation), then the implementation commit swaps its body to
  `model.ForwardPrefill(ids, cache)` — same row names in both JSONs, clean `--compare`.
- Loop rows limited to L = 8/16 on the baseline side: the loop costs L × ~0.4 s/op, so
  L = 64/256 baseline rows would take minutes per timed op. **Both** JSONs carry all five
  row names; rows 3-4 are loop-based (slow but sampled) in the baseline and batched in the
  postfix — that before/after on "Qwen prefill seed [L]" IS the P0 delta. L=16 ≈ 6 s/op and
  the harness tolerates slow ops (LM-head rows ran at 5 ops/s).
- Expected signal: `bytesPerOp` on the seed rows drops from **L × ~2.5 GB** to **~2.5 GB**
  (flat vs L, like the decode rows' B/op collapse), and `ops/s` rises roughly L×.
- Idle machine, `--runs 3` child-process medians, `--compare` gate; unrelated-row noise
  under load is the documented P0-2/P0 phenomenon (byte-identical B/op).

### C. E2E (real benchmark, conclusive for prefill)

The P0-2/P0 E2E instrument (~±22% noise on identical binaries) is too noisy for a ~1%
change, but **prefill is a ~10–20× change** — far above the noise band, so the E2E is
conclusive here (unlike the decode P0).

- `samples/NivaraInference/Qwen.cs`: split `TimeGeneration`/`Generate` timing into
  **seed (prefill) vs decode** phases (two `Stopwatch`s — pure benchmark output change).
  Report `prefill ms` and `decode ms/token` for both cache and full paths.
- A/B on the same machine, main vs branch, `--synthetic-weights` 64-token prompt +
  24-token decode. Expected: prefill ms collapses ~L× (64 × ~0.4 s → ~0.2 s); decode
  ms/token flat (no decode change — already fused).
- Controlled harness rows (§B) remain authoritative; the E2E split adds the real-shapes
  L=64 headline the harness cannot afford.

### D. Unit tests (`tests/Nivara.Tests/AutoDiff/`)

Mirror the TinyModel fixture in `LlamaCausalKVCacheTests` (vocab 128, hidden 32, 2 layers,
4/2 heads, maxPos 32). New fixture `LlamaForCausalLMPrefillTests.cs`:

1. `ForwardPrefill_SeedMatchesTokenByTokenSeed_KvCacheAndLogits` — same prompt seeded
   both ways; assert per-layer cache K/V rows **bit-equal** (expected: batched row i is the
   same per-row dot + RoPE as the single-token walk; P0-2 locked single-row == row-of-batch)
   and last-token logits within 1e-5 (existing cache-vs-full convention; tighten if
   bit-exact holds).
2. `ForwardPrefill_LastRowLogits_MatchFullForwardLastRow` — `[1, vocab]` result equals
   row L-1 of `model.Forward(ids)` within 1e-5 (single-row LM-head parity per P0-2).
3. `ForwardPrefill_SeedThenCachedDecode_MatchesFullForward` — prefill cache + N
   `ForwardCached` steps == `model.Forward(prompt ∪ gen)` within 1e-5 per step (generalizes
   the existing `ForwardCached_MatchesFullForward_WhenSeedingPromptTokenByToken`).
4. `ForwardPrefill_CapturesAllKvRows_AtCorrectOffsets` — cache capacity/row writes: rows
   `[0, L)` filled, nothing beyond, resp. at offset > 0 when called with `positionOffset`.
5. `ForwardPrefill_OutsideGrad_BuildsNoGraphNode` — last-row logits are leaf when outside
   `GradientUtils.Grad()`; cache capture writes into plain arrays.
6. GQA ratio coverage — run parity across 4/2, 8/2, 14/2 configs (parameterize the
   TinyModel builder like the decode tests).
7. Existing `LlamaCausalKVCacheTests` + `QwenInstructParityTests` (model-gated) stay green —
   the instruct parity pins the swapped sample path's trajectory against the Torch fixture
   (`qwen_tool_final_prompt_ids.bin`); flipped ids would trigger an escalation (tolerance
   audit; expected none — cache-vs-full parity already proves trajectory stability).

### E. Docs

- `tests/Nivara.PerformanceTests/README.md` — prefill rows (Prev = loop-based seed B/op +
  ops/s, Current = batched) + note mirroring P0-2's added-rows note.
- `CHANGELOG.md` — P0 entry under [Unreleased] (batched prompt prefill, before/after).
- `docs/QWEN-PERF.md` — append full P0 ledger entry (status/commits, item, numerics, §B + §C
  measured tables incl. the E2E split, gate note); remove the executed P0 prefill row from
  the future-entries table (the other future rows stay).
- `docs/QWEN.md` — update the "per prompt token → per-token ForwardCached" flow line to
  "one batched ForwardPrefill".
- `docs/AUTODIFF.md` — add `ForwardPrefill` to the `LlamaCausalAttention`/`LlamaKVCache`
  module lines.

## Verification steps (each gated on human confirmation per AGENTS.md)

1. `dotnet build Nivara.slnx` — clean.
2. Baseline: idle machine, `--runs 3` → `qwen-prefill-baseline.json` (on main, loop-based
   seed + full-fwd rows).
3. Implement §A + §B swap + §D tests; then `dotnet build` again.
4. Targeted fixture (`LlamaForCausalLMPrefillTests` + `LlamaCausalKVCacheTests` +
   `LlamaCausalAttentionTests`).
5. Full `dotnet test` suite (ask before running).
6. Postfix: `--runs 3 --compare` → `qwen-prefill-postfix.json`; P0 rows must PASS.
7. E2E A/B (main vs branch, split prefill/decode timing, same machine).
8. Model-gated `QwenInstructParityTests` if the checkpoint is present.
9. Docs commits; G2 review (branch as a whole + against this plan); `git rm` this plan;
   offer push + PR (human-confirmed).

## Planned commits

1. `docs: plan Qwen-fast P0 batched prompt prefill in TODO.md` (this plan)
2. `perf: add Qwen prefill scenarios to PerformanceTests` (loop-based `PrefillInto` on main)
3. `perf: split prefill/decode timing in Qwen benchmark` (E2E instrumentation — landed before
   the baseline so the main-side baseline already reports `prefill ms` + `decode ms/token`)
4. `perf: record qwen prefill baseline` (main → `qwen-prefill-baseline.json`; E2E split baseline
   numbers captured on the same pre-change build)
5. `perf: batched prompt prefill (ForwardPrefill)` — core attention + block + sample model +
   chat-client swaps + harness `PrefillInto` swap
6. `test: pin batched prefill parity, cache capture, and alloc-free path`
7. `perf: record qwen prefill postfix results` (`qwen-prefill-postfix.json`)
8. E2E A/B runs (no commit; branch vs main-side baseline)
9. `docs: record qwen-fast P0 prefill results (perf README, CHANGELOG, QWEN.md, AUTODIFF)`
10. `docs: add P0 prefill improvement-ledger entry to QWEN-PERF.md`
11. G2 → `docs: remove executed plan` (+ any issue-log updates)

## Blast radius

- Core: `LlamaCausalAttention.cs`, `LlamaDecoderBlock.cs` (additive methods + a
  shape-preserving refactor of `Forward`'s body).
- Samples: `LlamaForCausalLM.cs` (new method), `Qwen.cs`, `QwenChatClient.cs`,
  `SmolLMChatClient.cs` (one-line swaps).
- Tests: one new fixture. Harness: additive scenarios + JSONs.
- No kernels, no storage, no query-engine surface. `Forward`, `ForwardCached`, and the
  fused decode path are unchanged in behavior.

## Open items / decisions (confirmed at plan time)

- **Harness model size**: real Qwen shapes ⇒ ~2 GB resident synthetic model in the perf
  harness (same footprint the E2E already runs). Confirmed by human.
- **Loop-row L values**: baseline loop rows only at L = 8/16 (L = 64+ ≈ 30 s/op in the
  loop). The L = 64/256 prefill before/after is carried by the E2E split timing. Confirmed.
- **`ForwardPrefill` returns `[1, vocab]` last-row logits** (mirrors `ForwardCached`;
  single-row LM head is bit-identical per P0-2). Confirmed.
- **SmolLM chat client swap** included (same one-line change on the shared model class).
  Confirmed.
- **Parity**: 1e-5 logits; cache K/V asserted bit-equal with 1e-5 fallback documented.

## GitHub issues log

- (none at plan time; created during execution as deferred work is found)
- Related tracked work referenced: #384 (qkvBias, landed), #387/#391 (BF16 SIMD — orthogonal
  to this memory-traffic P0), #388 (fused BF16→F32 read), #390 (GGUF), #399 (large-kvLen
  DecodeAttention follow-up from the fused-GQA P0). A **GQA-aware batched attention** (drop
  `GqaRepeatKV` in prefill via a fused multi-row kernel) is a natural follow-up — file at
  creation time if the L=256 prefill rows show attention-materialization cost, else record
  in the ledger's future table.