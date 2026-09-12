# TODO — Fold QWEN-PERF into the authoritative Qwen docs and close the issue backlog

## Problem

The Qwen-fast performance work (P0–P1, all executed and merged) is documented in
`docs/QWEN-PERF.md`, a research + improvement-ledger doc that sits *beside* the
authoritative Qwen documentation (`docs/QWEN.md`) and the sample READMEs
(`samples/NivaraInference/README.md`, `samples/NivaraChat/README.md`). After the
perf work landed, `QWEN-PERF.md` is the only place that records the perf journey,
but it is a standing doc the user wants eliminated: **no overlapping docs should
exist** — only the Inference README + Chat README + `docs/QWEN.md` are
authoritative. The user also wants every not-yet-done item from `QWEN-PERF`
captured in GitHub issues so nothing is shelved when the doc is deleted.

## Goal

1. Fold `QWEN-PERF.md` into:
   - `samples/NivaraInference/README.md` — library additions/enhancements from the
     perf work, in the established per-model "Core library improvements" / 
     "Sample-scoped additions" / results format.
   - `docs/QWEN.md` — the whole Qwen story: initial implementation + perf work.
2. Verify/refresh GitHub issue coverage for every outstanding `QWEN-PERF` item
   (P2 items, stretch items, E2E measurement), creating/fixing issues where needed.
3. Delete `docs/QWEN-PERF.md` and fix all dangling references.
4. Only P2 items (+ backlog/stretch) remain open as prioritized issues.

## Proposed changes

### Step 1 — GitHub issues (capture everything before docs cite issue numbers)

- **#386** (`qwen distill` full cycle on a dedicated machine) — refresh with
  post-Qwen-fast reality (the ~178–206 s/sentence / ~1–1.5 h estimates predate the
  perf work; post-perf prefill ≈ 0.75–0.94 s, decode ≈ 116–243 ms/token). Add the
  bf16 memory-halving option as a small-VM lever, the float-only distill note, and
  pointers to where the post-perf numbers live after this fold. Acceptance stays.
  The user takes this issue from there.
- **New issue — INT8 block-quantized weights (Stretch)** — the only `QWEN-PERF`
  item with no GitHub home. Body cross-links `docs/INTEGERS.md` (blocked: AutoDiff
  is `IFloatingPointIeee754<T>`-constrained, no integer tensors) and #390 (GGUF);
  cites the llama.cpp-style Q8 context (~4× traffic cut, ~65+ tok/s ceiling).
  Label `future` (+ `performance`).
- **#402** — body says "the P2 row of the `docs/QWEN-PERF.md` improvement ledger";
  reword to point at `docs/QWEN.md` (QWEN-PERF is deleted).

### Step 2 — `docs/QWEN.md` becomes the whole Qwen story

- Library map: add perf-era building blocks — `LlamaForCausalLM<T>.ForwardPrefill`
  (batch prefill #401), `ForwardCachedFused`/`ForwardPrefillFused` (#404),
  `AttentionKernels<T>.DecodeAttention` (#400) / `.BatchedAttention` (#403),
  `LlamaFusedKernels.DecoderBlockFused` toggle, `NivaraPrimitives.UseWidenSimd` use
  in bf16 benchmark.
- Generation/client section: `SeedCache` → one batched `ForwardPrefill` (not
  per-token `ForwardCached`); fused decode path; `--precision bf16` benchmark-only
  surface (chat/inference toggles it too, via their own hosts).
- Mark the `342567 ms` transcript and 9,153 ms/tok decode figures as
  **pre-Qwen-fast (historical)**; add the post-perf measured figures.
- New section **"Making Qwen fast (Qwen-fast)"** (condensed from QWEN-PERF):
  - Review premise (2–3 bullets): prefill O(L)·2 GB, redundant weight copies +
    `clearArray` on return, KV-copy + `GqaRepeatKV` materialization, per-op boxing.
  - Executed items with deltas (condensed, one bullet each): P0-1 (prefill
    5,566→747 ms, ~7.4×), P0-2 (LM head 5→36 ops/s, +620%; pool premise
    correction), P0-3 (decode-attn +62–177%; B/op 21×/41×/81×↓), #403 (alloc
    −5.6…−6.4%), #404 (decode block 131,985→1 B/op; decode fwd → ~619 KB logits
    floor; E2E decode ~4.5–4.9×), #407 bf16 (measured slower on the i5 — tradeoff
    framing). Full row history stays in `tests/Nivara.PerformanceTests` README +
    `qwen-*.json` artifacts.
  - **Remaining-items table**: P2 #402/#413/#414, backlog #399/#411, stretch
    #390/#387/#391/INT8 (gated on `docs/INTEGERS.md`).
  - "Not worth investing now" record (transposed-view cache, parallel decode,
    speculative) so the reasoning survives.
- Verification evidence: add post-perf suites (prefill, fused parity 42/42, bf16
  parity, #408 plain fixtures). Update "Related Qwen work" list.
- Doc conventions used by QWEN-PERF that must survive in some form: measurement
  protocol note (child-process medians, `--only`, gate defaults) — already
  documented in the PerformanceTests README; leave a one-line pointer, do not
  duplicate.

### Step 3 — `samples/NivaraInference/README.md`, set format

- Usage: add `--precision bf16` rows for `qwen benchmark` / `--synthetic-weights`;
  reconcile the "bf16/fp16 rejected for qwen" precision note (rejected for
  tools/chat/plain generation; supported for benchmark/synthetic modes).
- New Qwen subsection **"Qwen inference performance (Qwen-fast)"**:
  - *Core library improvements (src/Nivara)* in the established bullet format:
    single-row GEMV fast path in `TensorsHelper.MultiplyCore` (#398);
    `AttentionKernels<T>.DecodeAttention` (#400) and `.BatchedAttention` (#403);
    `LlamaCausalAttention<T>.ForwardPrefill` core path with K/V capture (#401);
    fused block kernels `ForwardCachedFused`/`ForwardPrefillFused` + public
    `LlamaFusedKernels.DecoderBlockFused` toggle (default on, inference-only) (#404).
  - *Sample-scoped additions*: `LlamaForCausalLM<T>.ForwardPrefill` + model-level
    fused routing; genericized qwen benchmark over `T`, `Read<BFloat16>`, and
    `UseWidenSimd` enabling for bf16 benchmark/synthetic (#407).
  - *Results*: refresh the decode-throughput table and Results/benchmarks table
    with post-perf numbers (old rows marked historical); add #408 plain-prompt
    rows; add `qwen_plain_*` fixtures to the Sample data table; add capability
    rows. Note the dedicated-machine full E2E refresh stays tracked in #386 and is
    out of scope for this branch.
- Mark 9,153 ms/tok / 173,900 ms rows as **pre-Qwen-fast (historical)**.

### Step 4 — `samples/NivaraChat/README.md` (light touch)

- Update the "…live in `docs/QWEN.md`" sentence to also point at the perf-journey
  content. Chat-side `--precision f32|bf16` already documented; no option changes.

### Step 5 — CHANGELOG

- Add a small "docs consolidation" bullet under `## [Unreleased]` (`### Docs` or
  next to the other additions): QWEN-PERF folded into `docs/QWEN.md` + Inference
  README; INT8 stretch captured as a blocked issue gated on integer AutoDiff.

### Step 6 — delete `docs/QWEN-PERF.md` + fix references

- `tests/Nivara.PerformanceTests/README.md:213` and `Program.cs` comments at
  ~484/569/620/660/683 reference `docs/QWEN-PERF.md` → retarget to `docs/QWEN.md`.
- `git rm docs/QWEN-PERF.md`; `grep -ri "QWEN-PERF"` → zero matches.

## Verification steps

- `grep -ri "QWEN-PERF"` across the repo returns nothing after Step 6.
- `grep` for stale figures (9,153 ms/tok, 342567, 173,900) — only as explicitly
  marked "historical (pre-Qwen-fast)" where they survive.
- `gh issue view 402|386` bodies no longer reference QWEN-PERF; new INT8 issue
  exists with the integer-domain blocker stated.
- No code changes — docs/issues only (PerformanceTests 📄 comment edits are
  comments only). No `dotnet test` required.
- G2 review: every `QWEN-PERF` ledger item has a home (issue or folded doc).

## Planned commits

1. `docs: plan QWEN-PERF fold in TODO.md`
2. `chore: capture INT8 quantization stretch issue (blocked on integer AutoDiff); refresh #386; reword #402`
3. `docs: fold Qwen-fast perf work into QWEN.md`
4. `docs: fold Qwen inference performance into the Inference README (set format)`
5. `docs: point NivaraChat README at the Qwen perf journey`
6. `docs: CHANGELOG note for QWEN docs consolidation`
7. `docs: delete QWEN-PERF.md and retarget harness references`
8. `docs: remove TODO.md — QWEN docs consolidation complete`

## GitHub issues log

- [x] #416 — INT8 block-quantized Qwen stretch (created 2026-09-12; blocked on integer AutoDiff, `docs/INTEGERS.md`; cross-links #390)
- [x] #386 — comment added (2026-09-12) refreshing the post-Qwen-fast estimates + bf16 small-VM lever + float-only distill note; user takes the cycle from there
- [x] #402 — body reference reworded from QWEN-PERF to QWEN.md (2026-09-12)
- [ ] existing/verified open: #399, #411 (backlog), #413, #414 (P2), #402 (P2), #387, #390, #391 (stretch/future)

## Blast radius

- **Docs only.** Files touched: `docs/TODO.md` (created/removed), `docs/QWEN.md`,
  `docs/QWEN-PERF.md` (deleted), `samples/NivaraInference/README.md`,
  `samples/NivaraChat/README.md`, `CHANGELOG.md`,
  `tests/Nivara.PerformanceTests/README.md` + `Program.cs` (comment-only edits).
- GitHub issues: one new issue (INT8), two issue-body edits (#386, #402). Issues
  are cross-linked from the docs, so bodies must be updated before/with the docs.
- No source/test/project files change; no build or test run required.