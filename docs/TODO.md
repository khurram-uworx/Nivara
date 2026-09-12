# TODO — Qwen inference: plain-prompt coverage (#408) + fused batched prefill attention (#403)

Branch: `khurram/qwen-perf` (off `main`). Empirical, baseline-first: every change is
preceded by readings recorded on the current code and followed by re-readings on the
changed code, so the `docs/QWEN-PERF.md` ledger gains before→after evidence exactly like
P0-1/P0-2/P0-3/P1.

Reminder: as each task executes, if you find deferred work or a concern outside the two
issues below, create a GitHub issue immediately
(`gh issue create --repo khurram-uworx/Nivara`) and record its number in the issues log —
don't rely on memory.

## Problem

Two gaps from the QWEN-PERF review and planning session:

1. **No plain (no-tools) prompt surface (new issue #408).** NivaraInference `qwen
   benchmark` runs the weather-*tool* prompt only; every model-level parity test and the
   only PyTorch reference (`qwen_tool_reference.py`) exercise the tool trajectory. There
   is no plain-prompt benchmark, demo, fixture, or greedy-parity test, so the common
   non-tool chat path is unmeasured and unverified.
2. **Batched prefill still materializes GQA repeats (existing issue #403).**
   `LlamaForCausalLM<T>.ForwardPrefill` seeds the cache with one `[L, hidden]` forward but
   reuses `GqaRepeatKV` (+ `MultiHeadAttention` head packing / causal `[L,L]` mask),
   exactly what the fused `DecodeAttention` eliminated on the decode side. Residual:
   `Qwen prefill seed [256 tok]` = **785,702,840 B/op** (P0 postfix), scaling with L.

## Measurement protocol (applies to both phases)

- Machine identity captured (CPU brand, logical processors, .NET version) per session —
  the P1 pattern.
- Clean sequential sessions; one benchmark at a time; watch for thermal drift (P1 gate
  documented 70–88% clock throttling after sustained full-forward runs).
- `tests/Nivara.PerformanceTests` gates: `--json <baseline.json> --runs 3` →
  `--compare <baseline.json> --runs 3` (minOps 90%, alloc ≤×1.01, gen0 ≤+0.05
  conventions). Baseline JSON committed **before** any kernel change; postfix JSON after.
- NivaraInference console readings transcribed into the ledger/QWEN.md with the same
  before→after framing.
- `dotnet test` and all long-running benchmark runs require human confirmation first
  (AGENTS.md).

## Phase 1 — #408: plain-prompt benchmark + fixtures (S effort; `src/Nivara/` untouched)

**EXECUTED 2026-09-12 — see commits below.** Deviations from the sketch, recorded for G2:
- `RenderPlain` is **system-inclusive** (default Qwen system turn + user + generation
  prompt) — the checkpoint chat template's no-tools branch unconditionally emits the
  default system turn (verified against `tokenizer_config.json`), matching NivaraChat's
  pinned `Render_PlainChat_EmitsQwenDefaultSystemTurn`. Measured: 36 tokens.
- Harness gained `--only <substring>` (battery-optimal targeted gates; full-harness
  `--runs 3` ~20-30 min on the 16-thread Raptor Lake box) — a permanent harness facility
  documented in `tests/Nivara.PerformanceTests/README.md` and the QWEN-PERF harness-gate
  section.
- Real-checkpoint `qwen benchmark` (tool) baseline **deferred** (battery; ~6-8 min of
  sustained full-fwd CPU); the plain probe was taken (cheap); the harness gate is the
  authoritative pre-#403 before-reading. Ledger P1 already holds the tool-prompt prefill
  reference.
- Plain baseline before-readings (this machine, 2026-09-12): prefill 1,471 ms; cached
  decode 287 ms/tok; full-fwd 2,632 ms/tok; **5.8×** (7-token answer, 36-token prompt).

### 1a. NivaraInference `qwen --plain` (probe first, then readings)
- `QwenChatTemplate.RenderPlain(string userText)` — system + user + generation prompt:
  `"<|im_start|>system\n" + DefaultSystem + "<|im_end|>\n<|im_start|>user\n" + text +
  "<|im_end|>\n<|im_start|>assistant\n"`, byte-identical to the checkpoint template's
  no-tools branch and to NivaraChat's `QwenChatTemplate.Render([user],
  addGenerationPrompt: true)`.
- `Qwen.RunPlainBenchmark<T>` reusing `LoadModel<T>` + `RunDecodeBenchmark`
  (cached-vs-full, prefill/decode split, tok/s). Default prompt: `"What is the capital
  of France?"` (fixed so fixture parity is reproducible; `--text` overrides).
- `Program.cs`: `--plain` flag; `qwen benchmark --plain` → plain benchmark; bare `--plain`
  → single-shot demo printing the generated text + prefill/decode timing; works with
  `--synthetic-weights`; help text updated.

### 1b. PyTorch reference (mirror `qwen_tool_reference.py`)
- `samples/NivaraInference/Python/qwen_plain_reference.py` — HF `apply_chat_template`
  with **no tools**, greedy decode, dumps into the model dir:
  `qwen_plain_prompt.txt`, `qwen_plain_prompt_ids.bin`, `qwen_plain_ids_py.bin`,
  `qwen_plain_logits_py.bin` (same skip-when-absent convention as tool fixtures).

### 1c. Tests
- `QwenInstructParityTests` (or sibling fixture):
  - `Tokenizer_EncodePlainPrompt_MatchesTorchIds` — encode `qwen_plain_prompt.txt` ==
    `qwen_plain_prompt_ids.bin`.
  - `Model_GreedyPlainPrompt_MatchesTorchGeneratedIds` — greedy (existing `Greedy`
    helper) over the plain prompt == `qwen_plain_ids_py.bin`.
  - Always-run numeric seat (no checkpoint): plain prompt seed-then-decode ==
    `model.Forward(prompt ∪ gen)` within 1e-5 over a deterministic synthetic
    Qwen-shaped model (same pattern as `LlamaForCausalLMPrefillTests` cache-vs-full).

### 1d. Baseline-first readings record — EXECUTED
- Real checkpoint: plain probe `qwen benchmark --plain` taken (36-tok prompt; prefill
  1,471 ms, cached decode 287.3 ms/tok, full-fwd 2,631.5 ms/tok, 5.8× on a 7-token
  answer). Tool probe deferred (battery) — prefill reference already in the P1 ledger
  entry. Recorded in `docs/QWEN-PERF.md`.
- Synthetic: perf harness fresh same-machine `--only Qwen --json
  qwen-prefill-baseline.json --runs 3` committed (the Phase-2 gate baseline); seed [256
  tok] **785,532,248 B/op** (5,428 ms) · full-fwd [256 tok] 1,314,448,548 B/op. The
  2026-09-08 `qwen-prefill-baseline.json` was a full-harness cross-machine read and was
  replaced.

## Phase 2 — #403: fused GQA-aware batched attention for prefill (M effort; core)

### 2a. New kernel — `AttentionKernels<T>.BatchedAttention`
Multi-row prefill analogue of `DecodeAttention`:
- Inputs: Q `[qLen, numHeads·headDim]` (post-RoPE), K/V pre-repeat row-major
  `[qLen, numKvHeads·headDim]` (post-RoPE), output `[qLen, numHeads·headDim]`.
- Virtual GQA mapping `kvHead = qh / repeat`; scores computed per query head **once** per
  shared KV head; causal mask via iterating `j <= i` per query row `i` (folds scale +
  softmax; softmax over the reduced j-range avoids `-inf` masking entirely).
- Constant rented `[max(qLen,1)]` score scratch (ArrayPool); no `GqaRepeatKV`, no
  `PackHeads`, no `[numHeads,L,L]` score block, no mask allocation.

Sketch:
```csharp
public static void BatchedAttention(
    ReadOnlySpan<T> q, ReadOnlySpan<T> k, ReadOnlySpan<T> v, Span<T> output,
    int qLen, int numHeads, int numKvHeads, int headDim, T scale)
{
    int repeat = numHeads / numKvHeads;
    int kvWidth = numKvHeads * headDim;
    var scores = ArrayPool<T>.Shared.Rent(Math.Max(qLen, 1));
    try
    {
        output.Clear();
        var scoresSpan = scores.AsSpan(0, qLen);
        for (int qh = 0; qh < numHeads; qh++)
        {
            int kvHead = qh / repeat;
            for (int i = 0; i < qLen; i++)
            {
                var qRow = q.Slice(i * numHeads * headDim + qh * headDim, headDim);
                for (int j = 0; j <= i; j++)
                    scoresSpan[j] = scale * TensorPrimitives.Dot(qRow, k.Slice(j * kvWidth + kvHead * headDim, headDim));
                SoftmaxRows(scoresSpan[..(i + 1)], 1, i + 1);
                var outRow = output.Slice(i * numHeads * headDim + qh * headDim, headDim);
                for (int j = 0; j <= i; j++)
                {
                    T w = scoresSpan[j];
                    var vRow = v.Slice(j * kvWidth + kvHead * headDim, headDim);
                    for (int d = 0; d < headDim; d++) outRow[d] += w * vRow[d];
                }
            }
        }
    }
    finally { ArrayPool<T>.Shared.Return(scores); }
}
```

### 2b. Wire into `LlamaCausalAttention.ForwardCore`
Under `kCache != null && vCache != null && !GradientUtils.IsGradEnabled` → fused kernel,
skip `GqaRepeatKV`/`MultiHeadAttention`; grad-enabled and decode paths unchanged; cache
capture unchanged (`ForwardPrefill` public behavior preserved: last-row `[1, vocab]`
logits). `Forward(int[])`/`ForwardPrefill`/`ForwardCached` signatures unchanged.

### 2c. Parity tests (`LlamaForCausalLMPrefillTests`)
- Fused `BatchedAttention` vs current `ForwardPrefill` within 1e-5 across GQA ratios
  4/2, 8/2, 14/2 (as #403 acceptance).
- Cache rows: layer-0 bit-equal to the per-token walk; deeper within 1e-5 (existing
  convention).
- Graph guard: no graph nodes outside `Grad()`; steady-state alloc guard on the seed rows.

### 2d. Post-fix readings
- Perf harness `--json qwen-prefill-postfix.json --runs 3` → `--compare
  qwen-prefill-baseline.json --runs 3`. Target: seed [256] 785.7 MB B/op → one-forward
  allocation; all four seed rows PASS with margins.
- NivaraInference E2E A/B (synthetic + real checkpoint if cheap): prefill ms, decode
  ms/token before→after, plus the plain-prompt surface from Phase 1 re-measured (shared
  prefill path must improve both tool and plain prompts).

## Planned commits (one logical change each)

- ✓ `docs: plan qwen plain-prompt coverage + fused prefill attention in TODO.md` (`b9fbae8`)
- Phase 1 (all landed 2026-09-12):
  - ✓ `infer: add qwen --plain prompt benchmark and demo` (`d9a9f70`)
  - ✓ `perf: add --only scenario filter to harness gate for targeted runs` (`4cb1ea1` — added in execution, battery-optimal gates)
  - ✓ `samples: add qwen plain-prompt Torch reference generator` (`81a2dce`)
  - ✓ `docs: document qwen --plain/--synthetic-weights and --only targeted gates` (`7e9c33f`)
  - ✓ `tests: pin qwen plain-prompt parity (tokenizer, greedy, cache-vs-full)` (`c9b0bcd` — always-run seat uses the measured 36-tok plain prompt length)
  - ✓ `perf: record qwen baseline readings (tool + plain) before kernel work` (`01e1b4f`)
  - Remaining Phase-1 verification: `dotnet test` (await human confirmation) and optional
    plain fixture generation (`Python/qwen_plain_reference.py` — activates the
    skip-if-absent parity tests; same convention as the tool fixtures).
- Phase 2:
  - `perf: fused GQA-aware batched attention for prefill kernel` (AttentionKernels +
    ForwardCore wiring)
  - `tests: pin batched-prefill fused attention parity (GQA ratios, guards)`
  - `perf: record qwen prefill postfix results + E2E A/B (#403)`
  - `docs: add QWEN-PERF ledger entries for #408/#403`
- G2: `docs: remove TODO.md — plan executed`

## Blast radius

- **#408:** `samples/NivaraInference/Qwen.cs` (+ `Program.cs`, `Python/`), one test
  fixture; `src/Nivara/` untouched; existing behavior additive (new `--plain` flag only).
  Tests affected: `QwenInstructParityTests` (additive), perf README/docs.
- **#403:** `src/Nivara/AutoDiff/Nn/LlamaCausalAttention.cs` (`ForwardCore` inference
  branch) + `src/Nivara/AutoDiff/Operations/AttentionKernels.cs` (new internal, static).
  Downstream callers of `ForwardPrefill` (LlamaForCausalLM → QwenChatClient.SeedCache,
  NivaraInference `GenerateCore`) are numerical-parity-locked within 1e-5, cache layout
  preserved — decode-after-prefill unchanged. Coverage: `LlamaForCausalLMPrefillTests`
  (21), `LlamaCausalAttentionTests`, `QwenInstructParityTests` (checkpoint-gated),
  `QwenToolsWeatherLoopTests`; perf-harness Qwen seed rows. Fused path is inference-only
  (`!IsGradEnabled`); autograd users cannot observe it.

## GitHub issues log

- [#408](https://github.com/khurram-uworx/Nivara/issues/408) — Qwen plain-prompt
  benchmark + fixtures (created while planning this branch)
- [#403](https://github.com/khurram-uworx/Nivara/issues/403) — GQA-aware batched
  attention for prefill (pre-existing; tracked from the P0-1 ledger entry)