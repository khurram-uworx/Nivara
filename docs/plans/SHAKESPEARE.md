# SHAKESPEAR2 — TinyShakespeare Silk/GPU port plan

Umbrella issue: **#1270** (NivaraChat: TinyShakespeare transformer has no Silk/GPU path).

This document captures the phased plan, the GitHub issues that implement it, and the
order in which a coding agent should pick them up. Training stays on the CPU
AutoDiff path; Silk is inference-only, matching SmolLM/Qwen.

## Scope summary

Add an OpenCL/Silk inference path to `--tinyshakespeare` (a `BatchedTransformer<float>`
trained on CPU via AutoDiff), with KV caching, improved sampling, and parity
coverage. Do **not** reopen D-001 (no rank-3 MatMul in core) or move training to
GPU.

## Phases and issues

### Phase A — cleanup (low risk, do first)

| Issue | Title | Depends on |
|---|---|---|
| #1307 | Remove unused `dropout` parameter from `BatchedTransformerBlock` | — |
| #1308 | Replace hand-rolled sampling in `BatchedChatClient` with `TokenSampling` | — |
| #1309 | Add `--top-p` support to `--tinyshakespeare` | #1308 |

### Phase B — KV cache (CPU)

| Issue | Title | Depends on |
|---|---|---|
| #1310 | Add KV cache to `BatchedTransformer` for autoregressive generation | — |

### Phase C — tokenizer decision

| Issue | Title | Depends on |
|---|---|---|
| #1311 | Decide tokenizer strategy (word-level vs BPE) for the Silk port | — |

Recommendation recorded there: keep word-level for v1; revisit BPE separately.

### Phase D — weight export

| Issue | Title | Depends on |
|---|---|---|
| #1312 | Add `ToSilkWeights()` export helper on `BatchedTransformer` | — |

### Phase E — Silk runner (inference)

| Issue | Title | Depends on |
|---|---|---|
| #1313 | Add Silk/OpenCL inference runner for `BatchedTransformer` | #1312, #1310 |

### Phase F — parity

| Issue | Title | Depends on |
|---|---|---|
| #1314 | Add CPU-vs-Silk parity test for TinyShakespeare | #1313 |

### Phase G — docs/CLI coverage

| Issue | Title | Depends on |
|---|---|---|
| #1315 | Update docs/README/CLI coverage for Silk + KV cache + top-p | #1307, #1308, #1309, #1310, #1313 |

## Picking order for coding agents

1. Start with **#1307** and **#1308** (independent, small).
2. Then **#1309** (depends on #1308).
3. In parallel with the above, **#1310** and **#1312** can be picked up (independent).
4. **#1311** is a decision-only issue; can be closed once the decision is recorded.
5. **#1313** must wait for #1312 (and benefits from #1310).
6. **#1314** waits for #1313.
7. **#1315** is last.

## Key references for agents

- `samples/NivaraChat/Transformer/BatchedTransformer.cs` — model + `BuildCausalMask`
- `samples/NivaraChat/Transformer/BatchedChatClient.cs` — generation, sampling
- `samples/NivaraChat/Transformer/TransformerMode.cs` — CLI
- `src/Nivara.Extensions/Silk/SilkSmolLMGpuRunner.cs` — runner template
- `src/Nivara.Extensions/Silk/SilkSmolLMSession.cs` — session template
- `src/Nivara.Extensions/Silk/SilkLauncher.cs`, `SilkRuntime.cs` — kernel launch + device
- `samples/NivaraChat/SmolLM/SmollmMode.cs`, `SmolLMChatClient.cs` — `--silk` wiring + KV cache usage
- `samples/NivaraChat/ARCHITECTURE.md` — TinyShakespeare + Decisions sections
- `samples/NivaraChat/README.md` — CLI table + gaps

## Compromises to revisit (recorded decisions)

- D-007 word-vocab cost accepted; keep for v1 (issue #1311).
- D-001 no rank-3 MatMul in core; runner uses flattened GEMM — no change.
- D-004 no further kernel changes; Silk runner is new code, not a core kernel change.
- D-010 attention softmax stays fused; runner uses causal attention kernel.
- Dropout parameter unused → remove (#1307).
- No KV cache → add (#1310).
- Top-p/top-k missing → add via `TokenSampling` (#1308, #1309).

## Non-goals

- GPU training / AutoDiff on Silk.
- Rank-3 (batched) MatMul in core.
- BPE migration in this port (decision tracked in #1311).
- Quantization (separate issue if needed).
