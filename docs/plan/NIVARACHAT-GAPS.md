# NivaraChat generalization gaps (post-TinyShakespeare)

Umbrella context: issues #1270/#1307–#1315 added a Silk/GPU path, KV cache, top-p, and
CPU-vs-Silk parity for the TinyShakespeare `BatchedTransformer`. The original phased plan
lived in `docs/plans/TINY-SHAKESPEARE.md`, which has been retired now that all of those
issues are closed and merged.
This document captures the follow-up gaps identified when comparing that work against the
rest of NivaraChat (SmolLM, Qwen, and the trained classifier models), and the GitHub issues
filed for them. Training stays on the CPU AutoDiff path; Silk is inference-only, matching
SmolLM/Qwen.

## Assessment summary

The TinyShakespeare work mostly **caught up** to where SmolLM/Qwen already were. The genuine,
reusable advances are:

1. `ToSilkWeights()` + a Silk runner for a **Nivara-trained** transformer — the only path where
   a model trained inside Nivara AutoDiff can be exported to Silk. Currently TinyShakespeare-only.
2. The CPU `IChatClient` + KV cache + streaming pattern for a trained-from-scratch transformer.

The clearest gap is that TinyShakespeare has **no Silk `IChatClient`** (its `--silk` path drives
`SilkTinyShakespeareSession` by hand and only prints), unlike SmolLM/Qwen which expose
`Silk*ChatClient : IChatClient`.

## Issues filed

| Issue | Title | Priority | Depends on |
|---|---|---|---|
| #1324 | NivaraChat: add Silk `IChatClient` (`SilkBatchedChatClient`) for `--tinyshakespeare` | High | — (uses #1313 session) |
| #1325 | NivaraChat: align KV-cache default (`true`) for `--tinyshakespeare` | Medium | — |
| #1326 | NivaraChat: add CPU-vs-Silk parity test for SmolLM | Medium | — |
| #1327 | NivaraChat: extract shared causal-LM generation loop across chat clients | Low (optional) | #1324 |
| #1328 | NivaraChat: migrate `BatchedTransformer` (TinyShakespeare) to BPE tokenization | Low (optional, breaking) | — |

## Phases and picking order

### Phase 1 — close the Silk IChatClient gap (do first)

1. **#1324** — Add `SilkBatchedChatClient : IChatClient` mirroring `SilkSmolLMChatClient`
   (prefill + cached decode + overflow + `TokenSampling` + top-p + streaming), and rewire
   `--tinyshakespeare --silk` to use it (`--text` single-shot + DI demo). Unblocks agents/
   workflows/streaming for the Silk path.
2. **#1325** — Flip `BatchedChatClient` default to `useKvCache = true`, add `--no-kv-cache`,
   matching SmolLM/Qwen.

### Phase 2 — parity coverage

3. **#1326** — Add `SilkSmolLMParityTests` (CPU `LlamaForCausalLM` vs `SilkSmolLMSession`,
   prefill + cached step), mirroring `SilkQwenParityTests`. No production code changes.

### Phase 3 — optional refactors (only if more transformer chat clients are planned)

4. **#1327** — Extract a shared generation loop used by all six chat clients. Design-first;
   overflow policy may need to stay per-model.
5. **#1328** — Migrate `BatchedTransformer` to `Gpt2BpeTokenizer`. Model-breaking (saved
   `*.json`/`*.tokenizer.json` no longer load); coordinate with a version bump. Recorded
   decision D-014 currently defers this.

## Non-goals (intentionally not generalized)

- `ToSilkWeights` for SmolLM/Qwen: they do not train in Nivara; weights come from safetensors.
- Classifier models (sentiment/entity/validator): no autoregressive generation or KV cache;
  Silk paths already exist (`GpuTextLinearClassifier`, etc.).
- GPU training / AutoDiff on Silk (non-goal carried over from the retired TinyShakespeare plan).
- Rank-3 (batched) MatMul in core.

## Key references for coding agents

- `samples/NivaraChat/SmolLM/SilkSmolLMChatClient.cs` — template for #1324.
- `samples/NivaraChat/Qwen/SilkQwenChatClient.cs` — same pattern.
- `samples/NivaraChat/Transformer/TransformerMode.cs::RunSilk` — current hand-driven path.
- `src/Nivara.Extensions/Silk/SilkTinyShakespeareSession.cs` — session API.
- `tests/Nivara.Tests/Qwen/SilkQwenParityTests.cs` — template for #1326.
- `samples/NivaraChat/Shared/TokenSampling.cs` — sampling helper.
- `samples/NivaraChat/ARCHITECTURE.md`, `samples/NivaraChat/README.md` — conventions/CLI.

## Picking order

1. #1324 and #1325 can be picked up together (small, independent).
2. #1326 is independent and test-only.
3. #1327 and #1328 are optional; #1327 benefits from #1324 being done first.