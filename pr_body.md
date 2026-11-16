## Summary

NivaraChat `--smollm` and `--qwen` now load native weights only (BF16 storage / F32 compute) and expose no `--precision` flag, mirroring the `NivaraInference` native default (#1216, #1221).

- SmolLM: `SafeTensorsLoader.Read<BFloat16>` → `LlamaLoader.Load<float, BFloat16>`; `SmolLMChatClient<T>` → `SmolLMChatClient<TModel, TWeight>`; single BF16 Silk path.
- Qwen: same native load; `QwenChatClient<T>` → `QwenChatClient<TModel, TWeight>`; `--silk` coerces to the F32 device path with diagnostic while #1219 is open.
- Unknown flags (including `--precision`) now fail loudly instead of being swallowed as a mode.
- Docs: note chat host surface in `docs/SMOLLM.md` and `docs/QWEN.md`.

## Test plan (verified)

- `dotnet build samples/NivaraChat` — succeeded.
- `dotnet build tests/Nivara.Tests` — succeeded.
- Docs namespace tests — 66/66 passed.
- `--smollm plain --text "The capital of France is"` — streamed "Paris, which" (native load).
- `--qwen plain --text "The capital of France is"` — streamed "Paris." (native load).
- `--qwen plain --silk ...` — F32 device path with diagnostic, streamed "Paris".
- `--precision bf16`/`--precision f32` — rejected with "Unknown option: --precision" and help.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` — run.