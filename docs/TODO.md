# Plan: #1419 — LoRA: optional NivaraChat adapter consumption demo

## Problem

NivaraChat is the interactive showcase app. After NivaraFineTuning produces a LoRA adapter bundle, there is no way to load and prompt the adapted Qwen2.5-0.5B (or generic Llama-family) model inside NivaraChat to demonstrate "same app, adapted model." RAG modes must remain untouched.

## Scope (from issue)

- Add a narrow adapter-loading/demo path (flag or mode) for adapted Qwen2.5-0.5B; generic Llama-family where easy.
- Keep RAG modes unchanged.
- Document clearly as FT showcase touchpoint, not RAG-vs-FT.

## Proposed changes

### 1. Add `--adapter <path>` to Qwen mode (headline path)

File: `samples/NivaraChat/Qwen/QwenMode.cs`

- Extend `QwenOptions` with `AdapterPath` field.
- Parse `--adapter <path>` in `ParseArgs`.
- Print in `PrintHelp`.
- In CPU `Execute<TModel,TWeight>`: after `LlamaLoader.Load(...)`, if `AdapterPath` is set, load `Nivara.Silk.LoraSftAdapterBundle`, parse targets, `model.AttachLora(...)`, copy A/B matrices with validation (rank/alpha/count/shape/data length). Print status line.
- In Silk `ExecuteSilk`: if `AdapterPath` is set, throw `NotSupportedException` with clear message (no silent fallback).

### 2. Add `--adapter <path>` to SmolLM mode (generic Llama-family)

File: `samples/NivaraChat/SmolLM/SmollmMode.cs`

- Mirror the above for CPU path. Silk path: explicit error if adapter provided.

### 3. Shared helper for bundle-apply (sample-local)

New file: `samples/NivaraChat/Helpers/LoraAdapterLoader.cs`

- Static method `Apply(LlamaForCausalLM<float,float> model, string adapterPath)` that encapsulates bundle load + attach + copy + validation, matching `LoadCpuAdapterBundle` logic from NivaraFineTuning.

### 4. CLI/docs wiring

- `samples/NivaraChat/Program.cs`: update `PrintUsage()` Qwen/SmolLM lines.
- `samples/NivaraChat/README.md`: add "LoRA-adapted Qwen (FT showcase)" subsection + CLI table entry.
- `samples/NivaraChat/ARCHITECTURE.md`: update LoRA/fine-tuning note.

### 5. Tests

- Add targeted unit test in `tests/Nivara.Tests` for the bundle-apply helper: valid bundle attaches/loads and round-trips; rejects rank/alpha/count/shape mismatch.

## Planned commits

1. `docs: plan #1419 in TODO.md`
2. `feat: add --adapter support to QwenMode (CPU path)`
3. `feat: add --adapter support to SmolLM mode (CPU path)`
4. `feat: shared LoraAdapterLoader helper + validation`
5. `docs: update NivaraChat README/ARCHITECTURE for adapter demo`
6. `test: add adapter bundle apply tests`
7. `docs: remove TODO.md — plan executed`

## Blast radius

- `samples/NivaraChat/Qwen/QwenMode.cs` — new optional flag; default behavior unchanged.
- `samples/NivaraChat/SmolLM/SmollmMode.cs` — new optional flag; default behavior unchanged.
- `samples/NivaraChat/Helpers/LoraAdapterLoader.cs` — new file, sample-local.
- `samples/NivaraChat/Program.cs` — help text only.
- `samples/NivaraChat/README.md`, `ARCHITECTURE.md` — docs.
- `tests/Nivara.Tests` — new tests referencing LlamaForCausalLM + LoraSftAdapterBundle.
- RAG modes (`Rag*Mode`, `Modes/*`) — NOT touched.

## GitHub issues log

(none yet)
## Grounding notes (G1)

- Adapter format: `Nivara.Silk.LoraSftAdapterBundle` (`nivara-lora-sft-adapters-v1`) — shared CPU/Silk bundle.
- CPU consumption reference: `NivaraFineTuning.Program.RunFtEval` + `LoadCpuAdapterBundle`.
- LoRA attach API: `LlamaForCausalLM.AttachLora(LlamaLoRATarget, int rank, TModel? alpha)`.
- Silk Qwen session has no LoRA load path → explicit error, no silent fallback.
- Targets string in bundle is enum flags (e.g. "Q, V"); parser accepts both "Q,V" and "q,v".
- NivaraChat.csproj already references Nivara.Extensions (LoraSftAdapterBundle available).
- No docs/TODO.md reintroduction in final PR (deleted at G2).
