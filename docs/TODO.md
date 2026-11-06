# Plan: Enable `--silk` for NivaraChat `--smollm`

## Problem

`NivaraChat --smollm` currently always uses the CPU `SmolLMChatClient<T>` path. Silk/OpenCL already has a working
SmolLM causal decoder through `SilkSmolLMSession` and `SilkSmolLMGpuRunner`, but that path is only wired into
`NivaraInference`. The chat sample's `--text`/REPL experience has no `--silk` option, so `--silk` is silently ignored there.

## Proposed changes

### Step 1 — CLI parsing in `SmolLM/SmollmMode.cs`

- Add `bool UseSilk` to `SmollmOptions`.
- Add `case "--silk": options.UseSilk = true; break;`.
- Update `PrintHelp` with `--silk`.

### Step 2 — Branch loading before generic CPU `Run<T>`

In `SmollmMode.Run`, keep the existing model-directory and config/tokenizer loading. Then:

- If `options.UseSilk`:
  - load safetensors as `float` for `--precision f32`, as `BFloat16` for `--precision bf16`;
  - construct `SilkSmolLMSession.Open(ToSilkConfig(config), tensors)`;
  - run `RunSilkFloat(...)` / `RunSilkBFloat16(...)`.
- Otherwise keep the existing `Run<float>` / `Run<BFloat16>` CPU path.

No changes to `SilkSmolLMSession` or Silk internals are planned.

### Step 3 — Add `SmolLM/SilkSmolLMChatClient.cs`

Implement `IChatClient` using `SilkSmolLMSession`, matching `SmolLMChatClient<T>` behavior:

- Render prompt with `SmollmChatTemplate.Render`.
- Encode prompt preserving special tokens.
- KV-cache branch:
  - `ResetCache()`.
  - prefill logits from `session.ForwardPrefill(ids.ToArray())`.
  - decode using `session.ForwardCached(next, position++)`.
- Non-cache branch:
  - full `session.Forward(ids.ToArray())`;
  - select from the last `vocabSize` logits.
- Greedy and temperature/top-p sampling using the shared `TokenSampling` helper.
- Stop on EOS.
- Stream one update per generated token.
- Dispose the session.
- Store a small `EncodePrompt` helper shared with the CPU client or as a shared utility.

### Step 4 — Metadata and diagnostics

- Set `ChatClientMetadata` to mention Silk/OpenCL.
- Print `DeviceName` during load.

### Step 5 — Verification

- Build `samples/NivaraChat` in Release.
- Run smoke commands when model and OpenCL device are available:
  - `dotnet run --project samples/NivaraChat -- --smollm plain --text "The capital of France is" --silk`
  - interactive `--smollm chat --silk`.
- Confirm missing OpenCL still fails loudly rather than silently falling back to CPU.

## Verification

- `dotnet build samples/NivaraChat -c Release`
- smoke commands above after confirmation from human

## Planned commits

1. `docs: enable --silk for NivaraChat --smollm: plan in TODO.md`
2. `feat: parse --silk for SmolLM mode`
3. `feat: add SilkSmolLMChatClient`
4. `feat: wire RunSilk* paths for float and BF16`
5. `docs: note --silk support in SmolLM help`
6. `test/verify: build and smoke notes`
7. `docs: remove TODO.md — plan executed`

## Blast radius

- Touches:
  - `samples/NivaraChat/SmolLM/SmollmMode.cs`
  - `samples/NivaraChat/SmolLM/SilkSmolLMChatClient.cs` (new)
  - `samples/NivaraChat/SmolLM/SilkSmolLMChatClient` helper if added
  - `samples/NivaraChat/README.md` if documented
- Does not touch core/src; Silk public surface already supports sessions.
- Affects interactive/plain SmolLM chat only.
- Does not change CPU SmolLM path or other samples.

## GitHub issues log

- No issues created yet.
