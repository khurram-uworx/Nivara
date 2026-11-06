# Plan: NivaraChat --qwen --silk

Branch: `khurram/silk-qwen` (base `khurram/silk`).

Scope: add a `--silk` path to `NivaraChat --qwen` so the Qwen2.5-0.5B chat client can
run on the OpenCL GPU, mirroring the SmolLM integration. Tool-calling modes
(`tools-weather`, `tools-nivara`) reuse the same Silk-generation path through
`FunctionInvokingChatClient`.

## Changes
- `samples/NivaraChat/Qwen/SilkQwenChatClient.cs` — new IChatClient adapter over
  `SilkQwenSession` (prefill/cached/no-cache, temperature/top-p sampling, stop ids,
  tool-call parsing via `QwenToolCallParser`).
- `QwenMode` — `--silk` parse + `UseSilk` option, `ExecuteSilk<T>` branch before CPU
  model load, rebuilt `--silk` help line, session-to-config mapping.
- `RunSingleTurn`/`RunRepl` generalised to `IChatClient` so the Silk client plugs in.
- `NivaraChat` top-level help mentions Qwen in the `--silk` line.
- Follow-up issues created for fine-tuning samples Silk story (issue #949).
- Sweep: replace user-facing `--ilgpu` command guidance in `Laya.cs` and stale
  seam comments in `GpuSeam.cs`.

## Verification
- `dotnet build samples/NivaraChat/NivaraChat.csproj -c Release`
- manual smoke of `--qwen --silk plain/chat` if model dir present.

## GitHub issues log
- #949 — Silk backend story for NivaraFineTuning
