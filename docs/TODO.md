# Plan: NivaraChat — drop `--precision` (issue #1265)

## Problem

`README.md` documents a `--precision f32|bf16` option for both `--smollm` and `--qwen`, promising that `bf16` keeps BF16 weights native. Neither mode parses the flag. The CPU/Silk paths already run a single native path (`SafeTensorsLoader.Read<BFloat16>` + `LlamaLoader.Load<float, BFloat16>`), and both `PrintHelp()` outputs already omit `--precision`. The only false claim is in the README.

## Decision

Resolve #1265 by removing the doc claim rather than implementing the flag: NivaraChat runs one honest native path (BF16 storage / F32 compute) and exposes no precision switch. f32-widened loading remains available in `samples/NivaraInference`, which still owns `--precision` for benchmarking/parity work.

## Proposed changes

- `samples/NivaraChat/README.md`: remove `--precision f32|bf16` from the SmolLM and Qwen option lists; update tested-example precision labels to `bf16 (native)` (or drop the column).
- No changes to `SmollmMode.cs`, `QwenMode.cs`, or `Program.cs` — they already implement the intended native path.
- Optional wording touch-up in `docs/SMOLLM.md`, `docs/QWEN.md`, `docs/SILK.md` (already state no flag).

## Blast radius

- Documentation only. No source/test changes. No impact on `InferenceCliArgumentTests` or DocsCoveragePins counts (no new/removed docs files or citations).

## Verification steps

- `dotnet build samples/NivaraChat/NivaraChat.csproj -c Release`
- `grep -n "precision" samples/NivaraChat/README.md` (no NivaraChat option claim)
- `--smollm --help` / `--qwen --help` (already no `--precision` line)
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits

1. `docs: plan #1265 in TODO.md`
2. `docs: drop --precision claim from NivaraChat README`
3. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none created during this work)
