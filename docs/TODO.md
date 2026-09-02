# Plan: Add --smollm to NivaraChat PrintUsage (Issue #377)

## Problem

`PrintUsage()` in `samples/NivaraChat/Program.cs` lists all CLI modes but omits `--smollm`.
The dispatcher (line 78-79) and interactive menu (line 117-119) already handle `--smollm`,
so this is a pure documentation/UX gap — CLI users discovering options via `--help` will
not learn the mode exists.

## Proposed Changes

Single file: `samples/NivaraChat/Program.cs`, method `PrintUsage()` (lines 154-182).

### Change 1 — Add `--smollm` to the modes list

Insert after the `--tinyshakespeare` line (line 171):

```
  --smollm              SmolLM: serve the pretrained SmolLM-135M-Instruct causal LM as IChatClient (see --smollm --help)
```

Style mirrors the `--tinyshakespeare` line: mode name + description + `--help` pointer.

### Change 2 — Add smollm options line

Insert after the `--tinyshakespeare options:` block (after line 181):

```
  --smollm options: --model-dir --precision --text --max-new-tokens --help
```

These are the options parsed by `SmollmMode.ParseArgs()` that aren't global options.
`--text` and `--max-new-tokens` are listed for discoverability (same pattern as
`--tinyshakespeare options` listing overlapping global options).

## Verification

1. `dotnet build samples/NivaraChat/NivaraChat.csproj` — must succeed.
2. Visual inspection: run the built exe with no args and confirm `--smollm` appears in output.

## Planned Commits

1. `docs: plan #377 in TODO.md` — this file.
2. `Add --smollm mode and options to NivaraChat PrintUsage` — the actual fix.

## Blast Radius

- **Files changed:** `samples/NivaraChat/Program.cs` only (2 insertions in `PrintUsage()`).
- **Downstream callers:** none — `PrintUsage()` is only called from the `default:` switch arm.
- **Tests:** no automated tests for CLI usage text output; verification is manual (build + visual).
- **Risk:** trivial — pure string literal additions, no behavioral change.

## GitHub Issues Log

- [ ] #377 — NivaraChat: PrintUsage omits the --smollm mode (pre-existing)
