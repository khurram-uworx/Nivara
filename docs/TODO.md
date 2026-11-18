# Plan: NivaraChat docs/README/CLI coverage for TinyShakespeare Silk + KV cache + top-p (#1315)

## Problem

After #1307–#1313, TinyShakespeare gained `--silk`, `--kv-cache`, and `--top-p` (and lost `--dropout`), but the sample docs and `--tinyshakespeare --help` still describe the old surface. Docs must match implementation; the help text must list every parsed flag.

## Proposed changes

### 1. `samples/NivaraChat/Transformer/TransformerMode.cs` — PrintHelp
Add the missing `--silk` line so `--tinyshakespeare --help` advertises all parsed flags.

### 2. `samples/NivaraChat/README.md`
- CLI table `--silk` row: include `--tinyshakespeare` (Silk path via `SilkTinyShakespeareSession`).
- TinyShakespeare subsection: add Tested-examples rows for `--silk`, `--kv-cache`, `--top-p`; extend `Uses:` line.

### 3. `samples/NivaraChat/ARCHITECTURE.md`
- CLI block: add `--kv-cache` and `--silk` lines.
- "Not doing": update stale bullets (KV cache now available; top-p now available).
- Decisions: append D-015 (KV cache), D-016 (Silk runner), D-017 (top-p).

### 4. Tests
`InferenceCliArgumentTests` does not cover `--tinyshakespeare`, so no edits required.

## Verification
- `dotnet build Nivara.slnx`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
- Manual: `dotnet run --project samples/NivaraChat -- --tinyshakespeare --help` lists all flags.

## Planned commits
1. docs: plan #1315 in TODO.md
2. fix: advertise --silk in --tinyshakespeare --help
3. docs(NivaraChat): update README TinyShakespeare + --silk coverage
4. docs(NivaraChat): update ARCHITECTURE CLI + decisions
5. docs: remove TODO.md — plan executed

## Blast radius
Low — docs + one help-text line. No runtime behavior change beyond help output.

## GitHub issues log
- [ ] #1315 — NivaraChat docs/README/CLI coverage for TinyShakespeare Silk + KV cache + top-p (parent #1270)