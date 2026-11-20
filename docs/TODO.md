# Plan: issue #1361 — TinyShakespeare Silk: update CLI help and docs/SILK.md

## Problem
`--silk` now trains the TinyShakespeare transformer on device (PR #1359), but the CLI help
and Silk docs still describe it as generation-only.

## Proposed changes
1. `samples/NivaraChat/Transformer/TransformerMode.cs` — `PrintHelp()`: update the `--silk`
   line to state it also trains on device (f32 only).
2. `samples/NivaraChat/ARCHITECTURE.md` — mirror the same help-text update (consistency).
3. `docs/SILK.md` — add a section `13g. TinyShakespeare trainer (SilkTinyShakespeareTrainer, #1359)`
   covering the trainer and the new `causal_attention_backward_prep` kernel, matching the style of
   `13e. SilkDistilBertTrainer`. Optionally add a bullet to `13d. Wiring coverage`.

## Blast radius
- Only documentation/help text. No public API or runtime behavior changes.
- Docs coverage pins: no new citations or fenced code blocks, so no pin changes expected.
  Verify with Docs namespace tests after editing.

## Verification
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Docs namespace tests: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
- `dotnet build Nivara.slnx -c Release`

## Planned commits
1. docs: plan #1361 in TODO.md
2. docs: state --silk trains on device in TinyShakespeare CLI help
3. docs: add SilkTinyShakespeareTrainer and causal attention-backward kernel section

## GitHub issues log
- none created yet
