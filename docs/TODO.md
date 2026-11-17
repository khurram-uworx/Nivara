# Plan: Laya environment-gated skips (issue #1229)

## Problem

`LayaGoldenPromptTests` and `LayaNativeCheckpointTests` skip when the (gitignored, large) Laya inputs are absent. Their current remarks are vague and do not state the intended fate, so the skips read as "missing coverage". The wheel gate (`--inference-compare laya`) is already the authoritative parity check and runs where the wheel + tokenizer are provisioned.

## Decision

Option 2: both fixtures are **local / provisioned-machine only**. They are not expected to run in CI; their absence there is not a coverage gap. No reduced checkpoint subset and no new `.bin` parity fixture is shipped. The wheel gate remains authoritative.

## Proposed changes

1. `tests/Nivara.Tests/AutoDiff/LayaGoldenPromptTests.cs` — rewrite the `<remarks>` paragraph to state the decision, skip condition, wheel-gate authority, download step, and that `LayaPromptBuilderTests` still runs in CI.
2. `tests/Nivara.Tests/Inference/LayaNativeCheckpointTests.cs` — add a class-level `<summary>`/`<remarks>` stating the same decision (requires `model.safetensors` + `encoder/config.json`).
3. `docs/LAYA.md` — add `### Environment-gated tests (issue #1229)` under `## Verification` documenting the decision, inputs, wheel gate, download step, and always-on coverage.

No `.gitignore` exceptions, no new tests, no category changes, no fenced code blocks, no new docs files.

## Verification steps

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (snippet + coverage gate)
- Confirm both Laya fixtures still skip cleanly when inputs are absent.

## Planned commits

1. `docs: plan #1229 in TODO.md`
2. `docs/test: document Laya environment-gated skips as local/provisioned-only`
3. `docs: remove TODO.md — plan executed`

## Blast radius

Documentation + XML remarks only. No source/test-logic changes. DocsCoverage pins should not move (no new `.md`, no `File.cs:NN` citations, no fenced csharp blocks).

## GitHub issues log

- #1229 — Laya: decide fate of environment-gated skips (golden prompt + native checkpoint tests)