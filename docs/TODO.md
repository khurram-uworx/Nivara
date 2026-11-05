# Plan: Issue #933 — NivaraAIFunctionsTests fails (sentiment_model.json v1 vs v2)

## Problem
`NivaraAIFunctionsTests.SetUp` loads `samples/data/nivarachat/sentiment_model.json` via `ModelSerializer.Load`, which rejects the file because its `Format` is `nivara-ss-v1` while the serializer (post ADR-001) requires `nivara-ss-v2`. All parameter entries carry `HasNulls: false` / `NullMask: null`, so the weights are unaffected and only the format marker (+ stale null-mask fields) need migrating.

## Proposed changes
1. Migrate `samples/data/nivarachat/sentiment_model.json` to v2 (marker + remove `HasNulls`/`NullMask`).
2. Migrate the other three v1 assets in the same dir for consistency:
   - `agents_validator_model.json`
   - `entity_model.json` (loaded by NivaraChat sample)
   - `validator_model.json`
3. No source code changes and no serializer behavior change.

## Verification steps
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~NivaraAIFunctionsTests"`
- `dotnet build samples/NivaraChat/NivaraChat.csproj -c Release`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits
1. `docs: plan #933 model-format migration in TODO.md`
2. `data(nivarachat): migrate sentimental model assets to nivara-ss-v2`
3. `docs: remove TODO.md — plan executed`

## Blast radius
- Data files only. `ModelSerializer` unchanged (still rejects v1). Tests and sample app benefit. No downstream code.

## GitHub issues log
- (none created)
