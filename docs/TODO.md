# Plan: Promote SmolLM/Qwen chat templates to Nivara.Samples (issue #1372)

## Problem
`SmollmChatTemplate` and `QwenChatTemplate` (plus `QwenIds`, `QwenJson`) are `internal static`
renderers in `NivaraChat`. `NivaraFineTuning` is a separate assembly and cannot reach them.
Promote into `samples/Nivara.Samples` so both `NivaraChat` and `NivaraFineTuning` share one copy,
byte-exact rendering preserved (pinned by `QwenChatTemplateTests`).

## Proposed changes
1. Add `samples/Nivara.Samples/SmollmChatTemplate.cs` (namespace `Nivara.Samples`, public).
2. Add `samples/Nivara.Samples/QwenChatTemplate.cs` (namespace `Nivara.Samples`, public
   `QwenIds`/`QwenChatTemplate`/`QwenJson`).
3. Add backward-compat forwarding static classes under legacy namespaces
   (`NivaraChat.SmolLM`, `NivaraChat.Qwen`) so unchanged tests/call sites still resolve.
4. Update `NivaraChat` call sites to use `Nivara.Samples` types.
5. Remove duplicated definitions from `NivaraChat` (delete original files).
6. Keep templates `public` (cross-assembly reuse by NivaraFineTuning).

## Verification
- Build `Nivara.slnx`.
- Run `QwenChatTemplateTests` (and targeted SmolLM rendering tests) with human approval.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits
1. docs: plan #1372 in TODO.md
2. feat(samples): promote SmollmChatTemplate and QwenChatTemplate to Nivara.Samples
3. refactor(nivarachat): update call sites to promoted templates
4. chore: remove duplicated template definitions from NivaraChat
5. docs: remove TODO.md — plan executed

## Blast radius
- `samples/NivaraChat/SmolLM/*` and `samples/NivaraChat/Qwen/*` clients and modes.
- `tests/Nivara.Tests/Qwen/QwenChatTemplateTests.cs` (namespace resolution).
- `Nivara.Samples` public API surface gains 3 types.

## GitHub issues log
- (none yet)
