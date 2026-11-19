# Plan: NivaraInference FP16 story (issue #1281)

## Problem
`--silk --precision fp16` is rejected globally in `samples/NivaraInference/Program.cs` even though Silk runners
for minilm/distilbert/distilbert_sst/modernbert/laya already implement fp16 (narrowF16) and parity tests pass.
CLI dispatch has no fp16 Silk branch for those models, so the runners are unreachable from the CLI.

## Proposed changes
- Phase 1: targeted guard + dispatch branches in Program.cs.
- Phase 2: fp16 Silk helper methods.
- Phase 3: test updates (InferenceCliArgumentTests).
- Phase 4: docs.
- Phase 5: follow-up issues for gaps (vision, smollm/qwen).

## Verification
- dotnet build Nivara.slnx
- targeted InferenceCliArgumentTests
- snippet gate if docs change

## Planned commits
1. docs: plan 1281 in TODO.md
2. feat: wire Silk fp16 CLI routing for BERT-family + ModernBERT + Laya
3. test: update InferenceCliArgumentTests for Silk fp16 acceptance
4. docs: update Silk/acceleration fp16 statements
5. docs: remove TODO.md

## GitHub issues log
- #1281 — this issue (in progress)
- #1282 — vision Silk bf16 shims (open, related)
- #1274 — NivaraChat classifiers (open, related)