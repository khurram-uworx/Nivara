# Plan issue #1044 — Docs: Silk narrow-precision precision statements

## Problem
Issue #1019 landed selected-weight `--precision bf16` storage for the Silk backend (ModernBERT/Laya/Qwen/SmolLM
use the `*_bf16` kernels; tokEmb/LM-head or QKV matrices and attention/FFN weights stay bf16 while activations
remain f32). The user-facing docs still over-generalise the Silk path as "F32 only" and describe
`--silk --precision bf16` only as "weights widen on device", which no longer reflects the implementation.
fp16 stays explicitly rejected at the sample CLI (measured 0.97–1.07×, no speedup); that status must be stated
clearly.

## Blast radius
Docs only:
- `docs/SILK.md` (Part 2 scope, `--silk` wiring/precision text, open-work table)
- `docs/ACCELERATION.md` (scope, "why this route", §1.5 precision matrix, open work)
- `samples/NivaraInference/README.md` (argument-validation bullet, usage line)
No production symbols change. `DocsCoveragePins` unaffected unless citations are added; none planned.

## Proposed changes
See conversation plan. Three wording updates, scoped to Silk narrow-precision status:
1. Tier A (device bf16 storage): ModernBERT/Laya/Qwen/SmolLM `--precision bf16`.
2. Tier B ( bf16 on disk, widened to f32 before upload): minilm/distilbert/distilbert_sst.
3. fp16 rejected at sample CLI (no measured speedup). Fine-tuning still F32-only.

## Commit list (planned)
1. `docs: plan issue #1044 Silk narrow-precision doc updates in TODO.md`
2. `docs: update Silk.md narrow-precision section`
3. `docs: update ACCELERATION.md precision matrix`
4. `docs: update NivaraInference README precision line`
5. `docs: remove TODO.md - plan executed`

## Verification
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet test tests/Nivara.Tests -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (ask first)
- targeted CLI tests stay green (no code change)

## GitHub issues log
- #1044 — Docs: update Silk narrow-precision precision statements (follow-up #1019)
- #1019 — Silk/OpenCL: add narrow-precision (bf16/half) translation unit (parent)
