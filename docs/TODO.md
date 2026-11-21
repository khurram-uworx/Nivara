# Delete stale NivaraInference Silk gaps umbrella plan

**Branch:** `khurram/inference`
**Base:** `main`

## Problem

`docs/plan/NIVARAINFERENCE-SILK-GAPS.md` is an umbrella plan written 2026-11-19 ("Status: plan, not yet
executed") consolidating issues #1279–#1284 and #1343. Every gap it tracks is now either resolved in
code or explicitly deferred into a living design note:

- N1 (Laya CPU training path) — resolved: `Laya.RunTrain` in `Laya.Train.cs`.
- N2 (FP16 story) — resolved: #1281 exposed `--silk --precision fp16` (Option B); runners are live.
- N3 (vision BF16 shims) — resolved: shim files removed (#1282).
- N4 (ModernBERT fine-tune) — resolved: `modernbert train-head` mode.
- N5 (SmolLM/Qwen training) — deferred by design; design fully captured in `docs/plan/LORA.md` §14.
- N6 (BF16 training in NivaraFineTuning) — resolved: `--precision bf16` (#1279).

The document is now stale (claims Silk FP16 is "rejected globally", which is false) and redundant.
Keeping it misleads readers. The NivaraChat counterpart was retired the same way once its gaps closed.

## Proposed changes

1. Delete `docs/plan/NIVARAINFERENCE-SILK-GAPS.md`.
2. Remove the now-dangling pointer to it in `docs/plan/LORA.md` (line ~252).
3. Track the (out-of-scope) user-facing doc inconsistency — several docs still say
   `--silk --precision fp16` is rejected, while the CLI now accepts it for five models — as a
   GitHub issue rather than bundling a large doc rewrite into this branch.

No source code, tests, or pins should change: `docs/plan/*` is excluded from the snippet scan and
`DocsCoveragePins` counts only top-level `docs/*.md`.

## Blast radius

- `docs/plan/NIVARAINFERENCE-SILK-GAPS.md` (delete)
- `docs/plan/LORA.md` (one reference line)
- No code, no tests, no pins expected to move.

## Verification steps

- `dotnet build Nivara.slnx` (confirms nothing references the deleted file at build time).
- Targeted docs tests: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` to confirm no pin/count regression.
- Confirm `git grep NIVARAINFERENCE-SILK-GAPS` returns only the (to-be-removed) LORA.md reference.

## Planned commits

1. `docs: plan deletion of stale NIVARAINFERENCE-SILK-GAPS umbrella in TODO.md`
2. `docs: remove stale NivaraInference Silk gaps umbrella plan`
3. `docs: drop dangling reference in LORA.md`
4. (if created) record fp16 doc-consistency issue in TODO.md issues log

## GitHub issues log

- [x] #1391 — user-facing docs still claim `--silk --precision fp16` is rejected (docs/SILK.md, BFLOAT16.md, HALF-F16.md, ACCELERATION.md, DISTELBERT.md, LAYA.md, QWEN.md); CLI now accepts it for minilm/distilbert/distilbert_sst/modernbert/laya. Tracked as follow-up (out of scope for this branch).

## Grounding (G1)

- Confirmed via file inspection: all six gaps resolved or deferred into LORA.md.
- Confirmed `docs/plan/*` excluded from snippet scan (`DocSnippetExtractor.IsScanned`); `DocsCoveragePins` counts only top-level `docs/*.md`.
- Only reference to the file is `docs/plan/LORA.md:252`. No code/test references.
