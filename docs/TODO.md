# Plan: design note for TinyShakespeare transformer training on Silk (#1342)

**Branch:** khurram/1342 (off khurram/1341)
**Issue:** https://github.com/khurram-uworx/Nivara/issues/1342

## Problem

`--tinyshakespeare` training runs entirely on the CPU AutoDiff path
(`BatchedTransformer`, `ReverseGradOperations.BatchedMultiHeadAttention`, `Adam<T>`).
Silk is inference-only for this mode. With Silk now first-class, this should be tracked
explicitly rather than left as an implicit non-goal. Acceptance criteria: a design note
under `docs/plan/` scoping kernels, memory budget, precision lane, save/load round-trip,
parity gate, with explicit non-goals.

## Proposed changes

1. Create `docs/plan/NIVARACHAT-TINYSHAKESPEARE-SILK-TRAINING.md` containing the design note
   scoped from the CPU reference path and the existing Silk inference runner, plus the
   established training pattern in `SilkDistilBertTrainer`.
2. Update `docs/plan/NIVARACHAT-SILK-GAPS.md` Gap S6 to point at the new design note.
3. DocsCoveragePins: adding a docs file changes `ExpectedDocumentCount`; review and revise if needed.
   Adding no citations (no `File.cs:NN` citations in the new doc? It references files without
   line numbers, so no citation count change). Check.

## Verification steps

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Docs snippet gate `EveryGatedBlock_CompilesWithoutErrors` (no gated blocks added → unaffected)
- DocsCoverageTests (document count pin)

## Planned commits

1. `docs: plan TinyShakespeare Silk training design note in TODO.md`
2. `docs: add TinyShakespeare Silk training design note (#1342)`
3. `docs: link Gap S6 to new design note`
4. `docs: remove TODO.md — plan executed` (after G2)

## Blast radius

- New file under docs/plan/ only; one-line edit to NIVARACHAT-SILK-GAPS.md.
- DocsCoveragePins.ExpectedDocumentCount likely +1.

## GitHub issues log

- (none created yet; capture deferred work here if found)
