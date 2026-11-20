# Plan: Consolidate LLM fine-tuning design notes (#1273/#1283/#1284)

## Problem

The repo has three overlapping design notes for causal-LM training/fine-tuning:
- `docs/plan/LORA.md` (#1273) — LoRA vs RAG design note (NivaraChat).
- `docs/plan/GENERIC-SILK-TRAINER.md` (#1283) — Silk causal-LM/vision trainer scoping.
- `docs/plan/NIVARAINFERENCE-LLM-FINETUNING.md` was considered for #1284 but dropped as
  infeasible on CPU (full-param 135M+ models).

This is messy and duplicative: LORA.md already defers to #1284 and GENERIC-SILK duplicates
Silk scoping that LORA.md's non-goals only gesture at. #1273 is closed, so LORA.md is the
canonical home. Goal: one source of truth in `docs/plan/LORA.md`, folding the useful
Silk/causal-LM content from GENERIC-SILK-TRAINER.md, then deleting the latter.

## Proposed changes

1. Replace `docs/plan/LORA.md` with a consolidated version:
   - Keep all existing sections.
   - Staleness fixes: note chat templates are `internal static`; Silk status clarified;
     references updated.
   - Add §14 "Silk counterpart (future, from #1283)" consolidating trainer inventory,
     kernel gaps, tiers, precision lanes, parity gate, non-obvious points, open decisions,
     and relation to LoRA.
   - Update non-goals to drop the #1284 parenthetical and state full-param is out of scope.
   - Consolidation note in status line.
2. Delete `docs/plan/GENERIC-SILK-TRAINER.md`.
3. (Optional, low risk) Update `docs/plan/NIVARACHAT-SILK-GAPS.md` Gap S5 wording to
   reflect consolidation — already points at LORA.md; no change strictly required.

## Verification steps

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` (docs-only change;
  should be clean).
- Confirm no remaining references to `GENERIC-SILK-TRAINER` in docs.
- Confirm top-level docs count pins unaffected (changes are under docs/plan/, not docs/*.md).
- `git diff --stat` review.

## Planned commits

1. `docs: plan consolidation of LLM fine-tuning notes in TODO.md`
2. `docs: consolidate LLM fine-tuning design notes into LORA.md`
3. `docs: remove GENERIC-SILK-TRAINER.md (folded into LORA.md)`
4. `docs: remove TODO.md — plan executed`

## Blast radius

- Documentation only. No source/test changes. No docs-coverage pin changes (docs/plan/ is
  not in the top-level docs/*.md scan).
- Consumers of the deleted file: none found in docs (grep clean).

## GitHub issues log

- [ ] #1273 — LoRA design note (closed; LORA.md is its artifact)
- [ ] #1283 — generic Silk trainer (design note consolidated here)
- [ ] #1284 — LLM fine-tuning (covered by LORA.md; no separate note)
