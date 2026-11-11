# Plan: issue #1103 — BF16 training lane for BERT/Laya (moments in FP32)

## Problem

Issue #1103 tracks the BF16 training lane. Sub-issues #1133/#1134/#1135 are closed:
DistilBERT true BF16 path, BF16 backward GEMMs, and parity/smoke gates exist. Remaining scope:
1. BERT-family (BERT/MiniLM) BF16 training parity (gated by DistilBERT gates).
2. Laya head training on Silk (future, large).
3. Docs status update.

## Scope for this branch (Phase 1 + docs fold)

- Add BERT/MiniLM BF16 parity + smoke tests mirroring the DistilBERT ones.
- Fold Phase 2 (Laya head trainer design) and Phase 3 (docs closeout) into `docs/LAYA-HEAD-TRAINING.md` so the design is captured.
- Update `docs/SILK.md` narrow-precision section to state BERT/MiniLM BF16 training availability.

## Proposed changes

1. `tests/Nivara.Tests/Gpu/SilkBertBf16ParityTests.cs` — new test class, `BertGpuNaming.Bert`, tiny MiniLM shape, forward logits + gradients within narrow bound.
2. `tests/Nivara.Tests/Gpu/SilkBertBf16SmokeTests.cs` — loss decreases + F32 checkpoint export.
3. `docs/LAYA-HEAD-TRAINING.md` — design notes for a future Silk Laya head trainer.
4. `docs/SILK.md` — status wording update.

## Blast radius

- New test files only under `tests/Nivara.Tests/Gpu/`; no production code changes.
- Docs files only. No change to `DocsCoveragePins` expected (no new gated snippets/citations beyond existing style; verify).

## Verification

- `dotnet build Nivara.slnx`
- Targeted Gpu tests (requires device) — run manually/CI.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits

1. docs: plan #1103 in TODO.md
2. test: add BERT-family BF16 parity gate
3. test: add BERT-family BF16 smoke test
4. docs: fold Laya head training design into LAYA-HEAD-TRAINING.md
5. docs: update SILK.md BF16 training status
6. docs: remove TODO.md — plan executed

## GitHub issues log

- (none yet; create if deferred work discovered)
