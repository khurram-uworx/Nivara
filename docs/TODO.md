# Plan: Close issue #1273 (NivaraChat Llama training demo) as out of scope

Status: plan, executing on branch `khurram/1273`.

## Problem

Issue #1273 asks for a Llama (SmolLM/Qwen) training demo in the NivaraChat sample.
NivaraChat's SmolLM/Qwen paths are inference-only by design; training/fine-tuning
belongs in NivaraInference (backlog #1283 generic Silk trainer, #1284 LLM fine-tuning;
design note docs/plan/LORA.md). NivaraChat already documents this in ARCHITECTURE.md
("Fine-tuning/LoRA ... out of scope"; "SmolLM/Qwen import weights from safetensors and
never train in Nivara"). --tinyshakespeare remains the in-sample training demo.

Acceptance criteria for #1273: either a working training example or an explicit README
statement that Llama training is out of sample scope with a pointer to --tinyshakespeare
and core AutoDiff docs. We take the second path, and additionally point to NivaraInference.

## Proposed changes

1. `samples/NivaraChat/README.md` — add one explicit statement near the SmolLM section
   that SmolLM/Qwen are inference-only; Llama training/fine-tuning lives in
   NivaraInference (#1283/#1284, docs/plan/LORA.md); pointer to --tinyshakespeare and
   docs/AUTODIFF.md.
2. `samples/NivaraChat/ARCHITECTURE.md` — refine the "Not doing" bullet to point to
   NivaraInference/#1283/#1284/LORA.md.
3. `docs/plan/NIVARACHAT-SILK-GAPS.md` — mark Gap S5 as closed/scope-decided.
4. Close GitHub issue #1273 with a comment recording the scope decision.

## Verification

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` (docs-only change, should be clean).
- Docs snippet gate `EveryGatedBlock_CompilesWithoutErrors` unaffected (no new fenced blocks).
- DocsCoveragePins: no new docs files / citations / ADRs → no pin changes expected; verify.

## Planned commits

1. `docs: plan #1273 closure in TODO.md`
2. `docs: state Llama training out of NivaraChat scope in README`
3. `docs: refine ARCHITECTURE.md and NIVARACHAT-SILK-GAPS.md for #1273`
4. `docs: remove TODO.md — plan executed`

## Blast radius

Docs-only (README, ARCHITECTURE.md, plan doc). No source/test changes. No citation count
changes (no `File.cs:NN` citations added). No ADR changes. Low risk.

## GitHub issues log

- [x] #1273 — NivaraChat: no training demo for Llama (closing as out of scope)
- (tracked, open) #1283 — NivaraInference generic Silk trainer
- (tracked, open) #1284 — NivaraInference LLM fine-tuning
