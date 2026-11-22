# Plan: Fill gaps from #1419 — adapter demo completeness

## Problem

Issue #1419 added CPU-path `--adapter` support to NivaraChat Qwen/SmolLM. Several gaps remain:
1. `docs/QWEN.md` and `docs/SMOLLM.md` don't mention `--adapter`.
2. No demo path exercises `MergeAllAdapters()` (baking LoRA into base weights for deployment).
3. NivaraChat interactive menu doesn't offer Qwen/SmolLM/adapter options.
4. (Out of scope: NivaraInference adapter support — benchmarking tool, not showcase.)

## Proposed changes

### 1. Doc updates
- `docs/QWEN.md`: add `--adapter` to the option list and a short subsection.
- `docs/SMOLLM.md`: same.

### 2. MergeAllAdapters demo path
- Add `--merge-adapter <path>` flag to `--qwen` and `--smollm` modes.
- On CPU path: load model, apply adapter, call `MergeAllAdapters()`, then run inference with merged weights.
- Print status showing adapter was merged.

### 3. Interactive menu
- Add menu options 6 (Qwen) and 7 (SmolLM) to `RunInteractiveMenu`.
- Prompt for adapter path optionally.

## Planned commits

1. `docs: plan gaps in TODO.md`
2. `docs: add --adapter to QWEN.md and SMOLLM.md`
3. `feat: add --merge-adapter demo path to QwenMode and SmollmMode`
4. `feat: add Qwen/SmolLM options to interactive menu`
5. `docs: remove TODO.md — plan executed`

## GitHub issues log

(none yet)
## Blast radius

- `docs/QWEN.md`, `docs/SMOLLM.md` — doc only.
- `samples/NivaraChat/Qwen/QwenMode.cs` — new `--merge-adapter` flag.
- `samples/NivaraChat/SmolLM/SmollmMode.cs` — new `--merge-adapter` flag.
- `samples/NivaraChat/Program.cs` — interactive menu + help text.
- RAG modes — NOT touched.