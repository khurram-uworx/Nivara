# Plan: Issue #452 — SmolLM F32/BF16 parity measurement

## Problem

`smollm generate` reports F32 as worse than BF16 (25/32 tokens, cosine 0.243 vs BF16 22/32, 0.94).
Investigation shows the F32 path is actually exact against an F32 PyTorch reference (32/32, cosine 0.9999999).
The low F32 cosine is a cross-dtype, cross-sequence artifact: the default reference fixture is BF16
PyTorch, while the run is F32, and final-position logits are computed on each side's own generated stream.
Once greedy decoding diverges (token 25), the two logits are for different inputs.

## Proposed changes

1. `samples/NivaraInference/Python/smollm_generate_reference.py`: add `--dtype`-suffixed output filenames
   (`compare_smollm_py_{dtype}.bin`, `compare_smollm_logits_py_{dtype}.bin`), keep legacy names as BF16 default.
2. `samples/NivaraInference/Program.cs` (`ConsumeGenerated`): select fixture matching `--precision`;
   compute final-position logits on the PyTorch reference token stream when available; report both
   greedy-stream agreement and reference-sequence logits parity. Add cosine >= 0.99 sanity gate.
3. `docs/SMOLLM.md`, `docs/BFLOAT16.md`: update verification table with corrected per-dtype metrics
   and explain the prior 0.24 cosine was a cross-sequence artifact.

## Verification

- Build sample Release.
- Regenerate F32 and BF16 fixtures.
- Run `smollm generate --precision f32` and `--precision bf16`; expect high cosine on common sequence.
- Targeted: existing smollm tests if any.

## Planned commits

1. docs: plan #452 in TODO.md
2. feat(inference): dtype-aware SmolLM reference fixtures
3. fix(inference): compute final logits on reference token stream
4. docs: correct SmolLM F32/BF16 parity numbers
5. docs: remove TODO.md — plan executed

## GitHub issues log

- (none yet)
