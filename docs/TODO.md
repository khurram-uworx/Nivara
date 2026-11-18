# Plan: Align Silk trainer parity gates (issue #1305)

## Problem
Parity gates asymmetric between SilkDistilBertTrainer and SilkLayaHeadTrainer, and between F32 vs BF16 comparisons:
1. Laya BF16 gate checks logits only; DistilBert checks logits + per-param gradients.
2. Laya F32 gate checks forward logits only vs CPU LayaDecisionHead; backward path runs+reloads with no CPU gradient comparison.
3. Tolerance expressions differ: DistilBert uses `1e-2 * (1 + |reference|)`; Laya uses `1e-2 * scale` with `scale = 1 + |logit|`.

## Proposed changes
- Add `Gradients_TrueBf16_MatchF32Trainer` test to `SilkLayaHeadTrainerBf16ParityTests` mirroring DistilBert: export gradients from both trainers, assert per-key, per-element `1e-2 * (1 + |reference|)`.
- Standardize Laya BF16 forward tolerance expression to `1e-2f * (1f + MathF.Abs(...))`.
- Keep F32 run+reload gate; document Laya gradient correctness pinned by BF16-vs-F32 gate + F32 forward gate.

## Blast radius
- tests/Nivara.Tests/Gpu/SilkLayaHeadTrainerBf16ParityTests.cs (only). No source changes.

## Verification
- Build Nivara.Tests.
- Targeted test run (GPU-gated; skipped without OpenCL).

## Planned commits
1. docs: plan 1305 in TODO.md
2. test: align Laya BF16 parity gate tolerance + add gradient parity test

## GitHub issues log
- (none yet)
