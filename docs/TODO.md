# Plan: #1018 — default-on widen + drop public toggle

## Problem

The widen-compute-narrow path for `Half`/`BFloat16` is gated behind
`NivaraPrimitives.UseWidenSimd`, default `false`. Production measurements show
~10x improvement for BF16 while F32 is bit-identical and correctness stays within
documented tolerances. Keeping the toggle public leaves users with a footgun (slow
narrow inference unless they know to enable it) and complicates sample/test code.

## Proposed changes

1. Library: remove public `NivaraPrimitives.UseWidenSimd`; default narrow path on;
   retain a hidden diagnostic opt-out (`Nivara.Primitives.DisableWidenSimd`).
2. `ShouldWiden`: remove toggle condition, keep hardware/narrow/length gates and
   hidden opt-out.
3. Samples: remove `--simd-widen`, automatic widen in narrow runs, `smollm ab`
   via hidden opt-out.
4. Tests: rewrite toggle-related tests around the hidden opt-out; update CLI tests.
5. Docs: update default-on wording and hidden opt-out note.

## Blast radius

- Public API break: `NivaraPrimitives.UseWidenSimd` removed.
- Behavior default changes for all narrow AutoDiff/column kernels.
- CLI/test surface loses `--simd-widen`.

## Planned commits

1. docs: plan #1018 in TODO.md
2. refactor: default widen path on; remove public UseWidenSimd
3. refactor: simplify ShouldWiden + KernelSelector wording
4. test: update widen/KernelSelector/Llama/CLI tests
5. sample: remove --simd-widen; use hidden opt-out for A/B
6. docs: mark widen as default; hidden opt-out
7. docs: remove TODO.md

## GitHub issues log

- [x] #1018 — widen default + remove flag
- Follow-ups: none planned

## Verification

- targeted widen diagnostics tests
- CLI argument tests
- Llama BF16 parity tests
- NivaraInference sample bf16 runs
- docs tests if doc wording/pins move