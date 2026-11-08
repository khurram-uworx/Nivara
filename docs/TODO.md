# Plan: Silk narrow precision — wire ModernBERT/Laya sample routes to bf16/fp16 (#1040)

## Problem

- `samples/NivaraInference/Program.cs` still widens bf16 weights (`WidenBFloat16Tensors`) for `modernbert`/`laya` `--silk` routes.
- `ModernBert.RunGpu` / `Laya.RunGpu` accept only `Dictionary<string,(float[] Data,int[] Shape)>`, so native bf16/fp16 Silk sessions cannot be driven through them.
- No ModernBERT/Laya narrow-precision parity tests analogous to `SilkQwenBf16ParityTests`.

Backend support already exists: `SilkModernBertGpuRunner` (bf16/fp16 constructors), `SilkModernBertSession.Open` (BFloat16/Half/float), `SilkLayaSession.Open` (same).

## Proposed changes

1. Program.cs: restore `silkNarrowModel` (include `modernbert`, `laya`) for fp16 gating; for `--silk --precision bf16|fp16`, pass `tensorsBf16`/`tensorsHalf` straight through (no widening).
2. ModernBert samples: add `RunGpu` (and benchmark where useful) overloads for BFloat16/Half that open `SilkModernBertSession` directly with native tensors.
3. Laya samples: add `RunGpu` overloads for BFloat16/Half that call `SilkLayaSession.Open` directly.
4. Parity tests: tiny fused-weight state dicts (Wqkv/Wo/Wi keys; `encoder.` prefix for Laya), convert F32→BFloat16/Half, compare Silk sessions for argmax and bounded max-abs difference.
5. DocsCoveragePins: revise if docs/citations added.

## Planned commits

- docs: plan narrow ModernBERT/Laya Silk precision in TODO.md
- feat(silk): wire bf16/fp16 CLI routing for modernbert/laya
- feat(sample): narrow-aware ModernBert.RunGpu overloads
- feat(sample): narrow-aware Laya.RunGpu overloads
- test: ModernBERT/Laya narrow Silk parity vs F32
- (optional) docs/citation pins touch-ups

## Blast radius

- Samples: `Program.cs`, `ModernBert.cs`, `Laya.cs`.
- Extension API used (no Extension code changes expected): `SilkModernBertSession`, `SilkLayaSession`.
- Tests: new Gpu-category tests; may require model state dict synthesis.

## GitHub issues log

- #1040 — this plan
