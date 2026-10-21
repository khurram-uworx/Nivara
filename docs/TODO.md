# Plan: Issue #760 — Silk `--precision bf16` wiring + BF16 conv-kernel decision

## Problem

PR #756 added BF16 adapter constructors across all Silk runners, but the sample CLI still hard-rejects
`--silk --precision bf16` for every model (`--silk is F32-only …`). The adapters are therefore unreachable
from the CLI. Separately, there are no true BF16 conv2d kernels in `SilkLauncher`/`SilkKernels`; the vision
adapters widen BF16→F32. This plan wires the CLI to the adapters and records the conv-kernel decision.

## Decision (conv kernels)

Keep the **widening adapters** for vision/conv models (MobileNetV2, ResNet-18). True device-side BF16
conv2d kernels are a large, separate kernel effort and are deferred. Only SmolLM and Qwen already have
true BF16 paths (`bf16Weights = true`); those are wired to use BFloat16 tensors so the BF16 kernels run.

## Proposed changes

### A. Session `Open` overloads (`src/Nivara.Extensions/Silk/`)
- `SilkModernBertSession`: add `Open(ModernBertConfig, Dictionary<string,(BFloat16[],int[])>)`.
- `SilkLayaSession`: add `Open(ModernBertConfig, Dictionary<string,(BFloat16[],int[])>, int headLayerCount)`.

### B. CLI guards (`samples/NivaraInference/Program.cs`)
- `if (useSilk && (bf16 || fp16))` → `if (useSilk && fp16)` (allow bf16; still reject fp16 — no fp16 kernels).
- Adjust `ConsumesNarrow` guard: `if ((bf16 || fp16) && !(useSilk && bf16) && !ConsumesNarrow(...))`.

### C. Dispatch routing (`samples/NivaraInference/Program.cs`)
- Vision / BERT-family / ModernBERT / Laya (widening adapters): when `useSilk && bf16`, widen `tensorsBf16`
  to a float tensor dictionary and pass to the existing float Silk functions.
- SmolLM: add BFloat16 overloads of `RunSmolLMSilk`/`BenchmarkSmolLMSilk` (and synthetic path) that open
  the BFloat16 session.
- Qwen: add BFloat16 overloads of `RunToolsSilk`/`RunPlainSilk`/`RunBenchmarkSilk`/`RunDistillSilk`.

### D. Tests (`tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`)
- Retarget the `Silk*Precision_Rejected` cases to fp16 (or replace with accepted-mode cases).
- Add `--silk --precision bf16` entries to the reach-model-file-lookup gate.

### E. Docs
- `samples/NivaraInference/README.md`: Silk precision note (BF16 supported, fp16 rejected).

## Verification steps
1. `dotnet build Nivara.slnx`
2. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
3. Targeted: `InferenceCliArgumentTests`
4. Confirm `--silk --precision bf16` reaches "Model file not found" for all Silk models.

## Planned commits
1. `docs: plan issue #760 in TODO.md`
2. `feat(Silk): add BF16 Open overloads for ModernBert/Laya sessions`
3. `feat(NivaraInference): allow --silk --precision bf16 (reject fp16 only)`
4. `feat(NivaraInference): route Silk bf16 dispatch to BF16/widening paths`
5. `test(NivaraInference): update CLI precision gates for Silk bf16`
6. `docs: update Silk precision note in README`
7. `docs: remove TODO.md — plan executed`

## Blast radius
- `src/Nivara.Extensions/Silk/SilkModernBertSession.cs`, `SilkLayaSession.cs` (additive overloads).
- `samples/NivaraInference/Program.cs` (guards + dispatch + helper overloads).
- `samples/NivaraInference/ModernBert.cs`, `Laya.cs`, `Qwen.cs`, `SmolLM` (via Program.cs functions).
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`.

## GitHub issues log
- #760 — sample CLI wiring + vision-model BF16 conv-kernel follow-up (this issue).
