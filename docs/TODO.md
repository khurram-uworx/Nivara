# Silk: extend device-side BF16 weights to remaining models + sample CLI wiring (#756)

## Problem

Issue #748 landed the device-side BF16 weight path for `SilkQwenGpuRunner` only: widened (F32) norm/embedding weights, raw-ushort BF16 matrices uploaded through `SilkRuntime.Upload(ReadOnlySpan<ushort>)`, kernel variants `gemm_row4_bf16`, `rms_norm_gemv_bf16`, `gemm_gemv_add_bf16`, `mlp_block_bf16`, plus `SilkQwenSession` overloads and `SilkQwenBf16ParityTests`. Every other Silk runner still consumes only F32 weights, and samples reject `--precision bf16` on every `--silk` path (`--silk is F32-only`). Issue #756 tracks porting the same construction/dispatch pattern to `SilkSmolLM`, `SilkBertEncoder`, `SilkModernBert`, `SilkLayaHead`, `SilkMobileNetV2`, `SilkResNet18`, plus CLI wiring and parity tests.

## Proposed changes

For each runner, port the Qwen pattern:

1. Add `readonly bool bf16Weights` to base class; default false in the F32 constructor.
2. Add a partial-class constructor overload taking
   `Dictionary<string, (BFloat16[] Data, int[] Shape)>` in `<Runner>.Bf16.cs`,
   setting `bf16Weights = true`, widening only activation-side weights (norms, embeddings, biases),
   uploading pre-transposed/packed matrices as raw `ushort` via `UploadTBf16`.
3. In the base class, make linear/matvec/MLP dispatch conditional via
   `bf16Weights`, mirroring `SilkQwenGpuRunner.Gemm/RmsNormGemv/MlpBlock`.
4. Add `LayerWeights` BF16 construction helpers (copy Qwen's `UploadFloat`/`UploadTBf16`).
5. Add session constructor/Open overloads where a session class exists.
6. Add a GPU-guarded parity test (`tests/Nivara.Tests/<Model>/Silk<...>Bf16ParityTests.cs`)
   comparing BF16 argmax and logits tolerance to the F32 path.

Also fix a discovered prerequisite in Qwen: its `bf16Weights`-aware dispatch wrappers currently
recurse into themselves on the F32 branch instead of calling `SilkLauncher.*`. Correct to
launcher calls before (or as) ports begin (Decision D1).

Sample/CLI wiring (`samples/NivaraInference/*`, `samples/NivaraChat` Qwen/SmolLM) plus adjusting
`InferenceCliArgumentTests` expectations for supported models. Vision models need explicit scope
decision (see D2).

## Grounding / red flags (Resistance G1)

- BF16 kernel coverage is limited to `gemm_row4_bf16`, `rms_norm_gemv_bf16`,
  `gemm_gemv_add_bf16`, `mlp_block_bf16`. Runners whose hot path is
  convolution/pooling only (MobileNetV2, ResNet18) gain little without new BF16
  conv kernels.
- The recursion bug in Qwen wrappers must be fixed; otherwise any port that copies the
  pattern ships a stack-overflow-prone F32 path.
- Sample tests pin `--silk is F32-only`; changing supported models requires changing
  those tests' expectations deliberately (not to make a red test pass, but because the
  contract changed).
- `docs/` count pin `DocsCoveragePins` is unaffected unless new gated docs are added.

## Blast radius

- `src/Nivara.Extensions/Silk/*GpuRunner.cs` and partial `.Bf16.cs` files (6 runners).
- `SilkQwenGpuRunner.cs` wrapper methods.
- `src/Nivara.Extensions/Silk/*Session.cs` (Qwen, SmolLM, Bert, ModernBert, Laya session(s)).
- `samples/NivaraInference/*.cs`, sample CLI parsers.
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`.
- New tests under `tests/Nivara.Tests/`.

## Planned commits (logical units; adjust as needed)

1. docs: plan (#756) — this file.
2. fix(SilkQwen): make bf16-aware dispatch wrappers call launchers on F32 path.
3. feat(SilkSmolLM): device-side BF16 weight path + session overload + parity test.
4. feat(SilkBertEncoder): device-side BF16 weight path + session/runner parity test.
5. feat(SilkModernBert): device-side BF16 weight path + parity test.
6. feat(SilkLayaHead): device-side BF16 weight path + parity test.
7. (pending decision) vision models: conv-kernel or widen-fallback scope.
8. feat(samples): `--silk --precision bf16` wiring for supported narrow models.
9. test(inference-cli): update F32-only expectations to reflect supported BF16 models.
10. docs: final docs/coverage reconciliation.

## Decisions required from human (Resistance G1)

- D1: confirm fixing Qwen recursion before porting.
- D2: which vision models (MobileNetV2, ResNet18) should receive a real BF16 device path
  vs being explicitly excluded as "conv kernels not BF16 yet". Recommend exclusion +
  tracked issue.

## GitHub issues log

- #760 — sample CLI wiring + vision-model BF16 conv-kernel follow-up (created while working on 756).
