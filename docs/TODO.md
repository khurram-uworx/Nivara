# Silk narrow precision: BertEncoder (MiniLM/DistilBERT) bf16/fp16 (#1041)

Follow-up to #1019. MiniLM/DistilBERT/DistilBERT-SST Silk routes still call
`WidenBFloat16Tensors` because `SilkBertEncoderRunner` and `SilkBertWeights` only
upload F32. This plan adds native bf16/fp16 constructor paths using the existing narrow
GEMM epilogue kernels (`gemm_row4_*_bf16`/`_f16`), keeps `Export()` widening
compatible, and adds parity tests.

## Problem

- `SilkBertEncoderRunner.Bf16.cs` only widens bf16 tensors to f32.
- `SilkBertEncoderRunner.cs` has no fp16 path and no narrow-aware kernel dispatch.
- `SilkBertWeights.cs` only accepts `float[]` tensors; all matrix GEMM weights are
  uploaded as f32, so narrow checkpoints cannot stay narrow on device.

## Proposed changes

1. `SilkRuntime.cs`: add `DownloadUshort(nint, int)` for widening matrices in `Export()`.
2. `SilkBertWeights.cs`:
   - Add `narrowBf16`/`narrowFp16` fields + `IsNarrowBf16`/`IsNarrowFp16` props.
   - Add `NarrowMatrix` flag to `SilkWeightSlot`.
   - Add constructor overloading accepting `Dictionary<string,(ushort[],int[])>`,
     `BertGpuNaming`, `trainable`, `isBf16`, `isFp16`.
   - Widen vectors/embeddings/norms/biases to f32; store GEMM matrices + classifier head
     as `ushort` buffers; pack q/k/v via `WeightLayout.QkvPackedNarrow`.
   - `Export()`/`ExportCore()` widen narrow matrices on readback.
3. `SilkBertEncoderRunner.cs`:
   - Add `narrowBf16`/`narrowFp16` fields populated from `SilkBertWeights`.
   - Add public constructor accepting `(ushort[], int[])` tensors plus flags.
   - `LaunchGemm` kernel selection via `ChooseKernel`.
4. Partial files:
   - `SilkBertEncoderRunner.Bf16.cs`: bf16 -> ushort conversion path.
   - `SilkBertEncoderRunner.Fp16.cs`: fp16 (Half) -> ushort conversion path.
5. Tests: `SilkBertNarrowParityTests.cs` with bf16/fp16 parity vs f32
   (MiniLM synthetic Bert-named state dict; DistilBERT via
   `DistilBertForSequenceClassification<float>.ToSilkTensors()`).

Matrices narrow, vectors stay f32. Embeddings stay f32 (existing `embed_sum` kernel is
f32-only; narrowing them is out of scope). Inference-only (`trainable=false`);

## Planned commits

1. docs: plan issue #1041 in TODO.md
2. feat(silk): add DownloadUshort helper to SilkRuntime
3. feat(silk): add narrow (bf16/fp16) constructor to SilkBertWeights
4. feat(silk): add narrow constructor and kernel selection to SilkBertEncoderRunner
5. feat(silk): add Bf16/Fp16 partial constructor files
6. test: add SilkBertNarrowParityTests
7. docs: remove TODO.md — plan executed (conditional on G2)

## GitHub issues log

- (none yet — deferred work discovered during implementation will be recorded here)

## Blast radius

- `src/Nivara.Extensions/Silk/SilkRuntime.cs` (one new method, no behavior change)
- `src/Nivara.Extensions/Silk/SilkBertWeights.cs` (new constructor; existing float path unchanged)
- `src/Nivara.Extensions/Silk/SilkBertEncoderRunner.cs` (new fields; internal constructor extended)
- `src/Nivara.Extensions/Silk/SilkBertEncoderRunner.Bf16.cs` (rewired)
- `src/Nivara.Extensions/Silk/SilkBertEncoderRunner.Fp16.cs` (new)
- `tests/Nivara.Tests/Gpu/SilkBertNarrowParityTests.cs` (new, GPU-gated)
- Does not touch ILGPU runner, trainer, or CPU models.

## Verification

- `dotnet build Nivara.slnx -c Release` (Extensions + tests)
- Targeted: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkBertNarrowParityTests"` (GPU host)
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
