# Plan: --silk for MobileNetV2 and ResNet-18 (issue #630)

## Problem

`--silk` (and `--ilgpu`) do not run `mobilenet_v2` or `resnet18`. The vision models
are conv-dominated (depthwise separable convs in MobileNetV2), and `SilkKernels` has no
conv/batchnorm/pool kernels. No ILGPU runner exists to port; this is the first GPU
runner for the vision models.

## Non-goals

- No change to `src/Nivara` (issue #457 owns the CPU conv kernel).
- Do not add `mobilenet_v2`/`resnet18` to `SilkModes` until a runner exists.
- F32 only.
- Do not fold into `docs/plan/silk-modernbert-laya.md`.

## Proposed changes

### Slice 1 — kernels + launch helpers + probe
- Add OpenCL C kernels to `SilkKernels.Source` and `KernelNames`: `conv2d` (general K×K
  with stride/padding/groups), `batchnorm_eval`, `relu`, `relu6`, `maxpool2d`,
  `adaptive_avgpool2d`. Reuse existing `add` and `gemm_row4_bias`.
- Add typed launch helpers in `SilkLauncher`.
- Add a correctness probe (in `tests/Nivara.SimdProbe` or a temp harness) comparing
  against CPU `Conv2d`/`BatchNorm2d` on a tiny tensor.

### Slice 2 — MobileNetV2 runner
- `SilkMobileNetV2GpuRunner` uploading conv weights/bn stats, implementing forward,
  returning logits. Validate against CPU model on a sample image.

### Slice 3 — ResNet-18 runner
- `SilkResNet18GpuRunner` (7×7 stem, maxpool, strided shortcuts).

### Slice 4 — sample wiring + CLI
- Add `mobilenet_v2`/`resnet18` to `SilkModes`; dispatch branches; benchmark/compare.
- Update `InferenceCliArgumentTests` Silk model list.

### Slice 5 — tests + docs
- CLI arg gate updates, `docs/SILK.md`, `docs/VISION.md`.

## Verification

- Build `Nivara.slnx` in Release.
- Targeted: `InferenceCliArgumentTests`, new runner correctness (argmax + logit bound
  `|gpu-ref| <= 1e-3*(1+|ref|)`).
- Full suite only with human confirmation.

## Planned commits

1. docs: plan #630 in TODO.md
2. feat(silk): add conv/bn/pool/relu OpenCL kernels + launchers
3. feat(silk): add SilkMobileNetV2GpuRunner
4. feat(silk): add SilkResNet18GpuRunner
5. feat(sample): wire --silk for mobilenet_v2 and resnet18
6. test/docs: update CLI arg gate and docs for vision --silk

## Blast radius

- `src/Nivara.Extensions/Silk/SilkKernels.cs`, `SilkLauncher.cs` (kernel registry, source)
- new runner files in same dir
- `samples/NivaraInference/Program.cs` (SilkModes, dispatch)
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`
- docs

## GitHub issues log

- [ ] #630 — --silk for MobileNetV2 and ResNet-18 (this plan)
"