# Plan: Issue #457 — Conv2d grouped/depthwise forward parallelism

## Problem

Issue #457: Conv2d grouped/depthwise forward path is single-threaded. MobileNetV2
(1x3x224x224) takes ~731.7 ms vs PyTorch ~22.4 ms (~33x). The issue's cited
lines (625-655) are stale — they point at ConvInputGradGeneric (backward); the
actual forward grouped path is Forward's groups != 1 branch (~199-244).
ConvForwardKernel already uses TensorPrimitives.Dot (SIMD) via im2col; what is
missing is Parallel.For over groups and over im2col tiles.

## Proposed changes

- Conv2d.Forward grouped branch: parallelize over groups (gated), hoisting
  input/weight/bias spans into arrays so the delegate does not capture ref-structs.
- ConvForwardKernel: parallelize over im2col tiles (gated), renting scratch per tile.
- Leave ConvForward1x1, all backward kernels, layout, and grouped semantics unchanged
  (bit-identical parity).

## GitHub issues log

- (none yet)

## Verification

- build (dotnet build Nivara.slnx)
- targeted tests: Conv2dTests, DepthwiseSeparableConv2dTests (Release)
- format check (dotnet format whitespace --verify-no-changes)
- human-run MobileNetV2/ResNet-18 compare scripts (not agent-run)

## Planned commits

1. docs: plan #457 in TODO.md
2. perf(autodiff): parallelize Conv2d grouped forward over channel groups
3. perf(autodiff): parallelize ConvForwardKernel over im2col tiles
4. test: Conv2d parity under parallel paths (if needed)
5. docs: remove TODO.md — plan executed

## Blast radius

- src/Nivara/AutoDiff/Nn/Conv2d.cs (Forward grouped branch, ConvForwardKernel).
- Downstream: MobileNetV2 sample inference, DepthwiseSeparableConv2d, any grouped Conv2d user.
- Tests: Conv2dTests (6), DepthwiseSeparableConv2dTests (1).
- Backward untouched; training behavior unchanged.