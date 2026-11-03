# Plan: Issue #872 — ONNX export: grouped and asymmetric convolution support

## Problem

Issue #871 brought ONNX round-trip parity for the convolution exports, but `Conv2d`
export must explicitly guarantee (and be covered by tests for) grouped convolution
(`groups > 1`) and unequal (asymmetric) padding (`paddingTop != paddingBottom` or
`paddingLeft != paddingRight`). A runtime mismatch in ONNX weight layout/ordering is
already fixed for groups, and per-side `pads` are passed in ONNX spatial order
(`[top, left, bottom, right]`). This plan adds explicit export-time validation so
unsupported/invalid configurations fail loudly, plus dedicated grouped and asymmetric
round-trip tests.

## Acceptance criteria

- Grouped Conv2d (`groups > 1`, channels divisible by groups) exports and runs in
  ONNX Runtime with results matching Nivara.
- Asymmetric padding exports correctly and matches Nivara.
- Invalid/unexportable Conv2d configurations (bad groups, non-divisible channels,
  invalid padding/kernel/stride) throw `NotSupportedException` (or the module's own
  argument validation) rather than silently producing a wrong graph.
- No regression in existing `OnnxExportConvTests`.

## Blast radius

- `src/Nivara.Extensions/Onnx/OnnxExporter.cs` — `Context<T>.Conv2d` only.
- `tests/Nivara.Tests/Onnx/OnnxExportConvTests.cs` — add round-trip tests and
  update the `Conv2dExportModel` test wrapper to accept groups / per-side padding.
- `src/Nivara/AutoDiff/Nn/Conv2d.cs` — no change; properties consumed are
  `InChannels`, `OutChannels`, `KernelSize`, `Stride`, `PaddingTop/Bottom/Left/Right`,
  `Groups`, `Bias`, `Weight`.

## Proposed changes

1. Hardening in `OnnxExporter.Context<T>.Conv2d`:
   - Validate groups divisibility up front:
     ```
     if (module.Groups <= 0
         || module.InChannels % module.Groups != 0
         || module.OutChannels % module.Groups != 0)
         throw new NotSupportedException("Conv2d export requires channels divisible by groups.");
     ```
   - Validate kernel/stride positivity and non-negative per-side padding.
   - Keep grouped weight shape `[OutChannels, InChannels/Groups, K, K]`.
   - Keep `pads = { PaddingTop, PaddingLeft, PaddingBottom, PaddingRight }`.
   - Keep `group = module.Groups`.
   - Keep output-shape propagation using per-side padding sums; throw
     `ArgumentOutOfRangeException`/`NotSupportedException` for non-positive output
     (instead of silently falling back to input size) to fail loudly.

2. Tests:
   - `Conv2dExport_Grouped_RoundTrip_MatchesNivara` with `groups = 2` (and a
     depthwise `groups = InChannels` case if supported).
   - `Conv2dExport_AsymmetricPadding_RoundTrip_MatchesNivara` covering unequal
     top/bottom/left/right, strided, bias + bias-free.
   - Update `Conv2dExportModel` to expose `groups` and per-side padding parameters.

## Verification

- `dotnet build Nivara.slnx -c Release`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~OnnxExportConvTests"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits
1. `docs: plan #872 ONNX grouped/asymmetric conv support in TODO.md`
2. `fix(onnx): validate Conv2d grouped/asymmetric export parameters`
3. `test(onnx): add grouped and asymmetric Conv2d parity tests`
4. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (add entries here as deferred work is found)