# Plan: issue #874 — ONNX export: OperationDiagnostics coverage

Branch: khurram/874 (from khurram/873).

## Problem

ONNX export currently records nothing through `OperationDiagnostics`/`DiagnosticsTracker`, unlike AutoDiff and inference kernels. We need export decisions (which module/emitter was used, unsupported-op failures) recorded when diagnostics are enabled, so export can be diagnosed and unsupported ops fail loudly.

## Proposed changes

- New `src/Nivara.Extensions/Onnx/OnnxExportDiagnostics.cs` (`Nivara.Onnx`, internal static): facade mirroring `AutoDiffDiagnostics`/`InferenceDiagnostics`; `Enabled` plus `IsEnabled = Enabled || DiagnosticsTracker.IsEnabled`.
- `OnnxExporter.Export<T>`: measure whole export; record top-level `OperationType = "OnnxExport"`; on exception record `...Reason=...` and rethrow.
- `Context<T>` emitter methods: measure; record OperationTypes `"OnnxExport" + MethodName` and `"Onnx" + MethodName` with emitter notes. At unsupported validation throws, also record `"OnnxExportUnsupported"` / `"OnnxUnsupported"`.
- No core changes; exporter behavior unchanged.

## Blast radius

- Files: `src/Nivara.Extensions/Onnx/OnnxExporter.cs`, new diagnostics file.
- Symbols: `OnnxExporter.Export`, `OnnxExporter.Context<T>` public methods.
- Tests: `tests/Nivara.Tests/Onnx/*` (targeted), plus new diagnostics tests.
- No public API/type-shape changes except new internal diagnostic facade.

## Planned commits

1. docs: plan 874 in TODO.md
2. feat(onnx): add OnnxExportDiagnostics facade
3. feat(onnx): record OperationDiagnostics in OnnxExporter
4. test(onnx): add diagnostics coverage tests (if hidden not provided)
5. docs: remove TODO.md

## Verification steps

- `dotnet build Nivara.slnx`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Targeted ONNX tests in Release.

## GitHub issues log

- (none yet)
