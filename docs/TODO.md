# Plan: Issue #871 — ONNX export parity tests for Conv/Norm/Embedding

## Problem

PR #870 added ONNX export and only round-trips MLP and TransformerBlock.
Need dedicated NUnit parity tests for Conv1d, Conv2d, BatchNorm1d/2d,
LayerNorm, RMSNorm, Embedding, including null/edge cases.

## Blame / blast radius

- Tests: `tests/Nivara.Tests/Onnx/*` (new + existing MLP/Transformer).
- Exporter: `src/Nivara.Extensions/Onnx/OnnxExporter.cs` (output shape propagation,
  BN running-stats handling).
- Modules already implement `ExportToOnnx`, so no module edits expected.

## Proposed changes

1. Add `OnnxExportConvTests.cs`, `OnnxExportNormTests.cs`, `OnnxExportEmbeddingTests.cs`.
2. Enhance exporter shape propagation for Conv/BatchNorm/LayerNorm/RMSNorm/Embedding.
3. Guard non-tracking BatchNorm export.

## GitHub issues log

(none yet)

## Verification

- `dotnet build Nivara.slnx`
- `dotnet test ... --filter "FullyQualifiedName~Onnx"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
