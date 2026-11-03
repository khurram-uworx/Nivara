# Plan: Issue #875 — ONNX export: documentation and sample

## Problem

ONNX export is implemented and parity-tested (#866, #871-#874), but there is no user-facing documentation or sample demonstrating the intended workflow: train in Nivara, export to `.onnx`, deploy via ONNX Runtime/ML.NET. Users have no prerequisite/verification guidance.

## Blast radius

- New `docs/ONNX.md` (adds one top-level docs markdown file → `ExpectedDocumentCount` 55→56).
- New sample project under `samples/NivaraOnnxExport/` (4 files: `.csproj`, `Program.cs`, model class, `README.md`).
- New `Nivara.Onnx` API references only; no core changes.
- Doc gates: snippet classification + citation counts may move if doc contains citations.
- No changes to exporter, tests, or storage.

## Proposed changes

1. Add `docs/ONNX.md` covering overview, non-goals, prerequisites, API surface/emitter table, workflow, mixed precision, diagnostics, limits, verification commands.
2. Add `samples/NivaraOnnxExport/` mirroring `NivaraClassifier` style: deterministic MLP, short training loop in `GradientUtils.Grad()`, export, ONNX Runtime parity check.
3. Update `samples/README.md` (and optionally root README) pointer.
4. Update doc gates: classify new doc in `DocSnippetExtractor.UngatedDocuments`; update `DocsCoveragePins` if counts move.
5. Verify build + targeted doc tests + sample run.

## GitHub issues log

- (none yet)

## Verification

- `dotnet build Nivara.slnx`
- `dotnet build samples/NivaraOnnxExport/NivaraOnnxExport.csproj`
- `dotnet run --project samples/NivaraOnnxExport -c Release` (parity within 1e-4)
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (after human OK)
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
