# Plan: AISTACK Phase 1 — verify ONNX export runtime parity and document supported subset (#1293)

## Problem

`#866`/export sub-issues closed; exporter exists. Revised DoD requires explicit verification
that exported `.onnx` runs in ONNX Runtime (ML.NET deployment target) reproducing Nivara
inference within float tolerance for `Linear->ReLU->Linear` and `TransformerBlock`, plus
documented supported/unsupported subset consistent with exporter behavior.

## Proposed changes

1. Code consistency:
   - Clarify `MultiheadAttention` unsupported message in `OnnxExporter`.
   - Document `OnnxExportOptions.NativeDtype` as reserved (graph element type follows module `T`).
2. Docs (`docs/ONNX.md`): fix element-type/NativeDtype claims, split support matrix,
   add constraints + verification sections.
3. Tests: add `OnnxExportUnsupportedTests.cs` asserting unsupported modules throw.
4. Optional changelog entry.

## Verification steps

- `dotnet build Nivara.slnx`
- `dotnet test ... --filter "FullyQualifiedName~Onnx"` (ask before running)
- `dotnet format ... whitespace --verify-no-changes`
- docs snippet gate

## Planned commits

1. docs: plan #1293 in TODO.md
2. code: clarify unsupported MultiheadAttention + NativeDtype doc
3. tests: add unsupported export throw tests
4. docs: update ONNX.md supported subset/verification
5. docs: remove TODO.md

## Blast radius

- `src/Nivara.Extensions/Onnx/OnnxExporter.cs` (options + MultiheadAttention)
- `docs/ONNX.md`
- new test file only

## GitHub issues log

- (none created yet)
