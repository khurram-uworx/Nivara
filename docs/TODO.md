# Plan: Issue #899 - NivaraClassifier ONNX export support

## Problem
NivaraClassifier already exports its linear and conv models to ONNX successfully. Follow-up asks for clear documentation and tests around what is and is not supported by `IOnnxExportContext<T>`. Goal is to make ONNX export support discoverable, tested, and robust for future model changes.

## Proposed changes

### 1. Update documentation (docs/ONNX.md)
- Add missing emitters to API surface: `TransposeAxes`, `Reshape`, `Concat`, `ReduceMean`
- Add "Classifier model support" section documenting TextClassifierModel and ConvTextClassifierModel emitter usage
- Add "Support matrix and known gaps" section documenting supported ops, Dropout behavior (identity in Eval/export), attention/MHA/TransformerBlock support, and gaps like MaxPool/AvgPool
- Update Limits section as needed

### 2. Add parity tests (tests/Nivara.Tests/Onnx/OnnxExportClassifierTests.cs)
- Add TextClassifierModel parity test: export to ONNX, compare with ONNX Runtime
- Add ConvTextClassifierModel parity test: export to ONNX, compare with ONNX Runtime
- Follow existing test patterns (OnnxExportConvTests/OnnxExportMlpTests)
- Use Microsoft.ML.OnnxRuntime for parity checks

### 3. Project references (if needed)
- Ensure tests project can reference NivaraClassifier (already added earlier)

## Blast radius
- docs/ONNX.md (documentation only)
- tests/Nivara.Tests/Onnx/OnnxExportClassifierTests.cs (new test file)
- tests/Nivara.Tests/Nivara.Tests.csproj (may already have NivaraClassifier reference)

Low blast radius - documentation and tests only, no changes to core export logic.

## Verification steps
1. Build: `dotnet build Nivara.slnx`
2. Run targeted ONNX tests: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~OnnxExportClassifier"`
3. Run all ONNX tests to ensure no regressions: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~OnnxExport"`
4. Check docs coverage if needed

## Planned commits
1. docs: plan issue #899 in TODO.md
2. docs: update ONNX.md with classifier support matrix and emitters
3. test: add ONNX export parity tests for TextClassifierModel and ConvTextClassifierModel
4. docs: remove TODO.md (after reviews pass)

## GitHub issues log
- [ ] #899 - NivaraClassifier: document ONNX export support and add parity smoke tests (in progress)
- [ ] May create follow-up issues for gaps if found (MaxPool/AvgPool support) - to be determined during review

Reminder: Create GitHub issues immediately when discovering deferred work/follow-ups, record in this log.
