# Plan: Fold NivaraOnnxExport into NivaraClassifier

## Problem

`samples/NivaraOnnxExport` demonstrates exporting a tiny MLP and verifying parity against ONNX Runtime. The classifier sample already has `--save`/`--load`, training, and prediction, but no ONNX export/inference path. Keeping two separate samples duplicates infrastructure and hides the classifier's exportability. Folding export into the classifier makes the classifier a single end-to-end "train → save → export → predict via ONNX" demo.

Gaps found while scoping:
- `IOnnxExportContext<T>` / `Context<T>` in `src/Nivara.Extensions/Onnx/OnnxExporter.cs` have no emitters for `TransposeAxes`, `Reshape`, `Concat`, or `ReduceMean`, all of which the conv classifier needs.
- `TextClassifierModel<T>` and `ConvTextClassifierModel<T>` do not implement `ExportToOnnx`.
- Conv predictor does not call `Eval()` before inference, so `Dropout` stays stochastic.
- The fixed-batch input shape expected by ONNX Runtime must be predictable for `--load-onnx`.

## Proposed changes

1. Extend `IOnnxExportContext<T>` and `Context<T>` with:
   - `TransposeAxes(string inputName, int[] perm)`
   - `Reshape(string inputName, int[] newShape)`
   - `Concat(string[] inputNames, int axis)`
   - `ReduceMean(string inputName, int[] axes, bool keepDims = false)`
2. Implement `ExportToOnnx` in `TextClassifierModel<T>`:
   `Embedding → ReduceMean(axis=1) → Linear → Relu → Linear`.
3. Implement `ExportToOnnx` in `ConvTextClassifierModel<T>`:
   `Embedding → TransposeAxes(0,2,1) → three (Conv1d → Relu) branches → Reshape/Concat → Linear → Relu → Linear` (Dropout omitted at export; Eval() before export).
4. `NivaraClassifier.csproj`: add `Nivara.Extensions` and `Microsoft.ML.OnnxRuntime` 1.31.0.
5. `Program.cs`: add `--export-onnx <path>` and `--load-onnx <path>` options; add `onnx-compare` command; add Eval() to prediction paths.
6. `--load-onnx`: single-batch `[1, MaxSeqLen]` double input via ORT `InferenceSession`.
7. `onnx-compare`: Nivara inferences → export temp `.onnx` → ORT inferences → label + max-abs-diff report.
8. Remove `samples/NivaraOnnxExport/`; update `samples/README.md`, `docs/ONNX.md`, and classifier README.
9. Document export gaps resolved in classifier README library-gaps table.

## Verification steps

- `dotnet build Nivara.slnx`
- `dotnet build samples/NivaraClassifier/NivaraClassifier.csproj`
- Manual smoke:
  - `dotnet run --project samples/NivaraClassifier -c Release -- --command generate`
  - `dotnet run --project samples/NivaraClassifier -c Release -- --command train --epochs 3 --save samples/data/classifier.model.json --export-onnx samples/data/classifier.model.onnx`
  - `dotnet run --project samples/NivaraClassifier -c Release -- --command onnx-compare --load samples/data/classifier.model.json`
  - `dotnet run --project samples/NivaraClassifier -c Release -- --command predict --load-onnx samples/data/classifier.model.onnx` (interactive; exit)
- Docs snippet gate: run only if docs changed and requested by user.

## Planned commits

1. docs: plan classifier-onnx in TODO.md
2. feat(onnx): add TransposeAxes/Reshape/Concat/ReduceMean emitters
3. feat(samples): implement ExportToOnnx in TextClassifierModel
4. feat(samples): implement ExportToOnnx in ConvTextClassifierModel
5. refactor(samples): fold ONNX into NivaraClassifier (csproj, CLI, predict, compare)
6. docs: update ONNX/readme documentation and remove NivaraOnnxExport
7. docs: remove TODO.md — plan executed (after G2)

## Blast radius

- Interface change: `IOnnxExportContext<T>` affects all modules implementing `ExportToOnnx` indirectly; new methods are additive only.
- `Context<T>` shape maps must remain consistent for new ops.
- `Nivara.Samples.TextClassifierModel<T>` is shared by NivaraChat; adding ExportToOnnx is additive.
- `ConvTextClassifierModel<T>` is sample-local.
- Sample README/CI references to NivaraOnnxExport removed.

## GitHub issues log

- None yet.
