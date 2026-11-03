# Plan: #873 — ONNX export mixed precision and dtype export options

## Problem
Mixed-weight modules (`Linear<TModel, TWeight>`, `Embedding<TModel, TWeight>`) register weights via
`RegisterMixedParameters` but neither overrides `Module<T>.ExportToOnnx`, which defaults to
`NotSupportedException`. The `IOnnxExportContext<T>` interface also has no overload accepting
mixed-weight modules, so `context.Linear(inputName, fc)` (where `fc : Linear<float, BFloat16>`)
cannot compile. As a result, any model containing mixed precision `Linear`/`Embedding` cannot be
exported to ONNX.

## Proposed changes

1. Core interface `IOnnxExportContext<T>` add generic overloads:
   ```csharp
   string Linear<TWeight>(string inputName, Linear<T, TWeight> module)
       where TWeight : struct, IFloatingPointIeee754<TWeight>;
   string Embedding<TWeight>(string inputName, Embedding<T, TWeight> module)
       where TWeight : struct, IFloatingPointIeee754<TWeight>;
   ```
   (Aliases `LinearMixed`/`MixedLinear` may be added for safety.)
2. `Linear<TModel,TWeight>` / `Embedding<TModel,TWeight>` override `ExportToOnnx` calling
   the new generic overloads, widening weights to compute type `TModel` at export.
3. Extensions exporter `Context<T>` implements the new overloads by replicating existing
   Gemm/Gather emission but converting weight storage (`TWeight[]`) to compute type (`T[]`).
4. Optional dtype switch surfaced on `OnnxExportOptions` (`NativeDtype` default false) to document
   and gate native dtype export. Default remains compute/`float` (F32) friendly behavior.
5. Tests: mixed precision Linear parity (Nivara vs ONNX Runtime).
6. Docs note default F32.

## Blast radius
- Files: `src/Nivara/AutoDiff/Nn/IOnnxExportContext.cs`, `Linear.cs`, `Embedding.cs`,
  `src/Nivara.Extensions/Onnx/OnnxExporter.cs`.
- Downstream: any model using mixed Linear/Embedding and calling export; tests in
  `tests/Nivara.Tests/Onnx`.
- Watch: initializer byte sizing via compute type `T`, shape handling unchanged.

## Planned commits
1. docs: plan 873 in TODO.md
2. feat(onnx): expose mixed precision Linear/Embedding context overloads
3. feat(onnx): implement mixed Linear/Embedding export in exporter
4. test(onnx): add mixed precision Linear export parity test
5. docs: remove TODO.md — plan executed

## GitHub issues log
(empty)

## Verification
- dotnet build Nivara.slnx
- dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Onnx"
