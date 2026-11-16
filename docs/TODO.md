# TODO: Laya native loading — F16 storage, F32 compute (#1218)

Detailed plan: `docs/plan/LAYA-HONEST-WEIGHTS.md` (Phase 2 of `docs/plan/MODEL-LOADING-HONESTY.md`).
SmolLM shipped in #1215, Qwen in #1220. This is the Laya follow-up.

## Problem

A bare `laya` run widens every weight to F32 at load, so it holds ~1.69 GB of resident weights even
though the checkpoint is F16. The native path (`--precision native`) that would keep F16 storage
does not exist for Laya. Worse, the fused encoder path (`ModernBertLayer.ForwardFused`) widens
343,146,496 projection elements per forward pass (196 allocs, ~1.37 GB) and discards them, and the
loader copies every fused block. `GradKernels.MixedMatMul` is also missing its `Half` arm (asymmetry
with `MixedMatMulTransposedB`).

## Proposed changes (shape)

1. `ModernBertModel.cs`: split `WeightSpan` into `ParamSpan(Parameter<TWeight>)` (raw storage, no
   conversion) + `ModelSpan(Parameter<TModel>)` (no conversion branch, so a compute-dtype read can
   never allocate). Seven projection calls become `MixedMatMulTransposedB<TModel,TWeight>`.
   `ModernBertEncoder.Forward` embedding gather becomes `Embedding.GatherRows` in storage dtype.
2. `StateDictLoader.cs`: zero-copy fused-block views (identity assertion, not element equality);
   `LoadLinearSlice`/`LoadFusedLinearSlice` stop copying. Add `LoadLayerNorm<TModel,TWeight>` widen
   (1-D tensors only).
3. `GradKernels.cs`: add `Half` arm to `MixedMatMul` (asymmetry fix).
4. `Embedding.cs`: add `GatherRows(int[], Span<TModel>)`.
5. `Program.cs`: `isLaya` joins native default; per-model map selection (`Read<Half>` for CPU native
   Laya, no narrow map for `--silk` native via `layaNative`/`layaNativeCpu`); native guard message
   update; `--silk` + native note in validation block; help text update; `ParameterLine<TWeight>` fold.
6. `Laya.Narrow.cs`: fold BFloat16/Half `ParameterLine` overloads into one generic.
7. Docs: `docs/LAYA.md`, `docs/SILK.md` updates; CLI test inversions in
   `InferenceCliArgumentTests`.

## Verification steps

- Build: `dotnet build Nivara.slnx`
- Format: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- New tests: `LayaNativeCheckpointTests` (by-reference identity, allocation bound, forward parity
  vs F32 within documented bound, backward Half arm).
- CLI: `InferenceCliArgumentTests` accept `laya --precision native` (Model file not found path),
  reject minilm/mobilenet_v2/modernbert with updated message, `--silk` note assertable.
- Wheel gate unchanged: `dotnet run -c Release --project tests/Nivara.PerformanceTests --
  --inference-compare laya`.
- Docs namespace: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter
  "FullyQualifiedName~Nivara.Tests.Docs"`.

## Planned commits

1. docs: plan Laya honest weights in TODO.md
2. feat(autodiff): add Half arm to MixedMatMul
3. feat(extensions): ParamSpan/ModelSpan + mixed matmuls in ModernBertLayer.ForwardFused
4. feat(extensions): storage-dtype embedding gather in ModernBertEncoder
5. feat(extensions): zero-copy fused-block views in StateDictLoader
6. feat(inference): Laya native default + Read<Half> + guards + help text + ParameterLine
7. test: Laya native checkpoint + CLI argument coverage
8. docs: Laya/Silk updates for native default
9. docs: remove TODO.md — plan executed

## Blast radius

- `src/Nivara.Extensions/AI/ModernBertModel.cs` (ForwardFused, ModernBertEncoder.Forward, WeightSpan)
- `src/Nivara.Extensions/AI/StateDictLoader.cs` (mixed-target slice overloads)
- `src/Nivara.Extensions/AI/SafeTensorsLoader.cs` (Read<T>, ConvertF16, ConvertF32)
- `src/Nivara/AutoDiff/Operations/GradKernels.cs` (MixedMatMul)
- `src/Nivara/AutoDiff/Nn/Embedding.cs` (GatherRows)
- `src/Nivara/AutoDiff/Nn/Linear.cs` (documented transient only)
- `samples/NivaraInference/Program.cs`, `Laya.cs`, `Laya.Narrow.cs`
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`
- `docs/LAYA.md`, `docs/SILK.md`

## GitHub issues log

(none yet — create issues as deferred work is found during execution)
