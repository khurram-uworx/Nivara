# CPU INT8 block-quantized decode weights (issue #1502)

## Problem

Issue #1469 delivered INT8 block-32 decode weights for Silk/OpenCL GPU runners (PR #1500, `6ff42382`). The managed CPU inference path (`LlamaForCausalLM<TModel,TWeight>` / `LlamaDecoderBlock` / `LlamaFusedKernels`) still uses dense F32/BF16/Half weights. CPU decode is the fallback/viability path; INT8 should cut weight bandwidth and memory footprint while keeping parity with the Silk INT8 contract.

Target issue: **#1502** (`Silk/CPU: INT8 block-quantized decode weights on the CPU inference path`, OPEN, `performance`). Related: #1469 (Silk INT8, closed), #1501 (INT4 feasibility, separate).

## Baseline (before implementation, synthetic weights only)

This machine is slow/different; capture synthetic-only CPU numbers in `release/vCurrent/` before changing code, then rerun the same commands after implementation and compare.

Planned baseline commands (run with explicit human approval; long-running):

```powershell
dotnet run --project samples/NivaraInference -c Release -- qwen profile --synthetic-weights > release/vCurrent/qwen-synthetic-cpu-profile.1502-before.txt
dotnet run --project samples/NivaraInference -c Release -- smollm profile --synthetic-weights > release/vCurrent/smollm-synthetic-cpu-profile.1502-before.txt
```

After implementation, capture:

```powershell
dotnet run --project samples/NivaraInference -c Release -- qwen profile --synthetic-weights > release/vCurrent/qwen-synthetic-cpu-profile.1502-after.txt
dotnet run --project samples/NivaraInference -c Release -- smollm profile --synthetic-weights > release/vCurrent/smollm-synthetic-cpu-profile.1502-after.txt
```

If INT8 opt-in flag/env is added, also capture the INT8-enabled CPU profile with the same synthetic path and record the command in the after file/PR.

## G1 grounding / decisions to clear before implementation

These must be resolved during G1 grounding (microsoft-learn MCP + code-memory + blast radius). Do not assume.

1. **Format ownership**: reuse `SilkInt8Block32 v1` semantics. `SilkInt8Quantizer` currently lives in `src/Nivara.Extensions/Silk/SilkInt8Quantizer.cs` (`Nivara.Silk`). CPU fused decode kernels live in core (`src/Nivara/AutoDiff/*`) and the model is in samples (`samples/Nivara.Samples/LlamaForCausalLM.cs`). Decide: move/promote quantizer+format type to a shared location, add a core-compatible copy/adapter, or reference Extensions from samples only. Avoid wrong-way assembly dependency and avoid duplicating format logic silently.
2. **Quantized weight representation**: `sbyte` is not `IFloatingPointIeee754<T>`, so INT8 weights cannot flow through `Parameter<TWeight>`/`ReverseGradTensor<TWeight>`/existing generic matmul paths. Decide carrier shape (e.g. `QuantizedLinearWeight` holding `sbyte[]` + `float[] scales` + dims + block size) and where it lives.
3. **Integration point**: CPU fused decode bypasses `Linear.Forward` and calls `GradKernels.MixedMatMulTransposedB` directly (`LlamaDecoderBlock.ForwardCachedFused/ForwardPrefillFused`, `LlamaForCausalLM.ForwardCachedFusedCore/ForwardPrefillFusedCore`). Decide whether to add INT8-aware overloads in `GradKernels`/`LlamaFusedKernels` or a dedicated quantized kernel class, and how `LlamaDecoderBlock`/`LlamaForCausalLM` select it.
4. **Scope**: decode weights only (Q/K/V/O, gate/up/down, LM head) mirroring Silk; embeddings/norms/biases/KV cache remain F32. LoRA/adapters/Grad-enabled training fall back to dense path. Prefill: include if it shares the same fused CPU path; confirm.
5. **ISA dispatch**: #1469 comment recommends `Vector128/256/512`, `System.Runtime.Intrinsics`, AVX2/AVX-512/VNNI where available, portable scalar fallback; `TensorPrimitives` may be insufficient for fused dequant+accum control. Decide minimum v1 dispatch (scalar + AVX2? scalar only?) and runtime dispatch mechanism.
6. **Parity contract**: define numeric parity target vs dense F32 reference and vs Silk INT8 where comparable. FP accumulation order may differ with SIMD; prefer exact where deterministic, otherwise record justified tolerance (max abs/cosine/argmax agreement).
7. **Public/API surface**: opt-in via renamed env var `NIVARA_QUANT_INT8` (same name for both CPU and Silk; `--silk` selects backend). Replaces `NIVARA_SILK_INT8`. Keep default unchanged.## Proposed changes

### A. Format & quantizer (shared, versioned)

- Keep `SilkInt8Block32 v1` semantics: symmetric, block size 32, no zero point, F32 scales, `sbyte` weights, reduction-dim blocks, scales `[outCols, numBlocks]`, weights `[inCols, outCols]` transposed layout for decode GEMV.
- Reuse/extract quantization logic so CPU and Silk share one implementation or one byte-compatible format definition. Candidate: move `SilkInt8Quantizer` to a neutral Extensions location (e.g. `Nivara.Extensions/AI` or `Nivara.Extensions/Quantization`) or make it internal/public appropriately; Silk keeps calling it. Record decision in docs.
- Add CPU-friendly quantized weight holder:

```csharp
// sketch - location TBD after G1
public sealed class Int8Block32Weight
{
    public required sbyte[] Data { get; init; }      // [inCols, outCols]
    public required float[] Scales { get; init; }     // [outCols, numBlocks]
    public required int InCols { get; init; }
    public required int OutCols { get; init; }
    public int NumBlocks => (InCols + 31) / 32;
}
```

### B. CPU INT8 fused GEMV kernels

- Add scalar reference first: fused dequant `(float)w * scale` + F32 accum, no full F32 materialization. Operate on spans/arrays.
- Add SIMD path(s) after correctness: AVX2/Vector256 baseline; AVX-512/VNNI if available and justified; runtime dispatch with portable fallback. Keep accumulation F32.
- Kernel surface (minimum for decode parity):
  - `MatMulTransposedBInt8` / GEMV for `[1,in] x [in,out]^T` INT8 weights.
  - Optional fused norm+gemv, gemv+add, norm+gemv+swiglu equivalents only if CPU fused path benefits and keeps parity manageable; otherwise compose primitives with no materialized dequant tensor.

### C. Wire into CPU decode path

- `LlamaDecoderBlock<TModel,TWeight>` / `LlamaForCausalLM<TModel,TWeight>`: carry optional INT8 weight payloads for eligible projections. Only active when `!GradientUtils.IsGradEnabled`, no LoRA/adapter on projection, and quantized weights present.
- `ForwardCachedFused` / `ForwardPrefillFused` / tied LM head: dispatch to INT8 GEMV when eligible.
- Loader/model construction: add opt-in path to quantize loaded dense weights or load prequantized weights (decision after G1). `NivaraInference` gets env/config toggle via `NIVARA_QUANT_INT8`.

### D. Tests & docs

- Unit tests: quantizer correctness, block tail/padding, scale behavior, scalar GEMV parity vs dense reference, SIMD parity vs scalar, no materialization/allocation sanity where practical.
- Integration/parity: synthetic small Llama model CPU dense vs CPU INT8 argmax/logits within defined tolerance; SmolLM/Qwen real-checkpoint smoke if available in CI/env, else gated/manual with recorded command.
- Docs: update `docs/SILK.md` (shared format note), `docs/QWEN.md`/`docs/SMOLLM.md` CPU INT8 toggle/status, add decision note. Update `DocsCoveragePins` if doc refs/counts change.

## Blast radius

| Area | Files/symbols | Risk |
|---|---|---|
| Silk quant format | `src/Nivara.Extensions/Silk/SilkInt8Quantizer.cs`, `SilkWeightQuantization.cs`, Silk runners/sessions | Medium if moved/renamed; keep byte-compatible and update callers. |
| CPU fused decode kernels | `src/Nivara/AutoDiff/Operations/GradKernels.cs`, `NarrowFloatKernels.cs` or new kernel file; `src/Nivara/AutoDiff/Nn/LlamaFusedKernels.cs` | Medium; hot path, must not regress dense path. |
| Llama CPU model | `src/Nivara/AutoDiff/Nn/LlamaDecoderBlock.cs`, `samples/Nivara.Samples/LlamaForCausalLM.cs`, `LlamaLoader.cs` | High - core decode path; guard with eligibility checks and existing parity tests. |
| Samples/CLI | `samples/NivaraInference/Qwen.cs`, `Program.cs`, README/docs | Low-medium. |
| Tests | new CPU INT8 tests; existing Llama/SmolLM/Qwen parity tests must stay green | Medium; numeric tolerance care needed. |
| Docs gates | `tests/Nivara.Tests/Docs/DocsCoveragePins.cs`, snippet/citation gates | Low if counts updated deliberately. |

## Verification

- [ ] `dotnet build Nivara.slnx -c Release`
- [ ] Targeted tests for new INT8 quant/kernel/model paths (ask before running).
- [ ] Existing Llama/SmolLM/Qwen/Silk parity tests relevant to touched paths (ask before running).
- [ ] `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- [ ] Docs gates if docs touched: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (ask before running).
- [ ] Before/after steady-state ms/token + memory footprint on SmolLM + Qwen real checkpoint (manual/recorded; do not run long benchmarks unprompted unless explicitly approved).
- [ ] #1502 acceptance criteria mapped: format decision recorded; CPU INT8 decode path with parity/quality checks; ms/token+memory before/after; runtime ISA dispatch with graceful fallback.

## Planned commits

1. `docs: plan CPU INT8 decode parity in TODO.md (#1502)`
2. `refactor: share INT8 block-32 quant format for CPU/Silk` (if G1 decides extraction; else fold into impl)
3. `feat: add CPU INT8 block-32 GEMV kernels with scalar fallback`
4. `feat: wire INT8 block-32 decode weights into Llama CPU path`
5. `feat: add CPU INT8 opt-in for NivaraInference Qwen/SmolLM`
6. `test: add CPU INT8 parity and quantization coverage`
7. `docs: document CPU INT8 decode path and update coverage pins`
8. `docs: remove TODO.md — plan executed` (only after G2 clears)

## GitHub issues log

- [ ] #1502 — CPU INT8 block-quantized decode weights on the CPU inference path (this plan)
- [ ] #1501 — INT4 block-quantized decode weights (separate follow-up, not this branch)

<!-- As each task executes, if you find deferred work or a concern, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number here. Don't rely on memory or wait until the end of the plan. -->
