# Plan: Issue #905 — Silk classifier device-resident embedding table with scatter backward

## Goal
Move the embedding table in `GpuTextLinearClassifier` and `GpuTextConvClassifier` from host-resident to device-resident, with device-side embedding backward (scatter/reduction), device Adam moments, and fused `adam_update` step, matching the treatment of the other parameters landed in #900. Refresh host mirrors at the end of `Train`.

## Non-goals / parity definition
- No training-trajectory equality with the current host-scatter implementation.
- Parity stays: fixed-weights forward + fixed-batch gradient + loss-decreasing smoke.
- Gradient parity tolerance remains `5e-4` by using a **fixed reduction order** (deterministic), not atomic CAS.

## Semantics decisions (stated, per issue)
- **Scatter semantics: deterministic reduction.** The embedding gradient for a vocab row is accumulated by a single work item that loops over the batch in ascending `(b, s)` order and sums the incoming gradient. No atomics, no CAS. This reproduces the host `for b, for s` accumulation order bit-for-bit enough to keep the existing `5e-4` bound meaningful, and avoids non-deterministic summation order.
- **Forward gather: on-device.** Linear uses a fused gather+mean-pool kernel; conv uses a gather into the `[B, E, S]` channel-major input layout.
- **Moments: device-resident**, zero-filled at construction and at the start of each `Train`.
- **Host mirrors: refreshed at end of `Train`** (and on export), as the other parameters.

## Files to change

### 1. `src/Nivara.Extensions/Silk/SilkKernels.cs`
- Add constants:
  - `EmbeddingGatherPool = "embedding_gather_pool"`
  - `EmbeddingGatherInput = "embedding_gather_input"`
  - `EmbeddingBackwardPool = "embedding_backward_pool"`
  - `EmbeddingBackwardInput = "embedding_backward_input"`
- Add the four constants to `KernelNames`.
- Add the four `__kernel` definitions to `Source` (OpenCL C). Suggested placement after `gather_rows`.

Kernel sketches:

```opencl
__kernel void embedding_gather_pool(
    __global const int* ids, __global const float* table, __global float* out,
    const uint batchSize, const uint seqLen, const uint embedDim)
{
    uint idx = get_global_id(0);
    uint total = batchSize * embedDim;
    if (idx >= total) return;
    uint b = idx / embedDim;
    uint e = idx % embedDim;
    float acc = 0.0f;
    for (uint s = 0; s < seqLen; ++s) {
        int tid = ids[b * seqLen + s];
        acc += table[(uint)tid * embedDim + e];
    }
    float denom = seqLen > 0u ? (float)seqLen : 1.0f;
    out[idx] = acc / denom;
}

__kernel void embedding_gather_input(
    __global const int* ids, __global const float* table, __global float* out,
    const uint batchSize, const uint seqLen, const uint embedDim)
{
    uint idx = get_global_id(0);
    uint total = batchSize * embedDim * seqLen;
    if (idx >= total) return;
    uint s = idx % seqLen;
    uint tmp = idx / seqLen;
    uint e = tmp % embedDim;
    uint b = tmp / embedDim;
    int tid = ids[b * seqLen + s];
    out[idx] = table[(uint)tid * embedDim + e];
}

__kernel void embedding_backward_pool(
    __global const int* ids, __global const float* dPooled, __global float* gradEmb,
    const uint vocabSize, const uint batchSize, const uint seqLen, const uint embedDim,
    const float poolScale)
{
    uint idx = get_global_id(0);
    uint total = vocabSize * embedDim;
    if (idx >= total) return;
    uint v = idx / embedDim;
    uint e = idx % embedDim;
    float acc = 0.0f;
    for (uint b = 0; b < batchSize; ++b)
        for (uint s = 0; s < seqLen; ++s)
            if ((uint)ids[b * seqLen + s] == v)
                acc += poolScale * dPooled[b * embedDim + e];
    gradEmb[idx] = acc;
}

__kernel void embedding_backward_input(
    __global const int* ids, __global const float* dX, __global float* gradEmb,
    const uint vocabSize, const uint batchSize, const uint seqLen, const uint embedDim)
{
    uint idx = get_global_id(0);
    uint total = vocabSize * embedDim;
    if (idx >= total) return;
    uint v = idx / embedDim;
    uint e = idx % embedDim;
    float acc = 0.0f;
    for (uint b = 0; b < batchSize; ++b)
        for (uint s = 0; s < seqLen; ++s)
            if ((uint)ids[b * seqLen + s] == v) {
                uint dIdx = b * embedDim * seqLen + e * seqLen + s;
                acc += dX[dIdx];
            }
    gradEmb[idx] = acc;
}
```

### 2. `src/Nivara.Extensions/Silk/SilkLauncher.cs`
Add four launcher methods following the existing `SetBuffer`/`SetUInt32`/`Run` pattern, with global size `LinearGlobal(total)`:
- `EmbeddingGatherPool(runtime, ids, table, outBuf, batchSize, seqLen, embedDim)` — total `batchSize*embedDim`
- `EmbeddingGatherInput(runtime, ids, table, outBuf, batchSize, seqLen, embedDim)` — total `batchSize*embedDim*seqLen`
- `EmbeddingBackwardPool(runtime, ids, dPooled, gradEmb, vocabSize, batchSize, seqLen, embedDim, poolScale)` — total `vocabSize*embedDim`
- `EmbeddingBackwardInput(runtime, ids, dX, gradEmb, vocabSize, batchSize, seqLen, embedDim)` — total `vocabSize*embedDim`

### 3. `GpuTextLinearClassifier.cs`
State changes:
- Add fields: `nint embBuf, embWm, embWv, dEmbBuf; nint tokenIdBuf; int tokenIdCap;`
- Remove `embAdamFirst`/`embAdamSecond` fields; keep `gradEmb` internal field (populated by downloading `dEmbBuf` in `Backward` for parity tests).
- Constructor: allocate `embBuf`, `embWm`, `embWv`, `dEmbBuf`; upload initial embedding; zero moments.
- `UploadWeights`: upload `embBuf`.
- `ZeroMomentBuffers`: fill `embWm`, `embWv`.
- `ForwardCore`: upload token ids; launch `EmbeddingGatherPool` into `pooledBuf`; then existing GEMMs. Download `pooledBuf` into `pooled` for `Forward` output.
- `Backward`: upload token ids; existing device chain into `dPooledBuf`; launch `EmbeddingBackwardPool` into `dEmbBuf` (poolScale `1f/Math.Max(1,seqLen)`); download `dEmbBuf` into `gradEmb`.
- `AdamUpdateAll`: add `SilkLauncher.AdamUpdate(runtime, embBuf, dEmbBuf, embWm, embWv, ...)`; remove `SilkLoss.AdamStep`.
- `RefreshHostMirrors`: download `embBuf` into `embedding`.
- `ExportEmbedding`: download `embBuf` into `embedding` before copy.
- `Dispose`: release new buffers and `tokenIdBuf`.
- Add `EnsureTokenBuffer` helper.
- Update class remarks/docstrings (embedding now device-resident).

### 4. `GpuTextConvClassifier.cs`
Analogous, with conv-specific layouts:
- Add fields `embBuf, embWm, embWv, dEmbBuf, tokenIdBuf, tokenIdCap`.
- Constructor/UploadWeights/ZeroMomentBuffers/Dispose updates.
- `ForwardBatch`: upload token ids; launch `EmbeddingGatherInput` into `inputBuf`; keep conv pipeline. Remove host gather.
- `Backward`: keep conv backward into `dXBuf`; upload token ids; launch `EmbeddingBackwardInput` into `dEmbBuf`; download into `gradEmb`.
- `AdamUpdateAll`: add embedding Adam step; remove `SilkLoss.AdamStep`.
- `RefreshHostMirrors`/`ExportEmbedding`: download `embBuf`.

### 5. `SilkLoss.cs`
- `AdamStep` becomes unused but may be retained as a shared helper (issue allows retiring or shrinking). Keep for now; optionally mark obsolete. No functional change required.

### 6. Docs
- Update `docs/SILK.md` §13c to describe device-resident embedding, deterministic reduction scatter, device moments, and refresh behavior.
- Update class remarks in both classifiers.

## Tests
- Existing tests should pass unchanged:
  - `GpuTextLinearClassifierTests.Gradients_Batch_MatchReverseAutograd` (uses `gpu.gradEmb`).
  - `GpuTextConvClassifierTests` gradient parity.
  - `SilkKernelsCatalogTests` (kernel names/source consistency).
- Add/adjust disposal tests if new buffers require release (existing tests cover disposed access).
- Run targeted: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~GpuTextLinearClassifier|FullyQualifiedName~GpuTextConvClassifier|FullyQualifiedName~SilkKernelsCatalog"`.

## Risks / notes
- `SilkBuffer` is float-count based; token id buffers are raw `nint` bytes, managed manually.
- Ensure `tokenIdBuf` upload uses `ReadOnlySpan<int>` overload.
- Conv input layout must remain `[B, E, S]` (`b*E*S + e*S + s`) to match existing conv kernels.
- Linear pool denominator uses `Math.Max(1, seqLen)` parity with host.
- Deterministic reduction order keeps `5e-4` tolerance meaningful; do not switch to atomics.
- Benchmarks for acceptance evidence are human-run.

## Measurement workflow (human-run)
Preferred harness: `samples/NivaraClassifier` benchmark mode:
  - `dotnet run -c Release --project samples/NivaraClassifier -- --benchmark --silk --mode linear`
  - `dotnet run -c Release --project samples/NivaraClassifier -- --benchmark --silk --mode conv`
Optional synthetic probe in `Nivara.PerformanceTests`. Measure before on `main`, implement on branch, measure after with identical args.

## Implementation order
1. Kernels + launcher + catalog.
2. Linear classifier.
3. Conv classifier.
4. Docs.
5. Build + targeted tests.

## GitHub issues log
- (none yet)
