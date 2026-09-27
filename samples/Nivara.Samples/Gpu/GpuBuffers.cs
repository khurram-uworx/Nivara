using ILGPU;
using ILGPU.Runtime;

namespace Nivara.Samples.Gpu;

/// <summary>
/// Device-buffer allocation, CPU upload and launch-configuration helpers shared by the GPU
/// encoder runners (<see cref="BertEncoderGpuRunner"/>, <see cref="ModernBertGpuRunner"/>) so
/// the transpose/alloc/launch math has one authoritative implementation.
/// </summary>
internal static class GpuBuffers
{
    /// <summary>Group size for the flat elementwise and row-reduction kernels.</summary>
    public const int LinearGroupSize = 256;

    /// <summary>
    /// Group size for the fused attention kernel. Lower than <see cref="LinearGroupSize"/>
    /// because that kernel stages a <c>headDim</c>-float accumulator tile per thread in local
    /// memory, and local memory is allocated per work item in the group.
    /// </summary>
    public const int AttentionGroupSize = 64;

    /// <summary>
    /// Compile-time bound on the fused attention kernel's per-thread accumulator tile. ILGPU
    /// requires a <c>SharedMemory.Allocate</c> size to be statically known, and the kernel's
    /// <c>headDim</c> is a runtime argument, so the tile is over-allocated to this constant and
    /// only the first <c>headDim</c> entries are used. Every encoder on the GPU path
    /// (DistilBERT, MiniLM, ModernBERT) has headDim 64, so this costs no extra local memory.
    /// <see cref="ValidateAttentionLocalMemory"/> rejects a larger headDim rather than letting
    /// it read past the tile.
    /// </summary>
    public const int MaxHeadDim = 64;

    /// <summary>
    /// Band value meaning "attend to every non-suppressed key", the sentinel
    /// <see cref="AttentionKernels"/> reads as global attention. Matches the CPU's own
    /// <c>band &lt; 0</c> convention in <c>ModernBertMasks.Build</c>.
    /// </summary>
    public const int GlobalAttentionBand = -1;

    /// <summary>
    /// Fails unless the fused attention kernel's local-memory tile fits the device: the
    /// per-thread tile must be within <see cref="MaxHeadDim"/>, a whole group of
    /// <see cref="AttentionGroupSize"/> work items must fit the accelerator's reported shared
    /// memory, and the group size must be within the work-group limit. Queried rather than
    /// assumed, because the OpenCL spec's minimum is not a value any particular driver is
    /// obliged to hit exactly.
    /// </summary>
    public static void ValidateAttentionLocalMemory(Accelerator accelerator, int headDim)
    {
        if (headDim > MaxHeadDim)
            throw new InvalidOperationException(
                $"The fused GPU attention kernel stages a {MaxHeadDim}-float accumulator tile per " +
                $"work item (an ILGPU compile-time limit), so headDim must be <= {MaxHeadDim}; got {headDim}.");

        if (AttentionGroupSize > accelerator.MaxNumThreadsPerGroup)
            throw new InvalidOperationException(
                $"The fused GPU attention kernel launches groups of {AttentionGroupSize}, but " +
                $"{accelerator.Name} allows at most {accelerator.MaxNumThreadsPerGroup} threads per group.");

        long bytes = (long)AttentionGroupSize * MaxHeadDim * sizeof(float);
        if (bytes > accelerator.MaxSharedMemoryPerGroup)
            throw new InvalidOperationException(
                $"The fused GPU attention kernel needs {bytes} B of shared memory per group " +
                $"({AttentionGroupSize} work items x {MaxHeadDim} floats), but {accelerator.Name} " +
                $"reports only {accelerator.MaxSharedMemoryPerGroup} B. Lower GpuBuffers.AttentionGroupSize.");
    }

    // ── launch configuration ──────────────────────────────────────

    /// <summary>One-dimensional launch covering <paramref name="total"/> items.</summary>
    public static KernelConfig Cfg1D(int total) => Cfg1D(total, LinearGroupSize);

    /// <summary>
    /// One-dimensional launch covering <paramref name="total"/> items in groups of
    /// <paramref name="groupSize"/>. Callers that stage local memory per thread must size the
    /// group against the device's local-memory limit.
    /// </summary>
    public static KernelConfig Cfg1D(int total, int groupSize)
        => (Math.Max(1, (total + groupSize - 1) / groupSize), groupSize);

    /// <summary>
    /// Tiled-GEMM launch: a <c>(ceil(aRows/16), ceil(bCols/64))</c> grid of 16x16 groups,
    /// matching <see cref="GemmKernels.TiledGemmKernelRow4"/>'s 1x4 register block.
    /// </summary>
    public static KernelConfig GemmCfg(int aRows, int bCols)
    {
        int blockCols = GemmKernels.TileSize * GemmKernels.BlockCols;
        var numGroups = new Index2D(
            (aRows + GemmKernels.TileSize - 1) / GemmKernels.TileSize,
            (bCols + blockCols - 1) / blockCols);
        return new KernelConfig(numGroups, new Index2D(GemmKernels.TileSize, GemmKernels.TileSize));
    }

    // ── allocation ────────────────────────────────────────────────

    public static MemoryBuffer1D<float, Stride1D.Dense> Alloc(Accelerator accelerator, int length)
        => accelerator.Allocate1D<float>(length);

    public static MemoryBuffer1D<int, Stride1D.Dense> AllocInt(Accelerator accelerator, int length)
        => accelerator.Allocate1D<int>(length);

    /// <summary>
    /// Grows <paramref name="buffer"/> in place when it is smaller than <paramref name="length"/>,
    /// allocating it on first use. Tolerates a null buffer so a runner can declare its workspace
    /// as <c>null!</c> and let the first forward size it, instead of having to pre-allocate
    /// against a sequence length the caller has not chosen yet.
    /// </summary>
    public static void Ensure(ref MemoryBuffer1D<float, Stride1D.Dense> buffer, int length, Accelerator accelerator)
    {
        if (buffer is null)
            buffer = accelerator.Allocate1D<float>(length);
        else if (buffer.Length < length)
        {
            buffer.Dispose();
            buffer = accelerator.Allocate1D<float>(length);
        }
    }

    /// <summary>Grows <paramref name="buffer"/> in place when it is smaller than <paramref name="length"/>.</summary>
    public static void Ensure(ref MemoryBuffer1D<int, Stride1D.Dense> buffer, int length, Accelerator accelerator)
    {
        if (buffer is null)
            buffer = accelerator.Allocate1D<int>(length);
        else if (buffer.Length < length)
        {
            buffer.Dispose();
            buffer = accelerator.Allocate1D<int>(length);
        }
    }

    // ── CPU upload ────────────────────────────────────────────────

    /// <summary>
    /// Uploads a weight as <c>Bt[in, out]</c> (the transpose of the checkpoint's
    /// <c>[out, in]</c> row-major layout) so the GEMM reads it as plain row-major and its global
    /// loads coalesce along K. Pre-fused weights — a checkpoint that already ships one
    /// concatenated matrix, such as ModernBERT's <c>Wqkv</c> / <c>Wi</c> — upload through here
    /// unchanged.
    /// </summary>
    public static MemoryBuffer1D<float, Stride1D.Dense> UploadTransposed(
        Accelerator accelerator,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        string key)
    {
        var t = Req(tensors, key);
        int outDim = t.Shape[0];
        int inDim = t.Shape[1];
        var bt = new float[outDim * inDim];
        for (int r = 0; r < outDim; r++)
            for (int c = 0; c < inDim; c++)
                bt[c * outDim + r] = t.Data[r * inDim + c];
        var buffer = accelerator.Allocate1D<float>(bt.Length);
        buffer.CopyFromCPU(bt);
        return buffer;
    }

    /// <summary>Uploads a tensor as-is (biases, LayerNorm gamma/beta, embeddings).</summary>
    public static MemoryBuffer1D<float, Stride1D.Dense> UploadPlain(
        Accelerator accelerator,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        string key)
    {
        var data = Req(tensors, key).Data;
        var buffer = accelerator.Allocate1D<float>(data.Length);
        buffer.CopyFromCPU(data);
        return buffer;
    }

    public static (float[] Data, int[] Shape) Req(
        Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
        => tensors.TryGetValue(key, out var t)
            ? t
            : throw new InvalidOperationException($"Missing weight key '{key}' for the --gpu forward.");

    // ── readback ──────────────────────────────────────────────────

    public static float[] Readback(MemoryBuffer1D<float, Stride1D.Dense> buffer, int length)
    {
        var arr = buffer.AsContiguous().GetAsArray();
        if (arr.Length < length)
            throw new InvalidOperationException("GPU readback returned a shorter buffer than requested.");
        var result = new float[length];
        Array.Copy(arr, result, length);
        return result;
    }
}
