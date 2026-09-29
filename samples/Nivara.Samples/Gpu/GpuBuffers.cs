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

    // ── GEMM tile geometries (#440) ───────────────────────────────

    /// <summary>
    /// One register-blocked GEMM geometry: a <c>BlockRows × BlockCols</c> output block per
    /// work item, a <c>KTile</c>-deep K step, over a 16×16 group. The tile is therefore
    /// <c>16·BlockRows</c> rows by <c>16·BlockCols</c> columns.
    /// </summary>
    /// <remarks>
    /// The group stays 16×16 = 256 work items across every geometry so the comparison against
    /// the incumbent <see cref="GemmGeometry.Row4"/> isolates the blocking factor rather than
    /// confounding it with a work-group resize. <see cref="SharedLoadsPerMac"/> is the
    /// quantity each geometry is trying to reduce: per K step a thread reads
    /// <c>BlockRows + BlockCols</c> shared floats to do <c>BlockRows·BlockCols</c> MACs, so
    /// widening the block cuts shared-memory traffic per MAC without changing global traffic.
    /// </remarks>
    public sealed record GemmGeometry(string Name, int BlockRows, int BlockCols, int KTile)
    {
        /// <summary>Output rows covered by one work group.</summary>
        public int TileRows => GemmKernels.TileSize * BlockRows;

        /// <summary>Output columns covered by one work group.</summary>
        public int TileCols => GemmKernels.TileSize * BlockCols;

        /// <summary>Register accumulators live per work item.</summary>
        public int Accumulators => BlockRows * BlockCols;

        /// <summary>
        /// Shared memory the A and B tiles need per group. ILGPU allocates per work
        /// <em>group</em>, not per work item (docs/ACCELERATION.md lesson 15), and the
        /// allocation size must be a compile-time constant — hence the geometry constants
        /// rather than runtime shape arguments.
        /// </summary>
        public int SharedBytes => (TileRows * KTile + KTile * TileCols) * sizeof(float);

        /// <summary>Shared-memory float reads per multiply-add, the blocking-factor payoff.</summary>
        public float SharedLoadsPerMac => (BlockRows + BlockCols) / (float)Accumulators;
    }

    /// <summary>
    /// The incumbent 1×4 @ K16 geometry (<see cref="GemmKernels.TiledGemmKernelRow4"/> and its
    /// fused siblings), kept as the baseline the #440 variants are measured against.
    /// </summary>
    public static readonly GemmGeometry Row4Geometry = new("1x4@KT16", 1, 4, 16);

    /// <summary>
    /// The four geometries added by #440. The K-tile is varied independently of the blocking
    /// factor (2×2 at both K16 and K32) so a win or loss can be attributed to one or the other
    /// rather than to "the new kernel" as a bundle. All four are plain — the fused
    /// Bias/GELU/ReLU/Qkv epilogues are ported for the winner only, since the epilogue is one
    /// extra register add over the same accumulators in the same ascending-K order and so is
    /// orthogonal to the blocking factor.
    /// </summary>
    public static readonly GemmGeometry[] GemmGeometries =
    [
        new("2x2@KT16", 2, 2, 16),
        new("2x2@KT32", 2, 2, 32),
        new("4x2@KT32", 4, 2, 32),
        new("1x8@KT16", 1, 8, 16),
    ];

    /// <summary>
    /// Fails unless <paramref name="geometry"/>'s shared-memory tile and group size fit the
    /// device, mirroring <see cref="ValidateAttentionLocalMemory"/>. Queried rather than
    /// assumed: #440 widens the GEMM tile from ~5 KB to up to 12 KB per group, and occupancy
    /// depends on how many groups the driver can co-resident, so the footprint is checked
    /// against what the device actually reports.
    /// </summary>
    public static void ValidateGemmSharedMemory(Accelerator accelerator, GemmGeometry geometry)
    {
        int groupSize = GemmKernels.TileSize * GemmKernels.TileSize;
        if (groupSize > accelerator.MaxNumThreadsPerGroup)
            throw new InvalidOperationException(
                $"GEMM geometry {geometry.Name} launches groups of {groupSize}, but " +
                $"{accelerator.Name} allows at most {accelerator.MaxNumThreadsPerGroup} threads per group.");

        if (geometry.SharedBytes > accelerator.MaxSharedMemoryPerGroup)
            throw new InvalidOperationException(
                $"GEMM geometry {geometry.Name} needs {geometry.SharedBytes} B of shared memory per group, " +
                $"but {accelerator.Name} reports only {accelerator.MaxSharedMemoryPerGroup} B. " +
                "Reduce the K tile or the blocking factor.");
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
    public static KernelConfig GemmCfg(int aRows, int bCols) => GemmCfg(aRows, bCols, Row4Geometry);

    /// <summary>
    /// Tiled-GEMM launch for an arbitrary #440 geometry: a
    /// <c>(ceil(aRows/tileRows), ceil(bCols/tileCols))</c> grid of 16x16 groups, where the tile
    /// is the geometry's <see cref="GemmGeometry.TileRows"/> x
    /// <see cref="GemmGeometry.TileCols"/> output block. The group is 16x16 for every geometry
    /// so a measurement difference is attributable to the blocking factor alone.
    /// </summary>
    public static KernelConfig GemmCfg(int aRows, int bCols, GemmGeometry geometry)
    {
        var numGroups = new Index2D(
            (aRows + geometry.TileRows - 1) / geometry.TileRows,
            (bCols + geometry.TileCols - 1) / geometry.TileCols);
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
