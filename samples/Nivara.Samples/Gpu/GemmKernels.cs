using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace Nivara.Samples.Gpu;

/// <summary>
/// Tiled f32 GEMM kernels for the DistilBERT GPU forward (docs/BERT-GPU.md §3).
/// Computes C[aRows × bCols] = A[aRows × aCols] · Bt[aCols × bCols] over plain
/// row-major 1D views; Bt is the weight matrix pre-transposed once at upload so
/// global reads are coalesced along the K axis. Both kernels are explicitly-grouped
/// (shared-memory tile staging + register accumulate) — implicitly-grouped kernels
/// cannot use local memory. The official ILGPU MatrixMultiply sample is the 1×1
/// reference; the 1×4 variant is the register-blocked form (measurement showed the
/// 1×1 form is local-memory bound at ~150 GMAC/s flat on Arc 140T regardless of
/// shape — each thread's 4 accumulators cut shared-memory traffic ~4×).
/// </summary>
internal static class GemmKernels
{
    public const int TileSize = 16;

    /// <summary>One output element per thread; grid (ceil(aRows/16) × ceil(bCols/16)) of 16×16.</summary>
    internal static void TiledGemmKernel(
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> c,
        int aRows,
        int aCols,
        int bCols)
    {
        var global = Grid.GlobalIndex.XY;
        int x = Group.IdxX;
        int y = Group.IdxY;

        var aTile = SharedMemory.Allocate2D<float, Stride2D.DenseX>(
            new Index2D(TileSize, TileSize), new Stride2D.DenseX(TileSize));
        var bTile = SharedMemory.Allocate2D<float, Stride2D.DenseX>(
            new Index2D(TileSize, TileSize), new Stride2D.DenseX(TileSize));

        int outRow = global.X;
        int outCol = global.Y;
        float acc = 0f;

        for (int k0 = 0; k0 < aCols; k0 += TileSize)
        {
            int aCol = k0 + y;
            int bRow = k0 + x;
            aTile[x, y] = (outRow < aRows && aCol < aCols) ? a[outRow * aCols + aCol] : 0f;
            bTile[x, y] = (bRow < aCols && outCol < bCols) ? b[bRow * bCols + outCol] : 0f;
            Group.Barrier();

            for (int k = 0; k < TileSize; k++)
                acc += aTile[x, k] * bTile[k, y];
            Group.Barrier();
        }

        if (outRow < aRows && outCol < bCols)
            c[outRow * bCols + outCol] = acc;
    }

    public const int BlockCols = 4;

    /// <summary>Register-blocked 1×4 GEMM: each thread computes 4 adjacent output columns
    /// of one row (16×16 threads cover a 16-row × 64-col output tile per group). Shared
    /// memory traffic drops ~4× per MAC versus the 1×1 form. Grid
    /// (ceil(aRows/16) × ceil(bCols/64)).</summary>
    internal static void TiledGemmKernelRow4(
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> c,
        int aRows,
        int aCols,
        int bCols)
    {
        const int TileCols = TileSize * BlockCols;
        var global = Grid.GlobalIndex.XY;
        int x = Group.IdxX;
        int y = Group.IdxY;

        var aTile = SharedMemory.Allocate2D<float, Stride2D.DenseX>(
            new Index2D(TileSize, TileSize), new Stride2D.DenseX(TileSize));
        var bTile = SharedMemory.Allocate2D<float, Stride2D.DenseX>(
            new Index2D(TileSize, TileCols), new Stride2D.DenseX(TileCols));

        int outRow = global.X;
        int colBase = Grid.IdxY * (TileSize * BlockCols);
        float acc0 = 0f, acc1 = 0f, acc2 = 0f, acc3 = 0f;

        for (int k0 = 0; k0 < aCols; k0 += TileSize)
        {
            int aCol = k0 + y;
            int bRow = k0 + x;
            aTile[x, y] = (outRow < aRows && aCol < aCols) ? a[outRow * aCols + aCol] : 0f;
            for (int w = 0; w < BlockCols; w++)
            {
                int outCol = colBase + y * BlockCols + w;
                bTile[x, y * BlockCols + w] = (bRow < aCols && outCol < bCols) ? b[bRow * bCols + outCol] : 0f;
            }
            Group.Barrier();

            for (int k = 0; k < TileSize; k++)
            {
                float aVal = aTile[x, k];
                int bBase = y * BlockCols;
                acc0 += aVal * bTile[k, bBase + 0];
                acc1 += aVal * bTile[k, bBase + 1];
                acc2 += aVal * bTile[k, bBase + 2];
                acc3 += aVal * bTile[k, bBase + 3];
            }
            Group.Barrier();
        }

        int rowBase = outRow * bCols + colBase + y * BlockCols;
        if (outRow < aRows)
        {
            if (colBase + y * BlockCols + 0 < bCols) c[rowBase + 0] = acc0;
            if (colBase + y * BlockCols + 1 < bCols) c[rowBase + 1] = acc1;
            if (colBase + y * BlockCols + 2 < bCols) c[rowBase + 2] = acc2;
            if (colBase + y * BlockCols + 3 < bCols) c[rowBase + 3] = acc3;
        }
    }

    /// <summary>
    /// Register-blocked 1×4 GEMM with a fused epilogue: y = act(A·Bt + bias), same tile
    /// geometry as <see cref="TiledGemmKernelRow4"/>. The bias row is read per output
    /// column and added to the register accumulator before the epilogue, so no separate
    /// bias launch is needed. The <paramref name="activation"/> byte is 0 (identity),
    /// 1 (exact GELU — same A–S 7.1.26 polynomial as <see cref="ElementwiseKernels.Gelu"/>),
    /// or 2 (ReLU). Elementwise results are bit-identical to the unfused
    /// GEMM + AddBias (+ GELU/ReLU) sequence: register acc + bias[c] versus
    /// stored-then-added, and the same activation polynomial.
    /// </summary>
    internal static void TiledGemmKernelRow4Fused(
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> c,
        ArrayView<float> bias,
        int aRows,
        int aCols,
        int bCols,
        int activation)
    {
        const int TileCols = TileSize * BlockCols;
        var global = Grid.GlobalIndex.XY;
        int x = Group.IdxX;
        int y = Group.IdxY;

        var aTile = SharedMemory.Allocate2D<float, Stride2D.DenseX>(
            new Index2D(TileSize, TileSize), new Stride2D.DenseX(TileSize));
        var bTile = SharedMemory.Allocate2D<float, Stride2D.DenseX>(
            new Index2D(TileSize, TileCols), new Stride2D.DenseX(TileCols));

        int outRow = global.X;
        int colBase = Grid.IdxY * (TileSize * BlockCols);
        float acc0 = 0f, acc1 = 0f, acc2 = 0f, acc3 = 0f;

        for (int k0 = 0; k0 < aCols; k0 += TileSize)
        {
            int aCol = k0 + y;
            int bRow = k0 + x;
            aTile[x, y] = (outRow < aRows && aCol < aCols) ? a[outRow * aCols + aCol] : 0f;
            for (int w = 0; w < BlockCols; w++)
            {
                int outCol = colBase + y * BlockCols + w;
                bTile[x, y * BlockCols + w] = (bRow < aCols && outCol < bCols) ? b[bRow * bCols + outCol] : 0f;
            }
            Group.Barrier();

            for (int k = 0; k < TileSize; k++)
            {
                float aVal = aTile[x, k];
                int bBase = y * BlockCols;
                acc0 += aVal * bTile[k, bBase + 0];
                acc1 += aVal * bTile[k, bBase + 1];
                acc2 += aVal * bTile[k, bBase + 2];
                acc3 += aVal * bTile[k, bBase + 3];
            }
            Group.Barrier();
        }

        int rowBase = outRow * bCols + colBase + y * BlockCols;
        if (outRow < aRows)
        {
            int c0 = colBase + y * BlockCols + 0;
            int c1 = colBase + y * BlockCols + 1;
            int c2 = colBase + y * BlockCols + 2;
            int c3 = colBase + y * BlockCols + 3;
            float b0 = c0 < bCols ? bias[c0] : 0f;
            float b1 = c1 < bCols ? bias[c1] : 0f;
            float b2 = c2 < bCols ? bias[c2] : 0f;
            float b3 = c3 < bCols ? bias[c3] : 0f;

            acc0 += b0;
            acc1 += b1;
            acc2 += b2;
            acc3 += b3;

            if (activation == 1)
            {
                acc0 = Gelu(acc0);
                acc1 = Gelu(acc1);
                acc2 = Gelu(acc2);
                acc3 = Gelu(acc3);
            }
            else if (activation == 2)
            {
                acc0 = acc0 > 0f ? acc0 : 0f;
                acc1 = acc1 > 0f ? acc1 : 0f;
                acc2 = acc2 > 0f ? acc2 : 0f;
                acc3 = acc3 > 0f ? acc3 : 0f;
            }

            if (c0 < bCols) c[rowBase + 0] = acc0;
            if (c1 < bCols) c[rowBase + 1] = acc1;
            if (c2 < bCols) c[rowBase + 2] = acc2;
            if (c3 < bCols) c[rowBase + 3] = acc3;
        }
    }

    /// <summary>Exact GELU via the A–S 7.1.26 erf port (same polynomial as <see cref="ElementwiseKernels.Gelu"/>).</summary>
    static float Gelu(float v)
    {
        float z = v * 0.7071067811865475f;
        float az = XMath.Abs(z);
        float t = 1f / (1f + 0.3275911f * az);
        float p = 1.061405429f * t - 1.453152027f;
        p = p * t + 1.421413741f;
        p = p * t - 0.284496736f;
        p = p * t + 0.254829592f;
        float erf = 1f - p * t * XMath.Exp(-az * az);
        if (z < 0f) erf = -erf;
        return 0.5f * v * (1f + erf);
    }
}

public enum IlgpuGemmVariant
{
    OneToOne,
    Row4,
}

/// <summary>
/// Persistent f32 tiled-GEMM workspace over the shared <see cref="IlgpuRuntime"/>:
/// the Bt weight buffer stays resident (uploaded once; the caller pre-transposes,
/// docs/BERT-GPU.md §3) and per-forward activations are copied in / read back
/// around a launch. Buffers are allocated once and reused — no per-call allocation
/// on the inference hot path (probe lesson, docs/ILGPU.md).
/// </summary>
public sealed class TiledGemm : IDisposable
{
    private readonly IlgpuRuntime runtime;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> gemm;
    private readonly IlgpuGemmVariant variant;
    private MemoryBuffer1D<float, Stride1D.Dense>? aBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense>? bBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense>? cBuffer;

    public TiledGemm(IlgpuRuntime runtime, IlgpuGemmVariant variant = IlgpuGemmVariant.Row4)
    {
        this.runtime = runtime;
        this.variant = variant;
        gemm = runtime.Accelerator.LoadKernel<
            ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
            variant == IlgpuGemmVariant.Row4 ? GemmKernels.TiledGemmKernelRow4 : GemmKernels.TiledGemmKernel);
    }

    public int ACols { get; private set; }
    public int BCols { get; private set; }

    /// <summary>Uploads a row-major [aCols × bCols] Bt weight matrix into the resident B workspace.</summary>
    public void UploadWeights(float[] bt, int aCols, int bCols)
    {
        Ensure(ref bBuffer, aCols * bCols);
        bBuffer!.CopyFromCPU(bt);
        ACols = aCols;
        BCols = bCols;
    }

    /// <summary>Runs C = A · Bt for host-side A (aRows × aCols) and returns the result read back.</summary>
    public float[] Run(float[] a, int aRows, int aCols)
    {
        Ensure(ref aBuffer, aRows * aCols);
        Ensure(ref cBuffer, aRows * (long)BCols);
        aBuffer!.CopyFromCPU(a);
        Launch(aBuffer!.View, aRows, aCols);
        return cBuffer!.AsContiguous().GetAsArray();
    }

    /// <summary>Runs C = A · Bt for an A view that is already resident on the GPU (no host copy).</summary>
    public void Launch(ArrayView<float> aView, int aRows, int aCols)
    {
        Ensure(ref cBuffer, aRows * (long)BCols);
        int blockCols = variant == IlgpuGemmVariant.Row4 ? GemmKernels.TileSize * GemmKernels.BlockCols : GemmKernels.TileSize;
        int groupsX = (aRows + GemmKernels.TileSize - 1) / GemmKernels.TileSize;
        int groupsY = (BCols + blockCols - 1) / blockCols;
        var numGroups = new Index2D(groupsX, groupsY);
        var groupSize = new Index2D(GemmKernels.TileSize, GemmKernels.TileSize);
        gemm(runtime.Stream, (numGroups, groupSize), aView, bBuffer!.View, cBuffer!.View, aRows, aCols, BCols);
        runtime.Synchronize();
    }

    public MemoryBuffer1D<float, Stride1D.Dense> ResultBuffer => cBuffer!;

    void Ensure(ref MemoryBuffer1D<float, Stride1D.Dense>? buffer, long length)
    {
        if (buffer is null || buffer.Length < length)
        {
            buffer?.Dispose();
            buffer = runtime.Accelerator.Allocate1D<float>((int)length);
        }
    }

    public void Dispose()
    {
        aBuffer?.Dispose();
        bBuffer?.Dispose();
        cBuffer?.Dispose();
    }
}