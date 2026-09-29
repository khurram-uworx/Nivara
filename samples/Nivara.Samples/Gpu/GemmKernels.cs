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

    // ── #440 register-blocked geometries — MEASURED, NOT ADOPTED ───────────────
    //
    // Do not route a runner at anything in this section. All four were gated and measured on
    // 2026-09-29 and every one of them is *slower* than Row4 on every shape: best cell 1.01x
    // on `laya head ff2` (the one already-weak shape, a wash inside run-to-run noise), typical
    // 0.6-0.85x, and the launch-bound narrow shapes (`head`, `scorer`) worst at 0.25-0.42x.
    // They are kept, and kept in the --gemm gate, so the result is reproducible and so the
    // next person to try a wider tile finds this instead of re-running the experiment. The
    // full ratio table is in tests/Nivara.PerformanceTests/README.md under "GEMM gate baseline
    // (2026-09-29)"; the analysis is docs/ACCELERATION.md section 5.2.
    //
    // They lose because the premise was wrong. These widen the per-thread output block beyond
    // Row4's 1x4 so a K step reads fewer shared floats per multiply-add
    // (GpuBuffers.GemmGeometry.SharedLoadsPerMac) - a 20-40% cut on paper. The binding
    // constraint on this iGPU is shared-memory *capacity per group*, not shared-memory
    // *traffic*. The isolating evidence: 2x2@K16 and 2x2@K32 have identical blocking factors,
    // so identical reads per MAC, and differ only in footprint (4 KB vs 8 KB per group), and
    // the 8 KB one is consistently 18-25% slower. 4x2@K32 has the best traffic ratio in the
    // family (0.75 reads/MAC), the worst footprint (12 KB), and lands mid-pack. Footprint
    // predicted the ordering; traffic did not.
    //
    // What they do still earn their keep: all four keep the incumbent's 16x16 group, K-tile
    // staging and strictly ascending-K accumulation, so each is *bit-identical* to Row4 on
    // every shape, not merely close - the same f32 values added to the same accumulator in
    // the same order. The --gemm gate asserts that byte-identity alongside its existing
    // double-precision truth check, which makes it a design regression detector rather than a
    // tolerance test. That assertion is what caught #468, an ILGPU OpenCL lowering bug that
    // returns plausible wrong values and cannot be seen in review: a tolerance-only gate would
    // have argued its way past it.
    //
    // The K tile is a separate axis from the blocking factor: 2x2 appears at both K16 and K32 so
    // the two effects can be attributed independently - which is exactly how the occupancy
    // result above was isolated. The body is duplicated per geometry rather than shared through
    // a multi-accumulator helper, because the epilogue and the unrolled inner loop are exactly
    // the parts a helper cannot express without reintroducing the local-memory spill that
    // register blocking exists to avoid.
    //
    // One implementation note that is not optional: each stages its A and B tiles through a
    // single SharedMemory.Allocate<float> with hand-computed 2D offsets. Two Allocate2D calls
    // in one kernel whose extents disagree get the second tile mis-placed by ILGPU's OpenCL
    // lowering (#468) - and the incumbent Row4 escapes only because its A tile is 16x16, i.e.
    // square, which happens to match its B tile's stride.

    /// <summary>2x2 @ K16: 32x32 output tile, 16x16 group. 4 accumulators, 1.0 shared reads/MAC. Measured slower than Row4 - not adopted.</summary>
    internal static void TiledGemmKernelReg2x2K16(
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> c,
        int aRows,
        int aCols,
        int bCols)
    {
        const int BR = 2, BC = 2, KT = 16;
        const int TileRows = TileSize * BR, TileCols = TileSize * BC;
        int x = Group.IdxX;
        int y = Group.IdxY;

        // Single 1D shared allocation, A tile then B tile; see the section banner for why
        // this is one Allocate and not two Allocate2D calls (#468).
        const int ASz = TileRows * KT;
        var smem = SharedMemory.Allocate<float>(ASz + KT * TileCols);

        int rowBase = Grid.IdxX * TileRows;
        int colBase = Grid.IdxY * TileCols;
        float acc00 = 0f, acc01 = 0f, acc10 = 0f, acc11 = 0f;

        for (int k0 = 0; k0 < aCols; k0 += KT)
        {
            // One 256-thread group stages TileRows*KT A values and KT*TileCols B values. Each
            // work item loads its own strided slice; the 2D bounds test zeroes the halo so
            // partial tiles contribute no spurious products, matching Row4's edge handling.
            for (int r = 0; r < TileRows; r += TileSize)
                for (int kk = 0; kk < KT; kk += TileSize)
                {
                    int srcRow = rowBase + r + x;
                    int srcCol = k0 + kk + y;
                    smem[(r + x) * KT + (kk + y)] = (srcRow < aRows && srcCol < aCols)
                        ? a[srcRow * aCols + srcCol] : 0f;
                }
            for (int kk = 0; kk < KT; kk += TileSize)
                for (int cc = 0; cc < TileCols; cc += TileSize)
                {
                    int srcRow = k0 + kk + x;
                    int srcCol = colBase + cc + y;
                    smem[ASz + (kk + x) * TileCols + (cc + y)] = (srcRow < aCols && srcCol < bCols)
                        ? b[srcRow * bCols + srcCol] : 0f;
                }
            Group.Barrier();

            for (int k = 0; k < KT; k++)
            {
                float a0 = smem[(x * BR + 0) * KT + k];
                float a1 = smem[(x * BR + 1) * KT + k];
                int cb = y * BC;
                acc00 += a0 * smem[ASz + k * TileCols + (cb + 0)];
                acc01 += a0 * smem[ASz + k * TileCols + (cb + 1)];
                acc10 += a1 * smem[ASz + k * TileCols + (cb + 0)];
                acc11 += a1 * smem[ASz + k * TileCols + (cb + 1)];
            }
            Group.Barrier();
        }

        int row0 = rowBase + x * BR + 0;
        if (row0 < aRows)
        {
            int dst = row0 * bCols + colBase + y * BC;
            if (colBase + y * BC + 0 < bCols) c[dst + 0] = acc00;
            if (colBase + y * BC + 1 < bCols) c[dst + 1] = acc01;
        }
        int row1 = rowBase + x * BR + 1;
        if (row1 < aRows)
        {
            int dst = row1 * bCols + colBase + y * BC;
            if (colBase + y * BC + 0 < bCols) c[dst + 0] = acc10;
            if (colBase + y * BC + 1 < bCols) c[dst + 1] = acc11;
        }
    }

    /// <summary>2x2 @ K32: 32x32 output tile, 16x16 group, double the K step. 4 accumulators. Measured slower than Row4 - not adopted.</summary>
    internal static void TiledGemmKernelReg2x2K32(
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> c,
        int aRows,
        int aCols,
        int bCols)
    {
        const int BR = 2, BC = 2, KT = 32;
        const int TileRows = TileSize * BR, TileCols = TileSize * BC;
        int x = Group.IdxX;
        int y = Group.IdxY;

        // Single 1D shared allocation, A tile then B tile; see the section banner for why
        // this is one Allocate and not two Allocate2D calls (#468).
        const int ASz = TileRows * KT;
        var smem = SharedMemory.Allocate<float>(ASz + KT * TileCols);

        int rowBase = Grid.IdxX * TileRows;
        int colBase = Grid.IdxY * TileCols;
        float acc00 = 0f, acc01 = 0f, acc10 = 0f, acc11 = 0f;

        for (int k0 = 0; k0 < aCols; k0 += KT)
        {
            for (int r = 0; r < TileRows; r += TileSize)
                for (int kk = 0; kk < KT; kk += TileSize)
                {
                    int srcRow = rowBase + r + x;
                    int srcCol = k0 + kk + y;
                    smem[(r + x) * KT + (kk + y)] = (srcRow < aRows && srcCol < aCols)
                        ? a[srcRow * aCols + srcCol] : 0f;
                }
            for (int kk = 0; kk < KT; kk += TileSize)
                for (int cc = 0; cc < TileCols; cc += TileSize)
                {
                    int srcRow = k0 + kk + x;
                    int srcCol = colBase + cc + y;
                    smem[ASz + (kk + x) * TileCols + (cc + y)] = (srcRow < aCols && srcCol < bCols)
                        ? b[srcRow * bCols + srcCol] : 0f;
                }
            Group.Barrier();

            for (int k = 0; k < KT; k++)
            {
                float a0 = smem[(x * BR + 0) * KT + k];
                float a1 = smem[(x * BR + 1) * KT + k];
                int cb = y * BC;
                acc00 += a0 * smem[ASz + k * TileCols + (cb + 0)];
                acc01 += a0 * smem[ASz + k * TileCols + (cb + 1)];
                acc10 += a1 * smem[ASz + k * TileCols + (cb + 0)];
                acc11 += a1 * smem[ASz + k * TileCols + (cb + 1)];
            }
            Group.Barrier();
        }

        for (int r = 0; r < BR; r++)
        {
            int outRow = rowBase + x * BR + r;
            if (outRow >= aRows) continue;
            int dst = outRow * bCols + colBase + y * BC;
            if (colBase + y * BC + 0 < bCols) c[dst + 0] = r == 0 ? acc00 : acc10;
            if (colBase + y * BC + 1 < bCols) c[dst + 1] = r == 0 ? acc01 : acc11;
        }
    }

    /// <summary>4x2 @ K32: 64x32 output tile, 16x16 group. 8 accumulators, 0.75 shared reads/MAC. Measured slower than Row4 - not adopted.</summary>
    internal static void TiledGemmKernelReg4x2K32(
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> c,
        int aRows,
        int aCols,
        int bCols)
    {
        const int BR = 4, BC = 2, KT = 32;
        const int TileRows = TileSize * BR, TileCols = TileSize * BC;
        int x = Group.IdxX;
        int y = Group.IdxY;

        // Single 1D shared allocation, A tile then B tile; see the section banner for why
        // this is one Allocate and not two Allocate2D calls (#468).
        const int ASz = TileRows * KT;
        var smem = SharedMemory.Allocate<float>(ASz + KT * TileCols);

        int rowBase = Grid.IdxX * TileRows;
        int colBase = Grid.IdxY * TileCols;
        float acc00 = 0f, acc01 = 0f, acc10 = 0f, acc11 = 0f;
        float acc20 = 0f, acc21 = 0f, acc30 = 0f, acc31 = 0f;

        for (int k0 = 0; k0 < aCols; k0 += KT)
        {
            for (int r = 0; r < TileRows; r += TileSize)
                for (int kk = 0; kk < KT; kk += TileSize)
                {
                    int srcRow = rowBase + r + x;
                    int srcCol = k0 + kk + y;
                    smem[(r + x) * KT + (kk + y)] = (srcRow < aRows && srcCol < aCols)
                        ? a[srcRow * aCols + srcCol] : 0f;
                }
            for (int kk = 0; kk < KT; kk += TileSize)
                for (int cc = 0; cc < TileCols; cc += TileSize)
                {
                    int srcRow = k0 + kk + x;
                    int srcCol = colBase + cc + y;
                    smem[ASz + (kk + x) * TileCols + (cc + y)] = (srcRow < aCols && srcCol < bCols)
                        ? b[srcRow * bCols + srcCol] : 0f;
                }
            Group.Barrier();

            for (int k = 0; k < KT; k++)
            {
                int cb = y * BC;
                float a0 = smem[(x * BR + 0) * KT + k];
                acc00 += a0 * smem[ASz + k * TileCols + (cb + 0)];
                acc01 += a0 * smem[ASz + k * TileCols + (cb + 1)];
                float a1 = smem[(x * BR + 1) * KT + k];
                acc10 += a1 * smem[ASz + k * TileCols + (cb + 0)];
                acc11 += a1 * smem[ASz + k * TileCols + (cb + 1)];
                float a2 = smem[(x * BR + 2) * KT + k];
                acc20 += a2 * smem[ASz + k * TileCols + (cb + 0)];
                acc21 += a2 * smem[ASz + k * TileCols + (cb + 1)];
                float a3 = smem[(x * BR + 3) * KT + k];
                acc30 += a3 * smem[ASz + k * TileCols + (cb + 0)];
                acc31 += a3 * smem[ASz + k * TileCols + (cb + 1)];
            }
            Group.Barrier();
        }

        int d = colBase + y * BC;
        int rowBase0 = rowBase + x * BR;
        if (rowBase0 < aRows)
        {
            int dst = rowBase0 * bCols + d;
            if (d + 0 < bCols) c[dst + 0] = acc00;
            if (d + 1 < bCols) c[dst + 1] = acc01;
        }
        if (rowBase0 + 1 < aRows)
        {
            int dst = (rowBase0 + 1) * bCols + d;
            if (d + 0 < bCols) c[dst + 0] = acc10;
            if (d + 1 < bCols) c[dst + 1] = acc11;
        }
        if (rowBase0 + 2 < aRows)
        {
            int dst = (rowBase0 + 2) * bCols + d;
            if (d + 0 < bCols) c[dst + 0] = acc20;
            if (d + 1 < bCols) c[dst + 1] = acc21;
        }
        if (rowBase0 + 3 < aRows)
        {
            int dst = (rowBase0 + 3) * bCols + d;
            if (d + 0 < bCols) c[dst + 0] = acc30;
            if (d + 1 < bCols) c[dst + 1] = acc31;
        }
    }

    /// <summary>1x8 @ K16: 16x128 output tile, 16x16 group. 8 accumulators. Measured slower than Row4 - not adopted.</summary>
    internal static void TiledGemmKernelReg1x8K16(
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> c,
        int aRows,
        int aCols,
        int bCols)
    {
        const int BR = 1, BC = 8, KT = 16;
        const int TileRows = TileSize * BR, TileCols = TileSize * BC;
        int x = Group.IdxX;
        int y = Group.IdxY;

        // Single 1D shared allocation, A tile then B tile; see the section banner for why
        // this is one Allocate and not two Allocate2D calls (#468).
        const int ASz = TileRows * KT;
        var smem = SharedMemory.Allocate<float>(ASz + KT * TileCols);

        int rowBase = Grid.IdxX * TileRows;
        int colBase = Grid.IdxY * TileCols;
        float acc0 = 0f, acc1 = 0f, acc2 = 0f, acc3 = 0f;
        float acc4 = 0f, acc5 = 0f, acc6 = 0f, acc7 = 0f;

        for (int k0 = 0; k0 < aCols; k0 += KT)
        {
            for (int r = 0; r < TileRows; r += TileSize)
                for (int kk = 0; kk < KT; kk += TileSize)
                {
                    int srcRow = rowBase + r + x;
                    int srcCol = k0 + kk + y;
                    smem[(r + x) * KT + (kk + y)] = (srcRow < aRows && srcCol < aCols)
                        ? a[srcRow * aCols + srcCol] : 0f;
                }
            for (int kk = 0; kk < KT; kk += TileSize)
                for (int cc = 0; cc < TileCols; cc += TileSize)
                {
                    int srcRow = k0 + kk + x;
                    int srcCol = colBase + cc + y;
                    smem[ASz + (kk + x) * TileCols + (cc + y)] = (srcRow < aCols && srcCol < bCols)
                        ? b[srcRow * bCols + srcCol] : 0f;
                }
            Group.Barrier();

            for (int k = 0; k < KT; k++)
            {
                float aVal = smem[(x * BR) * KT + k];
                int cb = y * BC;
                acc0 += aVal * smem[ASz + k * TileCols + (cb + 0)];
                acc1 += aVal * smem[ASz + k * TileCols + (cb + 1)];
                acc2 += aVal * smem[ASz + k * TileCols + (cb + 2)];
                acc3 += aVal * smem[ASz + k * TileCols + (cb + 3)];
                acc4 += aVal * smem[ASz + k * TileCols + (cb + 4)];
                acc5 += aVal * smem[ASz + k * TileCols + (cb + 5)];
                acc6 += aVal * smem[ASz + k * TileCols + (cb + 6)];
                acc7 += aVal * smem[ASz + k * TileCols + (cb + 7)];
            }
            Group.Barrier();
        }

        int outRow = rowBase + x;
        if (outRow < aRows)
        {
            int dst = outRow * bCols + colBase + y * BC;
            if (colBase + y * BC + 0 < bCols) c[dst + 0] = acc0;
            if (colBase + y * BC + 1 < bCols) c[dst + 1] = acc1;
            if (colBase + y * BC + 2 < bCols) c[dst + 2] = acc2;
            if (colBase + y * BC + 3 < bCols) c[dst + 3] = acc3;
            if (colBase + y * BC + 4 < bCols) c[dst + 4] = acc4;
            if (colBase + y * BC + 5 < bCols) c[dst + 5] = acc5;
            if (colBase + y * BC + 6 < bCols) c[dst + 6] = acc6;
            if (colBase + y * BC + 7 < bCols) c[dst + 7] = acc7;
        }
    }

    /// <summary>
    /// Register-blocked 1×4 GEMM with a fused bias epilogue: y = A·Bt + bias. Same tile
    /// geometry as <see cref="TiledGemmKernelRow4"/>; the bias row is read per output
    /// column and added to the register accumulator before the epilogue write, so no
    /// separate bias launch is needed. This is the lean identity epilogue — one of three
    /// fused siblings (bias-only, GELU, ReLU) so each launch only compiles the epilogue
    /// it needs (a single shared kernel with a runtime activation byte would inline all
    /// three paths, bloating registers on the hot lean launches). Elementwise results are
    /// bit-identical to the unfused GEMM + AddBias sequence: register acc + bias[c]
    /// versus stored-then-added.
    /// </summary>
    internal static void TiledGemmKernelRow4Bias(
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> c,
        ArrayView<float> bias,
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
            int c0 = colBase + y * BlockCols + 0;
            int c1 = colBase + y * BlockCols + 1;
            int c2 = colBase + y * BlockCols + 2;
            int c3 = colBase + y * BlockCols + 3;
            float b0 = c0 < bCols ? bias[c0] : 0f;
            float b1 = c1 < bCols ? bias[c1] : 0f;
            float b2 = c2 < bCols ? bias[c2] : 0f;
            float b3 = c3 < bCols ? bias[c3] : 0f;

            if (c0 < bCols) c[rowBase + 0] = acc0 + b0;
            if (c1 < bCols) c[rowBase + 1] = acc1 + b1;
            if (c2 < bCols) c[rowBase + 2] = acc2 + b2;
            if (c3 < bCols) c[rowBase + 3] = acc3 + b3;
        }
    }

    /// <summary>As <see cref="TiledGemmKernelRow4Bias"/> with a fused exact-GELU epilogue:
    /// y = gelu(A·Bt + bias), same A–S 7.1.26 polynomial as <see cref="ElementwiseKernels.Gelu"/>.</summary>
    internal static void TiledGemmKernelRow4Gelu(
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> c,
        ArrayView<float> bias,
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
            int c0 = colBase + y * BlockCols + 0;
            int c1 = colBase + y * BlockCols + 1;
            int c2 = colBase + y * BlockCols + 2;
            int c3 = colBase + y * BlockCols + 3;
            float b0 = c0 < bCols ? bias[c0] : 0f;
            float b1 = c1 < bCols ? bias[c1] : 0f;
            float b2 = c2 < bCols ? bias[c2] : 0f;
            float b3 = c3 < bCols ? bias[c3] : 0f;

            if (c0 < bCols) c[rowBase + 0] = ElementwiseKernels.GeluExact(acc0 + b0);
            if (c1 < bCols) c[rowBase + 1] = ElementwiseKernels.GeluExact(acc1 + b1);
            if (c2 < bCols) c[rowBase + 2] = ElementwiseKernels.GeluExact(acc2 + b2);
            if (c3 < bCols) c[rowBase + 3] = ElementwiseKernels.GeluExact(acc3 + b3);
        }
    }

    /// <summary>As <see cref="TiledGemmKernelRow4Bias"/> with a fused ReLU epilogue:
    /// y = max(A·Bt + bias, 0).</summary>
    internal static void TiledGemmKernelRow4Relu(
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> c,
        ArrayView<float> bias,
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
            int c0 = colBase + y * BlockCols + 0;
            int c1 = colBase + y * BlockCols + 1;
            int c2 = colBase + y * BlockCols + 2;
            int c3 = colBase + y * BlockCols + 3;
            float b0 = c0 < bCols ? bias[c0] : 0f;
            float b1 = c1 < bCols ? bias[c1] : 0f;
            float b2 = c2 < bCols ? bias[c2] : 0f;
            float b3 = c3 < bCols ? bias[c3] : 0f;

            float r0 = acc0 + b0; if (r0 < 0f) r0 = 0f;
            float r1 = acc1 + b1; if (r1 < 0f) r1 = 0f;
            float r2 = acc2 + b2; if (r2 < 0f) r2 = 0f;
            float r3 = acc3 + b3; if (r3 < 0f) r3 = 0f;

            if (c0 < bCols) c[rowBase + 0] = r0;
            if (c1 < bCols) c[rowBase + 1] = r1;
            if (c2 < bCols) c[rowBase + 2] = r2;
            if (c3 < bCols) c[rowBase + 3] = r3;
        }
    }

    /// <summary>
    /// Register-blocked 1×4 GEMM with a fused bias epilogue that writes into a
    /// block-separable layout: y = A·Bt + bias where the [aRows × bCols] result is
    /// scattered across bCols/blockWidth dense column blocks, each [aRows × blockWidth]
    /// stored consecutively in <paramref name="c"/>. Used by the q/k/v projections so one
    /// GEMM produces [q-block | k-block | v-block] and the attention kernel reads each
    /// block as a dense [rows × hidden] SubView (M2, issue #437). Same tile geometry and
    /// K-reduction order as <see cref="TiledGemmKernelRow4"/>, so elementwise results are
    /// bit-identical to three separate GEMM + AddBias sequences.
    /// </summary>
    internal static void TiledGemmKernelRow4Qkv(
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> c,
        ArrayView<float> bias,
        int aRows,
        int aCols,
        int bCols,
        int blockWidth)
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

            if (c0 < bCols) c[QkvDest(c0, outRow, aRows, blockWidth)] = acc0 + b0;
            if (c1 < bCols) c[QkvDest(c1, outRow, aRows, blockWidth)] = acc1 + b1;
            if (c2 < bCols) c[QkvDest(c2, outRow, aRows, blockWidth)] = acc2 + b2;
            if (c3 < bCols) c[QkvDest(c3, outRow, aRows, blockWidth)] = acc3 + b3;
        }
    }

    /// <summary>Destination index of output column j (of bCols = 3·blockWidth) in the
    /// block-separable layout: block j/blockWidth starts at block · (aRows · blockWidth)
    /// and stores column j % blockWidth of every row at aRows stride.</summary>
    static int QkvDest(int j, int outRow, int aRows, int blockWidth)
    {
        int block = j / blockWidth;
        int colInBlock = j - block * blockWidth;
        return block * aRows * blockWidth + outRow * blockWidth + colInBlock;
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
