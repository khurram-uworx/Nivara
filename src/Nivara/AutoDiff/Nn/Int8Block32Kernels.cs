namespace Nivara.AutoDiff.Nn;

/// <summary>
/// CPU INT8 block-32 fused GEMV kernels (format <c>SilkInt8Block32 v1</c>). Dequantization is
/// fused per 32-element block into a temporary buffer, then a SIMD matvec (TensorPrimitives/
/// TensorsHelper) accumulates. Weights stay INT8; only a 32-row block is materialized at a time.
/// </summary>
internal static class Int8Block32Kernels
{
    public static void MatMulTransposedB(ReadOnlySpan<float> a, Int8Block32Weight w, Span<float> output,
        int aRows, int aCols, int bCols)
    {
        if (aRows < 0) throw new ArgumentOutOfRangeException(nameof(aRows));
        if (aCols <= 0) throw new ArgumentOutOfRangeException(nameof(aCols));
        if (bCols <= 0) throw new ArgumentOutOfRangeException(nameof(bCols));
        if (a.Length < aRows * aCols) throw new ArgumentException("a too small", nameof(a));
        if (output.Length < aRows * bCols) throw new ArgumentException("output too small", nameof(output));
        if (w.InCols != aCols) throw new ArgumentException("InCols mismatch", nameof(w));
        if (w.OutCols != bCols) throw new ArgumentException("OutCols mismatch", nameof(w));

        int numBlocks = w.NumBlocks;
        int blockSize = Int8Block32Weight.BlockSize;

        // temp dequant buffer for one block: [blockSize, bCols]
        float[] dequant = System.Buffers.ArrayPool<float>.Shared.Rent(blockSize * bCols);
        // temp result for one block matvec
        float[] blockResult = System.Buffers.ArrayPool<float>.Shared.Rent(bCols);
        try
        {
            for (int r = 0; r < aRows; r++)
            {
                ReadOnlySpan<float> aRow = a.Slice(r * aCols, aCols);
                Span<float> outRow = output.Slice(r * bCols, bCols);
                for (int c = 0; c < bCols; c++) outRow[c] = 0f;

                for (int kb = 0; kb < numBlocks; kb++)
                {
                    int kStart = kb * blockSize;
                    int kEnd = Math.Min(kStart + blockSize, aCols);
                    int bs = kEnd - kStart;
                    if (bs <= 0) break;

                    // dequant this block: weights [kStart..kEnd, :] * scale
                    for (int t = 0; t < bs; t++)
                    {
                        int k = kStart + t;
                        int baseIdx = k * bCols;
                        for (int c = 0; c < bCols; c++)
                        {
                            float scale = w.Scales[c * numBlocks + kb];
                            dequant[t * bCols + c] = w.Data[baseIdx + c] * scale;
                        }
                    }

                    ReadOnlySpan<float> aBlock = aRow.Slice(kStart, bs);
                    // matvec: aBlock[1,bs] @ dequant[bs,bCols] (b not transposed) -> blockResult[bCols]
                    // Use TensorsHelper via GradKernels? GradKernels.MatMulTransposedB assumes transposed b.
                    // We have dequant as [bs, bCols] = [aCols, bCols] row-major = aCols-major.
                    // Need a[1,bs] @ dequant^T where dequant^T is [bCols, bs]. 
                    // So transpose dequant block to [bCols, bs] first, or use MultiplyCore bTransposed=false.
                    // Simpler: call TensorsHelper.MultiplyCore with bTransposed=false.
                    Nivara.Tensors.TensorsHelper.MultiplyCore<float>(
                        aBlock, dequant.AsSpan(0, bs * bCols), blockResult,
                        1, bs, bCols, bTransposed: false);

                    for (int c = 0; c < bCols; c++) outRow[c] += blockResult[c];
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(dequant, clearArray: false);
            System.Buffers.ArrayPool<float>.Shared.Return(blockResult, clearArray: false);
        }
    }
}
