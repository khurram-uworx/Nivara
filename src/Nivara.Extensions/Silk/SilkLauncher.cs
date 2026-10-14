namespace Nivara.Silk;

/// <summary>
/// Typed dispatch shared by the three Silk runners. Each method sets one kernel's arguments and
/// runs it over the geometry <see cref="SilkKernels"/> owns, so a runner cannot drift from the
/// kernel's argument order or its grid. The runners keep their own <c>Launch*</c> wrappers where
/// the call closes over runner state (hidden size, epsilon, scale, attention band).
/// </summary>
internal static class SilkLauncher
{
    public static void Gather(SilkRuntime runtime, nint table, nint ids, nint y, int n, int hidden)
    {
        runtime.SetBuffer(SilkKernels.GatherRows, 0, table);
        runtime.SetBuffer(SilkKernels.GatherRows, 1, ids);
        runtime.SetBuffer(SilkKernels.GatherRows, 2, y);
        runtime.SetUInt32(SilkKernels.GatherRows, 3, (uint)n);
        runtime.SetUInt32(SilkKernels.GatherRows, 4, (uint)hidden);
        Linear(runtime, SilkKernels.GatherRows, n);
    }

    public static void LayerNorm(
        SilkRuntime runtime, nint src, nint gamma, nint beta, nint y, int rows, int cols, float eps)
    {
        runtime.SetBuffer(SilkKernels.LayerNorm, 0, src);
        runtime.SetBuffer(SilkKernels.LayerNorm, 1, gamma);
        runtime.SetBuffer(SilkKernels.LayerNorm, 2, beta);
        runtime.SetBuffer(SilkKernels.LayerNorm, 3, y);
        runtime.SetUInt32(SilkKernels.LayerNorm, 4, (uint)rows);
        runtime.SetUInt32(SilkKernels.LayerNorm, 5, (uint)cols);
        runtime.SetFloat32(SilkKernels.LayerNorm, 6, eps);
        Linear(runtime, SilkKernels.LayerNorm, rows);
    }

    public static void LayerNormResidual(
        SilkRuntime runtime, nint a, nint b, nint gamma, nint beta, nint y, int rows, int cols, float eps)
    {
        runtime.SetBuffer(SilkKernels.LayerNormResidual, 0, a);
        runtime.SetBuffer(SilkKernels.LayerNormResidual, 1, b);
        runtime.SetBuffer(SilkKernels.LayerNormResidual, 2, gamma);
        runtime.SetBuffer(SilkKernels.LayerNormResidual, 3, beta);
        runtime.SetBuffer(SilkKernels.LayerNormResidual, 4, y);
        runtime.SetUInt32(SilkKernels.LayerNormResidual, 5, (uint)rows);
        runtime.SetUInt32(SilkKernels.LayerNormResidual, 6, (uint)cols);
        runtime.SetFloat32(SilkKernels.LayerNormResidual, 7, eps);
        Linear(runtime, SilkKernels.LayerNormResidual, rows);
    }

    /// <summary>
    /// The bias-free ModernBERT GEMM: three buffers and the three dimensions, no epilogue.
    /// </summary>
    public static void Gemm(SilkRuntime runtime, nint a, nint b, nint c, int aRows, int aCols, int bCols)
    {
        runtime.SetBuffer(SilkKernels.Gemm, 0, a);
        runtime.SetBuffer(SilkKernels.Gemm, 1, b);
        runtime.SetBuffer(SilkKernels.Gemm, 2, c);
        runtime.SetUInt32(SilkKernels.Gemm, 3, (uint)aRows);
        runtime.SetUInt32(SilkKernels.Gemm, 4, (uint)aCols);
        runtime.SetUInt32(SilkKernels.Gemm, 5, (uint)bCols);
        GemmGrid(runtime, SilkKernels.Gemm, aRows, bCols);
    }

    /// <summary>
    /// A Row4 epilogue GEMM (<c>gemm_row4_bias</c>, <c>gemm_row4_gelu</c>, <c>gemm_row4_relu</c>,
    /// <c>gemm_row4_qkv</c>): the six shared arguments plus, when <paramref name="blockWidth"/> is
    /// set, the extra block width <c>gemm_row4_qkv</c> takes.
    /// </summary>
    public static void GemmEpilogue(
        SilkRuntime runtime, string kernel, nint a, nint b, nint c, nint bias,
        int aRows, int aCols, int bCols, int? blockWidth = null)
    {
        runtime.SetBuffer(kernel, 0, a);
        runtime.SetBuffer(kernel, 1, b);
        runtime.SetBuffer(kernel, 2, c);
        runtime.SetBuffer(kernel, 3, bias);
        runtime.SetUInt32(kernel, 4, (uint)aRows);
        runtime.SetUInt32(kernel, 5, (uint)aCols);
        runtime.SetUInt32(kernel, 6, (uint)bCols);
        if (blockWidth is int width)
            runtime.SetUInt32(kernel, 7, (uint)width);
        GemmGrid(runtime, kernel, aRows, bCols);
    }

    public static void Attention(
        SilkRuntime runtime, nint q, nint k, nint v, nint mask, nint output,
        int batch, int seqLen, int numHeads, int headDim, int band, float scale)
    {
        runtime.SetBuffer(SilkKernels.Attention, 0, q);
        runtime.SetBuffer(SilkKernels.Attention, 1, k);
        runtime.SetBuffer(SilkKernels.Attention, 2, v);
        runtime.SetBuffer(SilkKernels.Attention, 3, mask);
        runtime.SetBuffer(SilkKernels.Attention, 4, output);
        runtime.SetUInt32(SilkKernels.Attention, 5, (uint)batch);
        runtime.SetUInt32(SilkKernels.Attention, 6, (uint)seqLen);
        runtime.SetUInt32(SilkKernels.Attention, 7, (uint)numHeads);
        runtime.SetUInt32(SilkKernels.Attention, 8, (uint)headDim);
        runtime.SetInt32(SilkKernels.Attention, 9, band);
        runtime.SetFloat32(SilkKernels.Attention, 10, scale);
        runtime.Run(
            SilkKernels.Attention,
            [(nuint)SilkKernels.AttentionGlobal(batch * numHeads * seqLen)],
            [(nuint)SilkKernels.AttentionGroupSize]);
    }

    public static void Add(SilkRuntime runtime, nint a, nint b, nint y, int n)
    {
        runtime.SetBuffer(SilkKernels.Add, 0, a);
        runtime.SetBuffer(SilkKernels.Add, 1, b);
        runtime.SetBuffer(SilkKernels.Add, 2, y);
        runtime.SetUInt32(SilkKernels.Add, 3, (uint)n);
        Linear(runtime, SilkKernels.Add, n);
    }

    public static void AddBias(SilkRuntime runtime, nint x, nint bias, nint y, int rows, int cols)
    {
        runtime.SetBuffer(SilkKernels.AddBias, 0, x);
        runtime.SetBuffer(SilkKernels.AddBias, 1, bias);
        runtime.SetBuffer(SilkKernels.AddBias, 2, y);
        runtime.SetUInt32(SilkKernels.AddBias, 3, (uint)rows);
        runtime.SetUInt32(SilkKernels.AddBias, 4, (uint)cols);
        Linear(runtime, SilkKernels.AddBias, rows * cols);
    }

    public static void Split(SilkRuntime runtime, nint src, nint dst, int rows, int cols, int parts, int part)
    {
        runtime.SetBuffer(SilkKernels.SplitColumns, 0, src);
        runtime.SetBuffer(SilkKernels.SplitColumns, 1, dst);
        runtime.SetUInt32(SilkKernels.SplitColumns, 2, (uint)rows);
        runtime.SetUInt32(SilkKernels.SplitColumns, 3, (uint)cols);
        runtime.SetUInt32(SilkKernels.SplitColumns, 4, (uint)parts);
        runtime.SetUInt32(SilkKernels.SplitColumns, 5, (uint)part);
        Linear(runtime, SilkKernels.SplitColumns, rows * cols);
    }

    public static void Rotary(
        SilkRuntime runtime, nint buffer, nint posIds, nint cos, nint sin, int rows, int numHeads, int headDim, int halfDim)
    {
        runtime.SetBuffer(SilkKernels.Rotary, 0, buffer);
        runtime.SetBuffer(SilkKernels.Rotary, 1, posIds);
        runtime.SetBuffer(SilkKernels.Rotary, 2, cos);
        runtime.SetBuffer(SilkKernels.Rotary, 3, sin);
        runtime.SetBuffer(SilkKernels.Rotary, 4, buffer);
        runtime.SetUInt32(SilkKernels.Rotary, 5, (uint)rows);
        runtime.SetUInt32(SilkKernels.Rotary, 6, (uint)numHeads);
        runtime.SetUInt32(SilkKernels.Rotary, 7, (uint)headDim);
        Linear(runtime, SilkKernels.Rotary, rows * numHeads * halfDim);
    }

    public static void GeGlu(SilkRuntime runtime, nint src, nint y, int rows, int intermediate)
    {
        runtime.SetBuffer(SilkKernels.GeGlu, 0, src);
        runtime.SetBuffer(SilkKernels.GeGlu, 1, y);
        runtime.SetUInt32(SilkKernels.GeGlu, 2, (uint)rows);
        runtime.SetUInt32(SilkKernels.GeGlu, 3, (uint)intermediate);
        Linear(runtime, SilkKernels.GeGlu, rows * intermediate);
    }

    public static void EmbedSum(
        SilkRuntime runtime, nint ids, nint posIds, nint word, nint pos, nint tokenType, nint y,
        int total, int hidden, int includeTokenType)
    {
        runtime.SetBuffer(SilkKernels.EmbedSum, 0, ids);
        runtime.SetBuffer(SilkKernels.EmbedSum, 1, posIds);
        runtime.SetBuffer(SilkKernels.EmbedSum, 2, word);
        runtime.SetBuffer(SilkKernels.EmbedSum, 3, pos);
        runtime.SetBuffer(SilkKernels.EmbedSum, 4, tokenType);
        runtime.SetBuffer(SilkKernels.EmbedSum, 5, y);
        runtime.SetUInt32(SilkKernels.EmbedSum, 6, (uint)total);
        runtime.SetUInt32(SilkKernels.EmbedSum, 7, (uint)hidden);
        runtime.SetUInt32(SilkKernels.EmbedSum, 8, (uint)includeTokenType);
        Linear(runtime, SilkKernels.EmbedSum, total);
    }

    /// <summary>
    /// Allocates a buffer, uploads <paramref name="data"/>, and releases the buffer if the upload
    /// throws. Callers that track owned buffers separately (the BERT encoder runner) do not use this.
    /// </summary>
    public static nint Upload(SilkRuntime runtime, float[] data)
    {
        nint buffer = runtime.Allocate(data.Length);
        try
        {
            runtime.Upload(buffer, data);
            return buffer;
        }
        catch
        {
            runtime.Release(buffer);
            throw;
        }
    }

    static void Linear(SilkRuntime runtime, string kernel, int total)
        => runtime.Run(
            kernel,
            [(nuint)SilkKernels.LinearGlobal(total)],
            [(nuint)SilkKernels.LinearGroupSize]);

    static void GemmGrid(SilkRuntime runtime, string kernel, int aRows, int bCols)
        => runtime.Run(
            kernel,
            [(nuint)SilkKernels.GemmGlobalRows(aRows), (nuint)SilkKernels.GemmGlobalCols(bCols)],
            [(nuint)SilkKernels.GemmTileSize, (nuint)SilkKernels.GemmTileSize]);
}
