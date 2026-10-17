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

    public static void RmsNorm(SilkRuntime runtime, nint src, nint gamma, nint y, int rows, int cols, float eps)
    {
        runtime.SetBuffer(SilkKernels.RmsNorm, 0, src);
        runtime.SetBuffer(SilkKernels.RmsNorm, 1, gamma);
        runtime.SetBuffer(SilkKernels.RmsNorm, 2, y);
        runtime.SetUInt32(SilkKernels.RmsNorm, 3, (uint)rows);
        runtime.SetUInt32(SilkKernels.RmsNorm, 4, (uint)cols);
        runtime.SetFloat32(SilkKernels.RmsNorm, 5, eps);
        Linear(runtime, SilkKernels.RmsNorm, rows);
    }

    public static void Silu(SilkRuntime runtime, nint src, nint y, int n)
    {
        runtime.SetBuffer(SilkKernels.Silu, 0, src);
        runtime.SetBuffer(SilkKernels.Silu, 1, y);
        runtime.SetUInt32(SilkKernels.Silu, 2, (uint)n);
        Linear(runtime, SilkKernels.Silu, n);
    }

    public static void SwiGlu(SilkRuntime runtime, nint gate, nint up, nint y, int rows, int intermediate)
    {
        runtime.SetBuffer(SilkKernels.SwiGlu, 0, gate);
        runtime.SetBuffer(SilkKernels.SwiGlu, 1, up);
        runtime.SetBuffer(SilkKernels.SwiGlu, 2, y);
        runtime.SetUInt32(SilkKernels.SwiGlu, 3, (uint)rows);
        runtime.SetUInt32(SilkKernels.SwiGlu, 4, (uint)intermediate);
        Linear(runtime, SilkKernels.SwiGlu, rows * intermediate);
    }

    public static void CausalGqaAttention(
        SilkRuntime runtime, nint q, nint k, nint v, nint output,
        int seqLen, int numHeads, int numKvHeads, int headDim, float scale)
    {
        runtime.SetBuffer(SilkKernels.CausalGqaAttention, 0, q);
        runtime.SetBuffer(SilkKernels.CausalGqaAttention, 1, k);
        runtime.SetBuffer(SilkKernels.CausalGqaAttention, 2, v);
        runtime.SetBuffer(SilkKernels.CausalGqaAttention, 3, output);
        runtime.SetUInt32(SilkKernels.CausalGqaAttention, 4, (uint)seqLen);
        runtime.SetUInt32(SilkKernels.CausalGqaAttention, 5, (uint)numHeads);
        runtime.SetUInt32(SilkKernels.CausalGqaAttention, 6, (uint)numKvHeads);
        runtime.SetUInt32(SilkKernels.CausalGqaAttention, 7, (uint)headDim);
        runtime.SetFloat32(SilkKernels.CausalGqaAttention, 8, scale);
        int total = seqLen * numHeads;
        runtime.Run(
            SilkKernels.CausalGqaAttention,
            [(nuint)SilkKernels.LinearGlobal(total)],
            [(nuint)SilkKernels.LinearGroupSize]);
    }

    public static void DecodeGqaAttention(
        SilkRuntime runtime, nint q, nint kCache, nint vCache, nint output,
        int kvLen, int numHeads, int numKvHeads, int headDim, float scale)
    {
        runtime.SetBuffer(SilkKernels.DecodeGqaAttention, 0, q);
        runtime.SetBuffer(SilkKernels.DecodeGqaAttention, 1, kCache);
        runtime.SetBuffer(SilkKernels.DecodeGqaAttention, 2, vCache);
        runtime.SetBuffer(SilkKernels.DecodeGqaAttention, 3, output);
        runtime.SetUInt32(SilkKernels.DecodeGqaAttention, 4, (uint)kvLen);
        runtime.SetUInt32(SilkKernels.DecodeGqaAttention, 5, (uint)numHeads);
        runtime.SetUInt32(SilkKernels.DecodeGqaAttention, 6, (uint)numKvHeads);
        runtime.SetUInt32(SilkKernels.DecodeGqaAttention, 7, (uint)headDim);
        runtime.SetFloat32(SilkKernels.DecodeGqaAttention, 8, scale);
        int total = numHeads;
        runtime.Run(
            SilkKernels.DecodeGqaAttention,
            [(nuint)SilkKernels.LinearGlobal(total)],
            [(nuint)SilkKernels.LinearGroupSize]);
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

    public static void Conv2d(
        SilkRuntime runtime, nint input, nint weight, nint bias, nint output,
        int N, int C, int H, int W, int outC, int kH, int kW,
        int strideH, int strideW, int padTop, int padBottom, int padLeft, int padRight,
        int groups, bool hasBias, int oH, int oW)
    {
        runtime.SetBuffer(SilkKernels.Conv2d, 0, input);
        runtime.SetBuffer(SilkKernels.Conv2d, 1, weight);
        runtime.SetBuffer(SilkKernels.Conv2d, 2, bias);
        runtime.SetBuffer(SilkKernels.Conv2d, 3, output);
        runtime.SetUInt32(SilkKernels.Conv2d, 4, (uint)N);
        runtime.SetUInt32(SilkKernels.Conv2d, 5, (uint)C);
        runtime.SetUInt32(SilkKernels.Conv2d, 6, (uint)H);
        runtime.SetUInt32(SilkKernels.Conv2d, 7, (uint)W);
        runtime.SetUInt32(SilkKernels.Conv2d, 8, (uint)outC);
        runtime.SetUInt32(SilkKernels.Conv2d, 9, (uint)kH);
        runtime.SetUInt32(SilkKernels.Conv2d, 10, (uint)kW);
        runtime.SetUInt32(SilkKernels.Conv2d, 11, (uint)strideH);
        runtime.SetUInt32(SilkKernels.Conv2d, 12, (uint)strideW);
        runtime.SetUInt32(SilkKernels.Conv2d, 13, (uint)padTop);
        runtime.SetUInt32(SilkKernels.Conv2d, 14, (uint)padBottom);
        runtime.SetUInt32(SilkKernels.Conv2d, 15, (uint)padLeft);
        runtime.SetUInt32(SilkKernels.Conv2d, 16, (uint)padRight);
        runtime.SetUInt32(SilkKernels.Conv2d, 17, (uint)groups);
        runtime.SetUInt32(SilkKernels.Conv2d, 18, hasBias ? 1u : 0u);
        runtime.SetUInt32(SilkKernels.Conv2d, 19, (uint)oH);
        runtime.SetUInt32(SilkKernels.Conv2d, 20, (uint)oW);
        int total = N * outC * oH * oW;
        runtime.Run(SilkKernels.Conv2d, [(nuint)SilkKernels.LinearGlobal(total)], [(nuint)SilkKernels.LinearGroupSize]);
    }

    public static void BatchNormEval(
        SilkRuntime runtime, nint x, nint mean, nint var, nint gamma, nint beta, nint y,
        int N, int C, int H, int W, float eps, bool hasAffine)
    {
        int HW = H * W;
        runtime.SetBuffer(SilkKernels.BatchNormEval, 0, x);
        runtime.SetBuffer(SilkKernels.BatchNormEval, 1, mean);
        runtime.SetBuffer(SilkKernels.BatchNormEval, 2, var);
        runtime.SetBuffer(SilkKernels.BatchNormEval, 3, gamma);
        runtime.SetBuffer(SilkKernels.BatchNormEval, 4, beta);
        runtime.SetBuffer(SilkKernels.BatchNormEval, 5, y);
        runtime.SetUInt32(SilkKernels.BatchNormEval, 6, (uint)N);
        runtime.SetUInt32(SilkKernels.BatchNormEval, 7, (uint)C);
        runtime.SetUInt32(SilkKernels.BatchNormEval, 8, (uint)HW);
        runtime.SetFloat32(SilkKernels.BatchNormEval, 9, eps);
        runtime.SetUInt32(SilkKernels.BatchNormEval, 10, hasAffine ? 1u : 0u);
        int total = N * C * HW;
        runtime.Run(SilkKernels.BatchNormEval, [(nuint)SilkKernels.LinearGlobal(total)], [(nuint)SilkKernels.LinearGroupSize]);
    }

    public static void Relu(SilkRuntime runtime, nint x, nint y, int n)
    {
        runtime.SetBuffer(SilkKernels.Relu, 0, x);
        runtime.SetBuffer(SilkKernels.Relu, 1, y);
        runtime.SetUInt32(SilkKernels.Relu, 2, (uint)n);
        runtime.Run(SilkKernels.Relu, [(nuint)SilkKernels.LinearGlobal(n)], [(nuint)SilkKernels.LinearGroupSize]);
    }

    public static void Relu6(SilkRuntime runtime, nint x, nint y, int n)
    {
        runtime.SetBuffer(SilkKernels.Relu6, 0, x);
        runtime.SetBuffer(SilkKernels.Relu6, 1, y);
        runtime.SetUInt32(SilkKernels.Relu6, 2, (uint)n);
        runtime.Run(SilkKernels.Relu6, [(nuint)SilkKernels.LinearGlobal(n)], [(nuint)SilkKernels.LinearGroupSize]);
    }

    public static void MaxPool2d(
        SilkRuntime runtime, nint input, nint output,
        int N, int C, int H, int W, int kH, int kW, int strideH, int strideW,
        int padTop, int padBottom, int padLeft, int padRight, int oH, int oW)
    {
        runtime.SetBuffer(SilkKernels.MaxPool2d, 0, input);
        runtime.SetBuffer(SilkKernels.MaxPool2d, 1, output);
        runtime.SetUInt32(SilkKernels.MaxPool2d, 2, (uint)N);
        runtime.SetUInt32(SilkKernels.MaxPool2d, 3, (uint)C);
        runtime.SetUInt32(SilkKernels.MaxPool2d, 4, (uint)H);
        runtime.SetUInt32(SilkKernels.MaxPool2d, 5, (uint)W);
        runtime.SetUInt32(SilkKernels.MaxPool2d, 6, (uint)kH);
        runtime.SetUInt32(SilkKernels.MaxPool2d, 7, (uint)kW);
        runtime.SetUInt32(SilkKernels.MaxPool2d, 8, (uint)strideH);
        runtime.SetUInt32(SilkKernels.MaxPool2d, 9, (uint)strideW);
        runtime.SetUInt32(SilkKernels.MaxPool2d, 10, (uint)padTop);
        runtime.SetUInt32(SilkKernels.MaxPool2d, 11, (uint)padBottom);
        runtime.SetUInt32(SilkKernels.MaxPool2d, 12, (uint)padLeft);
        runtime.SetUInt32(SilkKernels.MaxPool2d, 13, (uint)padRight);
        runtime.SetUInt32(SilkKernels.MaxPool2d, 14, (uint)oH);
        runtime.SetUInt32(SilkKernels.MaxPool2d, 15, (uint)oW);
        int total = N * C * oH * oW;
        runtime.Run(SilkKernels.MaxPool2d, [(nuint)SilkKernels.LinearGlobal(total)], [(nuint)SilkKernels.LinearGroupSize]);
    }

    public static void AdaptiveAvgPool2d(
        SilkRuntime runtime, nint input, nint output, int N, int C, int H, int W)
    {
        runtime.SetBuffer(SilkKernels.AdaptiveAvgPool2d, 0, input);
        runtime.SetBuffer(SilkKernels.AdaptiveAvgPool2d, 1, output);
        runtime.SetUInt32(SilkKernels.AdaptiveAvgPool2d, 2, (uint)N);
        runtime.SetUInt32(SilkKernels.AdaptiveAvgPool2d, 3, (uint)C);
        runtime.SetUInt32(SilkKernels.AdaptiveAvgPool2d, 4, (uint)H);
        runtime.SetUInt32(SilkKernels.AdaptiveAvgPool2d, 5, (uint)W);
        int total = N * C;
        runtime.Run(SilkKernels.AdaptiveAvgPool2d, [(nuint)SilkKernels.LinearGlobal(total)], [(nuint)SilkKernels.LinearGroupSize]);
    }
}
