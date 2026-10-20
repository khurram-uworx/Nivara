namespace Nivara.Silk;

/// <summary>
/// Typed dispatch shared by the three Silk runners. Each method sets one kernel's arguments and
/// runs it over the geometry <see cref="SilkKernels"/> owns, so a runner cannot drift from the
/// kernel's argument order or its grid. The runners keep their own <c>Launch*</c> wrappers where
/// the call closes over runner state (hidden size, epsilon, scale, attention band).
/// </summary>
internal static class SilkLauncher
{
    /// <summary>One output slot of <see cref="RmsNormGemv"/>: its weight matrix, an optional bias
    /// row, its destination, and its output width. Trailing slots are omitted rather than passed as
    /// four zeros, so a call site cannot wire a slot's width to another slot's buffer.</summary>
    public readonly record struct GemvSlot(nint Weight, nint Bias, nint Output, int Cols);

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
        int groupSize = SilkKernels.AttentionGroupSize;
        runtime.Run(
            SilkKernels.DecodeGqaAttention,
            [(nuint)(numHeads * groupSize)],
            [(nuint)groupSize]);
    }

    /// <summary>Combined cached-decode attention + O-projection (#672, Stage A2): runs
    /// <c>decode_gqa_attention</c> into <paramref name="contextScratch"/> then the flat O-proj
    /// GEMV. Device work-group limits prevent true single-buffer fusion, so the scratch buffer
    /// is retained internally.</summary>
    public static void DecodeGqaAttentionOproj(
        SilkRuntime runtime, nint q, nint kCache, nint vCache, nint contextScratch,
        nint Wo, nint residual, nint outBuf, int kvLen, int numHeads, int numKvHeads,
        int headDim, int qDim, int hiddenDim, float scale)
    {
        if (runtime.CanFuseDecodeOproj(numHeads, headDim))
        {
            DecodeGqaAttentionOprojFused(runtime, q, kCache, vCache, Wo, residual, outBuf,
                kvLen, numHeads, numKvHeads, headDim, hiddenDim, scale);
            return;
        }

        DecodeGqaAttention(runtime, q, kCache, vCache, contextScratch, kvLen, numHeads, numKvHeads, headDim, scale);
        GemmGemvAdd(runtime, contextScratch, Wo, outBuf, residual, qDim, hiddenDim);
    }

    /// <summary>True single-work-group fused decode attention + O-projection (#691). Requires
    /// <see cref="SilkRuntime.CanFuseDecodeOproj"/> to be true; otherwise use
    /// <see cref="DecodeGqaAttentionOproj"/>.</summary>
    public static void DecodeGqaAttentionOprojFused(
        SilkRuntime runtime, nint q, nint kCache, nint vCache, nint Wo, nint residual,
        nint outBuf, int kvLen, int numHeads, int numKvHeads, int headDim, int hiddenDim,
        float scale)
    {
        runtime.SetBuffer(SilkKernels.DecodeGqaAttentionOprojFused, 0, q);
        runtime.SetBuffer(SilkKernels.DecodeGqaAttentionOprojFused, 1, kCache);
        runtime.SetBuffer(SilkKernels.DecodeGqaAttentionOprojFused, 2, vCache);
        runtime.SetBuffer(SilkKernels.DecodeGqaAttentionOprojFused, 3, Wo);
        runtime.SetBuffer(SilkKernels.DecodeGqaAttentionOprojFused, 4, residual);
        runtime.SetBuffer(SilkKernels.DecodeGqaAttentionOprojFused, 5, outBuf);
        runtime.SetUInt32(SilkKernels.DecodeGqaAttentionOprojFused, 6, (uint)kvLen);
        runtime.SetUInt32(SilkKernels.DecodeGqaAttentionOprojFused, 7, (uint)numHeads);
        runtime.SetUInt32(SilkKernels.DecodeGqaAttentionOprojFused, 8, (uint)numKvHeads);
        runtime.SetUInt32(SilkKernels.DecodeGqaAttentionOprojFused, 9, (uint)headDim);
        runtime.SetUInt32(SilkKernels.DecodeGqaAttentionOprojFused, 10, (uint)hiddenDim);
        runtime.SetFloat32(SilkKernels.DecodeGqaAttentionOprojFused, 11, scale);
        runtime.EnsureFusedDecodeOprojLocalMemory(numHeads, headDim);
        int groupSize = numHeads * SilkKernels.AttentionGroupSize;
        runtime.Run(
            SilkKernels.DecodeGqaAttentionOprojFused,
            [(nuint)groupSize],
            [(nuint)groupSize]);
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

    public static void ArgMaxPartial(
        SilkRuntime runtime, nint input, nint partialIndex, nint partialValue, int n, int groups)
    {
        if (groups <= 0) throw new ArgumentOutOfRangeException(nameof(groups));
        if (n < 0) throw new ArgumentOutOfRangeException(nameof(n));
        runtime.SetBuffer(SilkKernels.ArgMax, 0, input);
        runtime.SetBuffer(SilkKernels.ArgMax, 1, partialIndex);
        runtime.SetBuffer(SilkKernels.ArgMax, 2, partialValue);
        runtime.SetUInt32(SilkKernels.ArgMax, 3, (uint)n);
        runtime.Run(
            SilkKernels.ArgMax,
            [(nuint)(groups * 256)],
            [(nuint)256]);
    }

    public static void RmsNormLmHead(
        SilkRuntime runtime, nint x, nint gamma, nint weight, nint logits,
        int aRows, int aCols, int bCols, float eps)
    {
        if (aRows < 0) throw new ArgumentOutOfRangeException(nameof(aRows));
        if (aCols <= 0) throw new ArgumentOutOfRangeException(nameof(aCols));
        if (bCols <= 0) throw new ArgumentOutOfRangeException(nameof(bCols));
        runtime.SetBuffer(SilkKernels.RmsNormLmHead, 0, x);
        runtime.SetBuffer(SilkKernels.RmsNormLmHead, 1, gamma);
        runtime.SetBuffer(SilkKernels.RmsNormLmHead, 2, weight);
        runtime.SetBuffer(SilkKernels.RmsNormLmHead, 3, logits);
        runtime.SetUInt32(SilkKernels.RmsNormLmHead, 4, (uint)aRows);
        runtime.SetUInt32(SilkKernels.RmsNormLmHead, 5, (uint)aCols);
        runtime.SetUInt32(SilkKernels.RmsNormLmHead, 6, (uint)bCols);
        runtime.SetFloat32(SilkKernels.RmsNormLmHead, 7, eps);
        runtime.Run(
            SilkKernels.RmsNormLmHead,
            [(nuint)SilkKernels.GemmGlobalRows(aRows), (nuint)SilkKernels.GemmGlobalCols(bCols)],
            [(nuint)SilkKernels.GemmTileSize, (nuint)SilkKernels.GemmTileSize]);
    }

    public static void GemmAdd(
        SilkRuntime runtime, nint a, nint b, nint c, nint residual,
        int aRows, int aCols, int bCols)
    {
        if (aRows < 0) throw new ArgumentOutOfRangeException(nameof(aRows));
        if (aCols <= 0) throw new ArgumentOutOfRangeException(nameof(aCols));
        if (bCols <= 0) throw new ArgumentOutOfRangeException(nameof(bCols));
        runtime.SetBuffer(SilkKernels.GemmAdd, 0, a);
        runtime.SetBuffer(SilkKernels.GemmAdd, 1, b);
        runtime.SetBuffer(SilkKernels.GemmAdd, 2, c);
        runtime.SetBuffer(SilkKernels.GemmAdd, 3, residual);
        runtime.SetUInt32(SilkKernels.GemmAdd, 4, (uint)aRows);
        runtime.SetUInt32(SilkKernels.GemmAdd, 5, (uint)aCols);
        runtime.SetUInt32(SilkKernels.GemmAdd, 6, (uint)bCols);
        runtime.Run(
            SilkKernels.GemmAdd,
            [(nuint)SilkKernels.GemmGlobalRows(aRows), (nuint)SilkKernels.GemmGlobalCols(bCols)],
            [(nuint)SilkKernels.GemmTileSize, (nuint)SilkKernels.GemmTileSize]);
    }

    /// <summary>Flat single-row GEMV + residual add for cached-decode O-projection (#672, Stage A1).
    /// a is [aCols], b is [aCols*bCols] (B row-major), result c = a·B + residual. Single row only.</summary>
    public static void GemmGemvAdd(
        SilkRuntime runtime, nint a, nint b, nint c, nint residual, int aCols, int bCols)
    {
        if (aCols <= 0) throw new ArgumentOutOfRangeException(nameof(aCols));
        if (bCols <= 0) throw new ArgumentOutOfRangeException(nameof(bCols));
        runtime.SetBuffer(SilkKernels.GemmGemvAdd, 0, a);
        runtime.SetBuffer(SilkKernels.GemmGemvAdd, 1, b);
        runtime.SetBuffer(SilkKernels.GemmGemvAdd, 2, c);
        runtime.SetBuffer(SilkKernels.GemmGemvAdd, 3, residual);
        runtime.SetUInt32(SilkKernels.GemmGemvAdd, 4, (uint)aCols);
        runtime.SetUInt32(SilkKernels.GemmGemvAdd, 5, (uint)bCols);
        int threads = SilkKernels.GemvGlobalCols(bCols);
        runtime.Run(
            SilkKernels.GemmGemvAdd,
            [(nuint)threads],
            [(nuint)SilkKernels.GemvGroupSize]);
    }

    /// <summary>Fused single-token MLP block for cached decode (#672, Stage B): RMSNorm(x) feeds
    /// gate/up, SwiGLU, down-projection, plus the residual add, in one launch. Single row only.</summary>
    public static void MlpBlock(
        SilkRuntime runtime, nint x, nint gamma, nint Wgate, nint Wup, nint Wdown,
        nint outBuf, nint residual, int hidden, int intermediate, float eps)
    {
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden));
        if (intermediate <= 0) throw new ArgumentOutOfRangeException(nameof(intermediate));
        if (intermediate > 5120) throw new ArgumentOutOfRangeException(nameof(intermediate), "exceeds MLP_MAX_INTERMEDIATE");
        runtime.SetBuffer(SilkKernels.MlpBlock, 0, x);
        runtime.SetBuffer(SilkKernels.MlpBlock, 1, gamma);
        runtime.SetBuffer(SilkKernels.MlpBlock, 2, Wgate);
        runtime.SetBuffer(SilkKernels.MlpBlock, 3, Wup);
        runtime.SetBuffer(SilkKernels.MlpBlock, 4, Wdown);
        runtime.SetBuffer(SilkKernels.MlpBlock, 5, outBuf);
        runtime.SetBuffer(SilkKernels.MlpBlock, 6, residual);
        runtime.SetUInt32(SilkKernels.MlpBlock, 7, (uint)hidden);
        runtime.SetUInt32(SilkKernels.MlpBlock, 8, (uint)intermediate);
        runtime.SetFloat32(SilkKernels.MlpBlock, 9, eps);
        runtime.Run(
            SilkKernels.MlpBlock,
            [(nuint)SilkKernels.LinearGroupSize],
            [(nuint)SilkKernels.LinearGroupSize]);
    }

    /// <summary>Fused <c>RMSNorm(x)</c> followed by one to three independent projections, which share
    /// the one normalization: cached-decode Q/K/V, gate/up, or the tied head. Decode only — the
    /// kernel is the flat one-row form, so <paramref name="aRows"/> must be 1 and the prefill heads
    /// keep their individual tile kernels (#657).</summary>
    public static void RmsNormGemv(
        SilkRuntime runtime,
        nint x,
        nint gamma,
        ReadOnlySpan<GemvSlot> slots,
        bool hasBias,
        int aRows,
        int aCols,
        float eps)
    {
        if (aRows != 1)
            throw new ArgumentOutOfRangeException(
                nameof(aRows), aRows, "rms_norm_gemv is the single-row decode shape; prefill keeps the tile kernels");
        if (aCols <= 0) throw new ArgumentOutOfRangeException(nameof(aCols));
        if (slots.Length is < 1 or > 3) throw new ArgumentOutOfRangeException(nameof(slots));

        GemvSlot s0 = slots[0];
        GemvSlot s1 = slots.Length > 1 ? slots[1] : default;
        GemvSlot s2 = slots.Length > 2 ? slots[2] : default;
        if (s0.Cols <= 0) throw new ArgumentOutOfRangeException(nameof(slots));
        if (s1.Cols < 0) throw new ArgumentOutOfRangeException(nameof(slots));
        if (s2.Cols < 0) throw new ArgumentOutOfRangeException(nameof(slots));

        runtime.SetBuffer(SilkKernels.RmsNormGemv, 0, x);
        runtime.SetBuffer(SilkKernels.RmsNormGemv, 1, gamma);
        runtime.SetBuffer(SilkKernels.RmsNormGemv, 2, s0.Weight);
        runtime.SetBuffer(SilkKernels.RmsNormGemv, 3, s0.Bias);
        runtime.SetBuffer(SilkKernels.RmsNormGemv, 4, s0.Output);
        runtime.SetBuffer(SilkKernels.RmsNormGemv, 5, s1.Weight);
        runtime.SetBuffer(SilkKernels.RmsNormGemv, 6, s1.Bias);
        runtime.SetBuffer(SilkKernels.RmsNormGemv, 7, s1.Output);
        runtime.SetBuffer(SilkKernels.RmsNormGemv, 8, s2.Weight);
        runtime.SetBuffer(SilkKernels.RmsNormGemv, 9, s2.Bias);
        runtime.SetBuffer(SilkKernels.RmsNormGemv, 10, s2.Output);
        runtime.SetUInt32(SilkKernels.RmsNormGemv, 11, (uint)aCols);
        runtime.SetUInt32(SilkKernels.RmsNormGemv, 12, (uint)s0.Cols);
        runtime.SetUInt32(SilkKernels.RmsNormGemv, 13, (uint)s1.Cols);
        runtime.SetUInt32(SilkKernels.RmsNormGemv, 14, (uint)s2.Cols);
        runtime.SetUInt32(SilkKernels.RmsNormGemv, 15, hasBias ? 1u : 0u);
        runtime.SetFloat32(SilkKernels.RmsNormGemv, 16, eps);

        // One work-group per GEMVCOLS output columns, and each slot's group count lands on a whole
        // group, so a group never straddles two slots and the kernel recovers its slot from
        // get_group_id(0) alone. A zero-width slot adds no group.
        int threads =
            SilkKernels.GemvGlobalCols(s0.Cols)
            + SilkKernels.GemvGlobalCols(s1.Cols)
            + SilkKernels.GemvGlobalCols(s2.Cols);
        runtime.Run(
            SilkKernels.RmsNormGemv,
            [(nuint)threads],
            [(nuint)SilkKernels.GemvGroupSize]);
    }

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

    public static void GemmBf16(SilkRuntime runtime, nint a, nint b, nint c, int aRows, int aCols, int bCols)
    {
        runtime.SetBuffer(SilkKernels.GemmBf16, 0, a);
        runtime.SetBuffer(SilkKernels.GemmBf16, 1, b);
        runtime.SetBuffer(SilkKernels.GemmBf16, 2, c);
        runtime.SetUInt32(SilkKernels.GemmBf16, 3, (uint)aRows);
        runtime.SetUInt32(SilkKernels.GemmBf16, 4, (uint)aCols);
        runtime.SetUInt32(SilkKernels.GemmBf16, 5, (uint)bCols);
        GemmGrid(runtime, SilkKernels.GemmBf16, aRows, bCols);
    }

    public static void GemmGemvAddBf16(
        SilkRuntime runtime, nint a, nint b, nint c, nint residual, int aCols, int bCols)
    {
        if (aCols <= 0) throw new ArgumentOutOfRangeException(nameof(aCols));
        if (bCols <= 0) throw new ArgumentOutOfRangeException(nameof(bCols));
        runtime.SetBuffer(SilkKernels.GemmGemvAddBf16, 0, a);
        runtime.SetBuffer(SilkKernels.GemmGemvAddBf16, 1, b);
        runtime.SetBuffer(SilkKernels.GemmGemvAddBf16, 2, c);
        runtime.SetBuffer(SilkKernels.GemmGemvAddBf16, 3, residual);
        runtime.SetUInt32(SilkKernels.GemmGemvAddBf16, 4, (uint)aCols);
        runtime.SetUInt32(SilkKernels.GemmGemvAddBf16, 5, (uint)bCols);
        int threads = SilkKernels.GemvGlobalCols(bCols);
        runtime.Run(
            SilkKernels.GemmGemvAddBf16,
            [(nuint)threads],
            [(nuint)SilkKernels.GemvGroupSize]);
    }

    public static void MlpBlockBf16(
        SilkRuntime runtime, nint x, nint gamma, nint Wgate, nint Wup, nint Wdown,
        nint outBuf, nint residual, int hidden, int intermediate, float eps)
    {
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden));
        if (intermediate <= 0) throw new ArgumentOutOfRangeException(nameof(intermediate));
        if (intermediate > 5120) throw new ArgumentOutOfRangeException(nameof(intermediate), "exceeds MLP_MAX_INTERMEDIATE");
        runtime.SetBuffer(SilkKernels.MlpBlockBf16, 0, x);
        runtime.SetBuffer(SilkKernels.MlpBlockBf16, 1, gamma);
        runtime.SetBuffer(SilkKernels.MlpBlockBf16, 2, Wgate);
        runtime.SetBuffer(SilkKernels.MlpBlockBf16, 3, Wup);
        runtime.SetBuffer(SilkKernels.MlpBlockBf16, 4, Wdown);
        runtime.SetBuffer(SilkKernels.MlpBlockBf16, 5, outBuf);
        runtime.SetBuffer(SilkKernels.MlpBlockBf16, 6, residual);
        runtime.SetUInt32(SilkKernels.MlpBlockBf16, 7, (uint)hidden);
        runtime.SetUInt32(SilkKernels.MlpBlockBf16, 8, (uint)intermediate);
        runtime.SetFloat32(SilkKernels.MlpBlockBf16, 9, eps);
        runtime.Run(
            SilkKernels.MlpBlockBf16,
            [(nuint)SilkKernels.LinearGroupSize],
            [(nuint)SilkKernels.LinearGroupSize]);
    }

    public static void RmsNormGemvBf16(
        SilkRuntime runtime,
        nint x,
        nint gamma,
        ReadOnlySpan<GemvSlot> slots,
        bool hasBias,
        int aRows,
        int aCols,
        float eps)
    {
        if (aRows != 1)
            throw new ArgumentOutOfRangeException(nameof(aRows), aRows, "rms_norm_gemv is the single-row decode shape");
        if (aCols <= 0) throw new ArgumentOutOfRangeException(nameof(aCols));
        if (slots.Length is < 1 or > 3) throw new ArgumentOutOfRangeException(nameof(slots));

        GemvSlot s0 = slots[0];
        GemvSlot s1 = slots.Length > 1 ? slots[1] : default;
        GemvSlot s2 = slots.Length > 2 ? slots[2] : default;
        if (s0.Cols <= 0) throw new ArgumentOutOfRangeException(nameof(slots));
        if (s1.Cols < 0) throw new ArgumentOutOfRangeException(nameof(slots));
        if (s2.Cols < 0) throw new ArgumentOutOfRangeException(nameof(slots));

        runtime.SetBuffer(SilkKernels.RmsNormGemvBf16, 0, x);
        runtime.SetBuffer(SilkKernels.RmsNormGemvBf16, 1, gamma);
        runtime.SetBuffer(SilkKernels.RmsNormGemvBf16, 2, s0.Weight);
        runtime.SetBuffer(SilkKernels.RmsNormGemvBf16, 3, s0.Bias);
        runtime.SetBuffer(SilkKernels.RmsNormGemvBf16, 4, s0.Output);
        runtime.SetBuffer(SilkKernels.RmsNormGemvBf16, 5, s1.Weight);
        runtime.SetBuffer(SilkKernels.RmsNormGemvBf16, 6, s1.Bias);
        runtime.SetBuffer(SilkKernels.RmsNormGemvBf16, 7, s1.Output);
        runtime.SetBuffer(SilkKernels.RmsNormGemvBf16, 8, s2.Weight);
        runtime.SetBuffer(SilkKernels.RmsNormGemvBf16, 9, s2.Bias);
        runtime.SetBuffer(SilkKernels.RmsNormGemvBf16, 10, s2.Output);
        runtime.SetUInt32(SilkKernels.RmsNormGemvBf16, 11, (uint)aCols);
        runtime.SetUInt32(SilkKernels.RmsNormGemvBf16, 12, (uint)s0.Cols);
        runtime.SetUInt32(SilkKernels.RmsNormGemvBf16, 13, (uint)s1.Cols);
        runtime.SetUInt32(SilkKernels.RmsNormGemvBf16, 14, (uint)s2.Cols);
        runtime.SetUInt32(SilkKernels.RmsNormGemvBf16, 15, hasBias ? 1u : 0u);
        runtime.SetFloat32(SilkKernels.RmsNormGemvBf16, 16, eps);

        int threads =
            SilkKernels.GemvGlobalCols(s0.Cols)
            + SilkKernels.GemvGlobalCols(s1.Cols)
            + SilkKernels.GemvGlobalCols(s2.Cols);
        runtime.Run(
            SilkKernels.RmsNormGemvBf16,
            [(nuint)threads],
            [(nuint)SilkKernels.GemvGroupSize]);
    }

}
