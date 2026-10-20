namespace Nivara.Silk;

/// <summary>
/// Host-side weight-layout math shared by the GPU encoder runners, so a checkpoint's transpose and
/// q/k/v packing have one authoritative implementation across backends. Producing byte-identical
/// weights from the same arithmetic is what makes a cross-backend numerical comparison meaningful.
/// </summary>
public static class WeightLayout
{
    /// <summary>
    /// Host-side transpose of a checkpoint weight: the checkpoint stores <c>[out, in]</c> row-major
    /// and the GEMM reads <c>Bt[in, out]</c> so its global loads coalesce along K. Pre-fused
    /// weights — a checkpoint that already ships one concatenated matrix, such as ModernBERT's
    /// <c>Wqkv</c> / <c>Wi</c> — come through unchanged.
    /// </summary>
    /// <param name="tensors">The checkpoint tensors, keyed by name.</param>
    /// <param name="key">The weight key to transpose.</param>
    /// <param name="backend">Backend label used only in the missing-key error message.</param>
    public static T[] Transposed<T>(
        Dictionary<string, (T[] Data, int[] Shape)> tensors, string key, string backend)
        where T : struct
    {
        var t = Req(tensors, key, backend);
        int outDim = t.Shape[0];
        int inDim = t.Shape[1];
        var bt = new T[outDim * inDim];
        for (int r = 0; r < outDim; r++)
            for (int c = 0; c < inDim; c++)
                bt[c * outDim + r] = t.Data[r * inDim + c];
        return bt;
    }

    /// <summary>
    /// Host-side packing of the q/k/v projections: one pre-transposed <c>[in × 3·out]</c> Bt buffer
    /// <c>[Bt_q | Bt_k | Bt_v]</c> plus one <c>[3·out]</c> bias <c>[Bq | Bk | Bv]</c>, so the
    /// per-layer q/k/v GEMMs collapse into a single launch producing <c>[rows × 3·out]</c> and q/k/v
    /// are contiguous blocks of the result.
    /// </summary>
    /// <remarks>
    /// This layout is the contract between the packed-qkv GEMM and the attention that reads its three
    /// blocks, so every backend takes it from here: a private copy per runner could drift, and the
    /// drift would be a silently wrong attention rather than a crash.
    /// </remarks>
    public static (float[] W, float[] Bias) QkvPacked(
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        string qKey, string kKey, string vKey,
        string backend)
    {
        var weights = new[]
        {
            Req(tensors, $"{qKey}.weight", backend),
            Req(tensors, $"{kKey}.weight", backend),
            Req(tensors, $"{vKey}.weight", backend)
        };
        int outDim = weights[0].Shape[0];
        int inDim = weights[0].Shape[1];
        int concat = 3 * outDim;
        var w = new float[inDim * concat];
        for (int block = 0; block < 3; block++)
            for (int r = 0; r < outDim; r++)
                for (int c = 0; c < inDim; c++)
                    w[c * concat + block * outDim + r] = weights[block].Data[r * inDim + c];

        var bias = new float[concat];
        Array.Copy(Req(tensors, $"{qKey}.bias", backend).Data, bias, outDim);
        Array.Copy(Req(tensors, $"{kKey}.bias", backend).Data, 0, bias, outDim, outDim);
        Array.Copy(Req(tensors, $"{vKey}.bias", backend).Data, 0, bias, 2 * outDim, outDim);
        return (w, bias);
    }

    /// <summary>
    /// Resolves a checkpoint tensor by key, failing with the backend label so a missing key names
    /// the path that asked for it.
    /// </summary>
    public static (T[] Data, int[] Shape) Req<T>(
        Dictionary<string, (T[] Data, int[] Shape)> tensors, string key, string backend)
        => tensors.TryGetValue(key, out var t)
            ? t
            : throw new InvalidOperationException($"Missing weight key '{key}' for the {backend} forward.");
}
