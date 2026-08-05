using Nivara;
using Nivara.AutoDiff;
using System.Numerics;

namespace NivaraChatClient;

/// <summary>
/// Fixed sinusoidal position encoding (non-trainable). Position <c>pos</c> in
/// dimension <c>i</c> uses the classic GPT/Transformer schedule
/// <c>sin(pos / 10000^(2i/D))</c> for even <c>i</c>, <c>cos(...)</c> for odd.
/// </summary>
public static class PositionEncoding
{
    readonly static Dictionary<(int SeqLen, int EmbedDim), Array> Cache = [];

    /// <summary>
    /// Returns the [1, seqLen, embedDim] encoding table, cached per (seqLen, embedDim).
    /// </summary>
    public static ReverseGradTensor<T> Sinusoidal<T>(int seqLen, int embedDim)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));

        var data = GetData<T>(seqLen, embedDim);
        var column = NivaraColumn<T>.Create(data);
        var tensor = new ReverseGradTensor<T>(column, requiresGrad: false);
        tensor.Reshape(1, seqLen, embedDim);
        return tensor;
    }

    /// <summary>
    /// Builds a [B, seqLen, embedDim] encoding by tiling the cached [1, seqLen, embedDim]
    /// table across the batch dimension (the encoding is position-only).
    /// </summary>
    public static ReverseGradTensor<T> Build<T>(int batch, int seqLen, int embedDim)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch));

        var data = GetData<T>(seqLen, embedDim);
        var tiled = new T[batch * seqLen * embedDim];
        for (int b = 0; b < batch; b++)
            Array.Copy(data, 0, tiled, b * seqLen * embedDim, seqLen * embedDim);

        var column = NivaraColumn<T>.Create(tiled);
        var tensor = new ReverseGradTensor<T>(column, requiresGrad: false);
        tensor.Reshape(batch, seqLen, embedDim);
        return tensor;
    }

    static T[] GetData<T>(int seqLen, int embedDim) where T : struct, IFloatingPointIeee754<T>
    {
        lock (Cache)
        {
            if (Cache.TryGetValue((seqLen, embedDim), out var cached))
                return (T[])cached;
        }

        var data = new T[seqLen * embedDim];
        for (int pos = 0; pos < seqLen; pos++)
        {
            for (int i = 0; i < embedDim; i++)
            {
                double exponent = 2.0 * (i / 2) / embedDim;
                double angle = pos / Math.Pow(10000.0, exponent);
                data[pos * embedDim + i] = i % 2 == 0
                    ? T.CreateChecked(Math.Sin(angle))
                    : T.CreateChecked(Math.Cos(angle));
            }
        }

        lock (Cache)
        {
            Cache[(seqLen, embedDim)] = data;
        }
        return data;
    }
}
