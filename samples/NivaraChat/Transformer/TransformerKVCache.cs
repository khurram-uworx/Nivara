using System.Numerics;

namespace NivaraChat.Transformer;

/// <summary>
/// Per-layer K/V cache for cached generation on <see cref="BatchedTransformer{T}"/>.
/// Each layer stores K and V over <c>[maxSeqLen, nEmbd]</c>. Buffers are allocated up
/// front to <c>maxSeqLen</c> (no growth beyond it).
/// </summary>
public sealed class TransformerKVCache<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
{
    public int NumLayers { get; }
    public int MaxSeqLen { get; }
    public int EmbedDim { get; }

    // Llama-style lowercase fields (public for test access).
    public readonly T[][] keys;
    public readonly T[][] values;

    public T[][] Keys => keys;
    public T[][] Values => values;

    public int Capacity => MaxSeqLen;
    public int MaxCapacity => MaxSeqLen;

    bool disposed;

    public TransformerKVCache(int numLayers, int maxSeqLen, int nEmbd)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
        if (nEmbd <= 0) throw new ArgumentOutOfRangeException(nameof(nEmbd));

        NumLayers = numLayers;
        MaxSeqLen = maxSeqLen;
        EmbedDim = nEmbd;

        keys = new T[numLayers][];
        values = new T[numLayers][];
        for (int i = 0; i < numLayers; i++)
        {
            keys[i] = new T[maxSeqLen * nEmbd];
            values[i] = new T[maxSeqLen * nEmbd];
        }
    }

    public TransformerKVCache(BatchedTransformer<T> model)
        : this(model.NumLayers, model.MaxSeqLen, model.NEmbd)
    {
    }

    public TransformerKVCache(BatchedTransformer<T> model, int maxSeqLen)
        : this(model.NumLayers, maxSeqLen, model.NEmbd)
    {
    }

    public TransformerKVCache(int maxSeqLen, int nEmbd)
        : this(1, maxSeqLen, nEmbd)
    {
    }

    public void Ensure(int neededLength)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (neededLength <= MaxSeqLen) return;
        throw new ArgumentOutOfRangeException(nameof(neededLength),
            $"Requested length {neededLength} exceeds cache maximum capacity of {MaxSeqLen}.");
    }

    public void Reset()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        for (int i = 0; i < NumLayers; i++)
        {
            Array.Clear(keys[i]);
            Array.Clear(values[i]);
        }
    }

    public void Dispose()
    {
        if (disposed) return;
        for (int i = 0; i < NumLayers; i++)
        {
            keys[i] = [];
            values[i] = [];
        }
        disposed = true;
    }
}
