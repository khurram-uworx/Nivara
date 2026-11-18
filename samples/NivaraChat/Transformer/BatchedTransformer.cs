using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace NivaraChat.Transformer;

/// <summary>
/// Causal batched transformer with weight tying and a fixed sinusoidal position
/// encoding. Input is a rank-2 token tensor [B, L]; output is a rank-2 logits
/// tensor [B*L, V] (positions flattened row-major), matching the TensorDataset /
/// CrossEntropyLoss training layout used by NivaraGpt.
/// </summary>
public sealed class BatchedTransformer<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int vocabSize;
    readonly int nEmbd;
    readonly int nHead;
    readonly int maxSeqLen;
    readonly T attnScale;

    readonly Embedding<T> tokenEmb;
    readonly BatchedTransformerBlock<T>[] blocks;
    readonly LayerNorm<T> finalNorm;

    public int MaxSeqLen => maxSeqLen;
    public int VocabSize => vocabSize;
    public int NumLayers => blocks.Length;
    public int NEmbd => nEmbd;
    public Embedding<T> TokenEmbedding => tokenEmb;

    public TransformerKVCache<T>? KVCache { get; private set; }
    public TransformerKVCache<T>? Cache => KVCache;
    public TransformerKVCache<T>? LastKVCache => KVCache;

    public BatchedTransformer(
        int vocabSize,
        int nEmbd,
        int nLayer,
        int nHead,
        int maxSeqLen)
    {
        if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
        if (nEmbd <= 0) throw new ArgumentOutOfRangeException(nameof(nEmbd));
        if (nLayer <= 0) throw new ArgumentOutOfRangeException(nameof(nLayer));
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (nEmbd % nHead != 0)
            throw new ArgumentException($"nEmbd ({nEmbd}) must be divisible by nHead ({nHead}).");
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));

        this.vocabSize = vocabSize;
        this.nEmbd = nEmbd;
        this.nHead = nHead;
        this.maxSeqLen = maxSeqLen;
        attnScale = T.CreateChecked(1.0 / Math.Sqrt(nHead > 0 ? nEmbd / nHead : 1));

        tokenEmb = new Embedding<T>(vocabSize, nEmbd);

        blocks = new BatchedTransformerBlock<T>[nLayer];
        for (int i = 0; i < nLayer; i++)
            blocks[i] = new BatchedTransformerBlock<T>(nEmbd, nHead, maxSeqLen);

        finalNorm = new LayerNorm<T>(nEmbd);

        RegisterModules(tokenEmb);
        foreach (var block in blocks) RegisterModules(block);
        RegisterModules(finalNorm);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 2)
            throw new ArgumentException($"Expected [B, L] input, got {input.Rank}D.");
        if (input.Shape[1] > maxSeqLen)
            throw new ArgumentException($"Sequence length {input.Shape[1]} exceeds maxSeqLen {maxSeqLen}.");

        int batch = input.Shape[0];
        int length = input.Shape[1];

        var x = tokenEmb.Forward(input);                    // [B, L, D]
        var positions = PositionEncoding.Build<T>(batch, length, nEmbd);
        x = ReverseGradOperations.Add(x, positions);

        var mask = BuildCausalMask(batch, length);
        foreach (var block in blocks)
            x = block.ForwardBlock(x, mask);                    // [B, L, D]

        x = finalNorm.Forward(x);                           // [B, L, D]
        x.Reshape(batch * length, nEmbd);                   // [B*L, D]

        var wteT = ReverseGradOperations.Transpose(tokenEmb.Weight!.Tensor); // [D, V]
        return ReverseGradOperations.MatMul(x, wteT);               // [B*L, V]
    }

    public ReverseGradTensor<T> ForwardPrefill(ReverseGradTensor<T> tokens)
        => ForwardPrefill(tokens, null);

    public ReverseGradTensor<T> ForwardPrefill(ReverseGradTensor<T> tokens, TransformerKVCache<T>? cache)
    {
        if (tokens == null) throw new ArgumentNullException(nameof(tokens));

        var workingTokens = tokens;
        if (tokens.Rank == 1)
        {
            workingTokens = new ReverseGradTensor<T>(tokens.Data, tokens.RequiresGrad);
            workingTokens.Reshape(1, tokens.Shape[0]);
        }
        if (workingTokens.Rank != 2)
            throw new ArgumentException($"Expected [B, L] input, got {workingTokens.Rank}D.");
        int batch = workingTokens.Shape[0];
        int length = workingTokens.Shape[1];
        if (batch != 1)
            throw new ArgumentException("Cached path supports batch size 1 only.");
        if (length <= 0 || length > maxSeqLen)
            throw new ArgumentException($"Sequence length {length} exceeds maxSeqLen {maxSeqLen}.");

        if (cache == null) cache = new TransformerKVCache<T>(blocks.Length, maxSeqLen, nEmbd);
        KVCache = cache;
        cache.Ensure(length);

        var x = tokenEmb.Forward(workingTokens);                // [1, L, D]
        var positions = PositionEncoding.Build<T>(1, length, nEmbd);
        x = ReverseGradOperations.Add(x, positions);

        var mask = BuildCausalMask(batch, length);
        for (int i = 0; i < blocks.Length; i++)
            x = blocks[i].ForwardPrefill(x, cache.keys[i], cache.values[i], 0); // [1, L, D]

        x = finalNorm.Forward(x);                               // [1, L, D]

        var lastRow = new T[nEmbd];
        if (x.Data.TryGetSpan(out var xSpan))
            xSpan.Slice((length - 1) * nEmbd, nEmbd).CopyTo(lastRow);
        else
            for (int d = 0; d < nEmbd; d++) lastRow[d] = x[(length - 1) * nEmbd + d];

        var last = ReverseGradTensor<T>.FromMatrix(lastRow, 1, nEmbd, requiresGrad: false);
        var wteT = ReverseGradOperations.Transpose(tokenEmb.Weight!.Tensor); // [D, V]
        return ReverseGradOperations.MatMul(last, wteT);               // [1, V]
    }

    public ReverseGradTensor<T> ForwardPrefill(int[] tokenIds, TransformerKVCache<T>? cache = null)
    {
        if (tokenIds == null) throw new ArgumentNullException(nameof(tokenIds));
        var data = new T[tokenIds.Length];
        for (int i = 0; i < tokenIds.Length; i++) data[i] = T.CreateChecked(tokenIds[i]);
        var column = NivaraColumn<T>.Create(data);
        var t = new ReverseGradTensor<T>(column, requiresGrad: false);
        t.Reshape(1, tokenIds.Length);
        return ForwardPrefill(t, cache);
    }

    public ReverseGradTensor<T> ForwardPrefill(float[] tokenIds, TransformerKVCache<T>? cache = null)
    {
        if (tokenIds == null) throw new ArgumentNullException(nameof(tokenIds));
        var data = new T[tokenIds.Length];
        for (int i = 0; i < tokenIds.Length; i++) data[i] = T.CreateChecked(tokenIds[i]);
        var column = NivaraColumn<T>.Create(data);
        var t = new ReverseGradTensor<T>(column, requiresGrad: false);
        t.Reshape(1, tokenIds.Length);
        return ForwardPrefill(t, cache);
    }

    public ReverseGradTensor<T> ForwardCached(int tokenId, int position)
    {
        if (KVCache == null)
            throw new InvalidOperationException("No KV cache available. Call ForwardPrefill first or pass a cache.");
        return ForwardCached(tokenId, position, KVCache);
    }

    public ReverseGradTensor<T> ForwardCached(int tokenId, int position, TransformerKVCache<T> cache)
    {
        if (cache == null) throw new ArgumentNullException(nameof(cache));
        if (position < 0 || position >= maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(position));

        KVCache = cache;
        cache.Ensure(position + 1);

        var data = new T[] { T.CreateChecked(tokenId) };
        var column = NivaraColumn<T>.Create(data);
        var input = new ReverseGradTensor<T>(column, requiresGrad: false);
        input.Reshape(1, 1);

        var x = tokenEmb.Forward(input);                        // [1, 1, D]
        var positions = PositionEncoding.Build<T>(1, position, 1, nEmbd);
        x = ReverseGradOperations.Add(x, positions);

        for (int i = 0; i < blocks.Length; i++)
            x = blocks[i].ForwardCached(x, position, cache.keys[i], cache.values[i], position); // [1, 1, D]

        x = finalNorm.Forward(x);                               // [1, 1, D]

        var lastRow = new T[nEmbd];
        if (x.Data.TryGetSpan(out var xSpan))
            xSpan.Slice(0, nEmbd).CopyTo(lastRow);
        else
            for (int d = 0; d < nEmbd; d++) lastRow[d] = x[d];

        var last = ReverseGradTensor<T>.FromMatrix(lastRow, 1, nEmbd, requiresGrad: false);
        var wteT = ReverseGradOperations.Transpose(tokenEmb.Weight!.Tensor); // [D, V]
        return ReverseGradOperations.MatMul(last, wteT);               // [1, V]
    }

    internal static ReverseGradTensor<T> BuildCausalMask(int batch, int length)
    {
        var data = new T[batch * length * length];
        int stride = length * length;
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < length; i++)
            {
                int rowStart = b * stride + i * length;
                for (int j = 0; j < length; j++)
                    data[rowStart + j] = j > i ? T.NegativeInfinity : T.Zero;
            }
        }

        var column = NivaraColumn<T>.Create(data);
        var tensor = new ReverseGradTensor<T>(column, requiresGrad: false);
        tensor.Reshape(batch, length, length);
        return tensor;
    }

    /// <summary>
    /// Exports this transformer's parameters into the conventional Silk/OpenCL weight
    /// dictionary shape (<c>tok_emb.weight</c>, <c>final_norm.weight/bias</c>, and
    /// <c>layers.{l}.*</c>), copying data out of the AutoDiff tensors (no aliasing).
    /// </summary>
    public Dictionary<string, (float[] Data, int[] Shape)> ToSilkWeights()
    {
        var result = new Dictionary<string, (float[] Data, int[] Shape)>(StringComparer.Ordinal);

        if (tokenEmb.Weight is null)
            throw new InvalidOperationException("Token embedding weight has not been initialized.");
        result["tok_emb.weight"] =
            (TensorToFloatArray(tokenEmb.Weight.Tensor), (int[])tokenEmb.Weight.Tensor.Shape.Clone());

        if (finalNorm.Weight is not null)
            result["final_norm.weight"] =
                (TensorToFloatArray(finalNorm.Weight.Tensor), (int[])finalNorm.Weight.Tensor.Shape.Clone());
        if (finalNorm.Bias is not null)
            result["final_norm.bias"] =
                (TensorToFloatArray(finalNorm.Bias.Tensor), (int[])finalNorm.Bias.Tensor.Shape.Clone());

        for (int i = 0; i < blocks.Length; i++)
            blocks[i].PopulateSilkWeights(i, result);

        return result;
    }

    internal static float[] TensorToFloatArray(ReverseGradTensor<T> tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        var values = new T[tensor.Length];
        tensor.Data.CopyTo(values, T.Zero);
        var result = new float[values.Length];
        for (int i = 0; i < result.Length; i++)
            result[i] = float.CreateChecked(values[i]);
        return result;
    }
}

/// <summary>
/// Pre-norm transformer block operating on rank-3 [B, L, D] tensors. Attention is
/// computed for the whole batch in a single call to
/// <see cref="ReverseGradOperations.BatchedMultiHeadAttention{T}"/>.
/// </summary>
public sealed class BatchedTransformerBlock<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int nEmbd;
    readonly int nHead;
    readonly T attnScale;

    readonly LayerNorm<T> ln1;
    readonly LayerNorm<T> ln2;
    readonly Linear<T> qProj;
    readonly Linear<T> kProj;
    readonly Linear<T> vProj;
    readonly Linear<T> oProj;
    readonly Linear<T> mlpFc1;
    readonly Linear<T> mlpFc2;

    public BatchedTransformerBlock(int nEmbd, int nHead, int maxSeqLen)
    {
        if (nEmbd <= 0) throw new ArgumentOutOfRangeException(nameof(nEmbd));
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (nEmbd % nHead != 0)
            throw new ArgumentException($"nEmbd ({nEmbd}) must be divisible by nHead ({nHead}).");

        this.nEmbd = nEmbd;
        this.nHead = nHead;
        attnScale = T.CreateChecked(1.0 / Math.Sqrt(nEmbd / nHead));

        ln1 = new LayerNorm<T>(nEmbd);
        ln2 = new LayerNorm<T>(nEmbd);
        qProj = new Linear<T>(nEmbd, nEmbd, bias: false);
        kProj = new Linear<T>(nEmbd, nEmbd, bias: false);
        vProj = new Linear<T>(nEmbd, nEmbd, bias: false);
        oProj = new Linear<T>(nEmbd, nEmbd, bias: false);
        mlpFc1 = new Linear<T>(nEmbd, 4 * nEmbd);
        mlpFc2 = new Linear<T>(4 * nEmbd, nEmbd);

        RegisterModules(ln1, ln2, qProj, kProj, vProj, oProj, mlpFc1, mlpFc2);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input) =>
        ForwardBlock(input, null);

    public ReverseGradTensor<T> ForwardBlock(ReverseGradTensor<T> input, ReverseGradTensor<T>? causalMask)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 3)
            throw new ArgumentException($"Expected [B, L, D] input, got {input.Rank}D.");

        int batch = input.Shape[0];
        int length = input.Shape[1];

        var mask = causalMask ?? BatchedTransformer<T>.BuildCausalMask(batch, length);

        var residual = input;

        var normed = ln1.Forward(input);                    // [B, L, D]
        var flat = normed;
        flat.Reshape(batch * length, nEmbd);                // [B*L, D]

        var q = qProj.Forward(flat);                        // [B*L, D]
        var k = kProj.Forward(flat);
        var v = vProj.Forward(flat);
        q.Reshape(batch, length, nEmbd);
        k.Reshape(batch, length, nEmbd);
        v.Reshape(batch, length, nEmbd);

        var attn = ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, nHead, attnScale, mask);
        attn.Reshape(batch * length, nEmbd);                // [B*L, D]
        var attnOut = oProj.Forward(attn);
        attnOut.Reshape(batch, length, nEmbd);              // [B, L, D]

        var x = ReverseGradOperations.Add(attnOut, residual);

        residual = x;
        var normed2 = ln2.Forward(x);
        var hidden = normed2;
        hidden.Reshape(batch * length, nEmbd);              // [B*L, D]
        hidden = mlpFc1.Forward(hidden);
        hidden = Activation.Gelu(hidden);
        hidden = mlpFc2.Forward(hidden);
        hidden.Reshape(batch, length, nEmbd);               // [B, L, D]

        return ReverseGradOperations.Add(hidden, residual);
    }

    public ReverseGradTensor<T> ForwardPrefill(ReverseGradTensor<T> input, T[] kCache, T[] vCache, int positionOffset = 0)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 3)
            throw new ArgumentException($"Expected [B, L, D] input, got {input.Rank}D.");

        int batch = input.Shape[0];
        int length = input.Shape[1];
        int D = nEmbd;
        if (batch != 1)
            throw new ArgumentException("Cached path supports batch size 1 only.");

        var mask = BatchedTransformer<T>.BuildCausalMask(batch, length);

        var residual = input;

        var normed = ln1.Forward(input);                    // [B, L, D]
        var flat = normed;
        flat.Reshape(batch * length, nEmbd);                // [B*L, D]

        var q = qProj.Forward(flat);                        // [B*L, D]
        var k = kProj.Forward(flat);
        var v = vProj.Forward(flat);
        q.Reshape(batch, length, nEmbd);
        k.Reshape(batch, length, nEmbd);
        v.Reshape(batch, length, nEmbd);

        // Capture K/V into the cache.
        if (k.Data.TryGetSpan(out var kSpan) && v.Data.TryGetSpan(out var vSpan))
        {
            for (int p = 0; p < length; p++)
            {
                for (int d = 0; d < D; d++)
                {
                    kCache[(positionOffset + p) * D + d] = kSpan[p * D + d];
                    vCache[(positionOffset + p) * D + d] = vSpan[p * D + d];
                }
            }
        }

        var attn = ReverseGradOperations.BatchedMultiHeadAttention(q, k, v, nHead, attnScale, mask);
        attn.Reshape(batch * length, nEmbd);                // [B*L, D]
        var attnOut = oProj.Forward(attn);
        attnOut.Reshape(batch, length, nEmbd);              // [B, L, D]

        var x = ReverseGradOperations.Add(attnOut, residual);

        residual = x;
        var normed2 = ln2.Forward(x);
        var hidden = normed2;
        hidden.Reshape(batch * length, nEmbd);              // [B*L, D]
        hidden = mlpFc1.Forward(hidden);
        hidden = Activation.Gelu(hidden);
        hidden = mlpFc2.Forward(hidden);
        hidden.Reshape(batch, length, nEmbd);               // [B, L, D]

        return ReverseGradOperations.Add(hidden, residual);
    }

    public ReverseGradTensor<T> ForwardCached(ReverseGradTensor<T> input, int position, T[] kCache, T[] vCache, int cacheLen)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 3)
            throw new ArgumentException($"Expected [B, L, D] input, got {input.Rank}D.");

        int batch = input.Shape[0];
        int length = input.Shape[1];
        int D = nEmbd;
        if (batch != 1 || length != 1)
            throw new ArgumentException("Cached decode expects a single token [1, 1, D].");

        var residual = input;

        var normed = ln1.Forward(input);                    // [1, 1, D]
        var flat = normed;
        flat.Reshape(batch * length, nEmbd);                // [1, D]

        var q = qProj.Forward(flat);                        // [1, D]
        var k = kProj.Forward(flat);
        var v = vProj.Forward(flat);
        q.Reshape(batch, length, nEmbd);                    // [1, 1, D]
        k.Reshape(batch, length, nEmbd);
        v.Reshape(batch, length, nEmbd);

        // Append new K/V at position.
        if (k.Data.TryGetSpan(out var kSpan) && v.Data.TryGetSpan(out var vSpan))
        {
            for (int d = 0; d < D; d++)
            {
                kCache[position * D + d] = kSpan[d];
                vCache[position * D + d] = vSpan[d];
            }
        }

        int kvLen = position + 1;
        var kArr = new T[kvLen * D];
        var vArr = new T[kvLen * D];
        Array.Copy(kCache, 0, kArr, 0, kvLen * D);
        Array.Copy(vCache, 0, vArr, 0, kvLen * D);

        var kTensor = ReverseGradTensor<T>.FromMatrix(kArr, 1, kvLen * D, requiresGrad: false);
        kTensor.Reshape(1, kvLen, D);
        var vTensor = ReverseGradTensor<T>.FromMatrix(vArr, 1, kvLen * D, requiresGrad: false);
        vTensor.Reshape(1, kvLen, D);

        var attn = ReverseGradOperations.BatchedMultiHeadAttention(q, kTensor, vTensor, nHead, attnScale, null);
        attn.Reshape(batch * length, nEmbd);                // [1, D]
        var attnOut = oProj.Forward(attn);
        attnOut.Reshape(batch, length, nEmbd);              // [1, 1, D]

        var x = ReverseGradOperations.Add(attnOut, residual);

        residual = x;
        var normed2 = ln2.Forward(x);
        var hidden = normed2;
        hidden.Reshape(batch * length, nEmbd);              // [1, D]
        hidden = mlpFc1.Forward(hidden);
        hidden = Activation.Gelu(hidden);
        hidden = mlpFc2.Forward(hidden);
        hidden.Reshape(batch, length, nEmbd);               // [1, 1, D]

        return ReverseGradOperations.Add(hidden, residual);
    }

    internal void PopulateSilkWeights(int layerIndex, Dictionary<string, (float[] Data, int[] Shape)> dict)
    {
        string prefix = $"layers.{layerIndex}.";
        void Add(string name, ReverseGradTensor<T>? tensor)
        {
            if (tensor is null) return;
            dict[prefix + name] =
                (BatchedTransformer<T>.TensorToFloatArray(tensor), (int[])tensor.Shape.Clone());
        }

        Add("ln1.weight", ln1.Weight?.Tensor);
        Add("ln1.bias", ln1.Bias?.Tensor);
        Add("ln2.weight", ln2.Weight?.Tensor);
        Add("ln2.bias", ln2.Bias?.Tensor);
        Add("q_proj.weight", qProj.Weight?.Tensor);
        Add("k_proj.weight", kProj.Weight?.Tensor);
        Add("v_proj.weight", vProj.Weight?.Tensor);
        Add("o_proj.weight", oProj.Weight?.Tensor);
        Add("mlp_fc1.weight", mlpFc1.Weight?.Tensor);
        Add("mlp_fc1.bias", mlpFc1.Bias?.Tensor);
        Add("mlp_fc2.weight", mlpFc2.Weight?.Tensor);
        Add("mlp_fc2.bias", mlpFc2.Bias?.Tensor);
    }
}
