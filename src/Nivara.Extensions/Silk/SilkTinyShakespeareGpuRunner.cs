using System.Numerics;

namespace Nivara.Silk;

/// <summary>
/// TinyShakespeare (batched causal transformer) forward through the Silk/OpenCL kernels, F32 only.
/// Mirrors <see cref="BatchedTransformer{T}"/>: token embedding + sinusoidal position table,
/// per-layer LayerNorm (x2), bias-free Q/K/V/O GEMMs, causal GQA attention (numKvHeads == numHeads),
/// GELU MLP (GemmGelu + Gemm), tied LM head via the transposed token embedding.
/// </summary>
public sealed class SilkTinyShakespeareGpuRunner : IDisposable
{
    readonly SilkRuntime runtime;
    readonly TinyShakespeareConfig config;
    readonly int hiddenDim;
    readonly int intermediateDim;
    readonly int numHeads;
    readonly int headDim;
    readonly int vocabSize;
    readonly int seqCap;
    readonly float eps;
    readonly float scale;

    readonly nint tokEmb;
    readonly nint tokEmbTransposed;
    readonly nint posEmb;
    readonly nint finalNormW;
    readonly nint finalNormB;
    readonly LayerWeights[] layers;

    readonly nint[] kCache;
    readonly nint[] vCache;
    int cachedLen;

    SilkRuntime.SilkBuffer x;
    SilkRuntime.SilkBuffer normed;
    SilkRuntime.SilkBuffer q;
    SilkRuntime.SilkBuffer k;
    SilkRuntime.SilkBuffer v;
    SilkRuntime.SilkBuffer context;
    SilkRuntime.SilkBuffer projected;
    SilkRuntime.SilkBuffer gate;
    SilkRuntime.SilkBuffer posBuf;
    SilkRuntime.SilkBuffer idsBuf;
    SilkRuntime.SilkBuffer posIdsBuf;

    int[]? posIdHost;

    public SilkTinyShakespeareGpuRunner(
        SilkRuntime runtime,
        TinyShakespeareConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tensors);

        this.runtime = runtime;
        this.config = config;
        hiddenDim = config.HiddenSize;
        intermediateDim = 4 * hiddenDim;
        numHeads = config.NumAttentionHeads;
        headDim = hiddenDim / numHeads;
        vocabSize = config.VocabSize;
        seqCap = config.MaxSeqLen;
        eps = config.LayerNormEps;
        scale = 1f / MathF.Sqrt(headDim);

        runtime.EnsureAttentionLocalMemory(headDim);

        try
        {
            tokEmb = UploadPlain(tensors, "tok_emb.weight");
            tokEmbTransposed = runtime.Allocate(hiddenDim * vocabSize);
            runtime.Upload(tokEmbTransposed, WeightLayout.Transposed(tensors, "tok_emb.weight", "Silk"));

            posEmb = runtime.Allocate(seqCap * hiddenDim);
            runtime.Upload(posEmb, BuildPositionTable(seqCap, hiddenDim));

            finalNormW = UploadPlain(tensors, "final_norm.weight");
            finalNormB = UploadPlain(tensors, "final_norm.bias");

            layers = new LayerWeights[config.NumHiddenLayers];
            for (int i = 0; i < layers.Length; i++)
                layers[i] = new LayerWeights(this, tensors, i);

            kCache = new nint[config.NumHiddenLayers];
            vCache = new nint[config.NumHiddenLayers];
            for (int i = 0; i < config.NumHiddenLayers; i++)
            {
                kCache[i] = runtime.Allocate(seqCap * hiddenDim);
                vCache[i] = runtime.Allocate(seqCap * hiddenDim);
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public string DeviceName => runtime.DeviceName;

    public float[] ForwardPrefill(int[] tokenIds)
    {
        int seqLen = tokenIds.Length;
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(tokenIds));
        if (seqLen > seqCap) throw new ArgumentOutOfRangeException(nameof(tokenIds), $"seqLen {seqLen} exceeds {seqCap}.");
        cachedLen = 0;

        EnsureWorkspace(seqLen);

        if (posIdHost is null || posIdHost.Length < seqLen)
            posIdHost = new int[seqLen];
        for (int p = 0; p < seqLen; p++) posIdHost[p] = p;

        runtime.Upload(idsBuf.Handle, tokenIds.AsSpan(0, seqLen));
        runtime.Upload(posIdsBuf.Handle, posIdHost.AsSpan(0, seqLen));

        SilkLauncher.Gather(runtime, tokEmb, idsBuf.Handle, x.Handle, seqLen, hiddenDim);
        SilkLauncher.Gather(runtime, posEmb, posIdsBuf.Handle, posBuf.Handle, seqLen, hiddenDim);
        SilkLauncher.Add(runtime, x.Handle, posBuf.Handle, x.Handle, seqLen * hiddenDim);

        for (int li = 0; li < layers.Length; li++)
        {
            LayerWeights w = layers[li];

            SilkLauncher.LayerNorm(runtime, x.Handle, w.Ln1W, w.Ln1B, normed.Handle, seqLen, hiddenDim, eps);

            SilkLauncher.Gemm(runtime, normed.Handle, w.Wq, q.Handle, seqLen, hiddenDim, hiddenDim);
            SilkLauncher.Gemm(runtime, normed.Handle, w.Wk, k.Handle, seqLen, hiddenDim, hiddenDim);
            SilkLauncher.Gemm(runtime, normed.Handle, w.Wv, v.Handle, seqLen, hiddenDim, hiddenDim);

            runtime.CopyBuffer(k.Handle, kCache[li], 0, 0, seqLen * hiddenDim);
            runtime.CopyBuffer(v.Handle, vCache[li], 0, 0, seqLen * hiddenDim);

            SilkLauncher.CausalGqaAttention(runtime, q.Handle, k.Handle, v.Handle, context.Handle, seqLen, numHeads, numHeads, headDim, scale);

            SilkLauncher.Gemm(runtime, context.Handle, w.Wo, projected.Handle, seqLen, hiddenDim, hiddenDim);
            SilkLauncher.Add(runtime, x.Handle, projected.Handle, x.Handle, seqLen * hiddenDim);

            SilkLauncher.LayerNorm(runtime, x.Handle, w.Ln2W, w.Ln2B, normed.Handle, seqLen, hiddenDim, eps);

            SilkLauncher.GemmEpilogue(runtime, SilkKernels.GemmGelu, normed.Handle, w.Wfc1, gate.Handle, w.Fc1Bias, seqLen, hiddenDim, intermediateDim);
            SilkLauncher.GemmEpilogue(runtime, SilkKernels.GemmBias, gate.Handle, w.Wfc2, projected.Handle, w.Fc2Bias, seqLen, intermediateDim, hiddenDim);
            SilkLauncher.Add(runtime, x.Handle, projected.Handle, x.Handle, seqLen * hiddenDim);
        }

        SilkLauncher.LayerNorm(runtime, x.Handle, finalNormW, finalNormB, normed.Handle, seqLen, hiddenDim, eps);

        nint lastRowView = runtime.CreateSubview(normed.Handle, (seqLen - 1) * hiddenDim, hiddenDim);
        nint logitsBuf = runtime.Allocate(vocabSize);
        SilkLauncher.Gemm(runtime, lastRowView, tokEmbTransposed, logitsBuf, 1, hiddenDim, vocabSize);
        var logits = runtime.Download(logitsBuf, vocabSize);
        runtime.Release(logitsBuf);
        runtime.Release(lastRowView);

        cachedLen = seqLen;
        return logits;
    }

    public float[] ForwardCached(int tokenId, int position)
    {
        if (position < 0) throw new ArgumentOutOfRangeException(nameof(position));
        if (position + 1 > seqCap) throw new ArgumentOutOfRangeException(nameof(position));

        EnsureWorkspace(1);

        if (posIdHost is null || posIdHost.Length < 1)
            posIdHost = new int[1];
        posIdHost[0] = position;

        runtime.Upload(idsBuf.Handle, new int[] { tokenId });
        runtime.Upload(posIdsBuf.Handle, posIdHost.AsSpan(0, 1));

        SilkLauncher.Gather(runtime, tokEmb, idsBuf.Handle, x.Handle, 1, hiddenDim);
        SilkLauncher.Gather(runtime, posEmb, posIdsBuf.Handle, posBuf.Handle, 1, hiddenDim);
        SilkLauncher.Add(runtime, x.Handle, posBuf.Handle, x.Handle, hiddenDim);

        for (int li = 0; li < layers.Length; li++)
        {
            LayerWeights w = layers[li];

            SilkLauncher.LayerNorm(runtime, x.Handle, w.Ln1W, w.Ln1B, normed.Handle, 1, hiddenDim, eps);

            SilkLauncher.Gemm(runtime, normed.Handle, w.Wq, q.Handle, 1, hiddenDim, hiddenDim);
            SilkLauncher.Gemm(runtime, normed.Handle, w.Wk, k.Handle, 1, hiddenDim, hiddenDim);
            SilkLauncher.Gemm(runtime, normed.Handle, w.Wv, v.Handle, 1, hiddenDim, hiddenDim);

            runtime.CopyBuffer(k.Handle, kCache[li], 0, position * hiddenDim, hiddenDim);
            runtime.CopyBuffer(v.Handle, vCache[li], 0, position * hiddenDim, hiddenDim);

            int kvLen = position + 1;
            DecodeOproj(q.Handle, kCache[li], vCache[li], context.Handle, w.Wo, x.Handle, x.Handle, kvLen, numHeads, numHeads, headDim, hiddenDim, hiddenDim, scale);

            SilkLauncher.LayerNorm(runtime, x.Handle, w.Ln2W, w.Ln2B, normed.Handle, 1, hiddenDim, eps);

            SilkLauncher.GemmEpilogue(runtime, SilkKernels.GemmGelu, normed.Handle, w.Wfc1, gate.Handle, w.Fc1Bias, 1, hiddenDim, intermediateDim);
            SilkLauncher.GemmEpilogue(runtime, SilkKernels.GemmBias, gate.Handle, w.Wfc2, projected.Handle, w.Fc2Bias, 1, intermediateDim, hiddenDim);
            SilkLauncher.Add(runtime, x.Handle, projected.Handle, x.Handle, hiddenDim);
        }

        SilkLauncher.LayerNorm(runtime, x.Handle, finalNormW, finalNormB, normed.Handle, 1, hiddenDim, eps);

        nint logitsBuf = runtime.Allocate(vocabSize);
        SilkLauncher.Gemm(runtime, normed.Handle, tokEmbTransposed, logitsBuf, 1, hiddenDim, vocabSize);
        var logits = runtime.Download(logitsBuf, vocabSize);
        runtime.Release(logitsBuf);

        if (position + 1 > cachedLen) cachedLen = position + 1;
        return logits;
    }

    public void ResetCache() => cachedLen = 0;

    public int CachedLength => cachedLen;

    void DecodeOproj(nint q, nint kCache, nint vCache, nint contextScratch, nint wo, nint residual, nint outBuf, int kvLen, int numHeads2, int numKvHeads2, int headDim2, int qDim2, int hiddenDim2, float scale2)
    {
        SilkLauncher.DecodeGqaAttentionOproj(runtime, q, kCache, vCache, contextScratch, wo, residual, outBuf, kvLen, numHeads2, numKvHeads2, headDim2, qDim2, hiddenDim2, scale2);
    }

    void EnsureWorkspace(int seqLen)
    {
        int rows = seqLen;
        x = runtime.Ensure(x, rows * hiddenDim);
        normed = runtime.Ensure(normed, rows * hiddenDim);
        q = runtime.Ensure(q, rows * hiddenDim);
        k = runtime.Ensure(k, rows * hiddenDim);
        v = runtime.Ensure(v, rows * hiddenDim);
        context = runtime.Ensure(context, rows * hiddenDim);
        projected = runtime.Ensure(projected, rows * hiddenDim);
        gate = runtime.Ensure(gate, rows * intermediateDim);
        posBuf = runtime.Ensure(posBuf, rows * hiddenDim);
        idsBuf = runtime.Ensure(idsBuf, rows);
        posIdsBuf = runtime.Ensure(posIdsBuf, rows);
    }

    static float[] BuildPositionTable(int seqLen, int embedDim)
    {
        var data = new float[seqLen * embedDim];
        for (int pos = 0; pos < seqLen; pos++)
        {
            for (int i = 0; i < embedDim; i++)
            {
                double exponent = 2.0 * (i / 2) / embedDim;
                double angle = pos / Math.Pow(10000.0, exponent);
                data[pos * embedDim + i] = i % 2 == 0
                    ? (float)Math.Sin(angle)
                    : (float)Math.Cos(angle);
            }
        }
        return data;
    }

    nint UploadPlain(Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
    {
        var t = WeightLayout.Req(tensors, key, "Silk");
        nint buf = runtime.Allocate(t.Data.Length);
        runtime.Upload(buf, t.Data);
        return buf;
    }

    public void Dispose()
    {
        foreach (var l in layers ?? Array.Empty<LayerWeights>()) l.Dispose(runtime);
        if (kCache != null) foreach (var b in kCache) runtime.Release(b);
        if (vCache != null) foreach (var b in vCache) runtime.Release(b);
        if (tokEmb != 0) runtime.Release(tokEmb);
        if (tokEmbTransposed != 0) runtime.Release(tokEmbTransposed);
        if (posEmb != 0) runtime.Release(posEmb);
        if (finalNormW != 0) runtime.Release(finalNormW);
        if (finalNormB != 0) runtime.Release(finalNormB);
    }

    sealed class LayerWeights
    {
        public nint Ln1W, Ln1B;
        public nint Ln2W, Ln2B;
        public nint Wq, Wk, Wv, Wo;
        public nint Wfc1, Fc1Bias;
        public nint Wfc2, Fc2Bias;

        public LayerWeights(SilkTinyShakespeareGpuRunner owner, Dictionary<string, (float[] Data, int[] Shape)> tensors, int i)
        {
            string p = $"layers.{i}.";
            Ln1W = Upload(owner, tensors, p + "ln1.weight");
            Ln1B = Upload(owner, tensors, p + "ln1.bias");
            Ln2W = Upload(owner, tensors, p + "ln2.weight");
            Ln2B = Upload(owner, tensors, p + "ln2.bias");
            Wq = UploadT(owner, tensors, p + "q_proj.weight");
            Wk = UploadT(owner, tensors, p + "k_proj.weight");
            Wv = UploadT(owner, tensors, p + "v_proj.weight");
            Wo = UploadT(owner, tensors, p + "o_proj.weight");
            Wfc1 = UploadT(owner, tensors, p + "mlp_fc1.weight");
            Fc1Bias = Upload(owner, tensors, p + "mlp_fc1.bias");
            Wfc2 = UploadT(owner, tensors, p + "mlp_fc2.weight");
            Fc2Bias = Upload(owner, tensors, p + "mlp_fc2.bias");
        }

        static nint Upload(SilkTinyShakespeareGpuRunner owner, Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
        {
            var t = WeightLayout.Req(tensors, key, "Silk");
            nint buf = owner.runtime.Allocate(t.Data.Length);
            owner.runtime.Upload(buf, t.Data);
            return buf;
        }

        static nint UploadT(SilkTinyShakespeareGpuRunner owner, Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
        {
            var data = WeightLayout.Transposed(tensors, key, "Silk");
            nint buf = owner.runtime.Allocate(data.Length);
            owner.runtime.Upload(buf, data);
            return buf;
        }

        public void Dispose(SilkRuntime rt)
        {
            rt.Release(Ln1W); rt.Release(Ln1B);
            rt.Release(Ln2W); rt.Release(Ln2B);
            rt.Release(Wq); rt.Release(Wk); rt.Release(Wv); rt.Release(Wo);
            rt.Release(Wfc1); rt.Release(Fc1Bias);
            rt.Release(Wfc2); rt.Release(Fc2Bias);
        }
    }
}

public sealed record TinyShakespeareConfig
{
    public int HiddenSize { get; init; } = 96;
    public int NumHiddenLayers { get; init; } = 2;
    public int NumAttentionHeads { get; init; } = 4;
    public int VocabSize { get; init; } = 8000;
    public int MaxSeqLen { get; init; } = 64;
    public float LayerNormEps { get; init; } = 1e-5f;
}
