using Nivara.AI;
using Nivara.AutoDiff.Nn;

namespace Nivara.Silk;

/// <summary>
/// ModernBERT encoder forward through the hand-authored OpenCL kernels, operation for operation
/// with <c>ModernBertGpuRunner</c>. <paramref name="prefix"/> is <c>"model"</c> for the
/// stock checkpoint and <c>"encoder"</c> for Laya.
/// </summary>
public sealed class SilkModernBertGpuRunner : IDisposable
{
    readonly SilkRuntime runtime;
    readonly int hiddenDim;
    readonly int intermediateDim;
    readonly int numHeads;
    readonly int headDim;
    readonly int halfDim;
    readonly float eps;
    readonly float scale;
    readonly int seqCap;

    readonly nint tokEmb;
    readonly nint embedNormW;
    readonly nint finalNormW;
    readonly nint zeroBeta;
    readonly LayerWeights[] layers;

    readonly RotaryEmbedding<float> ropeFull;
    readonly RotaryEmbedding<float> ropeSliding;

    SilkRuntime.SilkBuffer cosFull;
    SilkRuntime.SilkBuffer sinFull;
    SilkRuntime.SilkBuffer cosSliding;
    SilkRuntime.SilkBuffer sinSliding;
    int ropeTableRows;

    SilkRuntime.SilkBuffer x;
    SilkRuntime.SilkBuffer normed;
    SilkRuntime.SilkBuffer qkv;
    SilkRuntime.SilkBuffer q;
    SilkRuntime.SilkBuffer k;
    SilkRuntime.SilkBuffer v;
    SilkRuntime.SilkBuffer context;
    SilkRuntime.SilkBuffer projected;
    SilkRuntime.SilkBuffer mlpIn;
    SilkRuntime.SilkBuffer gated;
    SilkRuntime.SilkBuffer mlpOut;
    SilkRuntime.SilkBuffer maskBuf;
    SilkRuntime.SilkBuffer idsBuf;
    SilkRuntime.SilkBuffer posIdsBuf;

    int[]? posIdHost;
    float[]? maskHost;
    int hiddenRows;

    public SilkModernBertGpuRunner(
        SilkRuntime runtime,
        ModernBertConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        string prefix = "model")
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tensors);

        this.runtime = runtime;
        hiddenDim = config.HiddenSize;
        intermediateDim = config.IntermediateSize;
        numHeads = config.NumAttentionHeads;
        headDim = config.HeadDim;
        halfDim = headDim / 2;
        eps = config.NormEps;
        scale = 1f / MathF.Sqrt(headDim);
        seqCap = config.MaxPositionEmbeddings;
        runtime.EnsureAttentionLocalMemory(headDim);

        try
        {
            tokEmb = UploadPlain(tensors, $"{prefix}.embeddings.tok_embeddings.weight");
            embedNormW = UploadPlain(tensors, $"{prefix}.embeddings.norm.weight");
            finalNormW = UploadPlain(tensors, $"{prefix}.final_norm.weight");
            zeroBeta = Upload(new float[hiddenDim]);

            layers = new LayerWeights[config.NumHiddenLayers];
            for (int i = 0; i < layers.Length; i++)
                layers[i] = new LayerWeights(this, tensors, config, prefix, i);

            ropeFull = new RotaryEmbedding<float>(headDim, config.MaxPositionEmbeddings, config.RopeThetaFull);
            ropeSliding = new RotaryEmbedding<float>(headDim, config.MaxPositionEmbeddings, config.RopeThetaSliding);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>The final-normed hidden state of the last forward, still on device. The buffer is the runner's.</summary>
    public nint HiddenOnDevice => projected.Handle;

    /// <summary>Row count of <see cref="HiddenOnDevice"/>.</summary>
    public int HiddenRows => hiddenRows;

    public float[] Forward(int[] tokenIds, int validLength)
    {
        ForwardOnDevice(tokenIds, validLength);
        return runtime.Download(projected.Handle, hiddenRows * hiddenDim);
    }

    public void ForwardOnDevice(int[] tokenIds, int validLength)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        int seqLen = tokenIds.Length;
        if (seqLen <= 0)
            throw new ArgumentOutOfRangeException(nameof(tokenIds), $"tokenIds must be non-empty, got {seqLen}.");
        if (validLength <= 0 || validLength > seqLen)
            throw new ArgumentOutOfRangeException(nameof(validLength), $"validLength must be within [1, {seqLen}], got {validLength}.");
        if (seqLen > seqCap)
            throw new ArgumentOutOfRangeException(nameof(tokenIds), $"seqLen {seqLen} exceeds max_position_embeddings {seqCap}.");

        EnsureWorkspace(seqLen);
        EnsureRopeTables(seqLen);

        if (posIdHost is null || posIdHost.Length < seqLen)
            posIdHost = new int[seqLen];
        for (int p = 0; p < seqLen; p++)
            posIdHost[p] = p;
        if (maskHost is null || maskHost.Length < seqLen)
            maskHost = new float[seqLen];
        for (int p = 0; p < seqLen; p++)
            maskHost[p] = p < validLength ? 1f : 0f;

        runtime.Upload(idsBuf.Handle, tokenIds.AsSpan(0, seqLen));
        runtime.Upload(posIdsBuf.Handle, posIdHost.AsSpan(0, seqLen));
        runtime.Upload(maskBuf.Handle, maskHost.AsSpan(0, seqLen));

        int rows = seqLen;
        LaunchGather(tokEmb, idsBuf.Handle, x.Handle, rows * hiddenDim);
        LaunchLayerNorm(x.Handle, embedNormW, zeroBeta, x.Handle, rows);

        for (int i = 0; i < layers.Length; i++)
        {
            LayerWeights w = layers[i];
            nint qkvIn = w.AttnNormW == 0
                ? x.Handle
                : LayerNormInto(x.Handle, w.AttnNormW, rows);

            LaunchGemm(qkvIn, w.Wqkv, qkv.Handle, rows, hiddenDim, 3 * hiddenDim);
            for (int part = 0; part < 3; part++)
            {
                nint dst = part switch { 0 => q.Handle, 1 => k.Handle, _ => v.Handle };
                LaunchSplit(qkv.Handle, dst, rows, part);
            }

            nint cos = w.Full ? cosFull.Handle : cosSliding.Handle;
            nint sin = w.Full ? sinFull.Handle : sinSliding.Handle;
            LaunchRotary(q.Handle, cos, sin, rows);
            LaunchRotary(k.Handle, cos, sin, rows);
            LaunchAttention(q.Handle, k.Handle, v.Handle, maskBuf.Handle, context.Handle, rows, w.Band);

            LaunchGemm(context.Handle, w.Wo, projected.Handle, rows, hiddenDim, hiddenDim);
            LaunchAdd(x.Handle, projected.Handle, x.Handle, rows * hiddenDim);

            LayerNormInto(x.Handle, w.MlpNormW, rows);
            LaunchGemm(normed.Handle, w.Wi, mlpIn.Handle, rows, hiddenDim, 2 * intermediateDim);
            LaunchGeGlu(mlpIn.Handle, gated.Handle, rows);
            LaunchGemm(gated.Handle, w.WoMlp, mlpOut.Handle, rows, intermediateDim, hiddenDim);
            LaunchAdd(x.Handle, mlpOut.Handle, x.Handle, rows * hiddenDim);
        }

        LaunchLayerNorm(x.Handle, finalNormW, zeroBeta, projected.Handle, rows);
        hiddenRows = rows;
    }

    nint LayerNormInto(nint src, nint gamma, int rows)
    {
        LaunchLayerNorm(src, gamma, zeroBeta, normed.Handle, rows);
        return normed.Handle;
    }

    void EnsureWorkspace(int seqLen)
    {
        int rows = seqLen;
        x = runtime.Ensure(x, rows * hiddenDim);
        normed = runtime.Ensure(normed, rows * hiddenDim);
        qkv = runtime.Ensure(qkv, rows * 3 * hiddenDim);
        q = runtime.Ensure(q, rows * hiddenDim);
        k = runtime.Ensure(k, rows * hiddenDim);
        v = runtime.Ensure(v, rows * hiddenDim);
        context = runtime.Ensure(context, rows * hiddenDim);
        projected = runtime.Ensure(projected, rows * hiddenDim);
        mlpIn = runtime.Ensure(mlpIn, rows * 2 * intermediateDim);
        gated = runtime.Ensure(gated, rows * intermediateDim);
        mlpOut = runtime.Ensure(mlpOut, rows * hiddenDim);
        maskBuf = runtime.Ensure(maskBuf, rows);
        idsBuf = runtime.Ensure(idsBuf, rows);
        posIdsBuf = runtime.Ensure(posIdsBuf, rows);
    }

    void EnsureRopeTables(int seqLen)
    {
        if (ropeTableRows >= seqLen)
            return;

        int n = seqLen * halfDim;
        cosFull = runtime.Ensure(cosFull, n);
        sinFull = runtime.Ensure(sinFull, n);
        cosSliding = runtime.Ensure(cosSliding, n);
        sinSliding = runtime.Ensure(sinSliding, n);

        ropeFull.GetPositionTables(0, seqLen, out var fullCos, out var fullSin);
        runtime.Upload(cosFull.Handle, fullCos.ToArray());
        runtime.Upload(sinFull.Handle, fullSin.ToArray());
        ropeSliding.GetPositionTables(0, seqLen, out var slideCos, out var slideSin);
        runtime.Upload(cosSliding.Handle, slideCos.ToArray());
        runtime.Upload(sinSliding.Handle, slideSin.ToArray());
        ropeTableRows = seqLen;
    }

    void LaunchGather(nint table, nint ids, nint y, int n)
        => SilkLauncher.Gather(runtime, table, ids, y, n, hiddenDim);

    void LaunchLayerNorm(nint src, nint gamma, nint beta, nint y, int rows)
        => SilkLauncher.LayerNorm(runtime, src, gamma, beta, y, rows, hiddenDim, eps);

    void LaunchGemm(nint a, nint b, nint c, int aRows, int aCols, int bCols)
        => SilkLauncher.Gemm(runtime, a, b, c, aRows, aCols, bCols);

    void LaunchSplit(nint src, nint dst, int rows, int part)
        => SilkLauncher.Split(runtime, src, dst, rows, hiddenDim, 3, part);

    void LaunchRotary(nint buffer, nint cos, nint sin, int rows)
        => SilkLauncher.Rotary(runtime, buffer, posIdsBuf.Handle, cos, sin, rows, numHeads, headDim, halfDim);

    void LaunchAttention(nint qBuf, nint kBuf, nint vBuf, nint mask, nint output, int seqLen, int band)
        => SilkLauncher.Attention(runtime, qBuf, kBuf, vBuf, mask, output, 1, seqLen, numHeads, headDim, band, scale);

    void LaunchAdd(nint a, nint b, nint y, int n)
        => SilkLauncher.Add(runtime, a, b, y, n);

    void LaunchGeGlu(nint src, nint y, int rows)
        => SilkLauncher.GeGlu(runtime, src, y, rows, intermediateDim);

    nint UploadPlain(Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
        => Upload(WeightLayout.Req(tensors, key, "Silk").Data);

    nint UploadTransposed(Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
        => Upload(WeightLayout.Transposed(tensors, key, "Silk"));

    nint Upload(float[] data) => SilkLauncher.Upload(runtime, data);

    public void Dispose()
    {
        Release(tokEmb);
        Release(embedNormW);
        Release(finalNormW);
        Release(zeroBeta);
        if (layers is not null)
        {
            foreach (LayerWeights w in layers)
                w?.Dispose();
        }

        Release(cosFull);
        Release(sinFull);
        Release(cosSliding);
        Release(sinSliding);
        Release(x);
        Release(normed);
        Release(qkv);
        Release(q);
        Release(k);
        Release(v);
        Release(context);
        Release(projected);
        Release(mlpIn);
        Release(gated);
        Release(mlpOut);
        Release(maskBuf);
        Release(idsBuf);
        Release(posIdsBuf);
    }

    void Release(SilkRuntime.SilkBuffer buffer) => runtime.Release(buffer.Handle);

    void Release(nint buffer) => runtime.Release(buffer);

    sealed class LayerWeights : IDisposable
    {
        readonly SilkRuntime runtime;
        public readonly nint Wqkv;
        public readonly nint Wo;
        public readonly nint Wi;
        public readonly nint WoMlp;
        public readonly nint MlpNormW;
        public readonly nint AttnNormW;
        public readonly int Band;
        public readonly bool Full;

        public LayerWeights(
            SilkModernBertGpuRunner owner,
            Dictionary<string, (float[] Data, int[] Shape)> tensors,
            ModernBertConfig config,
            string prefix,
            int index)
        {
            runtime = owner.runtime;
            try
            {
                string layerPrefix = $"{prefix}.layers.{index}";
                Wqkv = owner.UploadTransposed(tensors, $"{layerPrefix}.attn.Wqkv.weight");
                Wo = owner.UploadTransposed(tensors, $"{layerPrefix}.attn.Wo.weight");
                Wi = owner.UploadTransposed(tensors, $"{layerPrefix}.mlp.Wi.weight");
                WoMlp = owner.UploadTransposed(tensors, $"{layerPrefix}.mlp.Wo.weight");
                MlpNormW = owner.UploadPlain(tensors, $"{layerPrefix}.mlp_norm.weight");
                AttnNormW = index == 0 ? 0 : owner.UploadPlain(tensors, $"{layerPrefix}.attn_norm.weight");
                Band = config.Band(index);
                Full = config.IsFullAttention(index);
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        public void Dispose()
        {
            runtime.Release(Wqkv);
            runtime.Release(Wo);
            runtime.Release(Wi);
            runtime.Release(WoMlp);
            runtime.Release(MlpNormW);
            runtime.Release(AttnNormW);
        }
    }
}
