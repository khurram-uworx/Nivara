using Nivara.AutoDiff.Nn;
using System.Diagnostics;

namespace Nivara.Silk;

/// <summary>
/// SmolLM (Llama-family causal LM) forward through the hand-authored OpenCL kernels, F32 only.
/// Mirrors <see cref="LlamaForCausalLM{T}"/> / <see cref="LlamaDecoderBlock{T}"/> operation
/// for operation. Prefill-only (no KV cache): generation runs the full forward each step, matching
/// the CPU cache-free path.
/// </summary>
public sealed partial class SilkQwenGpuRunner : IDisposable
{
    readonly SilkRuntime runtime;
    readonly QwenSilkConfig config;
    readonly int hiddenDim;
    readonly int intermediateDim;
    readonly int numHeads;
    readonly int numKvHeads;
    readonly int headDim;
    readonly int qDim;
    readonly int kvDim;
    readonly float eps;
    readonly float scale;
    readonly int maxPos;
    readonly float ropeTheta;
    readonly int seqCap;
    readonly int vocabSize;

    readonly nint tokEmb;
    readonly bool bf16Weights;
    readonly nint finalNormW;
    readonly LayerWeights[] layers;

    readonly RotaryEmbedding<float> rotary;
    SilkRuntime.SilkBuffer cosBuf;
    SilkRuntime.SilkBuffer sinBuf;
    int ropeTableRows;

    SilkRuntime.SilkBuffer x;
    SilkRuntime.SilkBuffer normed;
    SilkRuntime.SilkBuffer q;
    SilkRuntime.SilkBuffer k;
    SilkRuntime.SilkBuffer v;
    SilkRuntime.SilkBuffer context;
    SilkRuntime.SilkBuffer projected;
    SilkRuntime.SilkBuffer gate;
    SilkRuntime.SilkBuffer up;
    SilkRuntime.SilkBuffer mlp;
    SilkRuntime.SilkBuffer idsBuf;
    SilkRuntime.SilkBuffer posIdsBuf;

    int[]? posIdHost;
    int[]? tokenIdHost;

    // Reused logits device buffer for ForwardCachedArgmax (avoids per-token Allocate/Release).
    SilkRuntime.SilkBuffer logitsBuf;
    // KV cache (device, per layer). Capacity = seqCap positions.
    readonly nint[] kCache;
    readonly nint[] vCache;
    int cachedLen;

    // Device-side argmax partial buffers (#663).
    nint partialArgmaxIdx;
    nint partialArgmaxVal;
    int argmaxGroups;

    // LoRA inference slots (optional). A/B are F32 device buffers.
    int loraRank;
    float loraScale;
    nint loraTmpBuf; // [maxSeqLen, rank]
    nint loraOutBuf; // [maxSeqLen, maxOutDim]

    // Decode profiling (opt-in). Enabled only for diagnostic runs; normal path is unchanged.
    int profileTokens;
    int profileDispatches;
    int profileCopies;
    double profileEmbedMs;
    double profileQkvMs;
    double profileRotaryMs;
    double profileCacheCopyMs;
    double profileAttentionOProjMs;
    double profileMlpMs;
    double profileFinalNormLmHeadMs;
    double profileArgmaxMs;

    /// <summary>Enables per-phase decode profiling. When true, each profiled decode token records
    /// wall-clock phase timings with <see cref="SilkRuntime.Finish"/> boundaries and accumulates
    /// dispatch/copy counts.</summary>
    public bool ProfilingEnabled { get; set; }

    /// <summary>Clears accumulated decode profiling counters.</summary>
    public void ResetDecodeProfile()
    {
        profileTokens = 0;
        profileDispatches = 0;
        profileCopies = 0;
        profileEmbedMs = 0;
        profileQkvMs = 0;
        profileRotaryMs = 0;
        profileCacheCopyMs = 0;
        profileAttentionOProjMs = 0;
        profileMlpMs = 0;
        profileFinalNormLmHeadMs = 0;
        profileArgmaxMs = 0;
    }

    /// <summary>Returns the accumulated decode profiling snapshot.</summary>
    public SilkDecodeProfile GetDecodeProfile() => new(
        profileTokens,
        profileDispatches,
        profileCopies,
        profileEmbedMs,
        profileQkvMs,
        profileRotaryMs,
        profileCacheCopyMs,
        profileAttentionOProjMs,
        profileMlpMs,
        profileFinalNormLmHeadMs,
        profileArgmaxMs);

    bool disposed;

    public SilkQwenGpuRunner(
        SilkRuntime runtime,
        QwenSilkConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tensors);

        this.runtime = runtime;
        this.config = config;
        this.bf16Weights = false;
        hiddenDim = config.HiddenSize;
        intermediateDim = config.IntermediateSize;
        numHeads = config.NumAttentionHeads;
        numKvHeads = config.NumKeyValueHeads;
        headDim = hiddenDim / numHeads;
        qDim = numHeads * headDim;
        kvDim = numKvHeads * headDim;
        eps = config.RmsNormEps;
        scale = 1f / MathF.Sqrt(headDim);
        maxPos = config.MaxPositionEmbeddings;
        ropeTheta = config.RopeTheta;
        seqCap = maxPos;
        vocabSize = config.VocabSize;
        runtime.EnsureAttentionLocalMemory(headDim);

        try
        {
            tokEmb = UploadPlain(tensors, "model.embed_tokens.weight");
            var tokEmbT = WeightLayout.Transposed(tensors, "model.embed_tokens.weight", "Silk");
            tokEmbTransposed = runtime.Allocate(tokEmbT.Length);
            runtime.Upload(tokEmbTransposed, tokEmbT);
            finalNormW = UploadPlain(tensors, "model.norm.weight");

            layers = new LayerWeights[config.NumHiddenLayers];
            for (int i = 0; i < layers.Length; i++)
                layers[i] = new LayerWeights(this, tensors, i);

            kCache = new nint[config.NumHiddenLayers];
            vCache = new nint[config.NumHiddenLayers];
            for (int i = 0; i < config.NumHiddenLayers; i++)
            {
                kCache[i] = runtime.Allocate(maxPos * kvDim);
                vCache[i] = runtime.Allocate(maxPos * kvDim);
            }

            rotary = new RotaryEmbedding<float>(headDim, maxPos, ropeTheta);

            argmaxGroups = (vocabSize + 255) / 256;
            partialArgmaxIdx = runtime.Allocate(argmaxGroups);
            partialArgmaxVal = runtime.Allocate(argmaxGroups);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public string DeviceName => runtime.DeviceName;

    public float[] Forward(int[] tokenIds)
    {
        int seqLen = tokenIds.Length;
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(tokenIds));
        if (seqLen > seqCap) throw new ArgumentOutOfRangeException(nameof(tokenIds), $"seqLen {seqLen} exceeds {seqCap}.");

        EnsureWorkspace(seqLen);
        EnsureRopeTables(seqLen);

        if (posIdHost is null || posIdHost.Length < seqLen)
            posIdHost = new int[seqLen];
        for (int p = 0; p < seqLen; p++) posIdHost[p] = p;

        runtime.Upload(idsBuf.Handle, tokenIds.AsSpan(0, seqLen));
        runtime.Upload(posIdsBuf.Handle, posIdHost.AsSpan(0, seqLen));

        // gather embedding -> x
        runtime.SetBuffer(SilkKernels.GatherRows, 0, tokEmb);
        // reuse gather: table=tokEmb, ids=idsBuf, y=x
        GatherTokEmb(tokEmb, idsBuf.Handle, x.Handle, seqLen * hiddenDim, hiddenDim);

        for (int li = 0; li < layers.Length; li++)
        {
            LayerWeights w = layers[li];

            // input norm
            SilkLauncher.RmsNorm(runtime, x.Handle, w.InputNormW, normed.Handle, seqLen, hiddenDim, eps);

            // Q/K/V
            GemmWithLora(normed.Handle, w.Wq, w.QLora, q.Handle, seqLen, hiddenDim, qDim);
            SilkLauncher.AddBias(runtime, q.Handle, w.QBias, q.Handle, seqLen, qDim);
            GemmWithLora(normed.Handle, w.Wk, w.KLora, k.Handle, seqLen, hiddenDim, kvDim);
            SilkLauncher.AddBias(runtime, k.Handle, w.KBias, k.Handle, seqLen, kvDim);
            GemmWithLora(normed.Handle, w.Wv, w.VLora, v.Handle, seqLen, hiddenDim, kvDim);
            SilkLauncher.AddBias(runtime, v.Handle, w.VBias, v.Handle, seqLen, kvDim);

            // RoPE in place
            SilkLauncher.RotaryQk(runtime, q.Handle, k.Handle, posIdsBuf.Handle, cosBuf.Handle, sinBuf.Handle, seqLen, numHeads, numKvHeads, headDim, headDim / 2);

            // causal GQA attention -> context
            SilkLauncher.CausalGqaAttention(runtime, q.Handle, k.Handle, v.Handle, context.Handle, seqLen, numHeads, numKvHeads, headDim, scale);

            // O proj + residual
            GemmWithLora(context.Handle, w.Wo, w.OLora, projected.Handle, seqLen, qDim, hiddenDim);
            SilkLauncher.Add(runtime, x.Handle, projected.Handle, x.Handle, seqLen * hiddenDim);

            // post norm
            SilkLauncher.RmsNorm(runtime, x.Handle, w.PostNormW, normed.Handle, seqLen, hiddenDim, eps);

            // gate/up
            GemmWithLora(normed.Handle, w.Wgate, w.GateLora, gate.Handle, seqLen, hiddenDim, intermediateDim);
            GemmWithLora(normed.Handle, w.Wup, w.UpLora, up.Handle, seqLen, hiddenDim, intermediateDim);

            // swiglu -> mlp (intermediate)
            SilkLauncher.SwiGlu(runtime, gate.Handle, up.Handle, mlp.Handle, seqLen, intermediateDim);

            // down + residual
            GemmWithLora(mlp.Handle, w.Wdown, w.DownLora, projected.Handle, seqLen, intermediateDim, hiddenDim);
            SilkLauncher.Add(runtime, x.Handle, projected.Handle, x.Handle, seqLen * hiddenDim);
        }

        // final norm
        SilkLauncher.RmsNorm(runtime, x.Handle, finalNormW, normed.Handle, seqLen, hiddenDim, eps);

        // tied LM head: logits = normed @ tokEmb^T  (last row only to save memory)
        // Compute for all rows but read back all; caller (generation) uses last row.
        nint logitsBuf = runtime.Allocate(seqLen * vocabSize);
        Gemm(normed.Handle, tokEmbTransposed, logitsBuf, seqLen, hiddenDim, vocabSize);
        // tokEmb is [vocab,hidden]; need transposed [hidden,vocab]. We don't have it precomputed.
        // Fallback: compute via host transpose on demand is expensive; instead store tokEmbT.
        var logits = runtime.Download(logitsBuf, seqLen * vocabSize);
        runtime.Release(logitsBuf);
        return logits;
    }

    // We need tokEmb transposed for LM head. Store it.
    nint tokEmbTransposed;

    // placeholder to satisfy field init order; set in constructor after tokEmb upload
    // (handled below via late init in constructor)

    public float[] ForwardLastLogits(int[] tokenIds)
    {
        var all = Forward(tokenIds);
        int seqLen = tokenIds.Length;
        var last = new float[vocabSize];
        Array.Copy(all, (seqLen - 1) * vocabSize, last, 0, vocabSize);
        return last;
    }

    /// <summary>Device-side argmax over the last row of a full forward (#663).</summary>
    public int ForwardLastArgmax(int[] tokenIds) => ForwardArgmax(tokenIds);

    public int ForwardArgmax(int[] tokenIds)
    {
        int seqLen = tokenIds.Length;
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(tokenIds));
        if (seqLen > seqCap) throw new ArgumentOutOfRangeException(nameof(tokenIds));

        EnsureWorkspace(seqLen);
        EnsureRopeTables(seqLen);

        if (posIdHost is null || posIdHost.Length < seqLen)
            posIdHost = new int[seqLen];
        for (int p = 0; p < seqLen; p++) posIdHost[p] = p;

        runtime.Upload(idsBuf.Handle, tokenIds.AsSpan(0, seqLen));
        runtime.Upload(posIdsBuf.Handle, posIdHost.AsSpan(0, seqLen));

        GatherTokEmb(tokEmb, idsBuf.Handle, x.Handle, seqLen * hiddenDim, hiddenDim);

        for (int li = 0; li < layers.Length; li++)
        {
            LayerWeights w = layers[li];

            SilkLauncher.RmsNorm(runtime, x.Handle, w.InputNormW, normed.Handle, seqLen, hiddenDim, eps);

            GemmWithLora(normed.Handle, w.Wq, w.QLora, q.Handle, seqLen, hiddenDim, qDim);
            SilkLauncher.AddBias(runtime, q.Handle, w.QBias, q.Handle, seqLen, qDim);
            GemmWithLora(normed.Handle, w.Wk, w.KLora, k.Handle, seqLen, hiddenDim, kvDim);
            SilkLauncher.AddBias(runtime, k.Handle, w.KBias, k.Handle, seqLen, kvDim);
            GemmWithLora(normed.Handle, w.Wv, w.VLora, v.Handle, seqLen, hiddenDim, kvDim);
            SilkLauncher.AddBias(runtime, v.Handle, w.VBias, v.Handle, seqLen, kvDim);

            SilkLauncher.RotaryQk(runtime, q.Handle, k.Handle, posIdsBuf.Handle, cosBuf.Handle, sinBuf.Handle, seqLen, numHeads, numKvHeads, headDim, headDim / 2);

            SilkLauncher.CausalGqaAttention(runtime, q.Handle, k.Handle, v.Handle, context.Handle, seqLen, numHeads, numKvHeads, headDim, scale);

            GemmWithLora(context.Handle, w.Wo, w.OLora, projected.Handle, seqLen, qDim, hiddenDim);
            SilkLauncher.Add(runtime, x.Handle, projected.Handle, x.Handle, seqLen * hiddenDim);

            SilkLauncher.RmsNorm(runtime, x.Handle, w.PostNormW, normed.Handle, seqLen, hiddenDim, eps);

            GemmWithLora(normed.Handle, w.Wgate, w.GateLora, gate.Handle, seqLen, hiddenDim, intermediateDim);
            GemmWithLora(normed.Handle, w.Wup, w.UpLora, up.Handle, seqLen, hiddenDim, intermediateDim);

            SilkLauncher.SwiGlu(runtime, gate.Handle, up.Handle, mlp.Handle, seqLen, intermediateDim);

            GemmWithLora(mlp.Handle, w.Wdown, w.DownLora, projected.Handle, seqLen, intermediateDim, hiddenDim);
            SilkLauncher.Add(runtime, x.Handle, projected.Handle, x.Handle, seqLen * hiddenDim);
        }

        nint lastRowView = runtime.CreateSubview(x.Handle, (seqLen - 1) * hiddenDim, hiddenDim);
        nint logitsBuf = runtime.Allocate(vocabSize);
        RmsNormLmHead(lastRowView, finalNormW, tokEmbTransposed, logitsBuf, 1, hiddenDim, vocabSize, eps);
        int argmax = ArgMaxDevice(logitsBuf, vocabSize);
        runtime.Release(logitsBuf);
        runtime.Release(lastRowView);
        return argmax;
    }

    public int ForwardPrefillArgmax(int[] tokenIds)
    {
        int seqLen = tokenIds.Length;
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(tokenIds));
        if (seqLen > seqCap) throw new ArgumentOutOfRangeException(nameof(tokenIds));
        cachedLen = 0;

        EnsureWorkspace(seqLen);
        EnsureRopeTables(seqLen);

        if (posIdHost is null || posIdHost.Length < seqLen)
            posIdHost = new int[seqLen];
        for (int p = 0; p < seqLen; p++) posIdHost[p] = p;

        runtime.Upload(idsBuf.Handle, tokenIds.AsSpan(0, seqLen));
        runtime.Upload(posIdsBuf.Handle, posIdHost.AsSpan(0, seqLen));

        GatherTokEmb(tokEmb, idsBuf.Handle, x.Handle, seqLen * hiddenDim, hiddenDim);

        for (int li = 0; li < layers.Length; li++)
        {
            LayerWeights w = layers[li];

            SilkLauncher.RmsNorm(runtime, x.Handle, w.InputNormW, normed.Handle, seqLen, hiddenDim, eps);

            GemmWithLora(normed.Handle, w.Wq, w.QLora, q.Handle, seqLen, hiddenDim, qDim);
            SilkLauncher.AddBias(runtime, q.Handle, w.QBias, q.Handle, seqLen, qDim);
            GemmWithLora(normed.Handle, w.Wk, w.KLora, k.Handle, seqLen, hiddenDim, kvDim);
            SilkLauncher.AddBias(runtime, k.Handle, w.KBias, k.Handle, seqLen, kvDim);
            GemmWithLora(normed.Handle, w.Wv, w.VLora, v.Handle, seqLen, hiddenDim, kvDim);
            SilkLauncher.AddBias(runtime, v.Handle, w.VBias, v.Handle, seqLen, kvDim);

            SilkLauncher.RotaryQk(runtime, q.Handle, k.Handle, posIdsBuf.Handle, cosBuf.Handle, sinBuf.Handle, seqLen, numHeads, numKvHeads, headDim, headDim / 2);

            runtime.CopyBuffer(k.Handle, kCache[li], 0, 0, seqLen * kvDim);
            runtime.CopyBuffer(v.Handle, vCache[li], 0, 0, seqLen * kvDim);

            SilkLauncher.CausalGqaAttention(runtime, q.Handle, k.Handle, v.Handle, context.Handle, seqLen, numHeads, numKvHeads, headDim, scale);

            GemmWithLora(context.Handle, w.Wo, w.OLora, projected.Handle, seqLen, qDim, hiddenDim);
            SilkLauncher.Add(runtime, x.Handle, projected.Handle, x.Handle, seqLen * hiddenDim);

            SilkLauncher.RmsNorm(runtime, x.Handle, w.PostNormW, normed.Handle, seqLen, hiddenDim, eps);

            GemmWithLora(normed.Handle, w.Wgate, w.GateLora, gate.Handle, seqLen, hiddenDim, intermediateDim);
            GemmWithLora(normed.Handle, w.Wup, w.UpLora, up.Handle, seqLen, hiddenDim, intermediateDim);

            SilkLauncher.SwiGlu(runtime, gate.Handle, up.Handle, mlp.Handle, seqLen, intermediateDim);

            GemmWithLora(mlp.Handle, w.Wdown, w.DownLora, projected.Handle, seqLen, intermediateDim, hiddenDim);
            SilkLauncher.Add(runtime, x.Handle, projected.Handle, x.Handle, seqLen * hiddenDim);
        }

        nint lastRowView = runtime.CreateSubview(x.Handle, (seqLen - 1) * hiddenDim, hiddenDim);
        nint logitsBuf = runtime.Allocate(vocabSize);
        RmsNormLmHead(lastRowView, finalNormW, tokEmbTransposed, logitsBuf, 1, hiddenDim, vocabSize, eps);
        int argmax = ArgMaxDevice(logitsBuf, vocabSize);
        runtime.Release(logitsBuf);
        runtime.Release(lastRowView);

        cachedLen = seqLen;
        return argmax;
    }

    public int ForwardCachedArgmax(int tokenId, int position)
    {
        if (position < 0) throw new ArgumentOutOfRangeException(nameof(position));
        if (position + 1 > seqCap) throw new ArgumentOutOfRangeException(nameof(position));

        EnsureWorkspace(1);

        if (posIdHost is null || posIdHost.Length < 1)
            posIdHost = new int[1];
        posIdHost[0] = position;

        if (tokenIdHost is null || tokenIdHost.Length < 1)
            tokenIdHost = new int[1];
        tokenIdHost[0] = tokenId;

        runtime.Upload(idsBuf.Handle, tokenIdHost.AsSpan(0, 1));
        runtime.Upload(posIdsBuf.Handle, posIdHost.AsSpan(0, 1));

        bool profile = ProfilingEnabled;
        Stopwatch? sw = profile ? Stopwatch.StartNew() : null;
        int d0 = profile ? runtime.KernelDispatchCount : 0;
        int c0 = profile ? runtime.QueueCopyCount : 0;

        GatherTokEmb(tokEmb, idsBuf.Handle, x.Handle, hiddenDim, hiddenDim);
        if (profile) { runtime.Finish(); profileEmbedMs += sw!.Elapsed.TotalMilliseconds; sw.Restart(); }

        for (int li = 0; li < layers.Length; li++)
        {
            LayerWeights w = layers[li];

            RmsNormGemv(x.Handle, w.InputNormW,
                [
                    new(w.Wq, w.QBias, q.Handle, qDim),
                    new(w.Wk, w.KBias, k.Handle, kvDim),
                    new(w.Wv, w.VBias, v.Handle, kvDim),
                ],
                [w.QLora, w.KLora, w.VLora],
                hasBias: true, aRows: 1, aCols: hiddenDim, eps: eps);
            if (profile) { runtime.Finish(); profileQkvMs += sw!.Elapsed.TotalMilliseconds; sw.Restart(); }

            EnsureRopeTables(position + 1);
            SilkLauncher.RotaryQk(runtime, q.Handle, k.Handle, posIdsBuf.Handle, cosBuf.Handle, sinBuf.Handle, 1, numHeads, numKvHeads, headDim, headDim / 2);
            if (profile) { runtime.Finish(); profileRotaryMs += sw!.Elapsed.TotalMilliseconds; sw.Restart(); }

            runtime.CopyBuffer(k.Handle, kCache[li], 0, position * kvDim, kvDim);
            runtime.CopyBuffer(v.Handle, vCache[li], 0, position * kvDim, kvDim);
            if (profile) { runtime.Finish(); profileCacheCopyMs += sw!.Elapsed.TotalMilliseconds; sw.Restart(); }

            int kvLen = position + 1;
            DecodeOproj(q.Handle, kCache[li], vCache[li], context.Handle,
                w.Wo, w.OLora, x.Handle, x.Handle, kvLen, numHeads, numKvHeads, headDim, qDim, hiddenDim, scale);
            if (profile) { runtime.Finish(); profileAttentionOProjMs += sw!.Elapsed.TotalMilliseconds; sw.Restart(); }

            MlpBlock(x.Handle, w.PostNormW, w.Wgate, w.GateLora, w.Wup, w.UpLora, w.Wdown, w.DownLora,
                x.Handle, x.Handle, hiddenDim, intermediateDim, eps);
            if (profile) { runtime.Finish(); profileMlpMs += sw!.Elapsed.TotalMilliseconds; sw.Restart(); }
        }

        logitsBuf = runtime.Ensure(logitsBuf, vocabSize);
        RmsNormGemv(x.Handle, finalNormW,
            [new(tokEmbTransposed, 0, logitsBuf.Handle, vocabSize)],
            hasBias: false, aRows: 1, aCols: hiddenDim, eps: eps);
        if (profile) { runtime.Finish(); profileFinalNormLmHeadMs += sw!.Elapsed.TotalMilliseconds; sw.Restart(); }

        int argmax = ArgMaxDevice(logitsBuf.Handle, vocabSize);
        if (profile) { runtime.Finish(); profileArgmaxMs += sw!.Elapsed.TotalMilliseconds; }

        if (profile)
        {
            profileTokens++;
            profileDispatches += runtime.KernelDispatchCount - d0;
            profileCopies += runtime.QueueCopyCount - c0;
        }

        if (position + 1 > cachedLen) cachedLen = position + 1;
        return argmax;
    }

    int ArgMaxDevice(nint logits, int count)
    {
        SilkLauncher.ArgMaxPartial(runtime, logits, partialArgmaxIdx, partialArgmaxVal, count, argmaxGroups);
        var indices = runtime.DownloadInt(partialArgmaxIdx, argmaxGroups);
        var values = runtime.Download(partialArgmaxVal, argmaxGroups);
        int best = indices[0];
        float bestVal = values[0];
        for (int i = 1; i < argmaxGroups; i++)
            if (values[i] > bestVal) { bestVal = values[i]; best = indices[i]; }
        return best;
    }

    public void ResetCache() => cachedLen = 0;

    public int CachedLength => cachedLen;

    public void LoadAdapterBundle(string adapterPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(adapterPath);
        var bundle = LoraSftAdapterBundle.Load(adapterPath);

        var targets = SilkLoraAdapterLoader.ParseTargets(bundle.Targets);
        var expected = SilkLoraAdapterLoader.BuildExpectedSlots(targets, layers.Length, hiddenDim, qDim, kvDim, intermediateDim);
        var parsed = SilkLoraAdapterLoader.ValidateAndParse(bundle, expected);

        ReleaseAllLora();

        loraRank = bundle.Rank;
        loraScale = bundle.Alpha / bundle.Rank;
        int maxOut = Math.Max(hiddenDim, Math.Max(qDim, Math.Max(kvDim, intermediateDim)));
        loraTmpBuf = runtime.Allocate(seqCap * loraRank);
        loraOutBuf = runtime.Allocate(seqCap * maxOut);

        for (int i = 0; i < expected.Count; i++)
        {
            var slot = expected[i];
            var data = parsed[i];
            var pair = new LoraPair
            {
                A = runtime.Allocate(data.A.Length),
                B = runtime.Allocate(data.B.Length),
                InFeatures = slot.InFeatures,
                OutFeatures = slot.OutFeatures,
            };
            runtime.Upload(pair.A, data.A);
            runtime.Upload(pair.B, data.B);
            AssignLora(slot.Layer, slot.Target, pair);
        }

    }

    void ReleaseAllLora()
    {
        foreach (var layer in layers ?? Array.Empty<LayerWeights>())
        {
            ReleaseLoraPair(ref layer.QLora);
            ReleaseLoraPair(ref layer.KLora);
            ReleaseLoraPair(ref layer.VLora);
            ReleaseLoraPair(ref layer.OLora);
            ReleaseLoraPair(ref layer.GateLora);
            ReleaseLoraPair(ref layer.UpLora);
            ReleaseLoraPair(ref layer.DownLora);
        }
        if (loraTmpBuf != 0) { runtime.Release(loraTmpBuf); loraTmpBuf = 0; }
        if (loraOutBuf != 0) { runtime.Release(loraOutBuf); loraOutBuf = 0; }
        loraRank = 0;
        loraScale = 0f;
    }

    void ReleaseLoraPair(ref LoraPair? pair)
    {
        if (pair is null) return;
        if (pair.A != 0) runtime.Release(pair.A);
        if (pair.B != 0) runtime.Release(pair.B);
        pair = null;
    }

    void AssignLora(int layerIndex, SilkLlamaLoRATarget target, LoraPair pair)
    {
        var layer = layers[layerIndex];
        switch (target)
        {
            case SilkLlamaLoRATarget.Q: layer.QLora = pair; break;
            case SilkLlamaLoRATarget.K: layer.KLora = pair; break;
            case SilkLlamaLoRATarget.V: layer.VLora = pair; break;
            case SilkLlamaLoRATarget.O: layer.OLora = pair; break;
            case SilkLlamaLoRATarget.Gate: layer.GateLora = pair; break;
            case SilkLlamaLoRATarget.Up: layer.UpLora = pair; break;
            case SilkLlamaLoRATarget.Down: layer.DownLora = pair; break;
            default: throw new InvalidOperationException($"Unsupported LoRA target '{target}'.");
        }
    }

    /// <summary>Prefills the cache with <paramref name="tokenIds"/> and returns last-row logits.</summary>
    public float[] ForwardPrefill(int[] tokenIds)
    {
        int seqLen = tokenIds.Length;
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(tokenIds));
        if (seqLen > seqCap) throw new ArgumentOutOfRangeException(nameof(tokenIds));
        cachedLen = 0;

        EnsureWorkspace(seqLen);
        EnsureRopeTables(seqLen);

        if (posIdHost is null || posIdHost.Length < seqLen)
            posIdHost = new int[seqLen];
        for (int p = 0; p < seqLen; p++) posIdHost[p] = p;

        runtime.Upload(idsBuf.Handle, tokenIds.AsSpan(0, seqLen));
        runtime.Upload(posIdsBuf.Handle, posIdHost.AsSpan(0, seqLen));

        // gather embedding -> x
        GatherTokEmb(tokEmb, idsBuf.Handle, x.Handle, seqLen * hiddenDim, hiddenDim);

        for (int li = 0; li < layers.Length; li++)
        {
            LayerWeights w = layers[li];

            SilkLauncher.RmsNorm(runtime, x.Handle, w.InputNormW, normed.Handle, seqLen, hiddenDim, eps);

            GemmWithLora(normed.Handle, w.Wq, w.QLora, q.Handle, seqLen, hiddenDim, qDim);
            SilkLauncher.AddBias(runtime, q.Handle, w.QBias, q.Handle, seqLen, qDim);
            GemmWithLora(normed.Handle, w.Wk, w.KLora, k.Handle, seqLen, hiddenDim, kvDim);
            SilkLauncher.AddBias(runtime, k.Handle, w.KBias, k.Handle, seqLen, kvDim);
            GemmWithLora(normed.Handle, w.Wv, w.VLora, v.Handle, seqLen, hiddenDim, kvDim);
            SilkLauncher.AddBias(runtime, v.Handle, w.VBias, v.Handle, seqLen, kvDim);

            SilkLauncher.RotaryQk(runtime, q.Handle, k.Handle, posIdsBuf.Handle, cosBuf.Handle, sinBuf.Handle, seqLen, numHeads, numKvHeads, headDim, headDim / 2);

            // capture K/V (RoPE'd) into cache
            runtime.CopyBuffer(k.Handle, kCache[li], 0, 0, seqLen * kvDim);
            runtime.CopyBuffer(v.Handle, vCache[li], 0, 0, seqLen * kvDim);

            SilkLauncher.CausalGqaAttention(runtime, q.Handle, k.Handle, v.Handle, context.Handle, seqLen, numHeads, numKvHeads, headDim, scale);

            GemmWithLora(context.Handle, w.Wo, w.OLora, projected.Handle, seqLen, qDim, hiddenDim);
            SilkLauncher.Add(runtime, x.Handle, projected.Handle, x.Handle, seqLen * hiddenDim);

            SilkLauncher.RmsNorm(runtime, x.Handle, w.PostNormW, normed.Handle, seqLen, hiddenDim, eps);

            GemmWithLora(normed.Handle, w.Wgate, w.GateLora, gate.Handle, seqLen, hiddenDim, intermediateDim);
            GemmWithLora(normed.Handle, w.Wup, w.UpLora, up.Handle, seqLen, hiddenDim, intermediateDim);

            SilkLauncher.SwiGlu(runtime, gate.Handle, up.Handle, mlp.Handle, seqLen, intermediateDim);

            GemmWithLora(mlp.Handle, w.Wdown, w.DownLora, projected.Handle, seqLen, intermediateDim, hiddenDim);
            SilkLauncher.Add(runtime, x.Handle, projected.Handle, x.Handle, seqLen * hiddenDim);
        }

        // fused final norm + tied LM head on the last row
        nint lastRowView = runtime.CreateSubview(x.Handle, (seqLen - 1) * hiddenDim, hiddenDim);
        nint logitsBuf = runtime.Allocate(vocabSize);
        RmsNormLmHead(lastRowView, finalNormW, tokEmbTransposed, logitsBuf, 1, hiddenDim, vocabSize, eps);
        var logits = runtime.Download(logitsBuf, vocabSize);
        runtime.Release(logitsBuf);
        runtime.Release(lastRowView);

        cachedLen = seqLen;
        return logits;
    }

    /// <summary>Decodes one token at <paramref name="position"/>, appending to cache, returns logits.</summary>
    public float[] ForwardCached(int tokenId, int position)
    {
        if (position < 0) throw new ArgumentOutOfRangeException(nameof(position));
        if (position + 1 > seqCap) throw new ArgumentOutOfRangeException(nameof(position));

        EnsureWorkspace(1);

        if (posIdHost is null || posIdHost.Length < 1)
            posIdHost = new int[1];
        posIdHost[0] = position;

        if (tokenIdHost is null || tokenIdHost.Length < 1)
            tokenIdHost = new int[1];
        tokenIdHost[0] = tokenId;

        runtime.Upload(idsBuf.Handle, tokenIdHost.AsSpan(0, 1));
        runtime.Upload(posIdsBuf.Handle, posIdHost.AsSpan(0, 1));

        // gather embedding -> x (single row)
        GatherTokEmb(tokEmb, idsBuf.Handle, x.Handle, hiddenDim, hiddenDim);

        for (int li = 0; li < layers.Length; li++)
        {
            LayerWeights w = layers[li];

            RmsNormGemv(x.Handle, w.InputNormW,
                [
                    new(w.Wq, w.QBias, q.Handle, qDim),
                    new(w.Wk, w.KBias, k.Handle, kvDim),
                    new(w.Wv, w.VBias, v.Handle, kvDim),
                ],
                [w.QLora, w.KLora, w.VLora],
                hasBias: true, aRows: 1, aCols: hiddenDim, eps: eps);

            // RoPE at position
            EnsureRopeTables(position + 1);
            SilkLauncher.RotaryQk(runtime, q.Handle, k.Handle, posIdsBuf.Handle, cosBuf.Handle, sinBuf.Handle, 1, numHeads, numKvHeads, headDim, headDim / 2);

            // append k/v to cache at row position
            runtime.CopyBuffer(k.Handle, kCache[li], 0, position * kvDim, kvDim);
            runtime.CopyBuffer(v.Handle, vCache[li], 0, position * kvDim, kvDim);

            int kvLen = position + 1;
            DecodeOproj(q.Handle, kCache[li], vCache[li], context.Handle,
                w.Wo, w.OLora, x.Handle, x.Handle, kvLen, numHeads, numKvHeads, headDim, qDim, hiddenDim, scale);

            MlpBlock(x.Handle, w.PostNormW, w.Wgate, w.GateLora, w.Wup, w.UpLora, w.Wdown, w.DownLora,
                x.Handle, x.Handle, hiddenDim, intermediateDim, eps);
        }

        nint logitsBuf = runtime.Allocate(vocabSize);
        RmsNormGemv(x.Handle, finalNormW,
            [new(tokEmbTransposed, 0, logitsBuf, vocabSize)],
            hasBias: false, aRows: 1, aCols: hiddenDim, eps: eps);
        var logits = runtime.Download(logitsBuf, vocabSize);
        runtime.Release(logitsBuf);

        if (position + 1 > cachedLen) cachedLen = position + 1;
        return logits;
    }

    void EnsureWorkspace(int seqLen)
    {
        int rows = seqLen;
        x = runtime.Ensure(x, rows * hiddenDim);
        normed = runtime.Ensure(normed, rows * hiddenDim);
        q = runtime.Ensure(q, rows * qDim);
        k = runtime.Ensure(k, rows * kvDim);
        v = runtime.Ensure(v, rows * kvDim);
        context = runtime.Ensure(context, rows * qDim);
        projected = runtime.Ensure(projected, rows * hiddenDim);
        gate = runtime.Ensure(gate, rows * intermediateDim);
        up = runtime.Ensure(up, rows * intermediateDim);
        mlp = runtime.Ensure(mlp, rows * intermediateDim);
        idsBuf = runtime.Ensure(idsBuf, rows);
        posIdsBuf = runtime.Ensure(posIdsBuf, rows);
    }

    void EnsureRopeTables(int seqLen)
    {
        if (ropeTableRows >= seqLen) return;
        int rows = seqLen;
        int half = headDim / 2;
        var cos = new float[rows * half];
        var sin = new float[rows * half];
        rotary.GetPositionTables(0, rows, out var cosMem, out var sinMem);
        cosMem.CopyTo(cos);
        sinMem.CopyTo(sin);
        cosBuf = runtime.Ensure(cosBuf, rows * half);
        sinBuf = runtime.Ensure(sinBuf, rows * half);
        runtime.Upload(cosBuf.Handle, cos);
        runtime.Upload(sinBuf.Handle, sin);
        ropeTableRows = rows;
    }

    nint UploadPlain(Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
    {
        var t = WeightLayout.Req(tensors, key, "Silk");
        nint buf = runtime.Allocate(t.Data.Length);
        runtime.Upload(buf, t.Data);
        return buf;
    }

    void Gemm(nint a, nint b, nint c, int aRows, int aCols, int bCols)
    {
        if (bf16Weights) SilkLauncher.GemmBf16(runtime, a, b, c, aRows, aCols, bCols);
        else SilkLauncher.Gemm(runtime, a, b, c, aRows, aCols, bCols);
    }

    void GemmWithLora(nint input, nint baseWeight, LoraPair? lora, nint output, int rows, int inFeatures, int outFeatures)
    {
        Gemm(input, baseWeight, output, rows, inFeatures, outFeatures);
        if (lora is null) return;
        if (loraTmpBuf == 0 || loraOutBuf == 0)
            throw new InvalidOperationException("LoRA scratch buffers are not allocated.");

        SilkLauncher.Gemm(runtime, input, lora.A, loraTmpBuf, rows, inFeatures, loraRank);
        SilkLauncher.Gemm(runtime, loraTmpBuf, lora.B, loraOutBuf, rows, loraRank, outFeatures);
        SilkLauncher.Scale(runtime, loraOutBuf, loraOutBuf, loraScale, rows * outFeatures);
        SilkLauncher.Add(runtime, output, loraOutBuf, output, rows * outFeatures);
    }

    void GatherTokEmb(nint table, nint ids, nint y, int n, int hidden)
    {
        if (bf16Weights) SilkLauncher.GatherBf16(runtime, table, ids, y, n, hidden);
        else SilkLauncher.Gather(runtime, table, ids, y, n, hidden);
    }

    void RmsNormLmHead(nint x, nint gamma, nint weight, nint logits, int aRows, int aCols, int bCols, float eps)
    {
        if (bf16Weights) SilkLauncher.RmsNormLmHeadBf16(runtime, x, gamma, weight, logits, aRows, aCols, bCols, eps);
        else SilkLauncher.RmsNormLmHead(runtime, x, gamma, weight, logits, aRows, aCols, bCols, eps);
    }
    void RmsNormGemv(nint x, nint gamma, ReadOnlySpan<SilkLauncher.GemvSlot> slots, bool hasBias, int aRows, int aCols, float eps)
    {
        if (bf16Weights) SilkLauncher.RmsNormGemvSubgroupBf16(runtime, x, gamma, slots, hasBias, aRows, aCols, eps);
        else SilkLauncher.RmsNormGemvSubgroup(runtime, x, gamma, slots, hasBias, aRows, aCols, eps);
    }

    void RmsNormGemv(nint x, nint gamma, ReadOnlySpan<SilkLauncher.GemvSlot> slots, ReadOnlySpan<LoraPair?> loras, bool hasBias, int aRows, int aCols, float eps)
    {
        bool anyLora = false;
        for (int i = 0; i < loras.Length; i++)
            if (loras[i] is not null) { anyLora = true; break; }

        if (!anyLora)
        {
            RmsNormGemv(x, gamma, slots, hasBias, aRows, aCols, eps);
            return;
        }

        SilkLauncher.RmsNorm(runtime, x, gamma, normed.Handle, aRows, aCols, eps);
        for (int i = 0; i < slots.Length; i++)
        {
            var s = slots[i];
            var lp = i < loras.Length ? loras[i] : null;
            GemmWithLora(normed.Handle, s.Weight, lp, s.Output, aRows, aCols, s.Cols);
            if (hasBias && s.Bias != 0)
                SilkLauncher.AddBias(runtime, s.Output, s.Bias, s.Output, aRows, s.Cols);
        }
    }

    void MlpBlock(nint x, nint gamma, nint wg, nint wu, nint wd, nint outBuf, nint residual, int hidden, int inter, float eps2)
    {
        MlpBlock(x, gamma, wg, null, wu, null, wd, null, outBuf, residual, hidden, inter, eps2);
    }

    void MlpBlock(nint x, nint gamma, nint wg, LoraPair? gateLora, nint wu, LoraPair? upLora, nint wd, LoraPair? downLora, nint outBuf, nint residual, int hidden, int inter, float eps2)
    {
        if (gateLora is null && upLora is null && downLora is null)
        {
            // Decode MLP decomposition: RmsNormGemv (gate+up, multi-WG) + SwiGlu (multi-WG) +
            // GemmGemvAdd (down+residual, multi-WG). Replaces the single-WG fused mlp_block
            // kernel which serialized decode on iGPUs (P1, #1453).
            if (bf16Weights)
            {
                SilkLauncher.RmsNormGemvSubgroupBf16(runtime, x, gamma,
                    [new(wg, 0, gate.Handle, inter), new(wu, 0, up.Handle, inter)],
                    hasBias: false, aRows: 1, aCols: hidden, eps: eps2);
                SilkLauncher.SwiGlu(runtime, gate.Handle, up.Handle, mlp.Handle, 1, inter);
                SilkLauncher.GemmGemvAddSubgroupBf16(runtime, mlp.Handle, wd, outBuf, residual, inter, hidden);
            }
            else
            {
                SilkLauncher.RmsNormGemvSubgroup(runtime, x, gamma,
                    [new(wg, 0, gate.Handle, inter), new(wu, 0, up.Handle, inter)],
                    hasBias: false, aRows: 1, aCols: hidden, eps: eps2);
                SilkLauncher.SwiGlu(runtime, gate.Handle, up.Handle, mlp.Handle, 1, inter);
                SilkLauncher.GemmGemvAddSubgroup(runtime, mlp.Handle, wd, outBuf, residual, inter, hidden);
            }
            return;
        }

        SilkLauncher.RmsNorm(runtime, x, gamma, normed.Handle, 1, hidden, eps2);
        GemmWithLora(normed.Handle, wg, gateLora, gate.Handle, 1, hidden, inter);
        GemmWithLora(normed.Handle, wu, upLora, up.Handle, 1, hidden, inter);
        SilkLauncher.SwiGlu(runtime, gate.Handle, up.Handle, mlp.Handle, 1, inter);
        GemmWithLora(mlp.Handle, wd, downLora, projected.Handle, 1, inter, hidden);
        SilkLauncher.Add(runtime, residual, projected.Handle, outBuf, hidden);
    }

    void DecodeOproj(nint q, nint kCache, nint vCache, nint contextScratch, nint wo, nint residual, nint outBuf, int kvLen, int numHeads2, int numKvHeads2, int headDim2, int qDim2, int hiddenDim2, float scale2)
    {
        DecodeOproj(q, kCache, vCache, contextScratch, wo, null, residual, outBuf, kvLen, numHeads2, numKvHeads2, headDim2, qDim2, hiddenDim2, scale2);
    }

    void DecodeOproj(nint q, nint kCache, nint vCache, nint contextScratch, nint wo, LoraPair? oLora, nint residual, nint outBuf, int kvLen, int numHeads2, int numKvHeads2, int headDim2, int qDim2, int hiddenDim2, float scale2)
    {
        if (oLora is null)
        {
            if (bf16Weights)
            {
                SilkLauncher.DecodeGqaAttention(runtime, q, kCache, vCache, contextScratch, kvLen, numHeads2, numKvHeads2, headDim2, scale2);
                SilkLauncher.GemmGemvAddSubgroupBf16(runtime, contextScratch, wo, outBuf, residual, qDim2, hiddenDim2);
            }
            else
            {
                SilkLauncher.DecodeGqaAttentionOproj(runtime, q, kCache, vCache, contextScratch, wo, residual, outBuf,
                    kvLen, numHeads2, numKvHeads2, headDim2, qDim2, hiddenDim2, scale2);
            }
            return;
        }

        SilkLauncher.DecodeGqaAttention(runtime, q, kCache, vCache, contextScratch, kvLen, numHeads2, numKvHeads2, headDim2, scale2);
        GemmWithLora(contextScratch, wo, oLora, projected.Handle, 1, qDim2, hiddenDim2);
        SilkLauncher.Add(runtime, residual, projected.Handle, outBuf, hiddenDim2);
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;

        foreach (var l in layers ?? Array.Empty<LayerWeights>()) l.Dispose(runtime);
        if (kCache != null) foreach (var b in kCache) runtime.Release(b);
        if (vCache != null) foreach (var b in vCache) runtime.Release(b);
        runtime.Release(tokEmb);
        runtime.Release(finalNormW);
        if (tokEmbTransposed != 0) runtime.Release(tokEmbTransposed);
        if (partialArgmaxIdx != 0) runtime.Release(partialArgmaxIdx);
        if (partialArgmaxVal != 0) runtime.Release(partialArgmaxVal);
        if (logitsBuf.Handle != 0) runtime.Release(logitsBuf.Handle);
        if (loraTmpBuf != 0) runtime.Release(loraTmpBuf);
        if (loraOutBuf != 0) runtime.Release(loraOutBuf);
        // workspace buffers released by runtime dispose (owned by runner? they are allocated via Ensure -> Allocate, not tracked). Release best-effort:
    }

    // Need to upload tokEmb transposed for LM head. Do in constructor after tokEmb.
    // We'll handle by adding a field initializer via partial constructor? Use post-construction in constructor try block.

    sealed class LoraPair
    {
        public required nint A;
        public required nint B;
        public required int InFeatures;
        public required int OutFeatures;
    }

    sealed class LayerWeights
    {
        public nint InputNormW;
        public nint PostNormW;
        public nint Wq, Wk, Wv, Wo;
        public nint QBias, KBias, VBias;
        public nint Wgate, Wup, Wdown;
        public LoraPair? QLora, KLora, VLora, OLora, GateLora, UpLora, DownLora;

        public LayerWeights(SilkQwenGpuRunner owner, Dictionary<string, (float[] Data, int[] Shape)> tensors, int i)
        {
            string p = $"model.layers.{i}";
            InputNormW = Upload(owner, tensors, $"{p}.input_layernorm.weight");
            PostNormW = Upload(owner, tensors, $"{p}.post_attention_layernorm.weight");
            Wq = UploadT(owner, tensors, $"{p}.self_attn.q_proj.weight");
            Wk = UploadT(owner, tensors, $"{p}.self_attn.k_proj.weight");
            Wv = UploadT(owner, tensors, $"{p}.self_attn.v_proj.weight");
            Wo = UploadT(owner, tensors, $"{p}.self_attn.o_proj.weight");
            QBias = UploadBias(owner, tensors, $"{p}.self_attn.q_proj.bias");
            KBias = UploadBias(owner, tensors, $"{p}.self_attn.k_proj.bias");
            VBias = UploadBias(owner, tensors, $"{p}.self_attn.v_proj.bias");
            Wgate = UploadT(owner, tensors, $"{p}.mlp.gate_proj.weight");
            Wup = UploadT(owner, tensors, $"{p}.mlp.up_proj.weight");
            Wdown = UploadT(owner, tensors, $"{p}.mlp.down_proj.weight");
        }

        static nint UploadBias(SilkQwenGpuRunner owner, Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
        {
            var t = WeightLayout.Req(tensors, key, "Silk");
            nint buf = owner.runtime.Allocate(t.Data.Length);
            owner.runtime.Upload(buf, t.Data);
            return buf;
        }

        static nint Upload(SilkQwenGpuRunner owner, Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
        {
            var t = WeightLayout.Req(tensors, key, "Silk");
            nint buf = owner.runtime.Allocate(t.Data.Length);
            owner.runtime.Upload(buf, t.Data);
            return buf;
        }

        static nint UploadT(SilkQwenGpuRunner owner, Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
        {
            var data = WeightLayout.Transposed(tensors, key, "Silk");
            nint buf = owner.runtime.Allocate(data.Length);
            owner.runtime.Upload(buf, data);
            return buf;
        }

        static float Widen(System.Numerics.BFloat16 b) => (float)b;
        // helper not needed beyond below

        public LayerWeights(SilkQwenGpuRunner owner, Dictionary<string, (System.Numerics.BFloat16[] Data, int[] Shape)> tensors, int i)
        {
            string p = $"model.layers.{i}";
            InputNormW = UploadFloat(owner, tensors, $"{p}.input_layernorm.weight");
            PostNormW = UploadFloat(owner, tensors, $"{p}.post_attention_layernorm.weight");
            Wq = UploadTBf16(owner, tensors, $"{p}.self_attn.q_proj.weight");
            Wk = UploadTBf16(owner, tensors, $"{p}.self_attn.k_proj.weight");
            Wv = UploadTBf16(owner, tensors, $"{p}.self_attn.v_proj.weight");
            Wo = UploadTBf16(owner, tensors, $"{p}.self_attn.o_proj.weight");
            QBias = UploadBiasFloat(owner, tensors, $"{p}.self_attn.q_proj.bias");
            KBias = UploadBiasFloat(owner, tensors, $"{p}.self_attn.k_proj.bias");
            VBias = UploadBiasFloat(owner, tensors, $"{p}.self_attn.v_proj.bias");
            Wgate = UploadTBf16(owner, tensors, $"{p}.mlp.gate_proj.weight");
            Wup = UploadTBf16(owner, tensors, $"{p}.mlp.up_proj.weight");
            Wdown = UploadTBf16(owner, tensors, $"{p}.mlp.down_proj.weight");
        }

        static nint UploadFloat(SilkQwenGpuRunner owner, Dictionary<string, (System.Numerics.BFloat16[] Data, int[] Shape)> tensors, string key)
        {
            var t = WeightLayout.Req(tensors, key, "Silk");
            var f = new float[t.Data.Length];
            for (int i = 0; i < f.Length; i++) f[i] = (float)t.Data[i];
            nint buf = owner.runtime.Allocate(f.Length);
            owner.runtime.Upload(buf, f);
            return buf;
        }
        static nint UploadBiasFloat(SilkQwenGpuRunner owner, Dictionary<string, (System.Numerics.BFloat16[] Data, int[] Shape)> tensors, string key)
            => UploadFloat(owner, tensors, key);
        static nint UploadTBf16(SilkQwenGpuRunner owner, Dictionary<string, (System.Numerics.BFloat16[] Data, int[] Shape)> tensors, string key)
        {
            var data = WeightLayout.Transposed(tensors, key, "Silk");
            // Device storage is BF16 bit-pattern (ushort); bit-preserving reinterpretation.
            var us = System.Runtime.InteropServices.MemoryMarshal.Cast<System.Numerics.BFloat16, ushort>(data.AsSpan());
            var arr = new ushort[us.Length];
            us.CopyTo(arr);
            nint buf = owner.runtime.Allocate(arr.Length);
            owner.runtime.Upload(buf, arr.AsSpan());
            return buf;
        }

        public void Dispose(SilkRuntime rt)
        {
            rt.Release(InputNormW); rt.Release(PostNormW);
            rt.Release(Wq); rt.Release(Wk); rt.Release(Wv); rt.Release(Wo);
            rt.Release(QBias); rt.Release(KBias); rt.Release(VBias);
            rt.Release(Wgate); rt.Release(Wup); rt.Release(Wdown);
            ReleaseLora(rt, QLora); ReleaseLora(rt, KLora); ReleaseLora(rt, VLora); ReleaseLora(rt, OLora);
            ReleaseLora(rt, GateLora); ReleaseLora(rt, UpLora); ReleaseLora(rt, DownLora);
        }

        static void ReleaseLora(SilkRuntime rt, LoraPair? p)
        {
            if (p is null) return;
            if (p.A != 0) rt.Release(p.A);
            if (p.B != 0) rt.Release(p.B);
        }
    }
}

public sealed record QwenSilkConfig
{
    public int HiddenSize { get; init; } = 896;
    public int NumHiddenLayers { get; init; } = 24;
    public int NumAttentionHeads { get; init; } = 14;
    public int NumKeyValueHeads { get; init; } = 2;
    public int IntermediateSize { get; init; } = 4864;
    public int VocabSize { get; init; } = 151936;
    public int MaxPositionEmbeddings { get; init; } = 32768;
    public float RmsNormEps { get; init; } = 1e-6f;
    public float RopeTheta { get; init; } = 1000000f;
}
