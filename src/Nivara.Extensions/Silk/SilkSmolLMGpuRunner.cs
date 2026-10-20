using Nivara.AutoDiff.Nn;

namespace Nivara.Silk;

/// <summary>
/// SmolLM (Llama-family causal LM) forward through the hand-authored OpenCL kernels, F32 only.
/// Mirrors <see cref="LlamaForCausalLM{T}"/> / <see cref="LlamaDecoderBlock{T}"/> operation
/// for operation. Prefill-only (no KV cache): generation runs the full forward each step, matching
/// the CPU cache-free path.
/// </summary>
public sealed partial class SilkSmolLMGpuRunner : IDisposable{
    readonly SilkRuntime runtime;
    readonly SmolLMConfig config;
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
    readonly bool qkvBias;
    readonly bool bf16Weights;

    readonly nint tokEmb;
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

    // KV cache (device, per layer). Capacity = seqCap positions.
    readonly nint[] kCache;
    readonly nint[] vCache;
    int cachedLen;

    // Device-side argmax partial buffers (#663).
    nint partialArgmaxIdx;
    nint partialArgmaxVal;
    int argmaxGroups;

    public SilkSmolLMGpuRunner(
        SilkRuntime runtime,
        SmolLMConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tensors);

        this.runtime = runtime;
        this.config = config;
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
        qkvBias = config.AttentionBias || tensors.ContainsKey("model.layers.0.self_attn.q_proj.bias");
        this.bf16Weights = false;
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
        SilkLauncher.Gather(runtime, tokEmb, idsBuf.Handle, x.Handle, seqLen * hiddenDim, hiddenDim);

        for (int li = 0; li < layers.Length; li++)
        {
            LayerWeights w = layers[li];

            // input norm
            SilkLauncher.RmsNorm(runtime, x.Handle, w.InputNormW, normed.Handle, seqLen, hiddenDim, eps);

            // Q/K/V
            Gemm(normed.Handle, w.Wq, q.Handle, seqLen, hiddenDim, qDim);
            Gemm(normed.Handle, w.Wk, k.Handle, seqLen, hiddenDim, kvDim);
            Gemm(normed.Handle, w.Wv, v.Handle, seqLen, hiddenDim, kvDim);
            if (qkvBias)
            {
                SilkLauncher.AddBias(runtime, q.Handle, w.QBias, q.Handle, seqLen, qDim);
                SilkLauncher.AddBias(runtime, k.Handle, w.KBias, k.Handle, seqLen, kvDim);
                SilkLauncher.AddBias(runtime, v.Handle, w.VBias, v.Handle, seqLen, kvDim);
            }
            // RoPE in place
            SilkLauncher.Rotary(runtime, q.Handle, posIdsBuf.Handle, cosBuf.Handle, sinBuf.Handle, seqLen, numHeads, headDim, headDim / 2);
            SilkLauncher.Rotary(runtime, k.Handle, posIdsBuf.Handle, cosBuf.Handle, sinBuf.Handle, seqLen, numKvHeads, headDim, headDim / 2);

            // causal GQA attention -> context
            SilkLauncher.CausalGqaAttention(runtime, q.Handle, k.Handle, v.Handle, context.Handle, seqLen, numHeads, numKvHeads, headDim, scale);

            // O proj + residual
            Gemm(context.Handle, w.Wo, projected.Handle, seqLen, qDim, hiddenDim);
            SilkLauncher.Add(runtime, x.Handle, projected.Handle, x.Handle, seqLen * hiddenDim);

            // post norm
            SilkLauncher.RmsNorm(runtime, x.Handle, w.PostNormW, normed.Handle, seqLen, hiddenDim, eps);

            // gate/up
            Gemm(normed.Handle, w.Wgate, gate.Handle, seqLen, hiddenDim, intermediateDim);
            Gemm(normed.Handle, w.Wup, up.Handle, seqLen, hiddenDim, intermediateDim);

            // swiglu -> mlp (intermediate)
            SilkLauncher.SwiGlu(runtime, gate.Handle, up.Handle, mlp.Handle, seqLen, intermediateDim);

            // down + residual
            Gemm(mlp.Handle, w.Wdown, projected.Handle, seqLen, intermediateDim, hiddenDim);
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

        SilkLauncher.Gather(runtime, tokEmb, idsBuf.Handle, x.Handle, seqLen * hiddenDim, hiddenDim);

        for (int li = 0; li < layers.Length; li++)
        {
            LayerWeights w = layers[li];

            SilkLauncher.RmsNorm(runtime, x.Handle, w.InputNormW, normed.Handle, seqLen, hiddenDim, eps);

            Gemm(normed.Handle, w.Wq, q.Handle, seqLen, hiddenDim, qDim);
            Gemm(normed.Handle, w.Wk, k.Handle, seqLen, hiddenDim, kvDim);
            Gemm(normed.Handle, w.Wv, v.Handle, seqLen, hiddenDim, kvDim);
            if (qkvBias)
            {
                SilkLauncher.AddBias(runtime, q.Handle, w.QBias, q.Handle, seqLen, qDim);
                SilkLauncher.AddBias(runtime, k.Handle, w.KBias, k.Handle, seqLen, kvDim);
                SilkLauncher.AddBias(runtime, v.Handle, w.VBias, v.Handle, seqLen, kvDim);
            }
            SilkLauncher.Rotary(runtime, q.Handle, posIdsBuf.Handle, cosBuf.Handle, sinBuf.Handle, seqLen, numHeads, headDim, headDim / 2);
            SilkLauncher.Rotary(runtime, k.Handle, posIdsBuf.Handle, cosBuf.Handle, sinBuf.Handle, seqLen, numKvHeads, headDim, headDim / 2);

            SilkLauncher.CausalGqaAttention(runtime, q.Handle, k.Handle, v.Handle, context.Handle, seqLen, numHeads, numKvHeads, headDim, scale);

            Gemm(context.Handle, w.Wo, projected.Handle, seqLen, qDim, hiddenDim);
            SilkLauncher.Add(runtime, x.Handle, projected.Handle, x.Handle, seqLen * hiddenDim);

            SilkLauncher.RmsNorm(runtime, x.Handle, w.PostNormW, normed.Handle, seqLen, hiddenDim, eps);

            Gemm(normed.Handle, w.Wgate, gate.Handle, seqLen, hiddenDim, intermediateDim);
            Gemm(normed.Handle, w.Wup, up.Handle, seqLen, hiddenDim, intermediateDim);

            SilkLauncher.SwiGlu(runtime, gate.Handle, up.Handle, mlp.Handle, seqLen, intermediateDim);

            Gemm(mlp.Handle, w.Wdown, projected.Handle, seqLen, intermediateDim, hiddenDim);
            SilkLauncher.Add(runtime, x.Handle, projected.Handle, x.Handle, seqLen * hiddenDim);
        }

        SilkLauncher.RmsNorm(runtime, x.Handle, finalNormW, normed.Handle, seqLen, hiddenDim, eps);

        nint lastRowView = runtime.CreateSubview(normed.Handle, (seqLen - 1) * hiddenDim, hiddenDim);
        nint logitsBuf = runtime.Allocate(vocabSize);
        Gemm(lastRowView, tokEmbTransposed, logitsBuf, 1, hiddenDim, vocabSize);
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

        SilkLauncher.Gather(runtime, tokEmb, idsBuf.Handle, x.Handle, seqLen * hiddenDim, hiddenDim);

        for (int li = 0; li < layers.Length; li++)
        {
            LayerWeights w = layers[li];

            SilkLauncher.RmsNorm(runtime, x.Handle, w.InputNormW, normed.Handle, seqLen, hiddenDim, eps);

            Gemm(normed.Handle, w.Wq, q.Handle, seqLen, hiddenDim, qDim);
            Gemm(normed.Handle, w.Wk, k.Handle, seqLen, hiddenDim, kvDim);
            Gemm(normed.Handle, w.Wv, v.Handle, seqLen, hiddenDim, kvDim);
            if (qkvBias)
            {
                SilkLauncher.AddBias(runtime, q.Handle, w.QBias, q.Handle, seqLen, qDim);
                SilkLauncher.AddBias(runtime, k.Handle, w.KBias, k.Handle, seqLen, kvDim);
                SilkLauncher.AddBias(runtime, v.Handle, w.VBias, v.Handle, seqLen, kvDim);
            }
            SilkLauncher.Rotary(runtime, q.Handle, posIdsBuf.Handle, cosBuf.Handle, sinBuf.Handle, seqLen, numHeads, headDim, headDim / 2);
            SilkLauncher.Rotary(runtime, k.Handle, posIdsBuf.Handle, cosBuf.Handle, sinBuf.Handle, seqLen, numKvHeads, headDim, headDim / 2);

            runtime.CopyBuffer(k.Handle, kCache[li], 0, 0, seqLen * kvDim);
            runtime.CopyBuffer(v.Handle, vCache[li], 0, 0, seqLen * kvDim);

            SilkLauncher.CausalGqaAttention(runtime, q.Handle, k.Handle, v.Handle, context.Handle, seqLen, numHeads, numKvHeads, headDim, scale);

            Gemm(context.Handle, w.Wo, projected.Handle, seqLen, qDim, hiddenDim);
            SilkLauncher.Add(runtime, x.Handle, projected.Handle, x.Handle, seqLen * hiddenDim);

            SilkLauncher.RmsNorm(runtime, x.Handle, w.PostNormW, normed.Handle, seqLen, hiddenDim, eps);

            Gemm(normed.Handle, w.Wgate, gate.Handle, seqLen, hiddenDim, intermediateDim);
            Gemm(normed.Handle, w.Wup, up.Handle, seqLen, hiddenDim, intermediateDim);

            SilkLauncher.SwiGlu(runtime, gate.Handle, up.Handle, mlp.Handle, seqLen, intermediateDim);

            Gemm(mlp.Handle, w.Wdown, projected.Handle, seqLen, intermediateDim, hiddenDim);
            SilkLauncher.Add(runtime, x.Handle, projected.Handle, x.Handle, seqLen * hiddenDim);
        }

        SilkLauncher.RmsNorm(runtime, x.Handle, finalNormW, normed.Handle, seqLen, hiddenDim, eps);

        nint lastRowView = runtime.CreateSubview(normed.Handle, (seqLen - 1) * hiddenDim, hiddenDim);
        nint logitsBuf = runtime.Allocate(vocabSize);
        Gemm(lastRowView, tokEmbTransposed, logitsBuf, 1, hiddenDim, vocabSize);
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

        runtime.Upload(idsBuf.Handle, new int[] { tokenId });
        runtime.Upload(posIdsBuf.Handle, posIdHost.AsSpan(0, 1));

        SilkLauncher.Gather(runtime, tokEmb, idsBuf.Handle, x.Handle, hiddenDim, hiddenDim);

        for (int li = 0; li < layers.Length; li++)
        {
            LayerWeights w = layers[li];

            RmsNormGemv(x.Handle, w.InputNormW,
                [
                    new(w.Wq, w.QBias, q.Handle, qDim),
                    new(w.Wk, w.KBias, k.Handle, kvDim),
                    new(w.Wv, w.VBias, v.Handle, kvDim),
                ],
                hasBias: qkvBias, aRows: 1, aCols: hiddenDim, eps: eps);

            EnsureRopeTables(position + 1);
            SilkLauncher.Rotary(runtime, q.Handle, posIdsBuf.Handle, cosBuf.Handle, sinBuf.Handle, 1, numHeads, headDim, headDim / 2);
            SilkLauncher.Rotary(runtime, k.Handle, posIdsBuf.Handle, cosBuf.Handle, sinBuf.Handle, 1, numKvHeads, headDim, headDim / 2);

            runtime.CopyBuffer(k.Handle, kCache[li], 0, position * kvDim, kvDim);
            runtime.CopyBuffer(v.Handle, vCache[li], 0, position * kvDim, kvDim);

            int kvLen = position + 1;
            DecodeOproj(q.Handle, kCache[li], vCache[li], context.Handle,
                w.Wo, x.Handle, x.Handle, kvLen, numHeads, numKvHeads, headDim, qDim, hiddenDim, scale);

            MlpBlock(x.Handle, w.PostNormW, w.Wgate, w.Wup, w.Wdown,
                x.Handle, x.Handle, hiddenDim, intermediateDim, eps);
        }

        nint logitsBuf = runtime.Allocate(vocabSize);
        RmsNormGemv(x.Handle, finalNormW,
            [new(tokEmbTransposed, 0, logitsBuf, vocabSize)],
            hasBias: false, aRows: 1, aCols: hiddenDim, eps: eps);
        int argmax = ArgMaxDevice(logitsBuf, vocabSize);
        runtime.Release(logitsBuf);

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
        SilkLauncher.Gather(runtime, tokEmb, idsBuf.Handle, x.Handle, seqLen * hiddenDim, hiddenDim);

        for (int li = 0; li < layers.Length; li++)
        {
            LayerWeights w = layers[li];

            SilkLauncher.RmsNorm(runtime, x.Handle, w.InputNormW, normed.Handle, seqLen, hiddenDim, eps);

            Gemm(normed.Handle, w.Wq, q.Handle, seqLen, hiddenDim, qDim);
            Gemm(normed.Handle, w.Wk, k.Handle, seqLen, hiddenDim, kvDim);
            Gemm(normed.Handle, w.Wv, v.Handle, seqLen, hiddenDim, kvDim);
            if (qkvBias)
            {
                SilkLauncher.AddBias(runtime, q.Handle, w.QBias, q.Handle, seqLen, qDim);
                SilkLauncher.AddBias(runtime, k.Handle, w.KBias, k.Handle, seqLen, kvDim);
                SilkLauncher.AddBias(runtime, v.Handle, w.VBias, v.Handle, seqLen, kvDim);
            }
            SilkLauncher.Rotary(runtime, q.Handle, posIdsBuf.Handle, cosBuf.Handle, sinBuf.Handle, seqLen, numHeads, headDim, headDim / 2);
            SilkLauncher.Rotary(runtime, k.Handle, posIdsBuf.Handle, cosBuf.Handle, sinBuf.Handle, seqLen, numKvHeads, headDim, headDim / 2);

            // capture K/V (RoPE'd) into cache
            runtime.CopyBuffer(k.Handle, kCache[li], 0, 0, seqLen * kvDim);
            runtime.CopyBuffer(v.Handle, vCache[li], 0, 0, seqLen * kvDim);

            SilkLauncher.CausalGqaAttention(runtime, q.Handle, k.Handle, v.Handle, context.Handle, seqLen, numHeads, numKvHeads, headDim, scale);

            Gemm(context.Handle, w.Wo, projected.Handle, seqLen, qDim, hiddenDim);
            SilkLauncher.Add(runtime, x.Handle, projected.Handle, x.Handle, seqLen * hiddenDim);

            SilkLauncher.RmsNorm(runtime, x.Handle, w.PostNormW, normed.Handle, seqLen, hiddenDim, eps);

            Gemm(normed.Handle, w.Wgate, gate.Handle, seqLen, hiddenDim, intermediateDim);
            Gemm(normed.Handle, w.Wup, up.Handle, seqLen, hiddenDim, intermediateDim);

            SilkLauncher.SwiGlu(runtime, gate.Handle, up.Handle, mlp.Handle, seqLen, intermediateDim);

            Gemm(mlp.Handle, w.Wdown, projected.Handle, seqLen, intermediateDim, hiddenDim);
            SilkLauncher.Add(runtime, x.Handle, projected.Handle, x.Handle, seqLen * hiddenDim);
        }

        // final norm over whole chunk
        SilkLauncher.RmsNorm(runtime, x.Handle, finalNormW, normed.Handle, seqLen, hiddenDim, eps);

        // last row logits
        nint lastRowView = runtime.CreateSubview(normed.Handle, (seqLen - 1) * hiddenDim, hiddenDim);
        nint logitsBuf = runtime.Allocate(vocabSize);
        Gemm(lastRowView, tokEmbTransposed, logitsBuf, 1, hiddenDim, vocabSize);
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

        runtime.Upload(idsBuf.Handle, new int[] { tokenId });
        runtime.Upload(posIdsBuf.Handle, posIdHost.AsSpan(0, 1));

        // gather embedding -> x (single row)
        SilkLauncher.Gather(runtime, tokEmb, idsBuf.Handle, x.Handle, hiddenDim, hiddenDim);

        for (int li = 0; li < layers.Length; li++)
        {
            LayerWeights w = layers[li];

            RmsNormGemv(x.Handle, w.InputNormW,
                [
                    new(w.Wq, w.QBias, q.Handle, qDim),
                    new(w.Wk, w.KBias, k.Handle, kvDim),
                    new(w.Wv, w.VBias, v.Handle, kvDim),
                ],
                hasBias: qkvBias, aRows: 1, aCols: hiddenDim, eps: eps);

            // RoPE at position
            EnsureRopeTables(position + 1);
            SilkLauncher.Rotary(runtime, q.Handle, posIdsBuf.Handle, cosBuf.Handle, sinBuf.Handle, 1, numHeads, headDim, headDim / 2);
            SilkLauncher.Rotary(runtime, k.Handle, posIdsBuf.Handle, cosBuf.Handle, sinBuf.Handle, 1, numKvHeads, headDim, headDim / 2);

            // append k/v to cache at row position
            runtime.CopyBuffer(k.Handle, kCache[li], 0, position * kvDim, kvDim);
            runtime.CopyBuffer(v.Handle, vCache[li], 0, position * kvDim, kvDim);

            int kvLen = position + 1;
            DecodeOproj(q.Handle, kCache[li], vCache[li], context.Handle,
                w.Wo, x.Handle, x.Handle, kvLen, numHeads, numKvHeads, headDim, qDim, hiddenDim, scale);

            MlpBlock(x.Handle, w.PostNormW, w.Wgate, w.Wup, w.Wdown,
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

    void Gemm(nint a, nint b, nint c, int aRows, int aCols, int bCols)
    {
        if (bf16Weights) SilkLauncher.GemmBf16(runtime, a, b, c, aRows, aCols, bCols);
        else SilkLauncher.Gemm(runtime, a, b, c, aRows, aCols, bCols);
    }

    void RmsNormGemv(nint x, nint gamma, ReadOnlySpan<SilkLauncher.GemvSlot> slots, bool hasBias, int aRows, int aCols, float eps)
    {
        if (bf16Weights) SilkLauncher.RmsNormGemvBf16(runtime, x, gamma, slots, hasBias, aRows, aCols, eps);
        else SilkLauncher.RmsNormGemv(runtime, x, gamma, slots, hasBias, aRows, aCols, eps);
    }

    void MlpBlock(nint x, nint gamma, nint wg, nint wu, nint wd, nint outBuf, nint residual, int hidden, int inter, float eps2)
    {
        if (bf16Weights) SilkLauncher.MlpBlockBf16(runtime, x, gamma, wg, wu, wd, outBuf, residual, hidden, inter, eps2);
        else SilkLauncher.MlpBlock(runtime, x, gamma, wg, wu, wd, outBuf, residual, hidden, inter, eps2);
    }

    void DecodeOproj(nint q, nint kCache, nint vCache, nint contextScratch, nint wo, nint residual, nint outBuf, int kvLen, int numHeads2, int numKvHeads2, int headDim2, int qDim2, int hiddenDim2, float scale2)
    {
        if (bf16Weights)
        {
            SilkLauncher.DecodeGqaAttention(runtime, q, kCache, vCache, contextScratch, kvLen, numHeads2, numKvHeads2, headDim2, scale2);
            SilkLauncher.GemmGemvAddBf16(runtime, contextScratch, wo, outBuf, residual, qDim2, hiddenDim2);
        }
        else
        {
            SilkLauncher.DecodeGqaAttentionOproj(runtime, q, kCache, vCache, contextScratch, wo, residual, outBuf, kvLen, numHeads2, numKvHeads2, headDim2, qDim2, hiddenDim2, scale2);
        }
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

    public void Dispose()
    {
        foreach (var l in layers ?? Array.Empty<LayerWeights>()) l.Dispose(runtime);
        if (kCache != null) foreach (var b in kCache) runtime.Release(b);
        if (vCache != null) foreach (var b in vCache) runtime.Release(b);
        runtime.Release(tokEmb);
        runtime.Release(finalNormW);
        if (tokEmbTransposed != 0) runtime.Release(tokEmbTransposed);
        if (partialArgmaxIdx != 0) runtime.Release(partialArgmaxIdx);
        if (partialArgmaxVal != 0) runtime.Release(partialArgmaxVal);
        // workspace buffers released by runtime dispose (owned by runner? they are allocated via Ensure -> Allocate, not tracked). Release best-effort:
    }

    // Need to upload tokEmb transposed for LM head. Do in constructor after tokEmb.
    // We'll handle by adding a field initializer via partial constructor? Use post-construction in constructor try block.

    sealed class LayerWeights
    {
        public nint InputNormW;
        public nint PostNormW;
        public nint Wq, Wk, Wv, Wo;
        public nint QBias, KBias, VBias;
        public nint Wgate, Wup, Wdown;

        public LayerWeights(SilkSmolLMGpuRunner owner, Dictionary<string, (float[] Data, int[] Shape)> tensors, int i)
        {
            string p = $"model.layers.{i}";
            InputNormW = Upload(owner, tensors, $"{p}.input_layernorm.weight");
            PostNormW = Upload(owner, tensors, $"{p}.post_attention_layernorm.weight");
            Wq = UploadT(owner, tensors, $"{p}.self_attn.q_proj.weight");
            Wk = UploadT(owner, tensors, $"{p}.self_attn.k_proj.weight");
            Wv = UploadT(owner, tensors, $"{p}.self_attn.v_proj.weight");
            Wo = UploadT(owner, tensors, $"{p}.self_attn.o_proj.weight");
            if (owner.qkvBias)
            {
                QBias = UploadBias(owner, tensors, $"{p}.self_attn.q_proj.bias");
                KBias = UploadBias(owner, tensors, $"{p}.self_attn.k_proj.bias");
                VBias = UploadBias(owner, tensors, $"{p}.self_attn.v_proj.bias");
            }
            Wgate = UploadT(owner, tensors, $"{p}.mlp.gate_proj.weight");
            Wup = UploadT(owner, tensors, $"{p}.mlp.up_proj.weight");
            Wdown = UploadT(owner, tensors, $"{p}.mlp.down_proj.weight");
        }

        public LayerWeights(SilkSmolLMGpuRunner owner, Dictionary<string, (System.Numerics.BFloat16[] Data, int[] Shape)> tensors, int i)
        {
            string p = $"model.layers.{i}";
            InputNormW = UploadFloat(owner, tensors, $"{p}.input_layernorm.weight");
            PostNormW = UploadFloat(owner, tensors, $"{p}.post_attention_layernorm.weight");
            Wq = UploadTBf16(owner, tensors, $"{p}.self_attn.q_proj.weight");
            Wk = UploadTBf16(owner, tensors, $"{p}.self_attn.k_proj.weight");
            Wv = UploadTBf16(owner, tensors, $"{p}.self_attn.v_proj.weight");
            Wo = UploadTBf16(owner, tensors, $"{p}.self_attn.o_proj.weight");
            if (owner.qkvBias)
            {
                QBias = UploadFloat(owner, tensors, $"{p}.self_attn.q_proj.bias");
                KBias = UploadFloat(owner, tensors, $"{p}.self_attn.k_proj.bias");
                VBias = UploadFloat(owner, tensors, $"{p}.self_attn.v_proj.bias");
            }
            Wgate = UploadTBf16(owner, tensors, $"{p}.mlp.gate_proj.weight");
            Wup = UploadTBf16(owner, tensors, $"{p}.mlp.up_proj.weight");
            Wdown = UploadTBf16(owner, tensors, $"{p}.mlp.down_proj.weight");
        }

        static nint UploadFloat(SilkSmolLMGpuRunner owner, Dictionary<string, (System.Numerics.BFloat16[] Data, int[] Shape)> tensors, string key)
        {
            var t = WeightLayout.Req(tensors, key, "Silk");
            var f = new float[t.Data.Length];
            var u = System.Runtime.InteropServices.MemoryMarshal.Cast<System.Numerics.BFloat16, ushort>(t.Data.AsSpan());
            for (int i = 0; i < f.Length; i++) f[i] = BitConverter.UInt32BitsToSingle((uint)u[i] << 16);
            nint buf = owner.runtime.Allocate(f.Length);
            owner.runtime.Upload(buf, f);
            return buf;
        }

        static nint UploadTBf16(SilkSmolLMGpuRunner owner, Dictionary<string, (System.Numerics.BFloat16[] Data, int[] Shape)> tensors, string key)
        {
            var data = WeightLayout.Transposed(tensors, key, "Silk");
            var us = System.Runtime.InteropServices.MemoryMarshal.Cast<System.Numerics.BFloat16, ushort>(data.AsSpan());
            var arr = new ushort[us.Length];
            us.CopyTo(arr);
            nint buf = owner.runtime.Allocate(arr.Length);
            owner.runtime.Upload(buf, arr.AsSpan());
            return buf;
        }

        static nint Upload(SilkSmolLMGpuRunner owner, Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
        {
            var t = WeightLayout.Req(tensors, key, "Silk");
            nint buf = owner.runtime.Allocate(t.Data.Length);
            owner.runtime.Upload(buf, t.Data);
            return buf;
        }

        static nint UploadT(SilkSmolLMGpuRunner owner, Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
        {
            var data = WeightLayout.Transposed(tensors, key, "Silk");
            nint buf = owner.runtime.Allocate(data.Length);
            owner.runtime.Upload(buf, data);
            return buf;
        }

        static nint UploadBias(SilkSmolLMGpuRunner owner, Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
        {
            var t = WeightLayout.Req(tensors, key, "Silk");
            nint buf = owner.runtime.Allocate(t.Data.Length);
            owner.runtime.Upload(buf, t.Data);
            return buf;
        }

        public void Dispose(SilkRuntime rt)
        {
            rt.Release(InputNormW); rt.Release(PostNormW);
            rt.Release(Wq); rt.Release(Wk); rt.Release(Wv); rt.Release(Wo);
            if (QBias != 0) rt.Release(QBias);
            if (KBias != 0) rt.Release(KBias);
            if (VBias != 0) rt.Release(VBias);
            rt.Release(Wgate); rt.Release(Wup); rt.Release(Wdown);
        }
    }
}

public sealed record SmolLMConfig
{
    public int HiddenSize { get; init; } = 576;
    public int NumHiddenLayers { get; init; } = 30;
    public int NumAttentionHeads { get; init; } = 9;
    public int NumKeyValueHeads { get; init; } = 3;
    public int IntermediateSize { get; init; } = 1536;
    public int VocabSize { get; init; } = 49152;
    public int MaxPositionEmbeddings { get; init; } = 2048;
    public float RmsNormEps { get; init; } = 1e-5f;
    public float RopeTheta { get; init; } = 10000f;
    public bool AttentionBias { get; init; } = false;
}
