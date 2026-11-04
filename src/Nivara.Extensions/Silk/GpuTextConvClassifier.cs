namespace Nivara.Silk;

/// <summary>
/// TextCNN classifier (embedding → conv branches → FC1 → ReLU → FC2) on Silk/OpenCL. The whole
/// forward executes on the device: the three conv branches run through the existing NCHW
/// <c>Conv2d</c> kernel with H=1 into disjoint sub-buffers of one flat workspace, and the branch
/// outputs stay branch-major so FC1 accumulates as three GEMMs without a concat copy. Backward is
/// device-side too (#900): the batch-averaged dZ is computed on the host from the downloaded
/// logits and uploaded, then the transposed-operand GEMMs, the ReLU masks, the conv backward
/// kernels and the fused <c>adam_update</c> step all run against device buffers — every parameter
/// except the embedding is device-authoritative during training, and the host mirrors refresh
/// when <c>Train</c> returns. The embedding stays host-resident (gather and scatter are host
/// work; a device-resident table with a CAS-scatter backward is a measured follow-up) and takes
/// its Adam step on the host through <see cref="SilkLoss.AdamStep"/>.
/// </summary>
public sealed class GpuTextConvClassifier : IDisposable
{
    readonly SilkRuntime runtime;
    readonly int vocabSize;
    readonly int embeddingDim;
    readonly int hiddenDim;
    readonly int seqLen;

    const int BranchChannels = 64;
    const int BranchCount = 3;
    int convHidden => BranchChannels * seqLen * BranchCount;
    int branchFIn => BranchChannels * seqLen;

    // The embedding is host-only (see the class remark); every other weight lives on the device
    // during training and the host arrays are the export surface, refreshed after Train.
    readonly float[] embedding;
    readonly int[] kernelSizes = [3, 5, 7];
    readonly float[] conv1W, conv1B;
    readonly float[] conv2W, conv2B;
    readonly float[] conv3W, conv3B;
    readonly float[] fc1W, fc1B;
    readonly float[] fc2W, fc2B;

    nint conv1Buf, conv1BiasBuf, conv2Buf, conv2BiasBuf, conv3Buf, conv3BiasBuf;
    nint fc1Buf, fc1BiasBuf, fc2Buf, fc2BiasBuf;

    // FC1 weight row blocks for the three branches, read-only sub-buffers of fc1Buf.
    nint[] fc1BranchBufs = [];

    // Gradient buffers, one per device parameter tensor — the packed layout the host backward
    // used (#903) is gone; each tensor's gradient feeds its own fused Adam step.
    nint dConv1WBuf, dConv1BBuf, dConv2WBuf, dConv2BBuf, dConv3WBuf, dConv3BBuf;
    nint dFc1Buf, dFc1BBuf, dFc2Buf, dFc2BBuf;

    // Device Adam moments, one first/second pair per device parameter tensor. Allocate returns
    // uninitialized memory, so they are zero-filled at construction and whenever a Train run
    // starts a fresh optimizer.
    nint[] momentBufs = [];

    SilkRuntime.SilkBuffer inputBuf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer convOutBuf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer hBuf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer logitsBuf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer dzBuf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer dHBuf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer dABuf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer dXBuf = SilkRuntime.SilkBuffer.Empty;

    // Host scratch: the pooling/gather input, the embedding scatter and the embedding's Adam step.
    float[] hostInput = [];
    internal float[] gradEmb = [];
    float[] embAdamFirst = [];
    float[] embAdamSecond = [];
    int[] batchTokens = [];
    int[] batchLabels = [];

    List<nint> activeSubviews = [];

    int adamStep;
    bool disposed;

    public string DeviceName => runtime.DeviceName;

    int KernelSize(int branch) => kernelSizes[branch];
    int ConvWeightLen(int k) => BranchChannels * embeddingDim * k;

    public GpuTextConvClassifier(int vocabSize, int embeddingDim, int hiddenDim, int seqLen, int seed = 42)
    {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.hiddenDim = hiddenDim;
        this.seqLen = seqLen;

        runtime = new SilkRuntime();
        var rng = new Random(seed);

        embedding = new float[vocabSize * embeddingDim];
        conv1W = new float[ConvWeightLen(3)]; conv1B = new float[BranchChannels];
        conv2W = new float[ConvWeightLen(5)]; conv2B = new float[BranchChannels];
        conv3W = new float[ConvWeightLen(7)]; conv3B = new float[BranchChannels];
        fc1W = new float[embeddingDim == 0 ? 0 : convHidden * hiddenDim];
        fc1B = new float[hiddenDim];
        fc2W = new float[hiddenDim * 2];
        fc2B = new float[2];

        for (int i = 0; i < embedding.Length; i++) embedding[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        for (int i = 0; i < conv1W.Length; i++) conv1W[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        for (int i = 0; i < conv2W.Length; i++) conv2W[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        for (int i = 0; i < conv3W.Length; i++) conv3W[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        for (int i = 0; i < fc1W.Length; i++) fc1W[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        for (int i = 0; i < fc2W.Length; i++) fc2W[i] = (float)(rng.NextDouble() * 0.1 - 0.05);

        conv1Buf = runtime.Allocate(conv1W.Length);
        conv1BiasBuf = runtime.Allocate(conv1B.Length);
        conv2Buf = runtime.Allocate(conv2W.Length);
        conv2BiasBuf = runtime.Allocate(conv2B.Length);
        conv3Buf = runtime.Allocate(conv3W.Length);
        conv3BiasBuf = runtime.Allocate(conv3B.Length);
        fc1Buf = runtime.Allocate(fc1W.Length);
        fc1BiasBuf = runtime.Allocate(fc1B.Length);
        fc2Buf = runtime.Allocate(fc2W.Length);
        fc2BiasBuf = runtime.Allocate(fc2B.Length);
        dConv1WBuf = runtime.Allocate(conv1W.Length);
        dConv1BBuf = runtime.Allocate(conv1B.Length);
        dConv2WBuf = runtime.Allocate(conv2W.Length);
        dConv2BBuf = runtime.Allocate(conv2B.Length);
        dConv3WBuf = runtime.Allocate(conv3W.Length);
        dConv3BBuf = runtime.Allocate(conv3B.Length);
        dFc1Buf = runtime.Allocate(fc1W.Length);
        dFc1BBuf = runtime.Allocate(fc1B.Length);
        dFc2Buf = runtime.Allocate(fc2W.Length);
        dFc2BBuf = runtime.Allocate(fc2B.Length);

        momentBufs = new nint[20];
        for (int i = 0; i < momentBufs.Length; i++)
        {
            int len = i switch
            {
                0 or 1 => conv1W.Length,
                2 or 3 => conv1B.Length,
                4 or 5 => conv2W.Length,
                6 or 7 => conv2B.Length,
                8 or 9 => conv3W.Length,
                10 or 11 => conv3B.Length,
                12 or 13 => fc1W.Length,
                14 or 15 => fc1B.Length,
                16 or 17 => fc2W.Length,
                _ => fc2B.Length,
            };
            momentBufs[i] = runtime.Allocate(len);
        }
        ZeroMomentBuffers();

        fc1BranchBufs = new nint[BranchCount];
        for (int br = 0; br < BranchCount; br++)
            fc1BranchBufs[br] = runtime.CreateSubview(fc1Buf, br * branchFIn * hiddenDim, branchFIn * hiddenDim);

        UploadWeights();
    }

    void ThrowIfDisposed()
    {
        if (disposed) throw new ObjectDisposedException(nameof(GpuTextConvClassifier));
    }

    void ZeroMomentBuffers()
    {
        for (int i = 0; i < momentBufs.Length; i++)
        {
            int len = i switch
            {
                0 or 1 => conv1W.Length,
                2 or 3 => conv1B.Length,
                4 or 5 => conv2W.Length,
                6 or 7 => conv2B.Length,
                8 or 9 => conv3W.Length,
                10 or 11 => conv3B.Length,
                12 or 13 => fc1W.Length,
                14 or 15 => fc1B.Length,
                16 or 17 => fc2W.Length,
                _ => fc2B.Length,
            };
            SilkLauncher.Fill(runtime, momentBufs[i], 0f, len);
        }
    }

    void UploadWeights()
    {
        runtime.Upload(conv1Buf, conv1W.AsSpan());
        runtime.Upload(conv1BiasBuf, conv1B.AsSpan());
        runtime.Upload(conv2Buf, conv2W.AsSpan());
        runtime.Upload(conv2BiasBuf, conv2B.AsSpan());
        runtime.Upload(conv3Buf, conv3W.AsSpan());
        runtime.Upload(conv3BiasBuf, conv3B.AsSpan());
        runtime.Upload(fc1Buf, fc1W.AsSpan());
        runtime.Upload(fc1BiasBuf, fc1B.AsSpan());
        runtime.Upload(fc2Buf, fc2W.AsSpan());
        runtime.Upload(fc2BiasBuf, fc2B.AsSpan());
    }

    /// <summary>Downloads the trained weights back into the host mirrors. Called when
    /// <c>Train</c> returns; while training runs the device buffers are authoritative.</summary>
    void RefreshHostMirrors()
    {
        runtime.Download(conv1Buf, conv1W.Length).CopyTo(conv1W, 0);
        runtime.Download(conv1BiasBuf, conv1B.Length).CopyTo(conv1B, 0);
        runtime.Download(conv2Buf, conv2W.Length).CopyTo(conv2W, 0);
        runtime.Download(conv2BiasBuf, conv2B.Length).CopyTo(conv2B, 0);
        runtime.Download(conv3Buf, conv3W.Length).CopyTo(conv3W, 0);
        runtime.Download(conv3BiasBuf, conv3B.Length).CopyTo(conv3B, 0);
        runtime.Download(fc1Buf, fc1W.Length).CopyTo(fc1W, 0);
        runtime.Download(fc1BiasBuf, fc1B.Length).CopyTo(fc1B, 0);
        runtime.Download(fc2Buf, fc2W.Length).CopyTo(fc2W, 0);
        runtime.Download(fc2BiasBuf, fc2B.Length).CopyTo(fc2B, 0);
    }

    /// <summary>Sets all weights; FC and conv arrays use the GPU layout (same as Export*).</summary>
    public void SetWeights(
        float[] embeddingWeights,
        float[] conv1Weights, float[] conv1Biases,
        float[] conv2Weights, float[] conv2Biases,
        float[] conv3Weights, float[] conv3Biases,
        float[] fc1Weights, float[] fc1Biases,
        float[] fc2Weights, float[] fc2Biases)
    {
        ThrowIfDisposed();
        ValidateAndCopy(embeddingWeights, conv1Weights, conv1Biases, conv2Weights, conv2Biases,
            conv3Weights, conv3Biases, fc1Weights, fc1Biases, fc2Weights, fc2Biases);
        UploadWeights();
    }

    /// <summary>
    /// Sets all weights from the CPU <c>Module</c> layout (FC weights row-major
    /// <c>[outFeatures, inFeatures]</c>) so callers need not hand-roll the transpose.
    /// </summary>
    public void SetWeightsCpuLayout(
        int vocabSize, int embeddingDim, int maxSeqLen,
        float[] embeddingWeights,
        float[] conv1Weights, float[] conv1Biases,
        float[] conv2Weights, float[] conv2Biases,
        float[] conv3Weights, float[] conv3Biases,
        float[] fc1WeightsCpu, float[] fc1Biases,
        float[] fc2WeightsCpu, float[] fc2Biases)
    {
        ThrowIfDisposed();
        if (vocabSize != this.vocabSize || embeddingDim != this.embeddingDim || maxSeqLen != this.seqLen)
            throw new ArgumentException("Shape mismatch with the classifier's configured dimensions.");
        if (fc1WeightsCpu is null || fc1WeightsCpu.Length != hiddenDim * convHidden)
            throw new ArgumentException("FC1 weight shape mismatch.", nameof(fc1WeightsCpu));
        if (fc2WeightsCpu is null || fc2WeightsCpu.Length != 2 * hiddenDim)
            throw new ArgumentException("FC2 weight shape mismatch.", nameof(fc2WeightsCpu));

        var fc1Gpu = new float[convHidden * hiddenDim];
        for (int h = 0; h < hiddenDim; h++)
            for (int j = 0; j < convHidden; j++)
                fc1Gpu[j * hiddenDim + h] = fc1WeightsCpu[h * convHidden + j];
        var fc2Gpu = new float[hiddenDim * 2];
        for (int h = 0; h < hiddenDim; h++)
            for (int c = 0; c < 2; c++)
                fc2Gpu[h * 2 + c] = fc2WeightsCpu[c * hiddenDim + h];

        ValidateAndCopy(embeddingWeights, conv1Weights, conv1Biases, conv2Weights, conv2Biases,
            conv3Weights, conv3Biases, fc1Gpu, fc1Biases, fc2Gpu, fc2Biases);
        UploadWeights();
    }

    void ValidateAndCopy(
        float[] embeddingWeights,
        float[] conv1Weights, float[] conv1Biases,
        float[] conv2Weights, float[] conv2Biases,
        float[] conv3Weights, float[] conv3Biases,
        float[] fc1Weights, float[] fc1Biases,
        float[] fc2Weights, float[] fc2Biases)
    {
        if (embeddingWeights is null || embeddingWeights.Length != embedding.Length) throw new ArgumentException("embedding");
        if (conv1Weights is null || conv1Weights.Length != conv1W.Length) throw new ArgumentException("conv1 weights");
        if (conv1Biases is null || conv1Biases.Length != conv1B.Length) throw new ArgumentException("conv1 biases");
        if (conv2Weights is null || conv2Weights.Length != conv2W.Length) throw new ArgumentException("conv2 weights");
        if (conv2Biases is null || conv2Biases.Length != conv2B.Length) throw new ArgumentException("conv2 biases");
        if (conv3Weights is null || conv3Weights.Length != conv3W.Length) throw new ArgumentException("conv3 weights");
        if (conv3Biases is null || conv3Biases.Length != conv3B.Length) throw new ArgumentException("conv3 biases");
        if (fc1Weights is null || fc1Weights.Length != fc1W.Length) throw new ArgumentException("fc1 weights");
        if (fc1Biases is null || fc1Biases.Length != fc1B.Length) throw new ArgumentException("fc1 biases");
        if (fc2Weights is null || fc2Weights.Length != fc2W.Length) throw new ArgumentException("fc2 weights");
        if (fc2Biases is null || fc2Biases.Length != fc2B.Length) throw new ArgumentException("fc2 biases");

        Array.Copy(embeddingWeights, embedding, embedding.Length);
        Array.Copy(conv1Weights, conv1W, conv1W.Length);
        Array.Copy(conv1Biases, conv1B, conv1B.Length);
        Array.Copy(conv2Weights, conv2W, conv2W.Length);
        Array.Copy(conv2Biases, conv2B, conv2B.Length);
        Array.Copy(conv3Weights, conv3W, conv3W.Length);
        Array.Copy(conv3Biases, conv3B, conv3B.Length);
        Array.Copy(fc1Weights, fc1W, fc1W.Length);
        Array.Copy(fc1Biases, fc1B, fc1B.Length);
        Array.Copy(fc2Weights, fc2W, fc2W.Length);
        Array.Copy(fc2Biases, fc2B, fc2B.Length);
    }

    public float[] ExportEmbedding() { ThrowIfDisposed(); var a = new float[embedding.Length]; Array.Copy(embedding, a, a.Length); return a; }
    public float[] ExportConv1Weight() { ThrowIfDisposed(); var a = new float[conv1W.Length]; Array.Copy(conv1W, a, a.Length); return a; }
    public float[] ExportConv1Bias() { ThrowIfDisposed(); var a = new float[conv1B.Length]; Array.Copy(conv1B, a, a.Length); return a; }
    public float[] ExportConv2Weight() { ThrowIfDisposed(); var a = new float[conv2W.Length]; Array.Copy(conv2W, a, a.Length); return a; }
    public float[] ExportConv2Bias() { ThrowIfDisposed(); var a = new float[conv2B.Length]; Array.Copy(conv2B, a, a.Length); return a; }
    public float[] ExportConv3Weight() { ThrowIfDisposed(); var a = new float[conv3W.Length]; Array.Copy(conv3W, a, a.Length); return a; }
    public float[] ExportConv3Bias() { ThrowIfDisposed(); var a = new float[conv3B.Length]; Array.Copy(conv3B, a, a.Length); return a; }
    public float[] ExportFc1Weight() { ThrowIfDisposed(); var a = new float[fc1W.Length]; Array.Copy(fc1W, a, a.Length); return a; }
    public float[] ExportFc1Bias() { ThrowIfDisposed(); var a = new float[fc1B.Length]; Array.Copy(fc1B, a, a.Length); return a; }
    public float[] ExportFc2Weight() { ThrowIfDisposed(); var a = new float[fc2W.Length]; Array.Copy(fc2W, a, a.Length); return a; }
    public float[] ExportFc2Bias() { ThrowIfDisposed(); var a = new float[fc2B.Length]; Array.Copy(fc2B, a, a.Length); return a; }

    /// <summary>All weights transposed into the CPU <c>Module</c> layout.</summary>
    public (float[] Embedding, float[] Conv1W, float[] Conv1B, float[] Conv2W, float[] Conv2B,
        float[] Conv3W, float[] Conv3B, float[] Fc1W, float[] Fc1B, float[] Fc2W, float[] Fc2B)
        ExportCpuLayout()
    {
        ThrowIfDisposed();
        var fc1Cpu = new float[hiddenDim * convHidden];
        for (int h = 0; h < hiddenDim; h++)
            for (int j = 0; j < convHidden; j++)
                fc1Cpu[h * convHidden + j] = fc1W[j * hiddenDim + h];
        var fc2Cpu = new float[2 * hiddenDim];
        for (int h = 0; h < hiddenDim; h++)
            for (int c = 0; c < 2; c++)
                fc2Cpu[c * hiddenDim + h] = fc2W[h * 2 + c];
        return (ExportEmbedding(), ExportConv1Weight(), ExportConv1Bias(), ExportConv2Weight(),
            ExportConv2Bias(), ExportConv3Weight(), ExportConv3Bias(), fc1Cpu, ExportFc1Bias(),
            fc2Cpu, ExportFc2Bias());
    }

    public float[] Forward(int[] tokenIds, int batchSize)
    {
        ThrowIfDisposed();
        return ForwardBatch(tokenIds, batchSize);
    }

    internal float[] ForwardBatch(int[] tokenIds, int batchSize)
    {
        ThrowIfDisposed();
        if (tokenIds is null || tokenIds.Length != batchSize * seqLen)
            throw new ArgumentException("Batched token ids must be exactly batchSize × seqLen.");

        int inputLen = batchSize * embeddingDim * seqLen;
        int outLen = batchSize * convHidden;
        int hidLen = batchSize * hiddenDim;
        int branchBatch = batchSize * branchFIn;

        hostInput = SilkLoss.Obtain(ref hostInput, inputLen);
        var input = hostInput;
        for (int b = 0; b < batchSize; b++)
        {
            int bBase = b * embeddingDim * seqLen;
            for (int s = 0; s < seqLen; s++)
            {
                int tid = tokenIds[b * seqLen + s];
                if (tid < 0 || tid >= vocabSize) throw new ArgumentOutOfRangeException(nameof(tokenIds));
                int tidBase = tid * embeddingDim;
                for (int e = 0; e < embeddingDim; e++)
                    input[bBase + e * seqLen + s] = embedding[tidBase + e];
            }
        }

        convOutBuf = runtime.Ensure(convOutBuf, outLen);
        hBuf = runtime.Ensure(hBuf, hidLen);
        logitsBuf = runtime.Ensure(logitsBuf, batchSize * 2);
        inputBuf = runtime.Ensure(inputBuf, inputLen);
        runtime.Upload(inputBuf.Handle, input.AsSpan(0, inputLen));

        activeSubviews.Clear();
        try
        {
            for (int br = 0; br < BranchCount; br++)
            {
                int k = KernelSize(br);
                int pad = k / 2;
                nint outView = runtime.CreateSubview(convOutBuf.Handle, br * branchBatch, branchBatch, readOnly: false);
                activeSubviews.Add(outView);
                nint weightBuf = br switch { 0 => conv1Buf, 1 => conv2Buf, _ => conv3Buf };
                nint biasBuf = br switch { 0 => conv1BiasBuf, 1 => conv2BiasBuf, _ => conv3BiasBuf };
                SilkLauncher.Conv2d(runtime, inputBuf.Handle, weightBuf, biasBuf, outView,
                    N: batchSize, C: embeddingDim, H: 1, W: seqLen,
                    outC: BranchChannels, kH: 1, kW: k,
                    strideH: 1, strideW: 1, padTop: 0, padBottom: 0, padLeft: pad, padRight: k - pad - 1,
                    groups: 1, hasBias: true, oH: 1, oW: seqLen);
            }

            SilkLauncher.Relu(runtime, convOutBuf.Handle, convOutBuf.Handle, outLen);

            // h = ReLU(FC1 · concat) computed branch-wise: branch 0 seeds h with its GEMM plus
            // the FC1 bias, branches 1 and 2 accumulate, then one element-wise ReLU.
            SilkLauncher.GemmEpilogue(runtime, SilkKernels.GemmBias,
                activeSubviews[0], fc1BranchBufs[0], hBuf.Handle, fc1BiasBuf,
                batchSize, branchFIn, hiddenDim);
            for (int br = 1; br < BranchCount; br++)
                SilkLauncher.GemmAdd(runtime, activeSubviews[br], fc1BranchBufs[br], hBuf.Handle, hBuf.Handle,
                    batchSize, branchFIn, hiddenDim);
            SilkLauncher.Relu(runtime, hBuf.Handle, hBuf.Handle, hidLen);

            SilkLauncher.GemmEpilogue(runtime, SilkKernels.GemmBias,
                hBuf.Handle, fc2Buf, logitsBuf.Handle, fc2BiasBuf,
                batchSize, hiddenDim, 2);

            return runtime.Download(logitsBuf.Handle, batchSize * 2);
        }
        finally
        {
            foreach (var view in activeSubviews) runtime.Release(view);
            activeSubviews.Clear();
        }
    }

    public int[] Predict(int[] tokenIds, int seqLen)
    {
        ThrowIfDisposed();
        if (seqLen != this.seqLen)
            throw new ArgumentException("seqLen does not match the configured sequence length.", nameof(seqLen));
        if (tokenIds is null || tokenIds.Length % seqLen != 0)
            throw new ArgumentException("Token array size mismatch.", nameof(tokenIds));
        int batch = tokenIds.Length / seqLen;
        var logits = Forward(tokenIds, batch);
        int numClasses = logits.Length / batch;
        var result = new int[batch];
        for (int b = 0; b < batch; b++)
        {
            int best = 0; float bestVal = logits[b * numClasses];
            for (int c = 1; c < numClasses; c++)
                if (logits[b * numClasses + c] > bestVal) { bestVal = logits[b * numClasses + c]; best = c; }
            result[b] = best;
        }
        return result;
    }

    /// <summary>
    /// Trains in place against the device forward and the device backward (#900), with the fused
    /// <c>adam_update</c> step for every device parameter and the host Adam step for the
    /// host-resident embedding. Each call starts a fresh optimizer (moments zeroed, step counter
    /// reset).
    /// </summary>
    public IReadOnlyList<SilkTrainEpoch> Train(int[] tokens, int[] labels, int batchSize, int seqLen, int epochs, float learningRate, int seed)
    {
        ThrowIfDisposed();
        if (tokens is null || labels is null) throw new ArgumentNullException();
        if (tokens.Length % seqLen != 0) throw new ArgumentException("Token shape mismatch.");
        if (seqLen != this.seqLen) throw new ArgumentException("seqLen does not match the configured sequence length.");
        int sampleCount = tokens.Length / seqLen;
        if (labels.Length != sampleCount) throw new ArgumentException("Label count mismatch.");
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
        if (epochs <= 0) throw new ArgumentOutOfRangeException(nameof(epochs));

        adamStep = 0;
        ZeroMomentBuffers();
        embAdamFirst = new float[embedding.Length];
        embAdamSecond = new float[embedding.Length];
        batchTokens = [];
        batchLabels = [];

        var results = new List<SilkTrainEpoch>(epochs);
        int batchesPerEpoch = (sampleCount + batchSize - 1) / batchSize;
        var shuffle = new Random(seed);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            int[] indices = Enumerable.Range(0, sampleCount).ToArray();
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = shuffle.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            float lossSum = 0f;
            int lossBatches = 0;
            for (int b = 0; b < batchesPerEpoch; b++)
            {
                int start = b * batchSize;
                int n = Math.Min(start + batchSize, sampleCount) - start;
                if (n <= 0) continue;

                if (batchTokens is null || batchTokens.Length != n * seqLen) batchTokens = new int[n * seqLen];
                if (batchLabels is null || batchLabels.Length != n) batchLabels = new int[n];
                for (int i = 0; i < n; i++)
                {
                    int src = indices[start + i];
                    Array.Copy(tokens, src * seqLen, batchTokens, i * seqLen, seqLen);
                    batchLabels[i] = labels[src];
                }

                var logits = ForwardBatch(batchTokens, n);
                Backward(logits, batchTokens, batchLabels, n);
                AdamUpdateAll(learningRate);

                lossSum += SilkLoss.CrossEntropy(logits, batchLabels, n);
                lossBatches++;
            }

            sw.Stop();
            results.Add(new SilkTrainEpoch(epoch, lossBatches > 0 ? lossSum / lossBatches : 0f, sw.Elapsed));
        }

        RefreshHostMirrors();
        return results;
    }

    /// <summary>
    /// One backward pass over the device activations the last forward left resident: uploads the
    /// batch-averaged dZ, runs the FC/conv backward chain on the device, and scatters the input
    /// gradient into <see cref="gradEmb"/> on the host (the conv forward is a plain lookup, so no
    /// pooling factor applies). No optimizer step — gradients stay in the device gradient buffers
    /// and <see cref="gradEmb"/> until <see cref="AdamUpdateAll"/>.
    /// </summary>
    internal void Backward(float[] logits, int[] tokens, int[] labels, int batch)
    {
        ThrowIfDisposed();
        if (logits is null || logits.Length != batch * 2)
            throw new ArgumentException("Logit count mismatch.", nameof(logits));
        if (tokens is null || tokens.Length != batch * seqLen)
            throw new ArgumentException("Token array size mismatch.", nameof(tokens));
        if (labels is null || labels.Length != batch)
            throw new ArgumentException("Label count mismatch.", nameof(labels));

        int outLen = batch * convHidden;
        int hidLen = batch * hiddenDim;
        int inputLen = batch * embeddingDim * seqLen;
        int branchBatch = batch * branchFIn;

        dzBuf = runtime.Ensure(dzBuf, batch * 2);
        dHBuf = runtime.Ensure(dHBuf, hidLen);
        dABuf = runtime.Ensure(dABuf, outLen);
        dXBuf = runtime.Ensure(dXBuf, inputLen);

        var dz = new float[batch * 2];
        SilkLoss.InputGradient(logits, labels, batch, 1f / batch, dz);
        runtime.Upload(dzBuf.Handle, dz);

        SilkLauncher.ColSum(runtime, dzBuf.Handle, dFc2BBuf, batch, 2, 1f);
        SilkLauncher.GemmTn(runtime, hBuf.Handle, dzBuf.Handle, dFc2Buf,
            aRows: hiddenDim, aCols: batch, bCols: 2);
        SilkLauncher.GemmNt(runtime, dzBuf.Handle, fc2Buf, dHBuf.Handle,
            aRows: batch, aCols: 2, bCols: hiddenDim);
        SilkLauncher.ReluBackward(runtime, dHBuf.Handle, hBuf.Handle, dHBuf.Handle, hidLen);
        SilkLauncher.ColSum(runtime, dHBuf.Handle, dFc1BBuf, batch, hiddenDim, 1f);

        activeSubviews.Clear();
        try
        {
            for (int br = 0; br < BranchCount; br++)
            {
                int k = KernelSize(br);
                int pad = k / 2;

                // dW1 branch block = convOut(branch)ᵀ·dH, written into a writable row block of
                // the dFc1 buffer; dA branch block = dH·W1ᵀ(branch), written into a writable
                // branch block of the flat dA workspace.
                nint convOutView = runtime.CreateSubview(convOutBuf.Handle, br * branchBatch, branchBatch, readOnly: false);
                activeSubviews.Add(convOutView);
                nint dW1View = runtime.CreateSubview(dFc1Buf, br * branchFIn * hiddenDim, branchFIn * hiddenDim, readOnly: false);
                activeSubviews.Add(dW1View);
                nint dAView = runtime.CreateSubview(dABuf.Handle, br * branchBatch, branchBatch, readOnly: false);
                activeSubviews.Add(dAView);

                SilkLauncher.GemmTn(runtime, convOutView, dHBuf.Handle, dW1View,
                    aRows: branchFIn, aCols: batch, bCols: hiddenDim);
                SilkLauncher.GemmNt(runtime, dHBuf.Handle, fc1BranchBufs[br], dAView,
                    aRows: batch, aCols: hiddenDim, bCols: branchFIn);
                SilkLauncher.ReluBackward(runtime, dAView, convOutView, dAView, branchBatch);

                nint weightBuf = br switch { 0 => conv1Buf, 1 => conv2Buf, _ => conv3Buf };
                nint dWeightBuf = br switch { 0 => dConv1WBuf, 1 => dConv2WBuf, _ => dConv3WBuf };
                nint dBiasBuf = br switch { 0 => dConv1BBuf, 1 => dConv2BBuf, _ => dConv3BBuf };
                SilkLauncher.Conv2dBackwardWeight(runtime, dAView, inputBuf.Handle, dWeightBuf,
                    N: batch, C: embeddingDim, H: 1, W: seqLen,
                    outC: BranchChannels, kH: 1, kW: k,
                    strideH: 1, strideW: 1, padTop: 0, padLeft: pad, groups: 1, oH: 1, oW: seqLen);
                SilkLauncher.Conv2dBackwardBias(runtime, dAView, dBiasBuf,
                    N: batch, outC: BranchChannels, oH: 1, oW: seqLen);
            }

            // dX accumulates the three branches: zero first, then one launch per branch.
            SilkLauncher.Fill(runtime, dXBuf.Handle, 0f, inputLen);
            for (int br = 0; br < BranchCount; br++)
            {
                int k = KernelSize(br);
                int pad = k / 2;
                nint weightBuf = br switch { 0 => conv1Buf, 1 => conv2Buf, _ => conv3Buf };
                nint dAView = runtime.CreateSubview(dABuf.Handle, br * branchBatch, branchBatch, readOnly: false);
                activeSubviews.Add(dAView);
                SilkLauncher.Conv2dBackwardInput(runtime, dAView, weightBuf, dXBuf.Handle,
                    N: batch, C: embeddingDim, H: 1, W: seqLen,
                    outC: BranchChannels, kH: 1, kW: k,
                    strideH: 1, strideW: 1, padTop: 0, padLeft: pad, groups: 1, oH: 1, oW: seqLen);
            }
        }
        finally
        {
            foreach (var view in activeSubviews) runtime.Release(view);
            activeSubviews.Clear();
        }

        var dX = runtime.Download(dXBuf.Handle, inputLen);

        gradEmb = SilkLoss.Obtain(ref gradEmb, embedding.Length);
        Array.Clear(gradEmb);
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int tidBase = tokens[b * seqLen + s] * embeddingDim;
                int bBase = b * embeddingDim * seqLen;
                for (int e = 0; e < embeddingDim; e++)
                    gradEmb[tidBase + e] += dX[bBase + e * seqLen + s];
            }
        }
    }

    /// <summary>Applies one Adam step: the fused device kernel to every device parameter (conv
    /// weights/biases, FC1/FC2 weights/biases) and <see cref="SilkLoss.AdamStep"/> to the
    /// host-resident embedding. One step counter is shared; every parameter takes the same bias
    /// correction.</summary>
    internal void AdamUpdateAll(float learningRate)
    {
        adamStep++;
        float bc1 = 1f - MathF.Pow(0.9f, adamStep);
        float bc2 = 1f - MathF.Pow(0.999f, adamStep);

        (nint weight, nint grad, nint first, nint second, int count)[] tensors =
        [
            (conv1Buf, dConv1WBuf, momentBufs[0], momentBufs[1], conv1W.Length),
            (conv1BiasBuf, dConv1BBuf, momentBufs[2], momentBufs[3], conv1B.Length),
            (conv2Buf, dConv2WBuf, momentBufs[4], momentBufs[5], conv2W.Length),
            (conv2BiasBuf, dConv2BBuf, momentBufs[6], momentBufs[7], conv2B.Length),
            (conv3Buf, dConv3WBuf, momentBufs[8], momentBufs[9], conv3W.Length),
            (conv3BiasBuf, dConv3BBuf, momentBufs[10], momentBufs[11], conv3B.Length),
            (fc1Buf, dFc1Buf, momentBufs[12], momentBufs[13], fc1W.Length),
            (fc1BiasBuf, dFc1BBuf, momentBufs[14], momentBufs[15], fc1B.Length),
            (fc2Buf, dFc2Buf, momentBufs[16], momentBufs[17], fc2W.Length),
            (fc2BiasBuf, dFc2BBuf, momentBufs[18], momentBufs[19], fc2B.Length),
        ];
        foreach (var t in tensors)
            SilkLauncher.AdamUpdate(runtime, t.weight, t.grad, t.first, t.second, learningRate, bc1, bc2, 1e-8f, t.count);

        SilkLoss.AdamStep(embedding, embAdamFirst, embAdamSecond, gradEmb, learningRate, bc1, bc2);
    }

    /// <summary>Downloads every device gradient; the FC weights arrive in the GPU layout (the
    /// CPU layout is their transpose). Used by the parity fixture.</summary>
    internal (float[] Conv1W, float[] Conv1B, float[] Conv2W, float[] Conv2B,
        float[] Conv3W, float[] Conv3B, float[] Fc1W, float[] Fc1B, float[] Fc2W, float[] Fc2B)
        DownloadGradients()
    {
        ThrowIfDisposed();
        return (
            runtime.Download(dConv1WBuf, conv1W.Length),
            runtime.Download(dConv1BBuf, conv1B.Length),
            runtime.Download(dConv2WBuf, conv2W.Length),
            runtime.Download(dConv2BBuf, conv2B.Length),
            runtime.Download(dConv3WBuf, conv3W.Length),
            runtime.Download(dConv3BBuf, conv3B.Length),
            runtime.Download(dFc1Buf, fc1W.Length),
            runtime.Download(dFc1BBuf, fc1B.Length),
            runtime.Download(dFc2Buf, fc2W.Length),
            runtime.Download(dFc2BBuf, fc2B.Length));
    }

    public void Dispose()
    {
        if (disposed) return;
        foreach (var view in activeSubviews) runtime.Release(view);
        activeSubviews.Clear();
        ReleaseWeights();
        runtime.Release(inputBuf.Handle);
        runtime.Release(convOutBuf.Handle);
        runtime.Release(hBuf.Handle);
        runtime.Release(logitsBuf.Handle);
        runtime.Release(dzBuf.Handle);
        runtime.Release(dHBuf.Handle);
        runtime.Release(dABuf.Handle);
        runtime.Release(dXBuf.Handle);
        runtime.Dispose();
        disposed = true;
    }

    void ReleaseWeights()
    {
        runtime.Release(conv1Buf);
        runtime.Release(conv1BiasBuf);
        runtime.Release(conv2Buf);
        runtime.Release(conv2BiasBuf);
        runtime.Release(conv3Buf);
        runtime.Release(conv3BiasBuf);
        foreach (var block in fc1BranchBufs) runtime.Release(block);
        runtime.Release(fc1Buf);
        runtime.Release(fc1BiasBuf);
        runtime.Release(fc2Buf);
        runtime.Release(fc2BiasBuf);
        runtime.Release(dConv1WBuf);
        runtime.Release(dConv1BBuf);
        runtime.Release(dConv2WBuf);
        runtime.Release(dConv2BBuf);
        runtime.Release(dConv3WBuf);
        runtime.Release(dConv3BBuf);
        runtime.Release(dFc1Buf);
        runtime.Release(dFc1BBuf);
        runtime.Release(dFc2Buf);
        runtime.Release(dFc2BBuf);
        foreach (var moment in momentBufs) runtime.Release(moment);
    }
}
