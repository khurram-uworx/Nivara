namespace Nivara.Silk;

/// <summary>
/// TextCNN classifier (embedding -> conv branches -> FC1 -> ReLU -> FC2) on Silk/OpenCL.
/// The whole forward executes on the device: the three conv branches run through the existing
/// NCHW <c>Conv2d</c> kernel with H=1 into disjoint sub-buffers of one flat workspace. The
/// branch outputs stay branch-major, so FC1 accumulates as three GEMMs (one per branch's row
/// block of the FC1 weight, via <c>GemmBias</c> + <c>GemmAdd</c>) and no concat copy is needed.
/// Training computes gradients on the host (#900 tracks device-side backward kernels), so the
/// optimized path is the same path the weights run in.
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

    // Weights live on the host in the GPU-facing layout (export-ready) and are mirrored to the
    // device each update; the embedding stays host-only because the per-batch conv input upload
    // is small while a vocab-sized embedding upload each batch would not be.
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

    SilkRuntime.SilkBuffer inputBuf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer convOutBuf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer hBuf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer logitsBuf = SilkRuntime.SilkBuffer.Empty;

    // Persistent host activations and gradient scratch, grown to the largest batch seen.
    float[] hostInput = [];
    float[] hostConvOut = [];
    float[] hostHidden = [];
    float[] gradZ = [];
    float[] gradDh = [];
    float[] gradA = [];
    float[] gradInput = [];
    internal float[] gradEmb = [];
    internal float[] gradFc1 = [];
    internal float[] gradFc2 = [];
    internal float[] gradConv = [];
    int[] batchTokens = [];
    int[] batchLabels = [];

    List<nint> activeSubviews = [];

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

        fc1BranchBufs = new nint[BranchCount];
        for (int br = 0; br < BranchCount; br++)
            fc1BranchBufs[br] = runtime.CreateSubview(fc1Buf, br * branchFIn * hiddenDim, branchFIn * hiddenDim);

        UploadWeights();
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

    /// <summary>Sets all weights; FC and conv arrays use the GPU layout (same as Export*).</summary>
    public void SetWeights(
        float[] embeddingWeights,
        float[] conv1Weights, float[] conv1Biases,
        float[] conv2Weights, float[] conv2Biases,
        float[] conv3Weights, float[] conv3Biases,
        float[] fc1Weights, float[] fc1Biases,
        float[] fc2Weights, float[] fc2Biases)
    {
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

    public float[] ExportEmbedding() { var a = new float[embedding.Length]; Array.Copy(embedding, a, a.Length); return a; }
    public float[] ExportConv1Weight() { var a = new float[conv1W.Length]; Array.Copy(conv1W, a, a.Length); return a; }
    public float[] ExportConv1Bias() { var a = new float[conv1B.Length]; Array.Copy(conv1B, a, a.Length); return a; }
    public float[] ExportConv2Weight() { var a = new float[conv2W.Length]; Array.Copy(conv2W, a, a.Length); return a; }
    public float[] ExportConv2Bias() { var a = new float[conv2B.Length]; Array.Copy(conv2B, a, a.Length); return a; }
    public float[] ExportConv3Weight() { var a = new float[conv3W.Length]; Array.Copy(conv3W, a, a.Length); return a; }
    public float[] ExportConv3Bias() { var a = new float[conv3B.Length]; Array.Copy(conv3B, a, a.Length); return a; }
    public float[] ExportFc1Weight() { var a = new float[fc1W.Length]; Array.Copy(fc1W, a, a.Length); return a; }
    public float[] ExportFc1Bias() { var a = new float[fc1B.Length]; Array.Copy(fc1B, a, a.Length); return a; }
    public float[] ExportFc2Weight() { var a = new float[fc2W.Length]; Array.Copy(fc2W, a, a.Length); return a; }
    public float[] ExportFc2Bias() { var a = new float[fc2B.Length]; Array.Copy(fc2B, a, a.Length); return a; }

    /// <summary>All weights transposed into the CPU <c>Module</c> layout.</summary>
    public (float[] Embedding, float[] Conv1W, float[] Conv1B, float[] Conv2W, float[] Conv2B,
        float[] Conv3W, float[] Conv3B, float[] Fc1W, float[] Fc1B, float[] Fc2W, float[] Fc2B)
        ExportCpuLayout()
    {
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

    public float[] Forward(int[] tokenIds, int batchSize) =>
        ForwardBatch(tokenIds, batchSize, collectBackwardState: false).Logits;

    /// <summary>
    /// Device forward that also returns the activations host backward needs: the post-ReLU conv
    /// outputs (with the conv input replicated host-side) and the post-ReLU FC1 activations.
    /// </summary>
    internal (float[] Logits, float[] ConvOut, float[] Hidden) ForwardBatch(int[] tokenIds, int batchSize, bool collectBackwardState)
    {
        if (tokenIds is null || tokenIds.Length != batchSize * seqLen)
            throw new ArgumentException("Batched token ids must be exactly batchSize × seqLen.");

        int inputLen = batchSize * embeddingDim * seqLen;
        int outLen = batchSize * convHidden;
        int hidLen = batchSize * hiddenDim;
        int branchBatch = batchSize * branchFIn;

        var input = new float[inputLen];
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
        runtime.Upload(inputBuf.Handle, input.AsSpan());

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

            var logits = runtime.Download(logitsBuf.Handle, batchSize * 2);
            if (!collectBackwardState) return (logits, [], []);

            hostInput = input;
            hostConvOut = runtime.Download(convOutBuf.Handle, outLen);
            hostHidden = runtime.Download(hBuf.Handle, hidLen);
            return (logits, hostConvOut, hostHidden);
        }
        finally
        {
            foreach (var view in activeSubviews) runtime.Release(view);
            activeSubviews.Clear();
        }
    }

    public int[] Predict(int[] tokenIds, int seqLen)
    {
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

    /// <summary>Per-epoch training outcome: mean cross-entropy loss and wall-clock duration.</summary>
    public readonly record struct SilkTrainEpoch(int Epoch, float Loss, TimeSpan Elapsed);

    /// <summary>
    /// Trains in place against the device forward with host-computed gradients (two branch
    /// convs are only worth device backward once #900 lands; here the branch convs, FC and
    /// embedding scattering all run on the host so the on-device weights stay authoritative).
    /// </summary>
    public IReadOnlyList<SilkTrainEpoch> Train(int[] tokens, int[] labels, int batchSize, int seqLen, int epochs, float learningRate, int seed)
    {
        if (tokens is null || labels is null) throw new ArgumentNullException();
        if (tokens.Length % seqLen != 0) throw new ArgumentException("Token shape mismatch.");
        if (seqLen != this.seqLen) throw new ArgumentException("seqLen does not match the configured sequence length.");
        int sampleCount = tokens.Length / seqLen;
        if (labels.Length != sampleCount) throw new ArgumentException("Label count mismatch.");
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
        if (epochs <= 0) throw new ArgumentOutOfRangeException(nameof(epochs));

        var results = new List<SilkTrainEpoch>(epochs);
        int batchesPerEpoch = (sampleCount + batchSize - 1) / batchSize;
        var rng = new Random(seed);
        var adam = new[]
        {
            new AdamState(embedding),
            new AdamState(conv1W),
            new AdamState(conv1B),
            new AdamState(conv2W),
            new AdamState(conv2B),
            new AdamState(conv3W),
            new AdamState(conv3B),
            new AdamState(fc1W),
            new AdamState(fc1B),
            new AdamState(fc2W),
            new AdamState(fc2B),
        };
        batchTokens = [];
        batchLabels = [];

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            int[] indices = Enumerable.Range(0, sampleCount).ToArray();
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            float lossSum = 0f;
            int lossBatches = 0;
            for (int b = 0; b < batchesPerEpoch; b++)
            {
                int start = b * batchSize;
                int n = Math.Min(start + batchSize, sampleCount) - start;
                if (n <= 0) continue;

                batchTokens = Obtain(ref batchTokens, n * seqLen);
                batchLabels = Obtain(ref batchLabels, n);
                for (int i = 0; i < n; i++)
                {
                    int src = indices[start + i];
                    Array.Copy(tokens, src * seqLen, batchTokens, i * seqLen, seqLen);
                    batchLabels[i] = labels[src];
                }

                var (logits, _, _) = ForwardBatch(batchTokens, n, collectBackwardState: true);
                BackwardAndSetGradients(logits, batchTokens, batchLabels, n);

                ApplyAdam(adam, learningRate);
                UploadWeights();

                lossSum += CrossEntropy(logits, batchLabels, n);
                lossBatches++;
            }

            sw.Stop();
            results.Add(new SilkTrainEpoch(epoch, lossBatches > 0 ? lossSum / lossBatches : 0f, sw.Elapsed));
        }

        return results;
    }

    static float CrossEntropy(float[] logits, int[] labels, int n)
    {
        float loss = 0f;
        for (int i = 0; i < n; i++)
        {
            float e0 = MathF.Exp(logits[i * 2]);
            float e1 = MathF.Exp(logits[i * 2 + 1]);
            float p0 = e0 / (e0 + e1);
            loss += labels[i] == 0 ? -MathF.Log(p0) : -MathF.Log(1f - p0);
        }
        return loss / n;
    }

    static T[] Obtain<T>(ref T[] buffer, int length)
    {
        if (buffer.Length >= length) return buffer;
        buffer = new T[length];
        return buffer;
    }

    sealed class AdamState(float[] values)
    {
        readonly float[] first = new float[values.Length];
        readonly float[] second = new float[values.Length];
        int step;

        public void Update(ReadOnlySpan<float> gradient, float learningRate)
        {
            step++;
            float bc1 = 1f - MathF.Pow(0.9f, step);
            float bc2 = 1f - MathF.Pow(0.999f, step);
            for (int i = 0; i < values.Length; i++)
            {
                first[i] = 0.9f * first[i] + 0.1f * gradient[i];
                second[i] = 0.999f * second[i] + 0.001f * gradient[i] * gradient[i];
                values[i] -= learningRate * (first[i] / bc1) / (MathF.Sqrt(second[i] / bc2) + 1e-8f);
            }
        }
    }

    void ApplyAdam(AdamState[] adam, float learningRate)
    {
        int c1 = conv1W.Length, c2 = conv2W.Length, c3 = conv3W.Length;
        int branchBlock = BranchChannels;
        adam[0].Update(gradEmb, learningRate);
        adam[1].Update(gradConv.AsSpan(0, c1), learningRate);
        adam[2].Update(gradConv.AsSpan(c1, BranchChannels), learningRate);
        adam[3].Update(gradConv.AsSpan(c1 + branchBlock, c2), learningRate);
        adam[4].Update(gradConv.AsSpan(c1 + branchBlock + c2, BranchChannels), learningRate);
        adam[5].Update(gradConv.AsSpan(c1 + branchBlock + c2 + branchBlock, c3), learningRate);
        adam[6].Update(gradConv.AsSpan(c1 + branchBlock + c2 + branchBlock + c3, BranchChannels), learningRate);
        adam[7].Update(gradFc1.AsSpan(0, fc1W.Length), learningRate);
        adam[8].Update(gradFc1.AsSpan(fc1W.Length, fc1B.Length), learningRate);
        adam[9].Update(gradFc2.AsSpan(0, fc2W.Length), learningRate);
        adam[10].Update(gradFc2.AsSpan(fc2W.Length, fc2B.Length), learningRate);
    }

    /// <summary>
    /// Computes gradients for one batch from the device forward's activations (host backward;
    /// #900 tracks moving this to the device). Gradients accumulate into the flat grad arrays
    /// the Adam step is bound to and an averaged copy of each is applied by <see cref="ApplyAdam"/>.
    /// </summary>
    internal void BackwardAndSetGradients(float[] logits, int[] tokens, int[] labels, int n)
    {
        int outLen = n * convHidden;
        int hidLen = n * hiddenDim;
        int embLen = n * embeddingDim * seqLen;

        gradEmb = Obtain(ref gradEmb, embedding.Length);
        gradFc1 = Obtain(ref gradFc1, fc1W.Length + fc1B.Length);
        gradFc2 = Obtain(ref gradFc2, fc2W.Length + fc2B.Length);
        gradConv = Obtain(ref gradConv, conv1W.Length + conv1B.Length + conv2W.Length + conv2B.Length + conv3W.Length + conv3B.Length);
        gradZ = Obtain(ref gradZ, n * 2);
        gradDh = Obtain(ref gradDh, hidLen);
        gradA = Obtain(ref gradA, outLen);
        gradInput = Obtain(ref gradInput, embLen);
        Array.Clear(gradEmb);
        Array.Clear(gradFc1);
        Array.Clear(gradFc2);
        Array.Clear(gradConv);
        Array.Clear(gradZ, 0, n * 2);
        Array.Clear(gradDh, 0, hidLen);
        Array.Clear(gradA, 0, outLen);
        Array.Clear(gradInput, 0, embLen);

        // Softmax + cross-entropy input gradient (two classes; softmax stays numerically
        // bounded because logits come from a linear layer).
        for (int i = 0; i < n; i++)
        {
            float e0 = MathF.Exp(logits[i * 2]);
            float e1 = MathF.Exp(logits[i * 2 + 1]);
            float z = e0 + e1;
            float p0 = e0 / z;
            int label = labels[i];
            gradZ[i * 2 + 0] = p0 - (label == 0 ? 1f : 0f);
            gradZ[i * 2 + 1] = 1f - p0 - (label == 1 ? 1f : 0f);
        }

        Span<float> gFc1w = gradFc1.AsSpan(0, fc1W.Length);
        Span<float> gFc1b = gradFc1.AsSpan(fc1W.Length, fc1B.Length);
        Span<float> gFc2w = gradFc2.AsSpan(0, fc2W.Length);
        Span<float> gFc2b = gradFc2.AsSpan(fc2W.Length, 2);
        Span<float> gC1w = gradConv.AsSpan(0, conv1W.Length);
        Span<float> gC1b = gradConv.AsSpan(conv1W.Length, BranchChannels);
        Span<float> gC2w = gradConv.AsSpan(conv1W.Length + BranchChannels, conv2W.Length);
        Span<float> gC2b = gradConv.AsSpan(conv1W.Length + BranchChannels + conv2W.Length, BranchChannels);
        Span<float> gC3w = gradConv.AsSpan(conv1W.Length + BranchChannels + conv2W.Length + BranchChannels, conv3W.Length);
        Span<float> gC3b = gradConv.AsSpan(
            conv1W.Length + BranchChannels + conv2W.Length + BranchChannels + conv3W.Length, BranchChannels);

        for (int i = 0; i < n; i++)
        {
            float g0 = gradZ[i * 2], g1 = gradZ[i * 2 + 1];
            gFc2b[0] += g0; gFc2b[1] += g1;
            for (int h = 0; h < hiddenDim; h++)
            {
                float hv = hostHidden[i * hiddenDim + h];
                gFc2w[h * 2 + 0] += hv * g0;
                gFc2w[h * 2 + 1] += hv * g1;
                gradDh[i * hiddenDim + h] = hv > 0f
                    ? fc2W[h * 2 + 0] * g0 + fc2W[h * 2 + 1] * g1
                    : 0f;
            }
        }

        int branchBatch = n * branchFIn;
        for (int b = 0; b < n; b++)
        {
            for (int h = 0; h < hiddenDim; h++)
            {
                float g = gradDh[b * hiddenDim + h];
                if (g == 0f) continue;
                gFc1b[h] += g;
                for (int j = 0; j < convHidden; j++)
                {
                    int jRem = j % branchFIn;
                    int aIndex = (j / branchFIn) * branchBatch + b * branchFIn + jRem;
                    gFc1w[j * hiddenDim + h] += hostConvOut[aIndex] * g;
                    gradA[aIndex] += fc1W[j * hiddenDim + h] * g;
                }
            }
        }

        // Conv weight grads and dInput; activations are branch-major, matching the device convOut.
        for (int br = 0; br < BranchCount; br++)
        {
            int k = KernelSize(br);
            int pad = k / 2;
            Span<float> gradW = br switch { 0 => gC1w, 1 => gC2w, _ => gC3w };
            Span<float> gradBias = br switch { 0 => gC1b, 1 => gC2b, _ => gC3b };
            float[] w = br switch { 0 => conv1W, 1 => conv2W, _ => conv3W };
            for (int b = 0; b < n; b++)
            {
                int bInBase = b * embeddingDim * seqLen;
                int bConvBase = br * branchBatch + b * branchFIn;
                for (int oc = 0; oc < BranchChannels; oc++)
                {
                    int wBase = oc * embeddingDim * k;
                    for (int s = 0; s < seqLen; s++)
                    {
                        int aIndex = bConvBase + oc * seqLen + s;
                        float g = gradA[aIndex];
                        if (hostConvOut[aIndex] <= 0f || g == 0f) continue;
                        gradBias[oc] += g;
                        for (int ic = 0; ic < embeddingDim; ic++)
                        {
                            for (int kx = 0; kx < k; kx++)
                            {
                                int inS = s - pad + kx;
                                if (inS < 0 || inS >= seqLen) continue;
                                int inIndex = bInBase + ic * seqLen + inS;
                                gradW[wBase + ic * k + kx] += hostInput[inIndex] * g;
                                gradInput[inIndex] += w[wBase + ic * k + kx] * g;
                            }
                        }
                    }
                }
            }
        }

        // Embedding scatter: a straight sum over positions — the forward is a plain lookup,
        // so no pooling factor applies.
        for (int b = 0; b < n; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int tidBase = tokens[b * seqLen + s] * embeddingDim;
                int bBase = b * embeddingDim * seqLen;
                for (int e = 0; e < embeddingDim; e++)
                    gradEmb[tidBase + e] += gradInput[bBase + e * seqLen + s];
            }
        }

        // Gradients are summed per batch; average before the optimizer step.
        float inv = 1f / n;
        Scale(gradEmb, inv);
        Scale(gradFc1, inv);
        Scale(gradFc2, inv);
        Scale(gradConv, inv);
    }

    static void Scale(Span<float> span, float factor)
    {
        for (int i = 0; i < span.Length; i++) span[i] *= factor;
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
        runtime.Release(fc2Buf);
        runtime.Release(fc2BiasBuf);
    }
}
