namespace Nivara.Silk;

/// <summary>
/// Linear text classifier (embedding → mean pool → FC1 → ReLU → FC2) trained on Silk/OpenCL.
/// The forward runs on the device. Backward runs on the device too (#900): the batch-averaged
/// softmax input gradient is computed on the host from the downloaded logits and uploaded, and
/// everything downstream — the transposed-operand GEMMs (<c>gemm_row4_tn</c>/<c>gemm_row4_nt</c>),
/// the ReLU mask, the bias reductions and the fused <c>adam_update</c> step — consumes and produces
/// device buffers, so FC1/FC2 are device-authoritative during training and the host mirrors are
/// refreshed when <c>Train</c> returns. The embedding stays host-resident (gather, mean pool and
/// scatter are host work; a device-resident table with a CAS-scatter backward is a measured
/// follow-up) and takes its Adam step on the host through <see cref="SilkLoss.AdamStep"/>.
/// </summary>
public sealed class GpuTextLinearClassifier : IDisposable
{
    readonly SilkRuntime runtime;
    readonly int vocabSize;
    readonly int embeddingDim;
    readonly int hiddenDim;
    readonly Random rng;

    readonly float[] embedding;
    readonly float[] fc1Weight;   // [embeddingDim, hiddenDim] — the GPU layout; the CPU layout is its transpose
    readonly float[] fc1Bias;
    readonly float[] fc2Weight;   // [hiddenDim, 2]
    readonly float[] fc2Bias;

    nint fc1Buf, fc1BiasBuf, fc2Buf, fc2BiasBuf;

    // Device Adam moments, one first/second pair per FC parameter tensor. Allocate returns
    // uninitialized memory, so they are zero-filled at construction and whenever a Train run
    // starts a fresh optimizer.
    nint fc1Wm, fc1Wv, fc1Bm, fc1Bv, fc2Wm, fc2Wv, fc2Bm, fc2Bv;

    // Workspace, grown to the largest batch seen (Ensure never shrinks).
    SilkRuntime.SilkBuffer pooledBuf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer hBuf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer logitsBuf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer dzBuf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer dHBuf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer dPooledBuf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer dW1Buf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer db1Buf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer dW2Buf = SilkRuntime.SilkBuffer.Empty;
    SilkRuntime.SilkBuffer db2Buf = SilkRuntime.SilkBuffer.Empty;

    // Host scratch for the pooling pass, the embedding scatter and the embedding's Adam step.
    float[] pooled = [];
    internal float[] gradEmb = [];
    float[] embAdamFirst = [];
    float[] embAdamSecond = [];
    int[] batchTokens = [];
    int[] batchLabels = [];

    int adamStep;
    bool disposed;

    public string DeviceName => runtime.DeviceName;

    public GpuTextLinearClassifier(int vocabSize, int embeddingDim, int hiddenDim, int seed = 42)
    {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.hiddenDim = hiddenDim;
        rng = new Random(seed);

        runtime = new SilkRuntime();
        embedding = new float[vocabSize * embeddingDim];
        fc1Weight = new float[embeddingDim * hiddenDim];
        fc1Bias = new float[hiddenDim];
        fc2Weight = new float[hiddenDim * 2];
        fc2Bias = new float[2];

        InitWeights();

        fc1Buf = runtime.Allocate(fc1Weight.Length);
        fc1BiasBuf = runtime.Allocate(fc1Bias.Length);
        fc2Buf = runtime.Allocate(fc2Weight.Length);
        fc2BiasBuf = runtime.Allocate(fc2Bias.Length);
        fc1Wm = runtime.Allocate(fc1Weight.Length);
        fc1Wv = runtime.Allocate(fc1Weight.Length);
        fc1Bm = runtime.Allocate(fc1Bias.Length);
        fc1Bv = runtime.Allocate(fc1Bias.Length);
        fc2Wm = runtime.Allocate(fc2Weight.Length);
        fc2Wv = runtime.Allocate(fc2Weight.Length);
        fc2Bm = runtime.Allocate(fc2Bias.Length);
        fc2Bv = runtime.Allocate(fc2Bias.Length);
        ZeroMomentBuffers();

        UploadWeights();
    }

    void ThrowIfDisposed()
    {
        if (disposed) throw new ObjectDisposedException(nameof(GpuTextLinearClassifier));
    }

    void InitWeights()
    {
        for (int i = 0; i < embedding.Length; i++) embedding[i] = NextFloat(-0.05f, 0.05f);
        for (int i = 0; i < fc1Weight.Length; i++) fc1Weight[i] = NextFloat(-0.05f, 0.05f);
        for (int i = 0; i < fc2Weight.Length; i++) fc2Weight[i] = NextFloat(-0.05f, 0.05f);
    }

    float NextFloat(float min, float max) => min + (float)rng.NextDouble() * (max - min);

    public void SetWeights(float[] embeddingWeights, float[] fc1Weights, float[] fc1Biases,
        float[] fc2Weights, float[] fc2Biases)
    {
        ThrowIfDisposed();
        ValidateShapes(embeddingWeights, fc1Weights, fc1Biases, fc2Weights, fc2Biases);
        CopyWeights(embeddingWeights, fc1Weights, fc1Biases, fc2Weights, fc2Biases);
        UploadWeights();
    }

    /// <summary>
    /// Sets all weights from the CPU <c>Module</c> layout (FC weights row-major
    /// <c>[outFeatures, inFeatures]</c>) so callers need not hand-roll the transpose.
    /// </summary>
    public void SetWeightsCpuLayout(
        int vocabSize, int embeddingDim, int maxSeqLen,
        float[] embeddingWeights,
        float[] fc1WeightsCpu, float[] fc1Biases,
        float[] fc2WeightsCpu, float[] fc2Biases)
    {
        ThrowIfDisposed();
        if (vocabSize != this.vocabSize || embeddingDim != this.embeddingDim || maxSeqLen < 1)
            throw new ArgumentException("Shape mismatch with the classifier's configured dimensions.");
        if (fc1WeightsCpu is null || fc1WeightsCpu.Length != hiddenDim * embeddingDim)
            throw new ArgumentException("FC1 weight shape mismatch.", nameof(fc1WeightsCpu));
        if (fc2WeightsCpu is null || fc2WeightsCpu.Length != 2 * hiddenDim)
            throw new ArgumentException("FC2 weight shape mismatch.", nameof(fc2WeightsCpu));

        var fc1Gpu = new float[embeddingDim * hiddenDim];
        for (int e = 0; e < embeddingDim; e++)
            for (int h = 0; h < hiddenDim; h++)
                fc1Gpu[e * hiddenDim + h] = fc1WeightsCpu[h * embeddingDim + e];
        var fc2Gpu = new float[hiddenDim * 2];
        for (int h = 0; h < hiddenDim; h++)
            for (int c = 0; c < 2; c++)
                fc2Gpu[h * 2 + c] = fc2WeightsCpu[c * hiddenDim + h];

        ValidateShapes(embeddingWeights, fc1Gpu, fc1Biases, fc2Gpu, fc2Biases);
        CopyWeights(embeddingWeights, fc1Gpu, fc1Biases, fc2Gpu, fc2Biases);
        UploadWeights();
    }

    void ValidateShapes(float[] embeddingWeights, float[] fc1Weights, float[] fc1Biases,
        float[] fc2Weights, float[] fc2Biases)
    {
        if (embeddingWeights is null || embeddingWeights.Length != embedding.Length)
            throw new ArgumentException("Embedding weight shape mismatch.", nameof(embeddingWeights));
        if (fc1Weights is null || fc1Weights.Length != fc1Weight.Length)
            throw new ArgumentException("FC1 weight shape mismatch.", nameof(fc1Weights));
        if (fc1Biases is null || fc1Biases.Length != fc1Bias.Length)
            throw new ArgumentException("FC1 bias shape mismatch.", nameof(fc1Biases));
        if (fc2Weights is null || fc2Weights.Length != fc2Weight.Length)
            throw new ArgumentException("FC2 weight shape mismatch.", nameof(fc2Weights));
        if (fc2Biases is null || fc2Biases.Length != fc2Bias.Length)
            throw new ArgumentException("FC2 bias shape mismatch.", nameof(fc2Biases));
    }

    void CopyWeights(float[] embeddingWeights, float[] fc1Weights, float[] fc1Biases,
        float[] fc2Weights, float[] fc2Biases)
    {
        Array.Copy(embeddingWeights, embedding, embedding.Length);
        Array.Copy(fc1Weights, fc1Weight, fc1Weight.Length);
        Array.Copy(fc1Biases, fc1Bias, fc1Bias.Length);
        Array.Copy(fc2Weights, fc2Weight, fc2Weight.Length);
        Array.Copy(fc2Biases, fc2Bias, fc2Bias.Length);
    }

    public float[] ExportEmbedding() { ThrowIfDisposed(); var a = new float[embedding.Length]; Array.Copy(embedding, a, a.Length); return a; }
    public float[] ExportFc1Weight() { ThrowIfDisposed(); var a = new float[fc1Weight.Length]; Array.Copy(fc1Weight, a, a.Length); return a; }
    public float[] ExportFc1Bias() { ThrowIfDisposed(); var a = new float[fc1Bias.Length]; Array.Copy(fc1Bias, a, a.Length); return a; }
    public float[] ExportFc2Weight() { ThrowIfDisposed(); var a = new float[fc2Weight.Length]; Array.Copy(fc2Weight, a, a.Length); return a; }
    public float[] ExportFc2Bias() { ThrowIfDisposed(); var a = new float[fc2Bias.Length]; Array.Copy(fc2Bias, a, a.Length); return a; }

    /// <summary>All weights transposed into the CPU <c>Module</c> layout.</summary>
    public (float[] Embedding, float[] Fc1W, float[] Fc1B, float[] Fc2W, float[] Fc2B) ExportCpuLayout()
    {
        ThrowIfDisposed();
        var fc1Cpu = new float[hiddenDim * embeddingDim];
        for (int e = 0; e < embeddingDim; e++)
            for (int h = 0; h < hiddenDim; h++)
                fc1Cpu[h * embeddingDim + e] = fc1Weight[e * hiddenDim + h];
        var fc2Cpu = new float[2 * hiddenDim];
        for (int h = 0; h < hiddenDim; h++)
            for (int c = 0; c < 2; c++)
                fc2Cpu[c * hiddenDim + h] = fc2Weight[h * 2 + c];
        return (ExportEmbedding(), fc1Cpu, ExportFc1Bias(), fc2Cpu, ExportFc2Bias());
    }

    /// <summary>Downloads the trained FC weights back into the host mirrors. Called when
    /// <c>Train</c> returns; while training runs the device buffers are authoritative.</summary>
    void RefreshHostMirrors()
    {
        var fc1 = runtime.Download(fc1Buf, fc1Weight.Length);
        fc1.CopyTo(fc1Weight, 0);
        var fc1b = runtime.Download(fc1BiasBuf, fc1Bias.Length);
        fc1b.CopyTo(fc1Bias, 0);
        var fc2 = runtime.Download(fc2Buf, fc2Weight.Length);
        fc2.CopyTo(fc2Weight, 0);
        var fc2b = runtime.Download(fc2BiasBuf, fc2Bias.Length);
        fc2b.CopyTo(fc2Bias, 0);
    }

    void ForwardCore(int[] tokenIds, int batchSize, int seqLen)
    {
        if (tokenIds is null || tokenIds.Length != batchSize * seqLen)
            throw new ArgumentException("Token array size mismatch.", nameof(tokenIds));

        pooled = SilkLoss.Obtain(ref pooled, batchSize * embeddingDim);
        for (int i = 0; i < batchSize * embeddingDim; i++) pooled[i] = 0f;
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int tid = tokenIds[b * seqLen + s];
                if (tid < 0 || tid >= vocabSize) throw new ArgumentOutOfRangeException(nameof(tokenIds));
                for (int e = 0; e < embeddingDim; e++)
                    pooled[b * embeddingDim + e] += embedding[tid * embeddingDim + e];
            }
            for (int e = 0; e < embeddingDim; e++)
                pooled[b * embeddingDim + e] /= Math.Max(1, seqLen);
        }

        pooledBuf = runtime.Ensure(pooledBuf, batchSize * embeddingDim);
        hBuf = runtime.Ensure(hBuf, batchSize * hiddenDim);
        logitsBuf = runtime.Ensure(logitsBuf, batchSize * 2);
        runtime.Upload(pooledBuf.Handle, pooled.AsSpan(0, batchSize * embeddingDim));

        SilkLauncher.GemmEpilogue(runtime, SilkKernels.GemmRelu,
            pooledBuf.Handle, fc1Buf, hBuf.Handle, fc1BiasBuf,
            batchSize, embeddingDim, hiddenDim);
        SilkLauncher.GemmEpilogue(runtime, SilkKernels.GemmBias,
            hBuf.Handle, fc2Buf, logitsBuf.Handle, fc2BiasBuf,
            batchSize, hiddenDim, 2);
    }

    public float[] Forward(int[] tokenIds, int batchSize, int seqLen,
        out float[] pooledOut, out float[] hiddenOut)
    {
        ThrowIfDisposed();
        ForwardCore(tokenIds, batchSize, seqLen);
        pooledOut = new float[batchSize * embeddingDim];
        Array.Copy(pooled, pooledOut, pooledOut.Length);
        hiddenOut = runtime.Download(hBuf.Handle, batchSize * hiddenDim);
        return runtime.Download(logitsBuf.Handle, batchSize * 2);
    }

    public float[] Forward(int[] tokenIds, int batchSize, int seqLen)
    {
        ThrowIfDisposed();
        ForwardCore(tokenIds, batchSize, seqLen);
        return runtime.Download(logitsBuf.Handle, batchSize * 2);
    }

    public int[] Predict(int[] tokenIds, int seqLen)
    {
        ThrowIfDisposed();
        if (tokenIds is null || tokenIds.Length % seqLen != 0)
            throw new ArgumentException("Token array size mismatch.", nameof(tokenIds));
        int batch = tokenIds.Length / seqLen;
        var logits = Forward(tokenIds, batch, seqLen);
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
    /// <c>adam_update</c> step for FC1/FC2 and the host Adam step for the host-resident embedding.
    /// Each call starts a fresh optimizer (moments zeroed, step counter reset), mirroring the conv
    /// classifier's per-call <c>AdamState</c> construction.
    /// </summary>
    public IReadOnlyList<SilkTrainEpoch> Train(int[] tokens, int[] labels, int batchSize, int seqLen, int epochs, float learningRate, int seed)
    {
        ThrowIfDisposed();
        if (tokens is null || labels is null) throw new ArgumentNullException();
        if (tokens.Length % seqLen != 0) throw new ArgumentException("Token shape mismatch.");
        int sampleCount = tokens.Length / seqLen;
        if (labels.Length != sampleCount) throw new ArgumentException("Label count mismatch.");
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
        if (epochs <= 0) throw new ArgumentOutOfRangeException(nameof(epochs));

        adamStep = 0;
        ZeroMomentBuffers();
        embAdamFirst = new float[embedding.Length];
        embAdamSecond = new float[embedding.Length];

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

                ForwardCore(batchTokens, n, seqLen);
                var logits = runtime.Download(logitsBuf.Handle, n * 2);
                Backward(logits, batchTokens, batchLabels, n, seqLen);
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
    /// batch-averaged dZ, runs the device GEMM/ReLU/bias-reduction chain, and scatters the mean-pool
    /// gradient into <see cref="gradEmb"/> on the host. No optimizer step — gradients stay in the
    /// device gradient buffers and <see cref="gradEmb"/> until <see cref="AdamUpdateAll"/>.
    /// </summary>
    internal void Backward(float[] logits, int[] tokens, int[] labels, int batch, int seqLen)
    {
        ThrowIfDisposed();
        if (logits is null || logits.Length != batch * 2)
            throw new ArgumentException("Logit count mismatch.", nameof(logits));
        if (tokens is null || tokens.Length != batch * seqLen)
            throw new ArgumentException("Token array size mismatch.", nameof(tokens));
        if (labels is null || labels.Length != batch)
            throw new ArgumentException("Label count mismatch.", nameof(labels));

        int hidLen = batch * hiddenDim;

        dzBuf = runtime.Ensure(dzBuf, batch * 2);
        dHBuf = runtime.Ensure(dHBuf, hidLen);
        dPooledBuf = runtime.Ensure(dPooledBuf, batch * embeddingDim);
        dW1Buf = runtime.Ensure(dW1Buf, fc1Weight.Length);
        db1Buf = runtime.Ensure(db1Buf, fc1Bias.Length);
        dW2Buf = runtime.Ensure(dW2Buf, fc2Weight.Length);
        db2Buf = runtime.Ensure(db2Buf, fc2Bias.Length);

        var dz = new float[batch * 2];
        SilkLoss.InputGradient(logits, labels, batch, 1f / batch, dz);
        runtime.Upload(dzBuf.Handle, dz);

        SilkLauncher.ColSum(runtime, dzBuf.Handle, db2Buf.Handle, batch, 2, 1f);
        SilkLauncher.GemmTn(runtime, hBuf.Handle, dzBuf.Handle, dW2Buf.Handle,
            aRows: hiddenDim, aCols: batch, bCols: 2);
        SilkLauncher.GemmNt(runtime, dzBuf.Handle, fc2Buf, dHBuf.Handle,
            aRows: batch, aCols: 2, bCols: hiddenDim);
        SilkLauncher.ReluBackward(runtime, dHBuf.Handle, hBuf.Handle, dHBuf.Handle, hidLen);
        SilkLauncher.ColSum(runtime, dHBuf.Handle, db1Buf.Handle, batch, hiddenDim, 1f);
        SilkLauncher.GemmTn(runtime, pooledBuf.Handle, dHBuf.Handle, dW1Buf.Handle,
            aRows: embeddingDim, aCols: batch, bCols: hiddenDim);
        SilkLauncher.GemmNt(runtime, dHBuf.Handle, fc1Buf, dPooledBuf.Handle,
            aRows: batch, aCols: hiddenDim, bCols: embeddingDim);

        var dPooled = runtime.Download(dPooledBuf.Handle, batch * embeddingDim);

        gradEmb = SilkLoss.Obtain(ref gradEmb, embedding.Length);
        Array.Clear(gradEmb);
        float poolScale = 1f / Math.Max(1, seqLen);
        for (int i = 0; i < batch; i++)
            for (int s = 0; s < seqLen; s++)
            {
                int tidBase = tokens[i * seqLen + s] * embeddingDim;
                int iBase = i * embeddingDim;
                for (int e = 0; e < embeddingDim; e++)
                    gradEmb[tidBase + e] += poolScale * dPooled[iBase + e];
            }
    }

    /// <summary>Applies one Adam step: the fused device kernel to FC1/FC2 (weights and biases) and
    /// <see cref="SilkLoss.AdamStep"/> to the host-resident embedding. One step counter is shared;
    /// every parameter takes the same bias correction.</summary>
    internal void AdamUpdateAll(float learningRate)
    {
        adamStep++;
        float bc1 = 1f - MathF.Pow(0.9f, adamStep);
        float bc2 = 1f - MathF.Pow(0.999f, adamStep);

        SilkLauncher.AdamUpdate(runtime, fc1Buf, dW1Buf.Handle, fc1Wm, fc1Wv, learningRate, bc1, bc2, 1e-8f, fc1Weight.Length);
        SilkLauncher.AdamUpdate(runtime, fc1BiasBuf, db1Buf.Handle, fc1Bm, fc1Bv, learningRate, bc1, bc2, 1e-8f, fc1Bias.Length);
        SilkLauncher.AdamUpdate(runtime, fc2Buf, dW2Buf.Handle, fc2Wm, fc2Wv, learningRate, bc1, bc2, 1e-8f, fc2Weight.Length);
        SilkLauncher.AdamUpdate(runtime, fc2BiasBuf, db2Buf.Handle, fc2Bm, fc2Bv, learningRate, bc1, bc2, 1e-8f, fc2Bias.Length);

        SilkLoss.AdamStep(embedding, embAdamFirst, embAdamSecond, gradEmb, learningRate, bc1, bc2);
    }

    internal (float[] Fc1W, float[] Fc1B, float[] Fc2W, float[] Fc2B) DownloadGradients()
    {
        ThrowIfDisposed();
        return (
            runtime.Download(dW1Buf.Handle, fc1Weight.Length),
            runtime.Download(db1Buf.Handle, fc1Bias.Length),
            runtime.Download(dW2Buf.Handle, fc2Weight.Length),
            runtime.Download(db2Buf.Handle, fc2Bias.Length));
    }

    void ZeroMomentBuffers()
    {
        SilkLauncher.Fill(runtime, fc1Wm, 0f, fc1Weight.Length);
        SilkLauncher.Fill(runtime, fc1Wv, 0f, fc1Weight.Length);
        SilkLauncher.Fill(runtime, fc1Bm, 0f, fc1Bias.Length);
        SilkLauncher.Fill(runtime, fc1Bv, 0f, fc1Bias.Length);
        SilkLauncher.Fill(runtime, fc2Wm, 0f, fc2Weight.Length);
        SilkLauncher.Fill(runtime, fc2Wv, 0f, fc2Weight.Length);
        SilkLauncher.Fill(runtime, fc2Bm, 0f, fc2Bias.Length);
        SilkLauncher.Fill(runtime, fc2Bv, 0f, fc2Bias.Length);
    }

    void UploadWeights()
    {
        runtime.Upload(fc1Buf, fc1Weight.AsSpan());
        runtime.Upload(fc1BiasBuf, fc1Bias.AsSpan());
        runtime.Upload(fc2Buf, fc2Weight.AsSpan());
        runtime.Upload(fc2BiasBuf, fc2Bias.AsSpan());
    }

    public void Dispose()
    {
        if (disposed) return;
        runtime.Release(fc1Buf);
        runtime.Release(fc1BiasBuf);
        runtime.Release(fc2Buf);
        runtime.Release(fc2BiasBuf);
        runtime.Release(fc1Wm);
        runtime.Release(fc1Wv);
        runtime.Release(fc1Bm);
        runtime.Release(fc1Bv);
        runtime.Release(fc2Wm);
        runtime.Release(fc2Wv);
        runtime.Release(fc2Bm);
        runtime.Release(fc2Bv);
        runtime.Release(pooledBuf.Handle);
        runtime.Release(hBuf.Handle);
        runtime.Release(logitsBuf.Handle);
        runtime.Release(dzBuf.Handle);
        runtime.Release(dHBuf.Handle);
        runtime.Release(dPooledBuf.Handle);
        runtime.Release(dW1Buf.Handle);
        runtime.Release(db1Buf.Handle);
        runtime.Release(dW2Buf.Handle);
        runtime.Release(db2Buf.Handle);
        runtime.Dispose();
        disposed = true;
    }
}
