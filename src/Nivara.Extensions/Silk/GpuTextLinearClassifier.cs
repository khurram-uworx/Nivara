using System.Numerics.Tensors;

namespace Nivara.Silk;

/// <summary>
/// Linear text classifier forward and (host-gradient scaffold) training on Silk/OpenCL.
/// Embeddings and mean pool are retained on the host; FC1/FC2 execute on the device.
/// Host-side backward and SGD updates provide a complete, testable training loop while
/// device-side backward kernels remain tracked by #886.
/// </summary>
public sealed class GpuTextLinearClassifier : IDisposable
{
    readonly SilkRuntime runtime;
    readonly int vocabSize;
    readonly int embeddingDim;
    readonly int hiddenDim;
    readonly Random rng;

    readonly float[] embedding;
    readonly float[] fc1Weight;
    readonly float[] fc1Bias;
    readonly float[] fc2Weight;
    readonly float[] fc2Bias;

    readonly nint embBuf;
    readonly nint fc1Buf;
    readonly nint fc1BiasBuf;
    readonly nint fc2Buf;
    readonly nint fc2BiasBuf;

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
        embBuf = runtime.Allocate(embedding.Length);
        fc1Buf = runtime.Allocate(fc1Weight.Length);
        fc1BiasBuf = runtime.Allocate(fc1Bias.Length);
        fc2Buf = runtime.Allocate(fc2Weight.Length);
        fc2BiasBuf = runtime.Allocate(fc2Bias.Length);

        UploadAll();
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

        Array.Copy(embeddingWeights, embedding, embedding.Length);
        Array.Copy(fc1Weights, fc1Weight, fc1Weight.Length);
        Array.Copy(fc1Biases, fc1Bias, fc1Bias.Length);
        Array.Copy(fc2Weights, fc2Weight, fc2Weight.Length);
        Array.Copy(fc2Biases, fc2Bias, fc2Bias.Length);
        UploadAll();
    }

    public float[] ExportEmbedding() { var a = new float[embedding.Length]; Array.Copy(embedding, a, a.Length); return a; }
    public float[] ExportFc1Weight() { var a = new float[fc1Weight.Length]; Array.Copy(fc1Weight, a, a.Length); return a; }
    public float[] ExportFc1Bias() { var a = new float[fc1Bias.Length]; Array.Copy(fc1Bias, a, a.Length); return a; }
    public float[] ExportFc2Weight() { var a = new float[fc2Weight.Length]; Array.Copy(fc2Weight, a, a.Length); return a; }
    public float[] ExportFc2Bias() { var a = new float[fc2Bias.Length]; Array.Copy(fc2Bias, a, a.Length); return a; }

    public float[] Forward(int[] tokenIds, int batchSize, int seqLen,
        out float[] pooledOut, out float[] hiddenOut)
    {
        pooledOut = [];
        hiddenOut = [];
        if (tokenIds is null || tokenIds.Length != batchSize * seqLen)
            throw new ArgumentException("Token array size mismatch.", nameof(tokenIds));

        var pooled = new float[batchSize * embeddingDim];
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

        nint pooledBuf = runtime.Allocate(pooled.Length);
        nint hBuf = runtime.Allocate(batchSize * hiddenDim);
        nint logitsBuf = runtime.Allocate(batchSize * 2);
        try
        {
            runtime.Upload(pooledBuf, pooled.AsSpan());

            runtime.SetBuffer(SilkKernels.GemmRelu, 0, pooledBuf);
            runtime.SetBuffer(SilkKernels.GemmRelu, 1, fc1Buf);
            runtime.SetBuffer(SilkKernels.GemmRelu, 2, hBuf);
            runtime.SetBuffer(SilkKernels.GemmRelu, 3, fc1BiasBuf);
            runtime.SetUInt32(SilkKernels.GemmRelu, 4, (uint)batchSize);
            runtime.SetUInt32(SilkKernels.GemmRelu, 5, (uint)embeddingDim);
            runtime.SetUInt32(SilkKernels.GemmRelu, 6, (uint)hiddenDim);
            runtime.Run(SilkKernels.GemmRelu,
                [(nuint)SilkKernels.GemmGlobalRows(batchSize), (nuint)SilkKernels.GemmGlobalCols(hiddenDim)],
                [(nuint)SilkKernels.GemmTileSize, (nuint)SilkKernels.GemmTileSize]);

            runtime.SetBuffer(SilkKernels.GemmBias, 0, hBuf);
            runtime.SetBuffer(SilkKernels.GemmBias, 1, fc2Buf);
            runtime.SetBuffer(SilkKernels.GemmBias, 2, logitsBuf);
            runtime.SetBuffer(SilkKernels.GemmBias, 3, fc2BiasBuf);
            runtime.SetUInt32(SilkKernels.GemmBias, 4, (uint)batchSize);
            runtime.SetUInt32(SilkKernels.GemmBias, 5, (uint)hiddenDim);
            runtime.SetUInt32(SilkKernels.GemmBias, 6, (uint)2);
            runtime.Run(SilkKernels.GemmBias,
                [(nuint)SilkKernels.GemmGlobalRows(batchSize), (nuint)SilkKernels.GemmGlobalCols(2)],
                [(nuint)SilkKernels.GemmTileSize, (nuint)SilkKernels.GemmTileSize]);

            pooledOut = pooled;
            hiddenOut = runtime.Download(hBuf, batchSize * hiddenDim);
            return runtime.Download(logitsBuf, batchSize * 2);
        }
        finally
        {
            runtime.Release(pooledBuf);
            runtime.Release(hBuf);
            runtime.Release(logitsBuf);
        }
    }

    public float[] Forward(int[] tokenIds, int batchSize, int seqLen)
        => Forward(tokenIds, batchSize, seqLen, out _, out _);

    public int[] Predict(int[] tokenIds, int seqLen)
    {
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

    public void Train(int[] tokens, int[] labels, int batchSize, int seqLen, int epochs, float learningRate)
    {
        if (tokens is null || labels is null) throw new ArgumentNullException();
        if (tokens.Length % seqLen != 0) throw new ArgumentException("Token shape mismatch.");
        int sampleCount = tokens.Length / seqLen;
        if (labels.Length != sampleCount) throw new ArgumentException("Label count mismatch.");

        int batchesPerEpoch = (sampleCount + batchSize - 1) / batchSize;
        var rng = new Random(12345);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            int[] indices = Enumerable.Range(0, sampleCount).ToArray();
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            for (int b = 0; b < batchesPerEpoch; b++)
            {
                int start = b * batchSize;
                int end = Math.Min(start + batchSize, sampleCount);
                int n = end - start;
                if (n <= 0) continue;

                int[] batchTokens = new int[n * seqLen];
                int[] batchLabels = new int[n];
                for (int i = 0; i < n; i++)
                {
                    int src = indices[start + i];
                    Array.Copy(tokens, src * seqLen, batchTokens, i * seqLen, seqLen);
                    batchLabels[i] = labels[src];
                }

                var logits = Forward(batchTokens, n, seqLen, out var pooled, out var hidden);

                // Compute softmax probs and dZ.
                var gradZ = new float[n * 2];
                for (int i = 0; i < n; i++)
                {
                    float e0 = MathF.Exp(logits[i * 2]);
                    float e1 = MathF.Exp(logits[i * 2 + 1]);
                    float z = e0 + e1;
                    float p0 = e0 / z;
                    float p1 = e1 / z;
                    gradZ[i * 2 + 0] = p0 - (batchLabels[i] == 0 ? 1f : 0f);
                    gradZ[i * 2 + 1] = p1 - (batchLabels[i] == 1 ? 1f : 0f);
                }

                // FC2 gradients
                var gradW2 = new float[hiddenDim * 2];
                var gradB2 = new float[2];
                for (int i = 0; i < n; i++)
                {
                    gradB2[0] += gradZ[i * 2 + 0];
                    gradB2[1] += gradZ[i * 2 + 1];
                    for (int h = 0; h < hiddenDim; h++)
                    {
                        gradW2[h * 2 + 0] += hidden[i * hiddenDim + h] * gradZ[i * 2 + 0];
                        gradW2[h * 2 + 1] += hidden[i * hiddenDim + h] * gradZ[i * 2 + 1];
                    }
                }

                // dHidden
                var dH = new float[n * hiddenDim];
                for (int i = 0; i < n; i++)
                    for (int h = 0; h < hiddenDim; h++)
                        dH[i * hiddenDim + h] =
                            fc2Weight[h * 2 + 0] * gradZ[i * 2 + 0] +
                            fc2Weight[h * 2 + 1] * gradZ[i * 2 + 1];

                // ReLU backward
                for (int i = 0; i < n; i++)
                    for (int h = 0; h < hiddenDim; h++)
                        if (hidden[i * hiddenDim + h] <= 0f)
                            dH[i * hiddenDim + h] = 0f;

                // FC1 gradients
                var gradW1 = new float[embeddingDim * hiddenDim];
                var gradB1 = new float[hiddenDim];
                for (int i = 0; i < n; i++)
                {
                    for (int h = 0; h < hiddenDim; h++)
                        gradB1[h] += dH[i * hiddenDim + h];
                    for (int e = 0; e < embeddingDim; e++)
                        for (int h = 0; h < hiddenDim; h++)
                            gradW1[e * hiddenDim + h] += pooled[i * embeddingDim + e] * dH[i * hiddenDim + h];
                }

                // Pooled gradient
                var dPooled = new float[n * embeddingDim];
                for (int i = 0; i < n; i++)
                    for (int e = 0; e < embeddingDim; e++)
                        for (int h = 0; h < hiddenDim; h++)
                            dPooled[i * embeddingDim + e] += dH[i * hiddenDim + h] * fc1Weight[e * hiddenDim + h];

                // Embedding gradient (scatter)
                var gradEmb = new float[vocabSize * embeddingDim];
                for (int i = 0; i < n; i++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        int tid = batchTokens[i * seqLen + s];
                        float scale = 1f / Math.Max(1, seqLen);
                        for (int e = 0; e < embeddingDim; e++)
                            gradEmb[tid * embeddingDim + e] += scale * dPooled[i * embeddingDim + e];
                    }
                }

                // Average
                for (int i = 0; i < gradW2.Length; i++) gradW2[i] /= n;
                for (int i = 0; i < gradB2.Length; i++) gradB2[i] /= n;
                for (int i = 0; i < gradW1.Length; i++) gradW1[i] /= n;
                for (int i = 0; i < gradB1.Length; i++) gradB1[i] /= n;
                for (int i = 0; i < gradEmb.Length; i++) gradEmb[i] /= n;

                // SGD update
                for (int i = 0; i < fc2Weight.Length; i++) fc2Weight[i] -= learningRate * gradW2[i];
                for (int i = 0; i < fc2Bias.Length; i++) fc2Bias[i] -= learningRate * gradB2[i];
                for (int i = 0; i < fc1Weight.Length; i++) fc1Weight[i] -= learningRate * gradW1[i];
                for (int i = 0; i < fc1Bias.Length; i++) fc1Bias[i] -= learningRate * gradB1[i];
                for (int i = 0; i < embedding.Length; i++) embedding[i] -= learningRate * gradEmb[i];

                UploadAll();
            }
        }
    }

    void UploadAll()
    {
        runtime.Upload(embBuf, embedding.AsSpan());
        runtime.Upload(fc1Buf, fc1Weight.AsSpan());
        runtime.Upload(fc1BiasBuf, fc1Bias.AsSpan());
        runtime.Upload(fc2Buf, fc2Weight.AsSpan());
        runtime.Upload(fc2BiasBuf, fc2Bias.AsSpan());
    }

    public void Dispose()
    {
        if (disposed) return;
        runtime.Release(embBuf);
        runtime.Release(fc1Buf);
        runtime.Release(fc1BiasBuf);
        runtime.Release(fc2Buf);
        runtime.Release(fc2BiasBuf);
        runtime.Dispose();
        disposed = true;
    }
}
