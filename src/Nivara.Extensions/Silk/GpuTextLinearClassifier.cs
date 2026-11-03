using System.Numerics.Tensors;

namespace Nivara.Silk;

/// <summary>
/// Linear text classifier forward (embedding -> mean pool -> FC1 -> ReLU -> FC2) on Silk/OpenCL.
/// Phase 0 foundation: embedding gather and mean pool are retained on the host; the linear
/// projections execute on the device. Training backward kernels remain tracked by #886.
/// </summary>
public sealed class GpuTextLinearClassifier : IDisposable
{
    readonly SilkRuntime runtime;
    readonly int vocabSize;
    readonly int embeddingDim;
    readonly int hiddenDim;

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

    public GpuTextLinearClassifier(int vocabSize, int embeddingDim, int hiddenDim)
    {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.hiddenDim = hiddenDim;

        runtime = new SilkRuntime();
        embedding = new float[vocabSize * embeddingDim];
        fc1Weight = new float[embeddingDim * hiddenDim];
        fc1Bias = new float[hiddenDim];
        fc2Weight = new float[hiddenDim * 2];
        fc2Bias = new float[2];

        embBuf = runtime.Allocate(embedding.Length);
        fc1Buf = runtime.Allocate(fc1Weight.Length);
        fc1BiasBuf = runtime.Allocate(fc1Bias.Length);
        fc2Buf = runtime.Allocate(fc2Weight.Length);
        fc2BiasBuf = runtime.Allocate(fc2Bias.Length);

        UploadAll();
    }

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
