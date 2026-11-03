using System.Numerics.Tensors;

namespace Nivara.Silk;

/// <summary>
/// TextCNN classifier forward (embedding -> Conv1d branches -> concat -> FC1 -> ReLU -> FC2) on
/// Silk/OpenCL. Phase 0 foundation: embedding and conv branches execute on the host; the FC
/// projections execute on the device. Training/backward kernels remain tracked separately.
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

    readonly float[] embedding;
    readonly int[] kernelSizes = [3, 5, 7];
    readonly float[] conv1W, conv1B;
    readonly float[] conv2W, conv2B;
    readonly float[] conv3W, conv3B;
    readonly float[] fc1W, fc1B;
    readonly float[] fc2W, fc2B;

    readonly nint fc1Buf, fc1BiasBuf, fc2Buf, fc2BiasBuf;
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

        fc1Buf = runtime.Allocate(fc1W.Length);
        fc1BiasBuf = runtime.Allocate(fc1B.Length);
        fc2Buf = runtime.Allocate(fc2W.Length);
        fc2BiasBuf = runtime.Allocate(fc2B.Length);

        runtime.Upload(fc1Buf, fc1W.AsSpan());
        runtime.Upload(fc1BiasBuf, fc1B.AsSpan());
        runtime.Upload(fc2Buf, fc2W.AsSpan());
        runtime.Upload(fc2BiasBuf, fc2B.AsSpan());
    }

    public void SetWeights(
        float[] embeddingWeights,
        float[] conv1Weights, float[] conv1Biases,
        float[] conv2Weights, float[] conv2Biases,
        float[] conv3Weights, float[] conv3Biases,
        float[] fc1Weights, float[] fc1Biases,
        float[] fc2Weights, float[] fc2Biases)
    {
        if (embeddingWeights is null || embeddingWeights.Length != embedding.Length) throw new ArgumentException("embedding");
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

        runtime.Upload(fc1Buf, fc1W.AsSpan());
        runtime.Upload(fc1BiasBuf, fc1B.AsSpan());
        runtime.Upload(fc2Buf, fc2W.AsSpan());
        runtime.Upload(fc2BiasBuf, fc2B.AsSpan());
    }

    public float[] Forward(int[] tokenIds, int batchSize)
    {
        if (tokenIds is null || tokenIds.Length != batchSize * seqLen)
            throw new ArgumentException("token shape mismatch");

        // Host embedding+conv scaffold, ending in concat buffer.
        var concat = new float[batchSize * convHidden];
        for (int b = 0; b < batchSize; b++)
        {
            // embedding gather -> channels [E,S]
            var emb = new float[embeddingDim * seqLen];
            for (int s = 0; s < seqLen; s++)
            {
                int tid = tokenIds[b * seqLen + s];
                for (int e = 0; e < embeddingDim; e++)
                    emb[e * seqLen + s] = embedding[tid * embeddingDim + e];
            }

            // branch 0/1/2
            float[][] branchOuts =
            [
                ConvBranch(emb, conv1W, conv1B, 3),
                ConvBranch(emb, conv2W, conv2B, 5),
                ConvBranch(emb, conv3W, conv3B, 7)
            ];

            int offset = 0;
            for (int br = 0; br < BranchCount; br++)
            {
                // [64,S] -> flatten [64*S]
                for (int c = 0; c < BranchChannels; c++)
                    for (int s = 0; s < seqLen; s++)
                        concat[b * convHidden + offset + c * seqLen + s] = branchOuts[br][c * seqLen + s];
                offset += BranchChannels * seqLen;
            }
        }

        int fcIn = convHidden;
        nint concBuf = runtime.Allocate(concat.Length);
        nint hBuf = runtime.Allocate(batchSize * hiddenDim);
        nint logitsBuf = runtime.Allocate(batchSize * 2);
        try
        {
            runtime.Upload(concBuf, concat.AsSpan());

            runtime.SetBuffer(SilkKernels.GemmRelu, 0, concBuf);
            runtime.SetBuffer(SilkKernels.GemmRelu, 1, fc1Buf);
            runtime.SetBuffer(SilkKernels.GemmRelu, 2, hBuf);
            runtime.SetBuffer(SilkKernels.GemmRelu, 3, fc1BiasBuf);
            runtime.SetUInt32(SilkKernels.GemmRelu, 4, (uint)batchSize);
            runtime.SetUInt32(SilkKernels.GemmRelu, 5, (uint)fcIn);
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

            return runtime.Download(logitsBuf, batchSize * 2);
        }
        finally
        {
            runtime.Release(concBuf);
            runtime.Release(hBuf);
            runtime.Release(logitsBuf);
        }
    }

    public int[] Predict(int[] tokenIds, int seqLen)
    {
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

    float[] ConvBranch(float[] inputChannels, float[] weight, float[] bias, int k)
    {
        // input: [E,S], output [64,S] with zero padding (k-1)/2
        int pad = (k - 1) / 2;
        var output = new float[BranchChannels * seqLen];
        for (int oc = 0; oc < BranchChannels; oc++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                float acc = bias[oc];
                for (int ic = 0; ic < embeddingDim; ic++)
                {
                    for (int kx = 0; kx < k; kx++)
                    {
                        int inS = s - pad + kx;
                        if (inS < 0 || inS >= seqLen) continue;
                        int wIndex = (oc * embeddingDim * k) + (ic * k) + kx;
                        acc += inputChannels[ic * seqLen + inS] * weight[wIndex];
                    }
                }
                // ReLU
                output[oc * seqLen + s] = MathF.Max(0f, acc);
            }
        }
        return output;
    }

    public void Dispose()
    {
        if (disposed) return;
        runtime.Release(fc1Buf);
        runtime.Release(fc1BiasBuf);
        runtime.Release(fc2Buf);
        runtime.Release(fc2BiasBuf);
        runtime.Dispose();
        disposed = true;
    }
}
