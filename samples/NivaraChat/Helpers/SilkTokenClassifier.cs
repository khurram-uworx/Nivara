using Nivara.AutoDiff.Nn;
using Nivara.Silk;

namespace NivaraChat.Helpers;

/// <summary>
/// Silk/OpenCL-backed adapter for the token-level <see cref="TokenClassifierModel{T}"/>
/// classifier used by NivaraChat (entity extraction). Wraps a <see cref="GpuTokenClassifier"/>
/// built from the saved CPU weights so inference can run on the device when <c>--silk</c> is set.
/// </summary>
public sealed class SilkTokenClassifier : IDisposable
{
    readonly GpuTokenClassifier _gpu;
    readonly int _maxSeqLen;
    readonly int _numClasses;

    public SilkTokenClassifier(GpuTokenClassifier gpu, int maxSeqLen, int numClasses)
    {
        _gpu = gpu;
        _maxSeqLen = maxSeqLen;
        _numClasses = numClasses;
    }

    public int NumClasses => _numClasses;
    public int MaxSeqLen => _maxSeqLen;

    public float[] ForwardLogits(int[] tokenIds)
        => _gpu.Forward(tokenIds, _maxSeqLen, 1);

    public static SilkTokenClassifier? TryCreate(TokenClassifierModel<float> cpuModel, int maxSeqLen, int numClasses, int seed = 42)
    {
        try
        {
            var ps = cpuModel.GetParameters().Values.ToArray();
            var emb = ps[0].Tensor.Data.ToArray();
            var fc1wCpu = ps[1].Tensor.Data.ToArray();
            var fc1b = ps[2].Tensor.Data.ToArray();
            var fc2wCpu = ps[3].Tensor.Data.ToArray();
            var fc2b = ps[4].Tensor.Data.ToArray();

            int embed = cpuModel.EmbeddingDim;
            int hidden = fc1b.Length;
            var gpuFc1 = new float[embed * hidden];
            for (int e = 0; e < embed; e++)
                for (int h = 0; h < hidden; h++)
                    gpuFc1[e * hidden + h] = fc1wCpu[h * embed + e];
            var gpuFc2 = new float[hidden * numClasses];
            for (int h = 0; h < hidden; h++)
                for (int c = 0; c < numClasses; c++)
                    gpuFc2[h * numClasses + c] = fc2wCpu[c * hidden + h];

            var gpu = new GpuTokenClassifier(cpuModel.VocabSize, embed, hidden, seed, numClasses);
            gpu.SetWeights(emb, gpuFc1, fc1b, gpuFc2, fc2b);
            return new SilkTokenClassifier(gpu, maxSeqLen, numClasses);
        }
        catch
        {
            return null;
        }
    }

    public void Dispose() => _gpu.Dispose();
}
