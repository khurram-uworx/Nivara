using System.Numerics;

namespace Nivara.AutoDiff.Training;

public sealed class Batch<T> where T : struct, INumber<T>
{
    public ReverseGradTensor<T> Features { get; }
    public ReverseGradTensor<T> Labels { get; }
    public int Size { get; }

    public Batch(ReverseGradTensor<T> features, ReverseGradTensor<T> labels)
    {
        Features = features ?? throw new ArgumentNullException(nameof(features));
        Labels = labels ?? throw new ArgumentNullException(nameof(labels));
        Size = features.Length;
    }
}
