using System.Numerics;

namespace Nivara.AutoDiff.Training;

/// <summary>A single training batch consisting of feature and label tensors.</summary>
public sealed class Batch<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>The feature tensor for the batch.</summary>
    public ReverseGradTensor<T> Features { get; }

    /// <summary>The label tensor for the batch.</summary>
    public ReverseGradTensor<T> Labels { get; }

    /// <summary>The number of rows in the batch.</summary>
    public int Size { get; }

    /// <summary>
    /// Creates a batch from feature and label tensors.
    /// </summary>
    /// <param name="features">The feature tensor</param>
    /// <param name="labels">The label tensor</param>
    /// <exception cref="ArgumentNullException">Thrown when either tensor is null</exception>
    public Batch(ReverseGradTensor<T> features, ReverseGradTensor<T> labels)
    {
        Features = features ?? throw new ArgumentNullException(nameof(features));
        Labels = labels ?? throw new ArgumentNullException(nameof(labels));
        Size = features.Length;
    }
}
