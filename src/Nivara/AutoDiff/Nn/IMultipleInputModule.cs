using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Represents a module whose forward pass takes two input tensors.
/// </summary>
public interface IMultipleInputModule<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>
    /// Runs the forward pass with two input tensors.
    /// </summary>
    /// <param name="input1">The first input tensor</param>
    /// <param name="input2">The second input tensor</param>
    /// <returns>The output tensor</returns>
    ReverseGradTensor<T> Forward(ReverseGradTensor<T> input1, ReverseGradTensor<T> input2);
}
