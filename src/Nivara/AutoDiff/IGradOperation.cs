using System.Numerics;

namespace Nivara.AutoDiff;

/// <summary>
/// Interface for gradient-aware operations in reverse-mode automatic differentiation.
/// Defines the contract for operations that can compute both forward values and gradients.
/// </summary>
/// <typeparam name="T">The numeric type of the operation</typeparam>
public interface IGradOperation<T> where T : struct, INumber<T>
{
    /// <summary>
    /// Gets the name of the operation for debugging and graph visualization
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Performs the forward pass of the operation
    /// </summary>
    /// <param name="inputs">The input ReverseGradTensors for the operation</param>
    /// <returns>The result of the forward computation as a ReverseGradTensor</returns>
    ReverseGradTensor<T> Forward(params ReverseGradTensor<T>[] inputs);

    /// <summary>
    /// Performs the backward pass of the operation, computing gradients
    /// </summary>
    /// <param name="gradOutput">The gradient flowing back from the output</param>
    /// <param name="inputs">The original input tensors</param>
    /// <param name="output">The output tensor from the forward pass</param>
    void Backward(NivaraColumn<T> gradOutput, ReverseGradTensor<T>[] inputs, ReverseGradTensor<T> output);
}
