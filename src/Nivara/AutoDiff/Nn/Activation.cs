using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Functional wrappers around activation operations (ReLU, sigmoid, tanh, GELU,
/// softmax, etc.) that delegate to <see cref="Nivara.AutoDiff.Operations.ReverseGradOperations"/>.
/// </summary>
public static class Activation
{
    /// <summary>Applies the rectified linear unit: <c>max(0, input)</c>.</summary>
    /// <param name="input">The input tensor</param>
    /// <returns>The activated tensor</returns>
    public static ReverseGradTensor<T> Relu<T>(ReverseGradTensor<T> input) where T : struct, IFloatingPointIeee754<T>
        => ReverseGradOperations.Relu(input);

    /// <summary>Applies the sigmoid function: <c>1 / (1 + e^-x)</c>.</summary>
    /// <param name="input">The input tensor</param>
    /// <returns>The activated tensor</returns>
    public static ReverseGradTensor<T> Sigmoid<T>(ReverseGradTensor<T> input) where T : struct, IFloatingPointIeee754<T>
        => ReverseGradOperations.Sigmoid(input);

    /// <summary>Applies the hyperbolic tangent activation.</summary>
    /// <param name="input">The input tensor</param>
    /// <returns>The activated tensor</returns>
    public static ReverseGradTensor<T> Tanh<T>(ReverseGradTensor<T> input) where T : struct, IFloatingPointIeee754<T>
        => ReverseGradOperations.Tanh(input);

    /// <summary>Applies the Gaussian error linear unit (GELU) using the tanh approximation.</summary>
    /// <param name="input">The input tensor</param>
    /// <returns>The activated tensor</returns>
    public static ReverseGradTensor<T> Gelu<T>(ReverseGradTensor<T> input) where T : struct, IFloatingPointIeee754<T>
        => ReverseGradOperations.Gelu(input);

    /// <summary>Applies the exact Gaussian error linear unit (GELU) via the error function.</summary>
    /// <param name="input">The input tensor</param>
    /// <returns>The activated tensor</returns>
    public static ReverseGradTensor<T> GeluExact<T>(ReverseGradTensor<T> input) where T : struct, IFloatingPointIeee754<T>
        => ReverseGradOperations.GeluExact(input);

    /// <summary>Applies the leaky rectified linear unit: <c>x</c> for <c>x &gt;= 0</c>, otherwise <c>negativeSlope * x</c>.</summary>
    /// <param name="input">The input tensor</param>
    /// <param name="negativeSlope">Slope applied to negative inputs; zero (the default) is interpreted as 0.01</param>
    /// <returns>The activated tensor</returns>
    public static ReverseGradTensor<T> LeakyRelu<T>(ReverseGradTensor<T> input, T negativeSlope = default)
        where T : struct, IFloatingPointIeee754<T>
        => ReverseGradOperations.LeakyRelu(input, negativeSlope);

    /// <summary>Applies the exponential function element-wise.</summary>
    /// <param name="input">The input tensor</param>
    /// <returns>The activated tensor</returns>
    public static ReverseGradTensor<T> Exp<T>(ReverseGradTensor<T> input) where T : struct, IFloatingPointIeee754<T>
        => ReverseGradOperations.Exp(input);

    /// <summary>Applies the natural logarithm element-wise.</summary>
    /// <param name="input">The input tensor</param>
    /// <returns>The activated tensor</returns>
    public static ReverseGradTensor<T> Log<T>(ReverseGradTensor<T> input) where T : struct, IFloatingPointIeee754<T>
        => ReverseGradOperations.Log(input);

    /// <summary>Applies softmax over the given dimension (default -1 = the last dimension).</summary>
    /// <param name="input">The input tensor</param>
    /// <param name="dim">The dimension to normalize over</param>
    /// <returns>The normalized tensor</returns>
    public static ReverseGradTensor<T> Softmax<T>(ReverseGradTensor<T> input, int dim = -1)
        where T : struct, IFloatingPointIeee754<T>
        => ReverseGradOperations.Softmax(input, dim);

    /// <summary>Applies log-softmax over the given dimension (default -1 = the last dimension).</summary>
    /// <param name="input">The input tensor</param>
    /// <param name="dim">The dimension to normalize over</param>
    /// <returns>The log-normalized tensor</returns>
    public static ReverseGradTensor<T> LogSoftmax<T>(ReverseGradTensor<T> input, int dim = -1)
        where T : struct, IFloatingPointIeee754<T>
        => ReverseGradOperations.LogSoftmax(input, dim);
}
