using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public static class Activation
{
    public static ReverseGradTensor<T> Relu<T>(ReverseGradTensor<T> input) where T : struct, INumber<T>
        => GradOperations.Relu(input);

    public static ReverseGradTensor<T> Sigmoid<T>(ReverseGradTensor<T> input) where T : struct, INumber<T>
        => GradOperations.Sigmoid(input);

    public static ReverseGradTensor<T> Tanh<T>(ReverseGradTensor<T> input) where T : struct, INumber<T>
        => GradOperations.Tanh(input);

    public static ReverseGradTensor<T> LeakyRelu<T>(ReverseGradTensor<T> input, T negativeSlope = default)
        where T : struct, INumber<T>
        => GradOperations.LeakyRelu(input, negativeSlope);

    public static ReverseGradTensor<T> Exp<T>(ReverseGradTensor<T> input) where T : struct, INumber<T>
        => GradOperations.Exp(input);

    public static ReverseGradTensor<T> Log<T>(ReverseGradTensor<T> input) where T : struct, INumber<T>
        => GradOperations.Log(input);
}
