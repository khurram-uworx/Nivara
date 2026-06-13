using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class Linear<T> : Module<T> where T : struct, INumber<T>
{
    readonly int inFeatures;
    readonly int outFeatures;
    readonly bool useBias;
    readonly Parameter<T> weight;
    readonly Parameter<T>? bias;

    public int InFeatures => inFeatures;
    public int OutFeatures => outFeatures;
    public ReverseGradTensor<T> Weight => weight.Tensor;
    public ReverseGradTensor<T>? Bias => bias?.Tensor;

    public Linear(int inFeatures, int outFeatures, bool bias = true,
        Action<Parameter<T>>? weightInitializer = null,
        Action<Parameter<T>>? biasInitializer = null)
    {
        if (inFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(inFeatures));
        if (outFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(outFeatures));

        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        useBias = bias;

        var weightData = new T[outFeatures * inFeatures];
        var weightTensor = ReverseGradTensor<T>.FromMatrix(weightData, outFeatures, inFeatures, requiresGrad: true);
        weight = new Parameter<T>("Weight", weightTensor);
        RegisterParameters(weight);

        if (bias)
        {
            var biasData = new T[outFeatures];
            var biasTensor = ReverseGradTensor<T>.FromMatrix(biasData, 1, outFeatures, requiresGrad: true);
            this.bias = new Parameter<T>("Bias", biasTensor);
            RegisterParameters(this.bias);
        }

        (weightInitializer ?? Initializers.KaimingUniform)(weight);

        if (bias && biasInitializer != null)
            biasInitializer(this.bias!);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var w = weight.Tensor;
        var transposed = GradOperations.Transpose(w);
        var output = GradOperations.MatMul(input, transposed);

        if (useBias && bias != null)
        {
            var biasTensor = bias.Tensor;
            var ones = GradientUtils.Ones<T>(input.shape[0]);
            ones.Reshape(ones.Length, 1);
            var biasBroadcast = GradOperations.MatMul(ones, biasTensor);
            output = GradOperations.Add(output, biasBroadcast);
        }

        return output;
    }
}
