using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using Nivara.Tensors;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class Linear<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int inFeatures;
    readonly int outFeatures;
    readonly bool useBias;
    readonly Parameter<T> weight;
    readonly Parameter<T>? bias;
    T[]? transposedWeight;
    long transposedWeightVersion = -1;

    public int InFeatures => inFeatures;
    public int OutFeatures => outFeatures;
    public ReverseGradTensor<T> Weight => weight.Tensor;
    public Parameter<T> WeightParam => weight;
    public ReverseGradTensor<T>? Bias => bias?.Tensor;

    public Linear(int inFeatures, int outFeatures, bool bias = true,
        IInitializer<T>? weightInitializer = null,
        IInitializer<T>? biasInitializer = null)
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

        (weightInitializer ?? KaimingUniformInitializer<T>.Instance).Initialize(weight);

        if (bias && biasInitializer != null)
            biasInitializer.Initialize(this.bias!);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var w = weight.Tensor;
        var output = GradientUtils.IsGradEnabled
            ? ReverseGradOperations.MatMul(input, GetTransposedWeight(w))
            : ReverseGradOperations.MatMulTransposedB(input, w);

        if (useBias && bias != null)
            output = ReverseGradOperations.AddBias(output, bias.Tensor);

        return output;
    }

    ReverseGradTensor<T> GetTransposedWeight(ReverseGradTensor<T> w)
    {
        if (transposedWeight is null || transposedWeightVersion != weight.Version)
        {
            if (transposedWeight is null || transposedWeight.Length != outFeatures * inFeatures)
                transposedWeight = new T[outFeatures * inFeatures];
            w.Data.TryGetSpan(out var wSpan);
            TensorsHelper.Transpose(wSpan, transposedWeight.AsSpan(), outFeatures, inFeatures);
            transposedWeightVersion = weight.Version;
        }

        return ReverseGradOperations.TransposeCached(w, transposedWeight);
    }
}
