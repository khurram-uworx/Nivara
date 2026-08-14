using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class LayerNorm<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int normalizedShape;
    readonly T eps;
    readonly bool affine;

    readonly Parameter<T>? weight;
    readonly Parameter<T>? bias;

    public int NormalizedShape => normalizedShape;
    public T Eps => eps;
    public bool Affine => affine;
    public Parameter<T>? Weight => weight;
    public Parameter<T>? Bias => bias;

    public LayerNorm(
        int normalizedShape,
        float eps = 1e-5f,
        bool affine = true)
    {
        if (normalizedShape <= 0) throw new ArgumentOutOfRangeException(nameof(normalizedShape));
        if (eps <= 0) throw new ArgumentOutOfRangeException(nameof(eps));

        this.normalizedShape = normalizedShape;
        this.eps = T.CreateChecked(eps);
        this.affine = affine;

        if (affine)
        {
            var weightData = new T[normalizedShape];
            var biasData = new T[normalizedShape];
            for (int i = 0; i < normalizedShape; i++)
            {
                weightData[i] = T.One;
                biasData[i] = T.Zero;
            }
            weight = new Parameter<T>("Weight", ReverseGradTensor<T>.FromArray(weightData, requiresGrad: true));
            bias = new Parameter<T>("Bias", ReverseGradTensor<T>.FromArray(biasData, requiresGrad: true));
            RegisterParameters(weight, bias);
        }
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank < 2) throw new ArgumentException($"LayerNorm expects at least 2D input, got {input.Rank}D");
        if (input.Shape[^1] != normalizedShape)
            throw new ArgumentException($"Expected last dimension {normalizedShape}, got {input.Shape[^1]}");

        var gamma = affine && weight != null
            ? GetParamSpan(weight.Tensor)
            : ReadOnlySpan<T>.Empty;
        var beta = affine && bias != null
            ? GetParamSpan(bias.Tensor)
            : ReadOnlySpan<T>.Empty;

        var inputData = GetInputSpan(input);
        int rows = input.Length / normalizedShape;

        if (!input.RequiresGrad)
        {
            var output = LayerNormKernel<T>.ForwardInference(inputData, rows, normalizedShape, gamma, beta, eps, affine);
            return new ReverseGradTensor<T>(
                NivaraColumn<T>.CreateFromOwnedArray(output),
                false, input.Shape);
        }

        var result = LayerNormKernel<T>.Forward(inputData, rows, normalizedShape, gamma, beta, eps, affine);

        var resultTensor = new ReverseGradTensor<T>(
            NivaraColumn<T>.Create(result.Output),
            input.RequiresGrad, input.Shape);

        if (input.RequiresGrad)
        {
            var savedXHat = result.XHat;
            var savedInvStd = result.InvStd;
            var savedGamma = gamma.Length > 0 ? gamma.ToArray() : [];
            bool useAffine = affine;
            int savedRows = rows;
            int savedNormShape = normalizedShape;

            var gradFn = new OpNode<T>("LayerNorm", [input], (typedGradOutput) =>
            {
                var gradOutData = new T[typedGradOutput.Length];
                typedGradOutput.CopyTo(gradOutData, default(T)!);

                var gradInputData = LayerNormKernel<T>.BackwardInput(
                    gradOutData, savedXHat, savedGamma, savedInvStd,
                    savedRows, savedNormShape, useAffine);

                ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.Create(gradInputData));

                if (useAffine)
                {
                    var gradGammaData = LayerNormKernel<T>.BackwardWeight(
                        gradOutData, savedXHat, savedRows, savedNormShape);
                    var gradBetaData = LayerNormKernel<T>.BackwardBias(
                        gradOutData, savedRows, savedNormShape);

                    if (weight != null)
                        ReverseGradOperations.AccumulateGradient(weight.Tensor, NivaraColumn<T>.Create(gradGammaData));
                    if (bias != null)
                        ReverseGradOperations.AccumulateGradient(bias.Tensor, NivaraColumn<T>.Create(gradBetaData));
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    static ReadOnlySpan<T> GetInputSpan(ReverseGradTensor<T> tensor)
        => ModuleHelpers<T>.GetSpan(tensor);

    static ReadOnlySpan<T> GetParamSpan(ReverseGradTensor<T> tensor)
        => ModuleHelpers<T>.GetSpan(tensor);
}
