using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Layer normalization over the last dimension of the input. Each row (the trailing
/// <c>normalizedShape</c> elements) is normalized using that row's own mean and variance,
/// optionally followed by a per-element affine transform.
/// </summary>
public sealed class LayerNorm<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int normalizedShape;
    readonly T eps;
    readonly bool affine;

    readonly Parameter<T>? weight;
    readonly Parameter<T>? bias;

    /// <summary>Gets the size of the normalized (last) dimension.</summary>
    public int NormalizedShape => normalizedShape;
    /// <summary>Gets the stability term added to the variance.</summary>
    public T Eps => eps;
    /// <summary>Gets whether the affine gamma/beta transform is applied.</summary>
    public bool Affine => affine;
    /// <summary>Gets the learnable gamma parameter, or null when <c>affine</c> is false.</summary>
    public Parameter<T>? Weight => weight;
    /// <summary>Gets the learnable beta parameter, or null when <c>affine</c> is false.</summary>
    public Parameter<T>? Bias => bias;

    /// <summary>
    /// Creates a layer normalization layer. Gamma is initialized to one and beta to zero.
    /// </summary>
    /// <param name="normalizedShape">Size of the normalized (last) dimension (must be positive)</param>
    /// <param name="eps">Stability term added to the variance (must be positive)</param>
    /// <param name="affine">Whether to include learnable gamma/beta parameters</param>
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

    /// <summary>
    /// Normalizes the last dimension of a tensor of rank at least 2.
    /// </summary>
    /// <param name="input">The input tensor (rank at least 2)</param>
    /// <returns>The normalized tensor</returns>
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
