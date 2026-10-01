using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Layer normalization over the last dimension of the input. Each row (the trailing
/// <c>normalizedShape</c> elements) is normalized using that row's own mean and variance,
/// optionally followed by a per-element affine transform. The transform's gamma and beta are
/// independently optional: gamma without beta is the bias-free pre-norm of modern pre-norm
/// architectures.
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
    /// <summary>Gets the learnable beta parameter, or null when <c>affine</c> or <c>bias</c> is false.</summary>
    public Parameter<T>? Bias => bias;

    /// <summary>
    /// Creates a layer normalization layer. Gamma is initialized to one and, when a bias is
    /// requested, beta to zero. <paramref name="affine"/> false removes both parameters;
    /// <paramref name="bias"/> false keeps the learnable gamma and drops beta, which is how a
    /// bias-free pre-norm is expressed (HuggingFace's <c>norm_bias: false</c>). A bias-free
    /// norm registers no bias parameter, so it is absent from <c>StateDict()</c> and from an
    /// optimizer's parameter list, and there is nothing for an optimizer to drift.
    /// </summary>
    /// <param name="normalizedShape">Size of the normalized (last) dimension (must be positive)</param>
    /// <param name="eps">Stability term added to the variance (must be positive)</param>
    /// <param name="affine">Whether to include learnable gamma/beta parameters</param>
    /// <param name="bias">Whether to include the learnable beta parameter (requires <paramref name="affine"/>)</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown for a non-positive <paramref name="normalizedShape"/> or <paramref name="eps"/></exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="bias"/> is true but <paramref name="affine"/> is false</exception>
    public LayerNorm(
        int normalizedShape,
        float eps = 1e-5f,
        bool affine = true,
        bool bias = true)
    {
        if (normalizedShape <= 0) throw new ArgumentOutOfRangeException(nameof(normalizedShape));
        if (eps <= 0) throw new ArgumentOutOfRangeException(nameof(eps));
        if (!affine && bias)
            throw new ArgumentException(
                "affine: false removes both gamma and beta, so bias cannot also be true.",
                nameof(bias));

        this.normalizedShape = normalizedShape;
        this.eps = T.CreateChecked(eps);
        this.affine = affine;

        if (affine)
        {
            var weightData = new T[normalizedShape];
            for (int i = 0; i < normalizedShape; i++)
                weightData[i] = T.One;
            weight = new Parameter<T>("Weight", ReverseGradTensor<T>.FromArray(weightData, requiresGrad: true));
            RegisterParameters(weight);

            if (bias)
            {
                this.bias = new Parameter<T>("Bias", ReverseGradTensor<T>.FromArray(new T[normalizedShape], requiresGrad: true));
                RegisterParameters(this.bias);
            }
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

        var gamma = weight != null
            ? GetParamSpan(weight.Tensor)
            : ReadOnlySpan<T>.Empty;
        var beta = bias != null
            ? GetParamSpan(bias.Tensor)
            : ReadOnlySpan<T>.Empty;

        var inputData = GetInputSpan(input);
        int rows = input.Length / normalizedShape;

        if (!input.RequiresGrad)
        {
            var output = LayerNormKernel<T>.ForwardInference(inputData, rows, normalizedShape, gamma, beta, eps);
            return new ReverseGradTensor<T>(
                NivaraColumn<T>.CreateFromOwnedArray(output),
                false, input.Shape);
        }

        var result = LayerNormKernel<T>.Forward(inputData, rows, normalizedShape, gamma, beta, eps);

        var resultTensor = new ReverseGradTensor<T>(
            NivaraColumn<T>.Create(result.Output),
            input.RequiresGrad, input.Shape);

        if (input.RequiresGrad)
        {
            var savedXHat = result.XHat;
            var savedInvStd = result.InvStd;
            var savedGamma = gamma.Length > 0 ? gamma.ToArray() : [];
            int savedRows = rows;
            int savedNormShape = normalizedShape;

            var gradFn = new OpNode<T>("LayerNorm",
                ModuleHelpers<T>.NodeInputs(input, weight?.Tensor, bias?.Tensor),
                (typedGradOutput) =>
            {
                var gradOutData = new T[typedGradOutput.Length];
                typedGradOutput.CopyTo(gradOutData, default(T)!);

                var gradInputData = LayerNormKernel<T>.BackwardInput(
                    gradOutData, savedXHat, savedGamma, savedInvStd,
                    savedRows, savedNormShape);

                ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.Create(gradInputData));

                if (weight != null)
                {
                    var gradGammaData = LayerNormKernel<T>.BackwardWeight(
                        gradOutData, savedXHat, savedRows, savedNormShape);
                    ReverseGradOperations.AccumulateGradient(weight.Tensor, NivaraColumn<T>.Create(gradGammaData));
                }

                if (bias != null)
                {
                    var gradBetaData = LayerNormKernel<T>.BackwardBias(
                        gradOutData, savedRows, savedNormShape);
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
