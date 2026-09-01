using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Root-mean-square (RMS) normalization over the last dimension of the input, followed by an
/// affine multiply by a per-dimension gamma. Each row (the trailing <c>normalizedShape</c>
/// elements) is normalized by the root-mean-square of that row's elements. This is the
/// pre-normalization layer used by Llama-family causal language models.
/// </summary>
public sealed class RMSNorm<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int normalizedShape;
    readonly T eps;

    readonly Parameter<T> weight;

    /// <summary>Gets the size of the normalized (last) dimension.</summary>
    public int NormalizedShape => normalizedShape;

    /// <summary>Gets the stability term added to the row mean-square.</summary>
    public T Eps => eps;

    /// <summary>Gets the learnable gamma parameter (shape <c>[normalizedShape]</c>).</summary>
    public Parameter<T>? Weight => weight;

    /// <summary>
    /// Creates an RMS normalization layer. Gamma is initialized to one.
    /// </summary>
    /// <param name="normalizedShape">Size of the normalized (last) dimension (must be positive)</param>
    /// <param name="eps">Stability term added to the mean-square (must be positive)</param>
    public RMSNorm(int normalizedShape, float eps = 1e-5f)
    {
        if (normalizedShape <= 0) throw new ArgumentOutOfRangeException(nameof(normalizedShape));
        if (eps <= 0) throw new ArgumentOutOfRangeException(nameof(eps));

        this.normalizedShape = normalizedShape;
        this.eps = T.CreateChecked(eps);

        var weightData = new T[normalizedShape];
        for (int i = 0; i < normalizedShape; i++)
            weightData[i] = T.One;

        weight = new Parameter<T>("Weight", ReverseGradTensor<T>.FromArray(weightData, requiresGrad: true));
        RegisterParameters(weight);
    }

    /// <summary>
    /// Applies RMS normalization then an affine gamma multiply over the last dimension of a
    /// tensor of rank at least 2.
    /// </summary>
    /// <param name="input">The input tensor (rank at least 2)</param>
    /// <returns>The RMS-normalized, gamma-scaled tensor</returns>
    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank < 2) throw new ArgumentException($"RMSNorm expects at least 2D input, got {input.Rank}D");
        if (input.Shape[^1] != normalizedShape)
            throw new ArgumentException($"Expected last dimension {normalizedShape}, got {input.Shape[^1]}");

        var gamma = ModuleHelpers<T>.GetSpan(weight.Tensor);
        var inputData = ModuleHelpers<T>.GetSpan(input);
        int rows = input.Length / normalizedShape;

        if (!input.RequiresGrad)
        {
            var output = ForwardInference(inputData, rows, normalizedShape, gamma, eps);
            return new ReverseGradTensor<T>(
                NivaraColumn<T>.CreateFromOwnedArray(output),
                false, input.Shape);
        }

        var outputData = new T[input.Length];
        RMSNormKernel<T>.PerRowRMSNormForwardKernel(
            ToArray(inputData), outputData, rows, normalizedShape, double.CreateChecked(eps));
        // Apply affine gamma: out[i,j] = y[i,j] * gamma[j]
        for (int i = 0; i < rows; i++)
        {
            int baseIdx = i * normalizedShape;
            for (int j = 0; j < normalizedShape; j++)
                outputData[baseIdx + j] = outputData[baseIdx + j] * gamma[j];
        }

        var resultTensor = new ReverseGradTensor<T>(
            NivaraColumn<T>.CreateFromOwnedArray(outputData),
            input.RequiresGrad, input.Shape);

        if (input.RequiresGrad)
        {
            var savedInput = ToArray(inputData);
            var savedGamma = gamma.ToArray();
            int savedRows = rows;
            int savedNormShape = normalizedShape;
            double savedEps = double.CreateChecked(eps);

            var gradFn = new OpNode<T>("RMSNorm", [input], (typedGradOutput) =>
            {
                var gradOutData = new T[typedGradOutput.Length];
                typedGradOutput.CopyTo(gradOutData, default(T)!);

                // dL/dy = dL/dOut * gamma  (since out = y * gamma)
                var gradNorm = new T[gradOutData.Length];
                for (int i = 0; i < rows; i++)
                {
                    int baseIdx = i * normalizedShape;
                    for (int j = 0; j < normalizedShape; j++)
                        gradNorm[baseIdx + j] = gradOutData[baseIdx + j] * savedGamma[j];
                }

                var gradInputData = new T[gradNorm.Length];
                RMSNormKernel<T>.PerRowRMSNormBackwardKernel(
                    savedInput, gradNorm, gradInputData, savedRows, savedNormShape, savedEps);
                ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.CreateFromOwnedArray(gradInputData));

                // dL/dgamma[j] = sum_i y[i,j] * dL/dOut[i,j]
                var gradWeightData = new T[normalizedShape];
                var yData = new T[gradNorm.Length];
                RMSNormKernel<T>.PerRowRMSNormForwardKernel(
                    savedInput, yData, savedRows, savedNormShape, savedEps);
                for (int i = 0; i < rows; i++)
                {
                    int baseIdx = i * normalizedShape;
                    for (int j = 0; j < normalizedShape; j++)
                        gradWeightData[j] += yData[baseIdx + j] * gradOutData[baseIdx + j];
                }
                ReverseGradOperations.AccumulateGradient(weight.Tensor, NivaraColumn<T>.CreateFromOwnedArray(gradWeightData));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    static T[] ToArray(ReadOnlySpan<T> span)
    {
        var arr = new T[span.Length];
        span.CopyTo(arr);
        return arr;
    }

    static T[] ForwardInference(ReadOnlySpan<T> input, int rows, int cols, ReadOnlySpan<T> gamma, T eps)
    {
        var src = ToArray(input);
        var y = new T[input.Length];
        RMSNormKernel<T>.PerRowRMSNormForwardKernel(src, y, rows, cols, double.CreateChecked(eps));
        for (int i = 0; i < rows; i++)
        {
            int baseIdx = i * cols;
            for (int j = 0; j < cols; j++)
                y[baseIdx + j] = y[baseIdx + j] * gamma[j];
        }
        return y;
    }
}
