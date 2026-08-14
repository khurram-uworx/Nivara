using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Fully-connected (dense) layer computing <c>output = input · Weight^T + bias</c>.
/// The weight parameter has shape <c>[outFeatures, inFeatures]</c>; the bias, when
/// enabled, has shape <c>[outFeatures]</c>.
/// </summary>
public sealed class Linear<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int inFeatures;
    readonly int outFeatures;
    readonly bool useBias;
    readonly Parameter<T> weight;
    readonly Parameter<T>? bias;

    /// <summary>Gets the number of input features.</summary>
    public int InFeatures => inFeatures;
    /// <summary>Gets the number of output features.</summary>
    public int OutFeatures => outFeatures;
    /// <summary>Gets the weight parameter (shape <c>[outFeatures, inFeatures]</c>).</summary>
    public Parameter<T>? Weight => weight;
    /// <summary>Gets the bias parameter (shape <c>[outFeatures]</c>), or null when bias is disabled.</summary>
    public Parameter<T>? Bias => bias;

    /// <summary>
    /// Creates a linear layer with the given dimensions. The weight is initialized with
    /// <see cref="KaimingUniformInitializer{T}"/> by default.
    /// </summary>
    /// <param name="inFeatures">Number of input features</param>
    /// <param name="outFeatures">Number of output features</param>
    /// <param name="bias">Whether to include a bias parameter</param>
    /// <param name="weightInitializer">Optional custom weight initializer (defaults to Kaiming uniform)</param>
    /// <param name="biasInitializer">Optional bias initializer (no default initialization when null)</param>
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

    /// <summary>
    /// Computes <c>input · Weight^T + bias</c> for a 2D input of shape <c>[batch, inFeatures]</c>,
    /// producing a tensor of shape <c>[batch, outFeatures]</c>.
    /// </summary>
    /// <param name="input">The input tensor</param>
    /// <returns>The layer output</returns>
    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var w = weight.Tensor;
        var output = ReverseGradOperations.MatMulTransposedB(input, w);

        if (useBias && bias != null)
            output = ReverseGradOperations.AddBias(output, bias.Tensor);

        return output;
    }
}
