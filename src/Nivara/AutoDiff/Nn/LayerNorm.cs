using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class LayerNorm<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int _normalizedShape;
    readonly T _eps;
    readonly bool _affine;

    readonly Parameter<T>? _weight;
    readonly Parameter<T>? _bias;

    public int NormalizedShape => _normalizedShape;
    public T Eps => _eps;
    public bool Affine => _affine;
    public Parameter<T>? Weight => _weight;
    public Parameter<T>? Bias => _bias;

    public LayerNorm(
        int normalizedShape,
        float eps = 1e-5f,
        bool affine = true)
    {
        if (normalizedShape <= 0) throw new ArgumentOutOfRangeException(nameof(normalizedShape));
        if (eps <= 0) throw new ArgumentOutOfRangeException(nameof(eps));

        _normalizedShape = normalizedShape;
        _eps = T.CreateChecked(eps);
        _affine = affine;

        if (affine)
        {
            var weightData = new T[normalizedShape];
            var biasData = new T[normalizedShape];
            for (int i = 0; i < normalizedShape; i++)
            {
                weightData[i] = T.One;
                biasData[i] = T.Zero;
            }
            _weight = new Parameter<T>("Weight", ReverseGradTensor<T>.FromArray(weightData, requiresGrad: true));
            _bias = new Parameter<T>("Bias", ReverseGradTensor<T>.FromArray(biasData, requiresGrad: true));
            RegisterParameters(_weight, _bias);
        }
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank < 2) throw new ArgumentException($"LayerNorm expects at least 2D input, got {input.Rank}D");
        if (input.Shape[^1] != _normalizedShape)
            throw new ArgumentException($"Expected last dimension {_normalizedShape}, got {input.Shape[^1]}");

        var gamma = _affine && _weight != null
            ? GetParamSpan(_weight.Tensor)
            : ReadOnlySpan<T>.Empty;
        var beta = _affine && _bias != null
            ? GetParamSpan(_bias.Tensor)
            : ReadOnlySpan<T>.Empty;

        var inputData = GetInputSpan(input);
        int rows = input.Length / _normalizedShape;

        var result = LayerNormKernel<T>.Forward(inputData, rows, _normalizedShape, gamma, beta, _eps, _affine);

        var resultTensor = new ReverseGradTensor<T>(
            NivaraColumn<T>.Create(result.Output),
            input.RequiresGrad, input.Shape);

        if (input.RequiresGrad)
        {
            var savedXHat = result.XHat;
            var savedInvStd = result.InvStd;
            var savedGamma = gamma.Length > 0 ? gamma.ToArray() : [];
            bool affine = _affine;
            int savedRows = rows;
            int savedNormShape = _normalizedShape;

            var gradFn = new OpNode<T>("LayerNorm", new object[] { input }, (typedGradOutput) =>
            {
                var gradOutData = new T[typedGradOutput.Length];
                typedGradOutput.CopyTo(gradOutData, default(T)!);

                var gradInputData = LayerNormKernel<T>.BackwardInput(
                    gradOutData, savedXHat, savedGamma, savedInvStd,
                    savedRows, savedNormShape, affine);

                ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.Create(gradInputData));

                if (affine)
                {
                    var gradGammaData = LayerNormKernel<T>.BackwardWeight(
                        gradOutData, savedXHat, savedRows, savedNormShape);
                    var gradBetaData = LayerNormKernel<T>.BackwardBias(
                        gradOutData, savedRows, savedNormShape);

                    if (_weight != null)
                        ReverseGradOperations.AccumulateGradient(_weight.Tensor, NivaraColumn<T>.Create(gradGammaData));
                    if (_bias != null)
                        ReverseGradOperations.AccumulateGradient(_bias.Tensor, NivaraColumn<T>.Create(gradBetaData));
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
