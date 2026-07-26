using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Nn;

public sealed class Conv1d<T> : Module<T> where T : struct, INumber<T>
{
    readonly int _inChannels;
    readonly int _outChannels;
    readonly int _kernelSize;
    readonly int _stride;
    readonly int _padding;
    readonly bool _useBias;

    readonly Parameter<T> _weight;
    readonly Parameter<T>? _bias;

    public int InChannels => _inChannels;
    public int OutChannels => _outChannels;
    public int KernelSize => _kernelSize;
    public int Stride => _stride;
    public int Padding => _padding;
    public Parameter<T> WeightParam => _weight;
    public Parameter<T>? BiasParam => _bias;

    public Conv1d(
        int inChannels,
        int outChannels,
        int kernelSize,
        int stride = 1,
        int padding = 0,
        bool bias = true)
    {
        if (inChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels));
        if (outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outChannels));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (padding < 0) throw new ArgumentOutOfRangeException(nameof(padding));

        _inChannels = inChannels;
        _outChannels = outChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;
        _useBias = bias;

        int patchSize = inChannels * kernelSize;
        int fanIn = patchSize;
        var weightData = new T[outChannels * patchSize];
        var kaimingBound = T.CreateChecked(Math.Sqrt(2.0 / fanIn));
        var rng = new Random(42);
        for (int i = 0; i < weightData.Length; i++)
            weightData[i] = T.CreateChecked((rng.NextDouble() * 2.0 - 1.0) * double.CreateChecked(kaimingBound));

        _weight = new Parameter<T>("Weight",
            ReverseGradTensor<T>.FromMatrix(weightData, outChannels, patchSize, requiresGrad: true));
        RegisterParameters(_weight);

        if (bias)
        {
            var biasData = new T[outChannels];
            _bias = new Parameter<T>("Bias",
                ReverseGradTensor<T>.FromArray(biasData, requiresGrad: true));
            RegisterParameters(_bias);
        }
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 3) throw new ArgumentException($"Conv1d expects 3D input [B, C, L], got {input.Rank}D");
        if (input.Shape[1] != _inChannels)
            throw new ArgumentException($"Expected {_inChannels} input channels, got {input.Shape[1]}");

        int n = input.Shape[0];
        int c = input.Shape[1];
        int l = input.Shape[2];

        int oL = (l + 2 * _padding - _kernelSize) / _stride + 1;
        if (oL <= 0)
            throw new ArgumentException($"Output length is non-positive ({oL}). Check input size, kernel, stride, and padding.");

        int patchSize = c * _kernelSize;
        int totalPatches = n * oL;

        ReadOnlySpan<T> inputSpan = input.Data.TryGetSpan(out var iSpan)
            ? iSpan
            : CopyToTemp(input.Data, n * c * l);

        ReadOnlySpan<T> weightSpan = _weight.Tensor.Data.TryGetSpan(out var wSpan)
            ? wSpan
            : CopyToTemp(_weight.Tensor.Data, _outChannels * patchSize);

        ReadOnlySpan<T> biasSpan = ReadOnlySpan<T>.Empty;
        if (_useBias && _bias != null)
        {
            biasSpan = _bias.Tensor.Data.TryGetSpan(out var bSpan)
                ? bSpan
                : CopyToTemp(_bias.Tensor.Data, _outChannels);
        }

        var outputData = new T[n * _outChannels * oL];

        Conv1dForwardKernel(inputSpan, weightSpan, biasSpan, outputData,
            n, c, l, _kernelSize, _stride, _padding, oL, patchSize, totalPatches, _outChannels);

        var outShape = new[] { n, _outChannels, oL };
        var result = NivaraColumn<T>.Create(outputData);
        bool shouldTrack = GradientUtils.ShouldTrackGrad(input, _weight.Tensor);
        if (_useBias && _bias != null)
            shouldTrack = shouldTrack || _bias.Tensor.RequiresGrad;
        var resultTensor = new ReverseGradTensor<T>(result, shouldTrack, outShape);

        if (shouldTrack)
        {
            var capturedInputData = inputSpan.ToArray();
            var capturedWeightData = weightSpan.ToArray();
            var capturedBiasData = biasSpan.Length > 0 ? biasSpan.ToArray() : null;

            var gradFn = new OpNode<T>("Conv1d", new object[] { input, _weight.Tensor }, (gradOutput, sgn) =>
            {
                var gradOutData = new T[n * _outChannels * oL];
                gradOutput.CopyTo(gradOutData, T.Zero);

                if (input.RequiresGrad)
                {
                    var inputGrad = new T[n * c * l];
                    Conv1dInputGradKernel(gradOutData, capturedWeightData, inputGrad,
                        n, c, l, _kernelSize, _stride, _padding, oL, _outChannels, patchSize);
                    ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.Create(inputGrad));
                }

                if (_weight.Tensor.RequiresGrad)
                {
                    var weightGrad = new T[_outChannels * patchSize];
                    Conv1dWeightGradKernel(capturedInputData, gradOutData, weightGrad,
                        n, c, l, _kernelSize, _stride, _padding, oL, _outChannels, patchSize, totalPatches);
                    ReverseGradOperations.AccumulateGradient(_weight.Tensor, NivaraColumn<T>.Create(weightGrad));
                }

                if (_useBias && _bias != null && _bias.Tensor.RequiresGrad)
                {
                    var biasGrad = new T[_outChannels];
                    Conv1dBiasGradKernel(gradOutData, biasGrad, n, _outChannels, oL);
                    ReverseGradOperations.AccumulateGradient(_bias.Tensor, NivaraColumn<T>.Create(biasGrad));
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    static T[] CopyToTemp(NivaraColumn<T> column, int length)
    {
        var arr = new T[length];
        column.CopyTo(arr, T.Zero);
        return arr;
    }

    static void Conv1dForwardKernel(
        ReadOnlySpan<T> inputData, ReadOnlySpan<T> weightSpan, ReadOnlySpan<T> biasSpan, Span<T> outputData,
        int n, int c, int l, int kL, int stride, int padding,
        int oL, int patchSize, int totalPatches, int outChannels)
    {
        for (int patch = 0; patch < totalPatches; patch++)
        {
            int batch = patch / oL;
            int oPos = patch % oL;
            int inStart = oPos * stride - padding;
            int outBase = batch * outChannels * oL + oPos;

            for (int oc = 0; oc < outChannels; oc++)
            {
                var weightSlice = weightSpan.Slice(oc * patchSize, patchSize);
                T dot = T.Zero;

                for (int kh = 0; kh < kL; kh++)
                {
                    int inPos = inStart + kh;
                    if ((uint)inPos >= (uint)l) continue;

                    int inputBase = batch * c * l + inPos;
                    int wBase = kh * c;

                    for (int ch = 0; ch < c; ch++)
                        dot += inputData[inputBase + ch * l] * weightSlice[wBase + ch];
                }

                if (biasSpan.Length > 0)
                    dot += biasSpan[oc];

                outputData[outBase + oc * oL] = dot;
            }
        }
    }

    static void Conv1dInputGradKernel(
        ReadOnlySpan<T> gradOutData, ReadOnlySpan<T> weightData, Span<T> inputGrad,
        int n, int c, int l, int kL, int stride, int padding,
        int oL, int outChannels, int patchSize)
    {
        for (int batch = 0; batch < n; batch++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int oPos = 0; oPos < oL; oPos++)
                {
                    T g = gradOutData[batch * outChannels * oL + oc * oL + oPos];
                    if (g == T.Zero) continue;

                    int inStart = oPos * stride - padding;
                    int weightBase = oc * patchSize;

                    for (int kh = 0; kh < kL; kh++)
                    {
                        int inPos = inStart + kh;
                        if ((uint)inPos >= (uint)l) continue;

                        int inputBase = batch * c * l + inPos;
                        int wBase = kh * c;

                        for (int ch = 0; ch < c; ch++)
                            inputGrad[inputBase + ch * l] += g * weightData[weightBase + wBase + ch];
                    }
                }
            }
        }
    }

    static void Conv1dWeightGradKernel(
        ReadOnlySpan<T> inputData, ReadOnlySpan<T> gradOutData, Span<T> weightGrad,
        int n, int c, int l, int kL, int stride, int padding,
        int oL, int outChannels, int patchSize, int totalPatches)
    {
        for (int batch = 0; batch < n; batch++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int oPos = 0; oPos < oL; oPos++)
                {
                    T g = gradOutData[batch * outChannels * oL + oc * oL + oPos];
                    if (g == T.Zero) continue;

                    int inStart = oPos * stride - padding;
                    int weightBase = oc * patchSize;

                    for (int kh = 0; kh < kL; kh++)
                    {
                        int inPos = inStart + kh;
                        if ((uint)inPos >= (uint)l) continue;

                        int inputBase = batch * c * l + inPos;
                        int wBase = kh * c;

                        for (int ch = 0; ch < c; ch++)
                            weightGrad[weightBase + wBase + ch] += g * inputData[inputBase + ch * l];
                    }
                }
            }
        }
    }

    static void Conv1dBiasGradKernel(ReadOnlySpan<T> gradOutData, Span<T> biasGrad, int n, int outChannels, int oL)
    {
        for (int oc = 0; oc < outChannels; oc++)
        {
            T sum = T.Zero;
            for (int batch = 0; batch < n; batch++)
            {
                int channelBase = batch * outChannels * oL + oc * oL;
                sum += T.CreateChecked(double.CreateChecked(TensorPrimitives.Sum(gradOutData.Slice(channelBase, oL))));
            }
            biasGrad[oc] += sum;
        }
    }
}
