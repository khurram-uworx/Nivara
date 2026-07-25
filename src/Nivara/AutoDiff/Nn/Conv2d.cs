using System.Buffers;
using System.Runtime.CompilerServices;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Nn;

public sealed class Conv2d<T> : Module<T> where T : struct, INumber<T>
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

    public Conv2d(
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

        int fanIn = inChannels * kernelSize * kernelSize;
        var weightData = new T[outChannels * inChannels * kernelSize * kernelSize];
        var kaimingBound = T.CreateChecked(Math.Sqrt(2.0 / fanIn));
        var rng = new Random(42);
        for (int i = 0; i < weightData.Length; i++)
            weightData[i] = T.CreateChecked((rng.NextDouble() * 2.0 - 1.0) * double.CreateChecked(kaimingBound));

        _weight = new Parameter<T>("Weight",
            ReverseGradTensor<T>.FromMatrix(weightData, outChannels, inChannels * kernelSize * kernelSize, requiresGrad: true));
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
        if (input.Rank != 4) throw new ArgumentException($"Conv2d expects 4D input [N, C, H, W], got {input.Rank}D");
        if (input.Shape[1] != _inChannels)
            throw new ArgumentException($"Expected {_inChannels} input channels, got {input.Shape[1]}");

        int n = input.Shape[0];
        int c = input.Shape[1];
        int h = input.Shape[2];
        int w = input.Shape[3];

        int oH = (h + 2 * _padding - _kernelSize) / _stride + 1;
        int oW = (w + 2 * _padding - _kernelSize) / _stride + 1;
        if (oH <= 0 || oW <= 0)
            throw new ArgumentException($"Output dimensions are non-positive ({oH}x{oW}). Check input size, kernel, stride, and padding.");

        int kW = _kernelSize;
        int patchSize = c * kW * kW;
        int totalPatches = n * oH * oW;

        var inputData = new T[n * c * h * w];
        input.Data.CopyTo(inputData, T.Zero);

        var weightData = new T[_outChannels * patchSize];
        _weight.Tensor.Data.CopyTo(weightData, T.Zero);

        T[]? biasData = null;
        if (_useBias && _bias != null)
        {
            biasData = new T[_outChannels];
            _bias.Tensor.Data.CopyTo(biasData, T.Zero);
        }

        var outputData = new T[n * _outChannels * oH * oW];

        ConvForwardKernel(inputData, weightData, biasData, outputData,
            n, c, h, w, kW, _stride, _padding, oH, oW, patchSize, totalPatches, _outChannels);

        var outShape = new[] { n, _outChannels, oH, oW };
        var result = NivaraColumn<T>.Create(outputData);
        bool shouldTrack = GradientUtils.ShouldTrackGrad(input, _weight.Tensor);
        if (_useBias && _bias != null)
            shouldTrack = shouldTrack || _bias.Tensor.RequiresGrad;
        var resultTensor = new ReverseGradTensor<T>(result, shouldTrack, outShape);

        if (shouldTrack)
        {
            var capturedInputData = inputData;
            var capturedWeightData = weightData;
            var capturedBiasData = biasData;

            var gradFn = new OpNode<T>("Conv2d", new object[] { input, _weight.Tensor }, (gradOutput, sgn) =>
            {
                var gradOutData = new T[n * _outChannels * oH * oW];
                gradOutput.CopyTo(gradOutData, T.Zero);

                if (input.RequiresGrad)
                {
                    var inputGrad = new T[n * c * h * w];
                    ConvInputGradKernel(gradOutData, capturedWeightData, inputGrad,
                        n, c, h, w, oH, oW, kW, _stride, _padding, _outChannels, patchSize);
                    ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.Create(inputGrad), sgn);
                }

                if (_weight.Tensor.RequiresGrad)
                {
                    var weightGrad = new T[_outChannels * patchSize];
                    ConvWeightGradKernel(capturedInputData, gradOutData, weightGrad,
                        n, c, h, w, oH, oW, kW, _stride, _padding, _outChannels, patchSize, totalPatches);
                    ReverseGradOperations.AccumulateGradient(_weight.Tensor, NivaraColumn<T>.Create(weightGrad), sgn);
                }

                if (_useBias && _bias != null && _bias.Tensor.RequiresGrad)
                {
                    var biasGrad = new T[_outChannels];
                    ConvBiasGradKernel(gradOutData, biasGrad, n, _outChannels, oH, oW);
                    ReverseGradOperations.AccumulateGradient(_bias.Tensor, NivaraColumn<T>.Create(biasGrad), sgn);
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    static int ComputeTileCapacity(int patchSize)
    {
        int bytesPerElement = Unsafe.SizeOf<T>();
        int targetBytes = 32 * 1024;
        return Math.Max(1, targetBytes / (patchSize * bytesPerElement));
    }

    static void ConvForwardKernel(
        T[] inputData, T[] weightData, T[]? biasData, T[] outputData,
        int n, int c, int h, int w, int kW, int stride, int padding,
        int oH, int oW, int patchSize, int totalPatches, int outChannels)
    {
        int tileSize = Math.Min(ComputeTileCapacity(patchSize), totalPatches);

        var scratchBuf = ArrayPool<T>.Shared.Rent(tileSize * patchSize);
        try
        {
            var scratch = scratchBuf.AsSpan(0, tileSize * patchSize);

            for (int tileStart = 0; tileStart < totalPatches; tileStart += tileSize)
            {
                int tileLen = Math.Min(tileSize, totalPatches - tileStart);

                Im2Col.Im2ColTile(inputData, scratch, c, h, w, kW, kW, stride, stride, padding, padding, oH, oW, tileStart, tileLen, n);

                for (int t = 0; t < tileLen; t++)
                {
                    int globalIdx = tileStart + t;
                    int batch = globalIdx / (oH * oW);
                    int spatialIdx = globalIdx % (oH * oW);
                    int oh = spatialIdx / oW;
                    int ow = spatialIdx % oW;

                    var patchSpan = scratch.Slice(t * patchSize, patchSize);

                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        var weightSlice = weightData.AsSpan(oc * patchSize, patchSize);
                        T dot = TensorPrimitives.Dot(patchSpan, weightSlice);

                        if (biasData != null)
                            dot += biasData[oc];

                        outputData[((batch * outChannels + oc) * oH + oh) * oW + ow] = dot;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(scratchBuf, clearArray: true);
        }
    }

    static void ConvWeightGradKernel(
        T[] inputData, T[] gradOutData, T[] weightGrad,
        int n, int c, int h, int w, int oH, int oW,
        int kW, int stride, int padding, int outChannels, int patchSize, int totalPatches)
    {
        int tileSize = Math.Min(ComputeTileCapacity(patchSize), totalPatches);

        var scratchBuf = ArrayPool<T>.Shared.Rent(tileSize * patchSize);
        try
        {
            var scratch = scratchBuf.AsSpan(0, tileSize * patchSize);

            for (int tileStart = 0; tileStart < totalPatches; tileStart += tileSize)
            {
                int tileLen = Math.Min(tileSize, totalPatches - tileStart);

                Im2Col.Im2ColTile(inputData, scratch, c, h, w, kW, kW, stride, stride, padding, padding, oH, oW, tileStart, tileLen, n);

                for (int t = 0; t < tileLen; t++)
                {
                    int globalIdx = tileStart + t;
                    int batch = globalIdx / (oH * oW);
                    int spatialIdx = globalIdx % (oH * oW);

                    var patchSpan = scratch.Slice(t * patchSize, patchSize);

                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        T g = gradOutData[((batch * outChannels + oc) * oH) * oW + spatialIdx];
                        if (g == T.Zero) continue;

                        var wSlice = weightGrad.AsSpan(oc * patchSize, patchSize);
                        TensorPrimitives.MultiplyAdd(patchSpan, g, wSlice, wSlice);
                    }
                }
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(scratchBuf, clearArray: true);
        }
    }

    static void ConvInputGradKernel(
        T[] gradOutData, T[] weightData, T[] inputGrad,
        int n, int c, int h, int w, int oH, int oW,
        int kW, int stride, int padding, int outChannels, int patchSize)
    {
        for (int batch = 0; batch < n; batch++)
        {
            for (int ic = 0; ic < c; ic++)
            {
                int inputChannelBase = (batch * c + ic) * h * w;

                for (int oc = 0; oc < outChannels; oc++)
                {
                    int gradChannelBase = (batch * outChannels + oc) * oH * oW;
                    int weightChannelBase = oc * patchSize + ic * kW * kW;

                    for (int oh = 0; oh < oH; oh++)
                    {
                        int baseIH = oh * stride - padding;

                        for (int ow = 0; ow < oW; ow++)
                        {
                            T g = gradOutData[gradChannelBase + oh * oW + ow];
                            if (g == T.Zero) continue;

                            int baseIW = ow * stride - padding;

                            for (int kh = 0; kh < kW; kh++)
                            {
                                int ih = baseIH + kh;
                                if ((uint)ih >= (uint)h) continue;

                                int iRowBase = inputChannelBase + ih * w;
                                int wRowBase = weightChannelBase + kh * kW;

                                for (int kw = 0; kw < kW; kw++)
                                {
                                    int iw = baseIW + kw;
                                    if ((uint)iw >= (uint)w) continue;

                                    inputGrad[iRowBase + iw] += g * weightData[wRowBase + kw];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    static void ConvBiasGradKernel(T[] gradOutData, T[] biasGrad, int n, int outChannels, int oH, int oW)
    {
        int spatialSize = oH * oW;

        for (int oc = 0; oc < outChannels; oc++)
        {
            T sum = T.Zero;
            for (int batch = 0; batch < n; batch++)
            {
                int channelBase = batch * outChannels * spatialSize + oc * spatialSize;
                sum += T.CreateChecked(double.CreateChecked(TensorPrimitives.Sum(gradOutData.AsSpan(channelBase, spatialSize))));
            }
            biasGrad[oc] += sum;
        }
    }
}

public sealed class ConvTranspose2d<T> : Module<T> where T : struct, INumber<T>
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

    public ConvTranspose2d(
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

        int fanIn = inChannels * kernelSize * kernelSize;
        var weightData = new T[inChannels * outChannels * kernelSize * kernelSize];
        var kaimingBound = T.CreateChecked(Math.Sqrt(2.0 / fanIn));
        var rng = new Random(42);
        for (int i = 0; i < weightData.Length; i++)
            weightData[i] = T.CreateChecked((rng.NextDouble() * 2.0 - 1.0) * double.CreateChecked(kaimingBound));

        _weight = new Parameter<T>("Weight",
            ReverseGradTensor<T>.FromMatrix(weightData, inChannels, outChannels * kernelSize * kernelSize, requiresGrad: true));
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
        if (input.Rank != 4) throw new ArgumentException($"ConvTranspose2d expects 4D input [N, C, H, W], got {input.Rank}D");
        if (input.Shape[1] != _inChannels)
            throw new ArgumentException($"Expected {_inChannels} input channels, got {input.Shape[1]}");

        int n = input.Shape[0];
        int c = input.Shape[1];
        int h = input.Shape[2];
        int w = input.Shape[3];

        int oH = (h - 1) * _stride - 2 * _padding + _kernelSize;
        int oW = (w - 1) * _stride - 2 * _padding + _kernelSize;
        if (oH <= 0 || oW <= 0)
            throw new ArgumentException($"Output dimensions are non-positive ({oH}x{oW}). Check input size, kernel, stride, and padding.");

        var inputData = new T[n * c * h * w];
        input.Data.CopyTo(inputData, T.Zero);

        var weightData = new T[c * _outChannels * _kernelSize * _kernelSize];
        _weight.Tensor.Data.CopyTo(weightData, T.Zero);

        T[]? biasData = null;
        if (_useBias && _bias != null)
        {
            biasData = new T[_outChannels];
            _bias.Tensor.Data.CopyTo(biasData, T.Zero);
        }

        var outputData = new T[n * _outChannels * oH * oW];
        int kW = _kernelSize;

        Im2Col.Col2ImForward(
            inputData, weightData, outputData,
            c, _outChannels, h, w, oH, oW,
            kW, _stride, _padding, n);

        if (biasData != null)
        {
            int spatialSize = oH * oW;
            for (int batch = 0; batch < n; batch++)
            {
                int outBase = batch * _outChannels * spatialSize;
                for (int oc = 0; oc < _outChannels; oc++)
                {
                    int rowStart = outBase + oc * spatialSize;
                    TensorPrimitives.Add(outputData.AsSpan(rowStart, spatialSize), biasData[oc], outputData.AsSpan(rowStart, spatialSize));
                }
            }
        }

        var outShape = new[] { n, _outChannels, oH, oW };
        var result = NivaraColumn<T>.Create(outputData);
        bool shouldTrack = GradientUtils.ShouldTrackGrad(input, _weight.Tensor);
        if (_useBias && _bias != null)
            shouldTrack = shouldTrack || _bias.Tensor.RequiresGrad;
        var resultTensor = new ReverseGradTensor<T>(result, shouldTrack, outShape);

        if (shouldTrack)
        {
            var capturedInputData = inputData;
            var capturedWeightData = weightData;

            var gradFn = new OpNode<T>("ConvTranspose2d", new object[] { input, _weight.Tensor }, (gradOutput, sgn) =>
            {
                var gradOutData = new T[n * _outChannels * oH * oW];
                gradOutput.CopyTo(gradOutData, T.Zero);

                if (input.RequiresGrad)
                {
                    var inputGrad = new T[n * c * h * w];
                    Im2Col.ConvTransposeInputGradKernel(
                        gradOutData, capturedWeightData, inputGrad,
                        c, _outChannels, h, w, oH, oW,
                        kW, _stride, _padding, n);
                    ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.Create(inputGrad), sgn);
                }

                if (_weight.Tensor.RequiresGrad)
                {
                    var weightGrad = new T[c * _outChannels * kW * kW];
                    Im2Col.ConvTransposeWeightGradKernel(
                        capturedInputData, gradOutData, weightGrad,
                        c, _outChannels, h, w, oH, oW,
                        kW, _stride, _padding, n);
                    ReverseGradOperations.AccumulateGradient(_weight.Tensor, NivaraColumn<T>.Create(weightGrad), sgn);
                }

                if (_useBias && _bias != null && _bias.Tensor.RequiresGrad)
                {
                    var biasGrad = new T[_outChannels];
                    int spatialSize = oH * oW;
                    for (int oc = 0; oc < _outChannels; oc++)
                    {
                        T sum = T.Zero;
                        for (int batch = 0; batch < n; batch++)
                        {
                            int channelBase = batch * _outChannels * spatialSize + oc * spatialSize;
                            sum += T.CreateChecked(double.CreateChecked(TensorPrimitives.Sum(gradOutData.AsSpan(channelBase, spatialSize))));
                        }
                        biasGrad[oc] = sum;
                    }
                    ReverseGradOperations.AccumulateGradient(_bias.Tensor, NivaraColumn<T>.Create(biasGrad), sgn);
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }
}
