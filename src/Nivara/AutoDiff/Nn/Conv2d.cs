using System.Buffers;
using System.Runtime.CompilerServices;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Nn;

public sealed class Conv2d<T> : Module<T> where T : struct, INumber<T>
{
    const int TargetL1Bytes = 32 * 1024;

    readonly int _inChannels;
    readonly int _outChannels;
    readonly int _kernelSize;
    readonly int _stride;
    readonly int _padding;
    readonly int _groups;
    readonly bool _useBias;

    readonly Parameter<T> _weight;
    readonly Parameter<T>? _bias;

    public int InChannels => _inChannels;
    public int OutChannels => _outChannels;
    public int KernelSize => _kernelSize;
    public int Stride => _stride;
    public int Padding => _padding;
    public int Groups => _groups;
    public Parameter<T> WeightParam => _weight;
    public Parameter<T>? BiasParam => _bias;

    public Conv2d(
        int inChannels,
        int outChannels,
        int kernelSize,
        int stride = 1,
        int padding = 0,
        bool bias = true,
        int groups = 1)
    {
        if (inChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels));
        if (outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outChannels));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (padding < 0) throw new ArgumentOutOfRangeException(nameof(padding));
        if (groups <= 0) throw new ArgumentOutOfRangeException(nameof(groups));
        if (inChannels % groups != 0) throw new ArgumentException($"inChannels ({inChannels}) must be divisible by groups ({groups})");
        if (outChannels % groups != 0) throw new ArgumentException($"outChannels ({outChannels}) must be divisible by groups ({groups})");

        _inChannels = inChannels;
        _outChannels = outChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;
        _groups = groups;
        _useBias = bias;

        int cPerGroup = inChannels / groups;
        int oPerGroup = outChannels / groups;
        int fanIn = cPerGroup * kernelSize * kernelSize;
        var weightData = new T[outChannels * cPerGroup * kernelSize * kernelSize];
        var kaimingBound = T.CreateChecked(Math.Sqrt(2.0 / fanIn));
        var rng = new Random(42);
        for (int i = 0; i < weightData.Length; i++)
            weightData[i] = T.CreateChecked((rng.NextDouble() * 2.0 - 1.0) * double.CreateChecked(kaimingBound));

        _weight = new Parameter<T>("Weight",
            ReverseGradTensor<T>.FromMatrix(weightData, outChannels, cPerGroup * kernelSize * kernelSize, requiresGrad: true));
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

        ReadOnlySpan<T> inputSpan = input.Data.TryGetSpan(out var iSpan)
            ? iSpan
            : CopyToTemp(input.Data, n * c * h * w);

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

        var outputData = new T[n * _outChannels * oH * oW];

        if (_groups == 1)
        {
            ConvForwardKernel(inputSpan, weightSpan, biasSpan, outputData,
                n, c, h, w, kW, _stride, _padding, oH, oW, patchSize, totalPatches, _outChannels);
        }
        else
        {
            int spatialIn = h * w;
            int spatialOut = oH * oW;
            int cPerGroup = c / _groups;
            int oPerGroup = _outChannels / _groups;
            int groupPatchSize = cPerGroup * kW * kW;
            int groupInputSize = n * cPerGroup * spatialIn;
            int groupOutputSize = n * oPerGroup * spatialOut;

            var groupInBuf = ArrayPool<T>.Shared.Rent(groupInputSize);
            var groupOutBuf = ArrayPool<T>.Shared.Rent(groupOutputSize);
            try
            {
                var groupInput = groupInBuf.AsSpan(0, groupInputSize);
                var groupOutput = groupOutBuf.AsSpan(0, groupOutputSize);

                for (int g = 0; g < _groups; g++)
                {
                    for (int b = 0; b < n; b++)
                        inputSpan.Slice((b * c + g * cPerGroup) * spatialIn, cPerGroup * spatialIn)
                            .CopyTo(groupInput.Slice(b * cPerGroup * spatialIn));

                    var weightGroup = weightSpan.Slice(g * oPerGroup * groupPatchSize, oPerGroup * groupPatchSize);
                    var biasGroup = biasSpan.IsEmpty ? ReadOnlySpan<T>.Empty : biasSpan.Slice(g * oPerGroup, oPerGroup);
                    groupOutput.Clear();

                    ConvForwardKernel(groupInput, weightGroup, biasGroup, groupOutput,
                        n, cPerGroup, h, w, kW, _stride, _padding, oH, oW, groupPatchSize, totalPatches, oPerGroup);

                    for (int b = 0; b < n; b++)
                        groupOutput.Slice(b * oPerGroup * spatialOut, oPerGroup * spatialOut)
                            .CopyTo(outputData.AsSpan((b * _outChannels + g * oPerGroup) * spatialOut));
                }
            }
            finally
            {
                ArrayPool<T>.Shared.Return(groupInBuf, clearArray: true);
                ArrayPool<T>.Shared.Return(groupOutBuf, clearArray: true);
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
            var capturedInputData = inputSpan.ToArray();
            var capturedWeightData = weightSpan.ToArray();
            var capturedBiasData = biasSpan.Length > 0 ? biasSpan.ToArray() : null;

            var gradFn = new OpNode<T>("Conv2d", new object[] { input, _weight.Tensor }, (gradOutput, sgn) =>
            {
                var gradOutData = new T[n * _outChannels * oH * oW];
                gradOutput.CopyTo(gradOutData, T.Zero);

                if (_groups == 1)
                {
                    if (input.RequiresGrad)
                    {
                        var inputGrad = new T[n * c * h * w];
                        ConvInputGradKernel(gradOutData, capturedWeightData, inputGrad,
                            n, c, h, w, oH, oW, kW, _stride, _padding, _outChannels, patchSize);
                        ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.Create(inputGrad));
                    }

                    if (_weight.Tensor.RequiresGrad)
                    {
                        var weightGrad = new T[_outChannels * patchSize];
                        ConvWeightGradKernel(capturedInputData, gradOutData, weightGrad,
                            n, c, h, w, oH, oW, kW, _stride, _padding, _outChannels, patchSize, totalPatches);
                        ReverseGradOperations.AccumulateGradient(_weight.Tensor, NivaraColumn<T>.Create(weightGrad));
                    }
                }
                else
                {
                    int spatialInLocal = h * w;
                    int spatialOutLocal = oH * oW;
                    int cPerGroupLocal = c / _groups;
                    int oPerGroupLocal = _outChannels / _groups;
                    int groupPatchSizeLocal = cPerGroupLocal * kW * kW;
                    int groupInputSizeLocal = n * cPerGroupLocal * spatialInLocal;
                    int groupOutputSizeLocal = n * oPerGroupLocal * spatialOutLocal;

                    var groupInGradBuf = ArrayPool<T>.Shared.Rent(groupInputSizeLocal);
                    var groupOutGradBuf = ArrayPool<T>.Shared.Rent(groupOutputSizeLocal);
                    try
                    {
                        var groupInGrad = groupInGradBuf.AsSpan(0, groupInputSizeLocal);
                        var groupOutGrad = groupOutGradBuf.AsSpan(0, groupOutputSizeLocal);

                        if (input.RequiresGrad)
                        {
                            var inputGrad = new T[n * c * h * w];

                            for (int g = 0; g < _groups; g++)
                            {
                                for (int b = 0; b < n; b++)
                                    gradOutData.AsSpan((b * _outChannels + g * oPerGroupLocal) * spatialOutLocal, oPerGroupLocal * spatialOutLocal)
                                        .CopyTo(groupOutGrad.Slice(b * oPerGroupLocal * spatialOutLocal));

                                var weightGroup = capturedWeightData.AsSpan(g * oPerGroupLocal * groupPatchSizeLocal, oPerGroupLocal * groupPatchSizeLocal);
                                groupInGrad.Clear();

                                ConvInputGradKernel(groupOutGrad, weightGroup, groupInGrad,
                                    n, cPerGroupLocal, h, w, oH, oW, kW, _stride, _padding, oPerGroupLocal, groupPatchSizeLocal);

                                for (int b = 0; b < n; b++)
                                    groupInGrad.Slice(b * cPerGroupLocal * spatialInLocal, cPerGroupLocal * spatialInLocal)
                                        .CopyTo(inputGrad.AsSpan((b * c + g * cPerGroupLocal) * spatialInLocal));
                            }

                            ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.Create(inputGrad));
                        }

                        if (_weight.Tensor.RequiresGrad)
                        {
                            var weightGrad = new T[_outChannels * cPerGroupLocal * kW * kW];

                            for (int g = 0; g < _groups; g++)
                            {
                                var inputGroupBuf = ArrayPool<T>.Shared.Rent(groupInputSizeLocal);
                                try
                                {
                                    var inputGroup = inputGroupBuf.AsSpan(0, groupInputSizeLocal);
                                    for (int b = 0; b < n; b++)
                                        capturedInputData.AsSpan((b * c + g * cPerGroupLocal) * spatialInLocal, cPerGroupLocal * spatialInLocal)
                                            .CopyTo(inputGroup.Slice(b * cPerGroupLocal * spatialInLocal));

                                    for (int b = 0; b < n; b++)
                                        gradOutData.AsSpan((b * _outChannels + g * oPerGroupLocal) * spatialOutLocal, oPerGroupLocal * spatialOutLocal)
                                            .CopyTo(groupOutGrad.Slice(b * oPerGroupLocal * spatialOutLocal));

                                    var weightGradGroup = weightGrad.AsSpan(g * oPerGroupLocal * groupPatchSizeLocal, oPerGroupLocal * groupPatchSizeLocal);

                                    ConvWeightGradKernel(inputGroup, groupOutGrad, weightGradGroup,
                                        n, cPerGroupLocal, h, w, oH, oW, kW, _stride, _padding, oPerGroupLocal, groupPatchSizeLocal, totalPatches);
                                }
                                finally
                                {
                                    ArrayPool<T>.Shared.Return(inputGroupBuf, clearArray: true);
                                }
                            }

                            ReverseGradOperations.AccumulateGradient(_weight.Tensor, NivaraColumn<T>.Create(weightGrad));
                        }
                    }
                    finally
                    {
                        ArrayPool<T>.Shared.Return(groupInGradBuf, clearArray: true);
                        ArrayPool<T>.Shared.Return(groupOutGradBuf, clearArray: true);
                    }
                }

                if (_useBias && _bias != null && _bias.Tensor.RequiresGrad)
                {
                    var biasGrad = new T[_outChannels];
                    ConvBiasGradKernel(gradOutData, biasGrad, n, _outChannels, oH, oW);
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

    static int ComputeTileCapacity(int patchSize)
    {
        int bytesPerElement = Unsafe.SizeOf<T>();
        return Math.Max(1, TargetL1Bytes / (patchSize * bytesPerElement));
    }

    static void ConvForwardKernel(
        ReadOnlySpan<T> inputData, ReadOnlySpan<T> weightSpan, ReadOnlySpan<T> biasSpan, Span<T> outputData,
        int n, int c, int h, int w, int kW, int stride, int padding,
        int oH, int oW, int patchSize, int totalPatches, int outChannels)
    {
        if (kW == 1 && stride == 1 && padding == 0)
        {
            ConvForward1x1(inputData, weightSpan, biasSpan, outputData, n, c, h, w, oH, oW, outChannels);
            return;
        }

        int tileSize = Math.Min(ComputeTileCapacity(patchSize), totalPatches);
        int positionsPerBatch = oH * oW;

        var scratchBuf = ArrayPool<T>.Shared.Rent(tileSize * patchSize);
        try
        {
            var scratch = scratchBuf.AsSpan(0, tileSize * patchSize);

            for (int tileStart = 0; tileStart < totalPatches; tileStart += tileSize)
            {
                int tileLen = Math.Min(tileSize, totalPatches - tileStart);
                var locs = Im2Col.BuildPatchLocations(positionsPerBatch, oW, tileStart, tileLen);

                Im2Col.Im2ColTile(inputData, scratch, c, h, w, kW, kW, stride, stride, padding, padding, oH, oW, tileStart, tileLen, locs);

                for (int t = 0; t < tileLen; t++)
                {
                    var loc = locs[t];
                    var patchSpan = scratch.Slice(t * patchSize, patchSize);
                    int outBase = (loc.Batch * outChannels) * positionsPerBatch + loc.OH * oW + loc.OW;

                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        var weightSlice = weightSpan.Slice(oc * patchSize, patchSize);
                        T dot = TensorPrimitives.Dot(patchSpan, weightSlice);

                        if (biasSpan.Length > 0)
                            dot += biasSpan[oc];

                        outputData[outBase + oc * positionsPerBatch] = dot;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(scratchBuf, clearArray: true);
        }
    }

    static void ConvForward1x1(
        ReadOnlySpan<T> inputData, ReadOnlySpan<T> weightSpan, ReadOnlySpan<T> biasSpan, Span<T> outputData,
        int n, int c, int h, int w, int oH, int oW, int outChannels)
    {
        int spatialIn = h * w;
        int spatialOut = oH * oW;
        int patchSize = c;

        var patchBuf = ArrayPool<T>.Shared.Rent(patchSize);
        try
        {
            var patch = patchBuf.AsSpan(0, patchSize);

            for (int batch = 0; batch < n; batch++)
            {
                int inBase = batch * c * spatialIn;
                int outBase = batch * outChannels * spatialOut;

                for (int oh = 0; oh < oH; oh++)
                {
                    int inRowBase = inBase + oh * w;
                    int outRowBase = outBase + oh * oW;

                    for (int ow = 0; ow < oW; ow++)
                    {
                        int inPixel = inRowBase + ow;
                        int outPos = outRowBase + ow;

                        for (int ic = 0; ic < c; ic++)
                            patch[ic] = inputData[inPixel + ic * spatialIn];

                        for (int oc = 0; oc < outChannels; oc++)
                        {
                            T dot = TensorPrimitives.Dot(patch, weightSpan.Slice(oc * c, c));
                            if (biasSpan.Length > 0) dot += biasSpan[oc];
                            outputData[outPos + oc * spatialOut] = dot;
                        }
                    }
                }
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(patchBuf, clearArray: true);
        }
    }

    static void ConvWeightGradKernel(
        ReadOnlySpan<T> inputData, ReadOnlySpan<T> gradOutData, Span<T> weightGrad,
        int n, int c, int h, int w, int oH, int oW,
        int kW, int stride, int padding, int outChannels, int patchSize, int totalPatches)
    {
        int tileSize = Math.Min(ComputeTileCapacity(patchSize), totalPatches);
        int positionsPerBatch = oH * oW;

        var scratchBuf = ArrayPool<T>.Shared.Rent(tileSize * patchSize);
        try
        {
            var scratch = scratchBuf.AsSpan(0, tileSize * patchSize);

            for (int tileStart = 0; tileStart < totalPatches; tileStart += tileSize)
            {
                int tileLen = Math.Min(tileSize, totalPatches - tileStart);
                var locs = Im2Col.BuildPatchLocations(positionsPerBatch, oW, tileStart, tileLen);

                Im2Col.Im2ColTile(inputData, scratch, c, h, w, kW, kW, stride, stride, padding, padding, oH, oW, tileStart, tileLen, locs);

                for (int t = 0; t < tileLen; t++)
                {
                    var loc = locs[t];
                    var patchSpan = scratch.Slice(t * patchSize, patchSize);
                    int gradBase = (loc.Batch * outChannels) * positionsPerBatch + loc.OH * oW + loc.OW;

                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        T g = gradOutData[gradBase + oc * positionsPerBatch];
                        if (g == T.Zero) continue;

                        var wSlice = weightGrad.Slice(oc * patchSize, patchSize);
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
        ReadOnlySpan<T> gradOutData, ReadOnlySpan<T> weightData, Span<T> inputGrad,
        int n, int c, int h, int w, int oH, int oW,
        int kW, int stride, int padding, int outChannels, int patchSize)
    {
        if (kW == 1 && stride == 1 && padding == 0)
        {
            ConvInputGrad1x1(gradOutData, weightData, inputGrad, n, c, h, w, oH, oW, outChannels);
            return;
        }

        if (kW == 3 && stride == 1 && padding == 1)
        {
            ConvInputGrad3x3(gradOutData, weightData, inputGrad, n, c, h, w, oH, oW, outChannels);
            return;
        }

        ConvInputGradGeneric(gradOutData, weightData, inputGrad,
            n, c, h, w, oH, oW, kW, stride, padding, outChannels, patchSize);
    }

    static void ConvInputGrad1x1(
        ReadOnlySpan<T> gradOutData, ReadOnlySpan<T> weightData, Span<T> inputGrad,
        int n, int c, int h, int w, int oH, int oW, int outChannels)
    {
        int spatialIn = h * w;
        int spatialOut = oH * oW;

        for (int batch = 0; batch < n; batch++)
        {
            int inBase = batch * c * spatialIn;
            int outBase = batch * outChannels * spatialOut;

            for (int oc = 0; oc < outChannels; oc++)
            {
                var gradSlice = gradOutData.Slice(outBase + oc * spatialOut, spatialOut);

                for (int ic = 0; ic < c; ic++)
                {
                    T weightVal = weightData[oc * c + ic];
                    if (weightVal == T.Zero) continue;

                    var inSlice = inputGrad.Slice(inBase + ic * spatialIn, spatialIn);
                    TensorPrimitives.MultiplyAdd(gradSlice, weightVal, inSlice, inSlice);
                }
            }
        }
    }

    static void ConvInputGrad3x3(
        ReadOnlySpan<T> gradOutData, ReadOnlySpan<T> weightData, Span<T> inputGrad,
        int n, int c, int h, int w, int oH, int oW, int outChannels)
    {
        int patchSize = c * 9;

        for (int batch = 0; batch < n; batch++)
        {
            int outBase = batch * outChannels * oH * oW;
            int inBase = batch * c * h * w;

            for (int oc = 0; oc < outChannels; oc++)
            {
                int gradChBase = outBase + oc * oH * oW;

                for (int ic = 0; ic < c; ic++)
                {
                    int inputChBase = inBase + ic * h * w;
                    int weightBase = oc * patchSize + ic * 9;

                    for (int oh = 0; oh < oH; oh++)
                    {
                        int baseIH = oh;

                        for (int ow = 0; ow < oW; ow++)
                        {
                            T g = gradOutData[gradChBase + oh * oW + ow];
                            if (g == T.Zero) continue;

                            int iRow0 = inputChBase + baseIH * w + ow;
                            int wRow0 = weightBase;

                            for (int kh = 0; kh < 3; kh++)
                            {
                                int ih = baseIH + kh;
                                if ((uint)ih >= (uint)h) continue;

                                int iBase = inputChBase + ih * w;
                                int wBase = wRow0 + kh * 3;

                                for (int kw = 0; kw < 3; kw++)
                                {
                                    int iw = ow + kw;
                                    if ((uint)iw >= (uint)w) continue;

                                    inputGrad[iBase + iw] += g * weightData[wBase + kw];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    static void ConvInputGradGeneric(
        ReadOnlySpan<T> gradOutData, ReadOnlySpan<T> weightData, Span<T> inputGrad,
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

    static void ConvBiasGradKernel(ReadOnlySpan<T> gradOutData, Span<T> biasGrad, int n, int outChannels, int oH, int oW)
    {
        int spatialSize = oH * oW;

        for (int oc = 0; oc < outChannels; oc++)
        {
            T sum = T.Zero;
            for (int batch = 0; batch < n; batch++)
            {
                int channelBase = batch * outChannels * spatialSize + oc * spatialSize;
                sum += T.CreateChecked(double.CreateChecked(TensorPrimitives.Sum(gradOutData.Slice(channelBase, spatialSize))));
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

        ReadOnlySpan<T> inputSpan = input.Data.TryGetSpan(out var iSpan)
            ? iSpan
            : CopyToTemp(input.Data, n * c * h * w);

        ReadOnlySpan<T> weightSpan = _weight.Tensor.Data.TryGetSpan(out var wSpan)
            ? wSpan
            : CopyToTemp(_weight.Tensor.Data, c * _outChannels * _kernelSize * _kernelSize);

        ReadOnlySpan<T> biasSpan = ReadOnlySpan<T>.Empty;
        if (_useBias && _bias != null)
        {
            biasSpan = _bias.Tensor.Data.TryGetSpan(out var bSpan)
                ? bSpan
                : CopyToTemp(_bias.Tensor.Data, _outChannels);
        }

        var outputData = new T[n * _outChannels * oH * oW];
        int kW = _kernelSize;

        Im2Col.Col2ImForward(
            inputSpan, weightSpan, outputData,
            c, _outChannels, h, w, oH, oW,
            kW, _stride, _padding, n);

        if (biasSpan.Length > 0)
        {
            int spatialSize = oH * oW;
            for (int batch = 0; batch < n; batch++)
            {
                int outBase = batch * _outChannels * spatialSize;
                for (int oc = 0; oc < _outChannels; oc++)
                {
                    int rowStart = outBase + oc * spatialSize;
                    TensorPrimitives.Add(outputData.AsSpan(rowStart, spatialSize), biasSpan[oc], outputData.AsSpan(rowStart, spatialSize));
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
            var capturedInputData = inputSpan.ToArray();
            var capturedWeightData = weightSpan.ToArray();

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
                    ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.Create(inputGrad));
                }

                if (_weight.Tensor.RequiresGrad)
                {
                    var weightGrad = new T[c * _outChannels * kW * kW];
                    Im2Col.ConvTransposeWeightGradKernel(
                        capturedInputData, gradOutData, weightGrad,
                        c, _outChannels, h, w, oH, oW,
                        kW, _stride, _padding, n);
                    ReverseGradOperations.AccumulateGradient(_weight.Tensor, NivaraColumn<T>.Create(weightGrad));
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
}
