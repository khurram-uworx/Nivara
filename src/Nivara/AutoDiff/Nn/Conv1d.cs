using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace Nivara.AutoDiff.Nn;

public sealed class Conv1d<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    const int TargetL1Bytes = 32 * 1024;

    readonly int inChannels;
    readonly int outChannels;
    readonly int kernelSize;
    readonly int stride;
    readonly int padding;
    readonly bool useBias;

    readonly Parameter<T> weight;
    readonly Parameter<T>? bias;

    public int InChannels => inChannels;
    public int OutChannels => outChannels;
    public int KernelSize => kernelSize;
    public int Stride => stride;
    public int Padding => padding;
    public Parameter<T>? Weight => weight;
    public Parameter<T>? Bias => bias;

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

        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        useBias = bias;

        int patchSize = inChannels * kernelSize;
        int fanIn = patchSize;
        var weightData = new T[outChannels * patchSize];
        var kaimingBound = T.CreateChecked(Math.Sqrt(2.0 / fanIn));
        var rng = new Random(42);
        for (int i = 0; i < weightData.Length; i++)
            weightData[i] = T.CreateChecked((rng.NextDouble() * 2.0 - 1.0) * double.CreateChecked(kaimingBound));

        weight = new Parameter<T>("Weight",
            ReverseGradTensor<T>.FromMatrix(weightData, outChannels, patchSize, requiresGrad: true));
        RegisterParameters(weight);

        if (bias)
        {
            var biasData = new T[outChannels];
            this.bias = new Parameter<T>("Bias",
                ReverseGradTensor<T>.FromArray(biasData, requiresGrad: true));
            RegisterParameters(this.bias);
        }
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 3) throw new ArgumentException($"Conv1d expects 3D input [B, C, L], got {input.Rank}D");
        if (input.Shape[1] != inChannels)
            throw new ArgumentException($"Expected {inChannels} input channels, got {input.Shape[1]}");

        int n = input.Shape[0];
        int c = input.Shape[1];
        int l = input.Shape[2];

        int oL = (l + 2 * padding - kernelSize) / stride + 1;
        if (oL <= 0)
            throw new ArgumentException($"Output length is non-positive ({oL}). Check input size, kernel, stride, and padding.");

        int patchSize = c * kernelSize;
        int totalPatches = n * oL;

        ReadOnlySpan<T> inputSpan = input.Data.TryGetSpan(out var iSpan)
            ? iSpan
            : ModuleHelpers<T>.CopyToTemp(input.Data, n * c * l);

        ReadOnlySpan<T> weightSpan = weight.Tensor.Data.TryGetSpan(out var wSpan)
            ? wSpan
            : ModuleHelpers<T>.CopyToTemp(weight.Tensor.Data, outChannels * patchSize);

        ReadOnlySpan<T> biasSpan = ReadOnlySpan<T>.Empty;
        if (useBias && bias != null)
        {
            biasSpan = bias.Tensor.Data.TryGetSpan(out var bSpan)
                ? bSpan
                : ModuleHelpers<T>.CopyToTemp(bias.Tensor.Data, outChannels);
        }

        var outputData = new T[n * outChannels * oL];

        Conv1dForwardKernel(inputSpan, weightSpan, biasSpan, outputData,
            n, c, l, kernelSize, stride, padding, oL, patchSize, totalPatches, outChannels);

        var outShape = new[] { n, outChannels, oL };
        var result = NivaraColumn<T>.Create(outputData);
        bool shouldTrack = useBias && bias != null
            ? GradientUtils.ShouldTrackGrad(input, weight.Tensor, bias.Tensor)
            : GradientUtils.ShouldTrackGrad(input, weight.Tensor);
        var resultTensor = new ReverseGradTensor<T>(result, shouldTrack, outShape);

        if (shouldTrack)
        {
            var capturedInputData = inputSpan.ToArray();
            var capturedWeightData = weightSpan.ToArray();
            var capturedBiasData = biasSpan.Length > 0 ? biasSpan.ToArray() : null;

            var gradFn = new OpNode<T>("Conv1d", [input, weight.Tensor], (gradOutput) =>
            {
                var gradOutData = new T[n * outChannels * oL];
                gradOutput.CopyTo(gradOutData, T.Zero);

                if (input.RequiresGrad)
                {
                    var inputGrad = new T[n * c * l];
                    Conv1dInputGradKernel(gradOutData, capturedWeightData, inputGrad,
                        n, c, l, kernelSize, stride, padding, oL, outChannels, patchSize);
                    ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.Create(inputGrad));
                }

                if (weight.Tensor.RequiresGrad)
                {
                    var weightGrad = new T[outChannels * patchSize];
                    Conv1dWeightGradKernel(capturedInputData, gradOutData, weightGrad,
                        n, c, l, kernelSize, stride, padding, oL, outChannels, patchSize, totalPatches);
                    ReverseGradOperations.AccumulateGradient(weight.Tensor, NivaraColumn<T>.Create(weightGrad));
                }

                if (useBias && bias != null && bias.Tensor.RequiresGrad)
                {
                    var biasGrad = new T[outChannels];
                    Conv1dBiasGradKernel(gradOutData, biasGrad, n, outChannels, oL);
                    ReverseGradOperations.AccumulateGradient(bias.Tensor, NivaraColumn<T>.Create(biasGrad));
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    static int ComputeTileCapacity(int patchSize)
    {
        int bytesPerElement = Unsafe.SizeOf<T>();
        return Math.Max(1, TargetL1Bytes / (patchSize * bytesPerElement));
    }

    static void Conv1dForwardKernel(
        ReadOnlySpan<T> inputData, ReadOnlySpan<T> weightSpan, ReadOnlySpan<T> biasSpan, Span<T> outputData,
        int n, int c, int l, int kL, int stride, int padding,
        int oL, int patchSize, int totalPatches, int outChannels)
    {
        if (kL == 1 && stride == 1 && padding == 0)
        {
            Conv1dForward1x1(inputData, weightSpan, biasSpan, outputData, n, c, l, oL, outChannels);
            return;
        }

        int tileSize = Math.Min(ComputeTileCapacity(patchSize), totalPatches);

        var scratchBuf = ArrayPool<T>.Shared.Rent(tileSize * patchSize);
        try
        {
            var scratch = scratchBuf.AsSpan(0, tileSize * patchSize);

            for (int tileStart = 0; tileStart < totalPatches; tileStart += tileSize)
            {
                int tileLen = Math.Min(tileSize, totalPatches - tileStart);
                var locs = Im2Col.BuildPatchLocations1D(oL, tileStart, tileLen);

                Im2Col.Im2Col1DTile(inputData, scratch, c, l, kL, stride, padding, oL, tileStart, tileLen, locs);

                for (int t = 0; t < tileLen; t++)
                {
                    var loc = locs[t];
                    var patchSpan = scratch.Slice(t * patchSize, patchSize);
                    int outBase = loc.Batch * outChannels * oL + loc.OH;

                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        var weightSlice = weightSpan.Slice(oc * patchSize, patchSize);
                        T dot = TensorPrimitives.Dot(patchSpan, weightSlice);

                        if (biasSpan.Length > 0)
                            dot += biasSpan[oc];

                        outputData[outBase + oc * oL] = dot;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(scratchBuf, clearArray: true);
        }
    }

    static void Conv1dForward1x1(
        ReadOnlySpan<T> inputData, ReadOnlySpan<T> weightSpan, ReadOnlySpan<T> biasSpan, Span<T> outputData,
        int n, int c, int l, int oL, int outChannels)
    {
        int patchSize = c;

        var patchBuf = ArrayPool<T>.Shared.Rent(patchSize);
        try
        {
            var patch = patchBuf.AsSpan(0, patchSize);

            for (int batch = 0; batch < n; batch++)
            {
                int inBase = batch * c * l;
                int outBase = batch * outChannels * oL;

                for (int pos = 0; pos < oL; pos++)
                {
                    int inPixel = inBase + pos;

                    for (int ic = 0; ic < c; ic++)
                        patch[ic] = inputData[inPixel + ic * l];

                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        T dot = TensorPrimitives.Dot(patch, weightSpan.Slice(oc * patchSize, patchSize));
                        if (biasSpan.Length > 0) dot += biasSpan[oc];
                        outputData[outBase + oc * oL + pos] = dot;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(patchBuf, clearArray: true);
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

                        for (int ch = 0; ch < c; ch++)
                            inputGrad[inputBase + ch * l] += g * weightData[weightBase + ch * kL + kh];
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
        int tileSize = Math.Min(ComputeTileCapacity(patchSize), totalPatches);

        var scratchBuf = ArrayPool<T>.Shared.Rent(tileSize * patchSize);
        try
        {
            var scratch = scratchBuf.AsSpan(0, tileSize * patchSize);

            for (int tileStart = 0; tileStart < totalPatches; tileStart += tileSize)
            {
                int tileLen = Math.Min(tileSize, totalPatches - tileStart);
                var locs = Im2Col.BuildPatchLocations1D(oL, tileStart, tileLen);

                Im2Col.Im2Col1DTile(inputData, scratch, c, l, kL, stride, padding, oL, tileStart, tileLen, locs);

                for (int t = 0; t < tileLen; t++)
                {
                    var loc = locs[t];
                    var patchSpan = scratch.Slice(t * patchSize, patchSize);
                    int gradBase = loc.Batch * outChannels * oL + loc.OH;

                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        T g = gradOutData[gradBase + oc * oL];
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
