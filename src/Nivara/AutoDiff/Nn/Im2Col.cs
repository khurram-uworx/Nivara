using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Span-based Im2Col/Col2Im helpers for convolution.
/// No allocations — caller owns all storage.
/// Follows: sequential writes, interior/border separation, specialized common kernels.
/// </summary>
internal static class Im2Col
{
    internal readonly struct PatchLocation
    {
        public readonly int Batch;
        public readonly int OH;
        public readonly int OW;

        public PatchLocation(int batch, int oh, int ow)
        {
            Batch = batch;
            OH = oh;
            OW = ow;
        }
    }

    internal static PatchLocation[] BuildPatchLocations(int positionsPerBatch, int outW, int tileStart, int tileLen)
    {
        var locs = new PatchLocation[tileLen];
        for (int t = 0; t < tileLen; t++)
        {
            int globalIdx = tileStart + t;
            int batch = globalIdx / positionsPerBatch;
            int spatial = globalIdx % positionsPerBatch;
            locs[t] = new PatchLocation(batch, spatial / outW, spatial % outW);
        }
        return locs;
    }

    /// <summary>
    /// Dispatches to the optimal Im2Col kernel based on kernel size.
    /// Builds patches for output positions [tileStart, tileStart+tileLen) only.
    /// Input: [N, C, H, W] flat row-major.
    /// Output: [tileLen, C * kH * kW] flat row-major, one row per receptive-field patch.
    /// </summary>
    public static void Im2ColTile<T>(
        ReadOnlySpan<T> input,
        Span<T> output,
        int channels, int height, int width,
        int kernelH, int kernelW,
        int strideH, int strideW,
        int padH, int padW,
        int outH, int outW,
        int tileStart, int tileLen,
        PatchLocation[] locs) where T : struct, IFloatingPointIeee754<T>
    {
        if (kernelH == 1 && kernelW == 1 && strideH == 1 && strideW == 1 && padH == 0 && padW == 0)
        {
            Im2ColTile1x1(input, output, channels, height, width, outH, outW, tileStart, tileLen, locs);
            return;
        }

        if (kernelH == 3 && kernelW == 3)
        {
            Im2ColTile3x3(input, output, channels, height, width, strideH, strideW, padH, padW, outH, outW, tileStart, tileLen, locs);
            return;
        }

        Im2ColTileGeneric(input, output, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW, outH, outW, tileStart, tileLen, locs);
    }

    /// <summary>
    /// Builds ALL patches for all batches. Caller must ensure output has N * oH * oW * patchSize elements.
    /// Used when the full im2col fits in cache and is reused across many filters.
    /// </summary>
    public static void Im2ColFull<T>(
        ReadOnlySpan<T> input,
        Span<T> output,
        int channels, int height, int width,
        int kernelH, int kernelW,
        int strideH, int strideW,
        int padH, int padW,
        int outH, int outW,
        int batchCount) where T : struct, IFloatingPointIeee754<T>
    {
        int totalPatches = batchCount * outH * outW;
        int positionsPerBatch = outH * outW;
        var locs = BuildPatchLocations(positionsPerBatch, outW, 0, totalPatches);
        Im2ColTile(input, output, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW, outH, outW, 0, totalPatches, locs);
    }

    public static void Im2ColTileGeneric<T>(
        ReadOnlySpan<T> input, Span<T> output,
        int channels, int height, int width,
        int kernelH, int kernelW,
        int strideH, int strideW,
        int padH, int padW,
        int outH, int outW,
        int tileStart, int tileLen,
        PatchLocation[] locs) where T : struct, IFloatingPointIeee754<T>
    {
        int patchSize = channels * kernelH * kernelW;

        for (int t = 0; t < tileLen; t++)
        {
            var loc = locs[t];
            int inBase = loc.Batch * channels * height * width;
            int baseH = loc.OH * strideH - padH;
            int baseW = loc.OW * strideW - padW;
            int outRow = t * patchSize;

            CopyPatch(input, output, inBase, outRow, channels, height, width, kernelH, kernelW, baseH, baseW);
        }
    }

    public static void Im2ColTile1x1<T>(
        ReadOnlySpan<T> input, Span<T> output,
        int channels, int height, int width,
        int outH, int outW,
        int tileStart, int tileLen,
        PatchLocation[] locs) where T : struct, IFloatingPointIeee754<T>
    {
        for (int t = 0; t < tileLen; t++)
        {
            var loc = locs[t];
            int inBase = loc.Batch * channels * height * width;
            int srcPixel = inBase + loc.OH * width + loc.OW;
            int dstIdx = t * channels;

            for (int ic = 0; ic < channels; ic++)
                output[dstIdx + ic] = input[srcPixel + ic * height * width];
        }
    }

    public static void Im2ColTile3x3<T>(
        ReadOnlySpan<T> input, Span<T> output,
        int channels, int height, int width,
        int strideH, int strideW,
        int padH, int padW,
        int outH, int outW,
        int tileStart, int tileLen,
        PatchLocation[] locs) where T : struct, IFloatingPointIeee754<T>
    {
        int patchSize = channels * 9;

        int interiorStartH = (padH + strideH - 1) / strideH;
        int interiorEndH = Math.Min(outH, (height - 3 + padH) / strideH + 1);
        int interiorStartW = (padW + strideW - 1) / strideW;
        int interiorEndW = Math.Min(outW, (width - 3 + padW) / strideW + 1);

        for (int t = 0; t < tileLen; t++)
        {
            var loc = locs[t];
            int inBase = loc.Batch * channels * height * width;
            int baseH = loc.OH * strideH - padH;
            int baseW = loc.OW * strideW - padW;
            int outRow = t * patchSize;

            bool interior = loc.OH >= interiorStartH && loc.OH < interiorEndH
                         && loc.OW >= interiorStartW && loc.OW < interiorEndW;

            if (interior)
                CopyPatch3x3Interior(input, output, inBase, outRow, channels, height, width, baseH, baseW);
            else
                CopyPatch3x3Border(input, output, inBase, outRow, channels, height, width, baseH, baseW);
        }
    }

    static void CopyPatch3x3Interior<T>(
        ReadOnlySpan<T> input, Span<T> output,
        int inBase, int outRow,
        int channels, int height, int width,
        int baseH, int baseW) where T : struct, IFloatingPointIeee754<T>
    {
        int patchIdx = 0;

        for (int ic = 0; ic < channels; ic++)
        {
            int cOff = inBase + ic * height * width;
            int r0 = cOff + baseH * width + baseW;
            int r1 = r0 + width;
            int r2 = r1 + width;

            output[outRow + patchIdx] = input[r0];
            output[outRow + patchIdx + 1] = input[r0 + 1];
            output[outRow + patchIdx + 2] = input[r0 + 2];
            output[outRow + patchIdx + 3] = input[r1];
            output[outRow + patchIdx + 4] = input[r1 + 1];
            output[outRow + patchIdx + 5] = input[r1 + 2];
            output[outRow + patchIdx + 6] = input[r2];
            output[outRow + patchIdx + 7] = input[r2 + 1];
            output[outRow + patchIdx + 8] = input[r2 + 2];
            patchIdx += 9;
        }
    }

    static void CopyPatch3x3Border<T>(
        ReadOnlySpan<T> input, Span<T> output,
        int inBase, int outRow,
        int channels, int height, int width,
        int baseH, int baseW) where T : struct, IFloatingPointIeee754<T>
    {
        int patchIdx = 0;

        for (int ic = 0; ic < channels; ic++)
        {
            int cOff = inBase + ic * height * width;

            for (int kh = 0; kh < 3; kh++)
            {
                int ih = baseH + kh;
                bool ihOk = (uint)ih < (uint)height;

                for (int kw = 0; kw < 3; kw++)
                {
                    int iw = baseW + kw;
                    output[outRow + patchIdx++] = (ihOk && (uint)iw < (uint)width)
                        ? input[cOff + ih * width + iw]
                        : T.Zero;
                }
            }
        }
    }

    static void CopyPatch<T>(
        ReadOnlySpan<T> input, Span<T> output,
        int inBase, int outRow,
        int channels, int height, int width,
        int kernelH, int kernelW,
        int baseH, int baseW) where T : struct, IFloatingPointIeee754<T>
    {
        int patchIdx = 0;

        for (int ic = 0; ic < channels; ic++)
        {
            int cOff = inBase + ic * height * width;

            for (int kh = 0; kh < kernelH; kh++)
            {
                int ih = baseH + kh;

                if ((uint)ih >= (uint)height)
                {
                    output.Slice(outRow + patchIdx, kernelW).Fill(T.Zero);
                    patchIdx += kernelW;
                    continue;
                }

                int rowStart = cOff + ih * width;

                if ((uint)baseW < (uint)(width - kernelW + 1))
                {
                    input.Slice(rowStart + baseW, kernelW).CopyTo(output.Slice(outRow + patchIdx));
                    patchIdx += kernelW;
                }
                else
                {
                    for (int kw = 0; kw < kernelW; kw++)
                    {
                        int iw = baseW + kw;
                        output[outRow + patchIdx++] = (uint)iw < (uint)width
                            ? input[rowStart + iw]
                            : T.Zero;
                    }
                }
            }
        }
    }

    public static void Col2ImWithWeight<T>(
        ReadOnlySpan<T> gradPatches,
        ReadOnlySpan<T> weight,
        Span<T> gradInput,
        int channels, int height, int width,
        int kernelH, int kernelW,
        int strideH, int strideW,
        int padH, int padW,
        int outH, int outW,
        int outChannels,
        int batchCount) where T : struct, IFloatingPointIeee754<T>
    {
        int patchSize = channels * kernelH * kernelW;
        int inputBatchStride = channels * height * width;

        for (int batch = 0; batch < batchCount; batch++)
        {
            int inBase = batch * inputBatchStride;
            int outBase = batch * outH * outW;

            for (int oh = 0; oh < outH; oh++)
            {
                int baseH = oh * strideH - padH;

                for (int ow = 0; ow < outW; ow++)
                {
                    int baseW = ow * strideW - padW;
                    int patchRow = (outBase + oh * outW + ow) * patchSize;
                    int patchIdx = 0;

                    for (int ic = 0; ic < channels; ic++)
                    {
                        int cOff = inBase + ic * height * width;

                        for (int kh = 0; kh < kernelH; kh++)
                        {
                            int ih = baseH + kh;
                            if ((uint)ih >= (uint)height) { patchIdx += kernelW; continue; }
                            int rOff = cOff + ih * width;

                            for (int kw = 0; kw < kernelW; kw++)
                            {
                                int iw = baseW + kw;
                                if ((uint)iw >= (uint)width) { patchIdx++; continue; }

                                T g = gradPatches[patchRow + patchIdx++];
                                if (g == T.Zero) continue;

                                int wBase = ic * kernelH * kernelW + kh * kernelW + kw;
                                int dst = rOff + iw;
                                for (int oc = 0; oc < outChannels; oc++)
                                    gradInput[dst] += g * weight[oc * patchSize + wBase];
                            }
                        }
                    }
                }
            }
        }
    }

    public static void Col2ImScatter<T>(
        ReadOnlySpan<T> gradPatches,
        Span<T> gradInput,
        int channels, int height, int width,
        int kernelH, int kernelW,
        int strideH, int strideW,
        int padH, int padW,
        int outH, int outW,
        int batchCount) where T : struct, IFloatingPointIeee754<T>
    {
        int patchSize = channels * kernelH * kernelW;
        int inputBatchStride = channels * height * width;

        for (int batch = 0; batch < batchCount; batch++)
        {
            int inBase = batch * inputBatchStride;
            int outBase = batch * outH * outW;

            for (int oh = 0; oh < outH; oh++)
            {
                int baseH = oh * strideH - padH;

                for (int ow = 0; ow < outW; ow++)
                {
                    int baseW = ow * strideW - padW;
                    int patchRow = (outBase + oh * outW + ow) * patchSize;
                    int patchIdx = 0;

                    for (int ic = 0; ic < channels; ic++)
                    {
                        int cOff = inBase + ic * height * width;

                        for (int kh = 0; kh < kernelH; kh++)
                        {
                            int ih = baseH + kh;
                            if ((uint)ih >= (uint)height) { patchIdx += kernelW; continue; }
                            int rOff = cOff + ih * width;

                            for (int kw = 0; kw < kernelW; kw++)
                            {
                                int iw = baseW + kw;
                                if ((uint)iw >= (uint)width) { patchIdx++; continue; }

                                gradInput[rOff + iw] += gradPatches[patchRow + patchIdx++];
                            }
                        }
                    }
                }
            }
        }
    }

    public static void WeightGradFromPatches<T>(
        ReadOnlySpan<T> patches,
        ReadOnlySpan<T> gradOutputFlat,
        Span<T> weightGrad,
        int tileStart, int tileLen,
        int patchSize, int outChannels) where T : struct, IFloatingPointIeee754<T>
    {
        for (int t = 0; t < tileLen; t++)
        {
            int globalIdx = tileStart + t;
            var patchSpan = patches.Slice(t * patchSize, patchSize);

            for (int oc = 0; oc < outChannels; oc++)
            {
                T g = gradOutputFlat[globalIdx * outChannels + oc];
                if (g == T.Zero) continue;

                var wSlice = weightGrad.Slice(oc * patchSize, patchSize);
                TensorPrimitives.MultiplyAdd(patchSpan, g, wSlice, wSlice);
            }
        }
    }

    public static void Col2ImForward<T>(
        ReadOnlySpan<T> input,
        ReadOnlySpan<T> weight,
        Span<T> output,
        int inChannels, int outChannels,
        int h, int w, int oH, int oW,
        int kW, int stride, int padding,
        int batchCount) where T : struct, IFloatingPointIeee754<T>
    {
        int inputBatchStride = inChannels * h * w;
        int outputBatchStride = outChannels * oH * oW;
        int weightChannelStride = outChannels * kW * kW;

        for (int batch = 0; batch < batchCount; batch++)
        {
            int inBase = batch * inputBatchStride;
            int outBase = batch * outputBatchStride;

            for (int ic = 0; ic < inChannels; ic++)
            {
                int inChBase = inBase + ic * h * w;

                for (int oc = 0; oc < outChannels; oc++)
                {
                    int wChBase = ic * weightChannelStride + oc * kW * kW;

                    for (int kh = 0; kh < kW; kh++)
                    {
                        int ohStart = kh - padding;
                        if (ohStart < 0) ohStart += ((-ohStart + stride - 1) / stride) * stride;
                        int wRowBase = wChBase + kh * kW;

                        for (int ih = ohStart / stride; ih < h; ih++)
                        {
                            int oh = ih * stride - padding + kh;
                            if ((uint)oh >= (uint)oH) break;

                            int inRowBase = inChBase + ih * w;
                            int outRowBase = outBase + oc * oH * oW + oh * oW;

                            for (int kw = 0; kw < kW; kw++)
                            {
                                T wVal = weight[wRowBase + kw];
                                if (wVal == T.Zero) continue;

                                int owStart = kw - padding;
                                if (owStart < 0) owStart += ((-owStart + stride - 1) / stride) * stride;

                                for (int iw = owStart / stride; iw < w; iw++)
                                {
                                    int ow = iw * stride - padding + kw;
                                    if ((uint)ow >= (uint)oW) break;

                                    output[outRowBase + ow] += input[inRowBase + iw] * wVal;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    public static void ConvTransposeInputGradKernel<T>(
        ReadOnlySpan<T> gradOutData,
        ReadOnlySpan<T> weight,
        Span<T> inputGrad,
        int inChannels, int outChannels,
        int h, int w, int oH, int oW,
        int kW, int stride, int padding,
        int batchCount) where T : struct, IFloatingPointIeee754<T>
    {
        int inputBatchStride = inChannels * h * w;
        int outputBatchStride = outChannels * oH * oW;
        int weightChannelStride = outChannels * kW * kW;

        for (int batch = 0; batch < batchCount; batch++)
        {
            int inBase = batch * inputBatchStride;
            int outBase = batch * outputBatchStride;

            for (int oc = 0; oc < outChannels; oc++)
            {
                int outChBase = outBase + oc * oH * oW;

                for (int ic = 0; ic < inChannels; ic++)
                {
                    int inChBase = inBase + ic * h * w;
                    int wBase = ic * weightChannelStride + oc * kW * kW;

                    for (int oh = 0; oh < oH; oh++)
                    {
                        int baseIH = oh + padding;

                        for (int ow = 0; ow < oW; ow++)
                        {
                            T g = gradOutData[outChBase + oh * oW + ow];
                            if (g == T.Zero) continue;

                            int baseIW = ow + padding;

                            for (int kh = 0; kh < kW; kh++)
                            {
                                int ih = baseIH - kh;
                                if (ih % stride != 0) continue;
                                ih /= stride;
                                if ((uint)ih >= (uint)h) continue;

                                int iRowBase = inChBase + ih * w;
                                int wRowBase = wBase + kh * kW;

                                for (int kw = 0; kw < kW; kw++)
                                {
                                    int iw = baseIW - kw;
                                    if (iw % stride != 0) continue;
                                    iw /= stride;
                                    if ((uint)iw >= (uint)w) continue;

                                    inputGrad[iRowBase + iw] += g * weight[wRowBase + kw];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    public static void ConvTransposeWeightGradKernel<T>(
        ReadOnlySpan<T> input,
        ReadOnlySpan<T> gradOutData,
        Span<T> weightGrad,
        int inChannels, int outChannels,
        int h, int w, int oH, int oW,
        int kW, int stride, int padding,
        int batchCount) where T : struct, IFloatingPointIeee754<T>
    {
        int inputBatchStride = inChannels * h * w;
        int outputBatchStride = outChannels * oH * oW;
        int weightChannelStride = outChannels * kW * kW;

        for (int batch = 0; batch < batchCount; batch++)
        {
            int inBase = batch * inputBatchStride;
            int outBase = batch * outputBatchStride;

            for (int ic = 0; ic < inChannels; ic++)
            {
                int inChBase = inBase + ic * h * w;

                for (int oc = 0; oc < outChannels; oc++)
                {
                    int outChBase = outBase + oc * oH * oW;
                    int wBase = ic * weightChannelStride + oc * kW * kW;

                    for (int kh = 0; kh < kW; kh++)
                    {
                        for (int kw = 0; kw < kW; kw++)
                        {
                            T wg = T.Zero;

                            for (int ih = 0; ih < h; ih++)
                            {
                                int oh = ih * stride - padding + kh;
                                if ((uint)oh >= (uint)oH) continue;

                                int inRowBase = inChBase + ih * w;
                                int outRowBase = outChBase + oh * oW;

                                for (int iw = 0; iw < w; iw++)
                                {
                                    int ow = iw * stride - padding + kw;
                                    if ((uint)ow >= (uint)oW) continue;

                                    wg += input[inRowBase + iw] * gradOutData[outRowBase + ow];
                                }
                            }

                            weightGrad[wBase + kh * kW + kw] += wg;
                        }
                    }
                }
            }
        }
    }

    // ── 1D Im2Col for Conv1d ────────────────────────────────────────────────

    internal static PatchLocation[] BuildPatchLocations1D(int positionsPerBatch, int tileStart, int tileLen)
    {
        var locs = new PatchLocation[tileLen];
        for (int t = 0; t < tileLen; t++)
        {
            int globalIdx = tileStart + t;
            int batch = globalIdx / positionsPerBatch;
            int oPos = globalIdx % positionsPerBatch;
            locs[t] = new PatchLocation(batch, oPos, 0);
        }
        return locs;
    }

    public static void Im2Col1DTile<T>(
        ReadOnlySpan<T> input,
        Span<T> output,
        int channels, int length,
        int kernelSize,
        int stride, int padding,
        int outLength,
        int tileStart, int tileLen,
        PatchLocation[] locs) where T : struct, IFloatingPointIeee754<T>
    {
        int patchSize = channels * kernelSize;

        for (int t = 0; t < tileLen; t++)
        {
            var loc = locs[t];
            int inBase = loc.Batch * channels * length;
            int basePos = loc.OH * stride - padding;
            int outRow = t * patchSize;

            CopyPatch1D(input, output, inBase, outRow, channels, length, kernelSize, basePos);
        }
    }

    static void CopyPatch1D<T>(
        ReadOnlySpan<T> input, Span<T> output,
        int inBase, int outRow,
        int channels, int length, int kernelSize,
        int basePos) where T : struct, IFloatingPointIeee754<T>
    {
        int patchIdx = 0;

        for (int ic = 0; ic < channels; ic++)
        {
            int cOff = inBase + ic * length;

            for (int kh = 0; kh < kernelSize; kh++)
            {
                int pos = basePos + kh;
                output[outRow + patchIdx++] = (uint)pos < (uint)length
                    ? input[cOff + pos]
                    : T.Zero;
            }
        }
    }
}
