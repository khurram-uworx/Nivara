using System;

namespace Nivara.Silk;

public sealed class SilkMobileNetV2GpuRunner : IDisposable
{
    readonly SilkRuntime runtime;
    readonly int numClasses;

    readonly nint stemFirstW, stemFirstMean, stemFirstVar, stemFirstGamma, stemFirstBeta;
    readonly nint stemDwW, stemDwMean, stemDwVar, stemDwGamma, stemDwBeta;
    readonly nint stemPwW, stemPwMean, stemPwVar, stemPwGamma, stemPwBeta;

    sealed class BlockWeights
    {
        public bool HasExpansion;
        public nint ExpandW, ExpandMean, ExpandVar, ExpandGamma, ExpandBeta;
        public nint DwW, DwMean, DwVar, DwGamma, DwBeta;
        public nint ProjW, ProjMean, ProjVar, ProjGamma, ProjBeta;
    }

    readonly BlockWeights[] blocks;

    readonly nint headConvW, headBnMean, headBnVar, headBnGamma, headBnBeta;
    readonly nint classifierW, classifierB;

    public SilkMobileNetV2GpuRunner(SilkRuntime runtime, Dictionary<string, (float[] Data, int[] Shape)> tensors, int numClasses = 1001)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(tensors);
        this.runtime = runtime;
        this.numClasses = numClasses;

        stemFirstW = U(tensors["mobilenet_v2.conv_stem.first_conv.convolution.weight"].Data);
        (stemFirstMean, stemFirstVar, stemFirstGamma, stemFirstBeta) = UBn(tensors, "mobilenet_v2.conv_stem.first_conv");
        stemDwW = U(tensors["mobilenet_v2.conv_stem.conv_3x3.convolution.weight"].Data);
        (stemDwMean, stemDwVar, stemDwGamma, stemDwBeta) = UBn(tensors, "mobilenet_v2.conv_stem.conv_3x3");
        stemPwW = U(tensors["mobilenet_v2.conv_stem.reduce_1x1.convolution.weight"].Data);
        (stemPwMean, stemPwVar, stemPwGamma, stemPwBeta) = UBn(tensors, "mobilenet_v2.conv_stem.reduce_1x1");

        int[] expands = { 96, 144, 144, 192, 192, 192, 384, 384, 384, 384, 576, 576, 576, 960, 960, 960 };
        int[] outs = { 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320 };
        blocks = new BlockWeights[16];
        int inCh = 16;
        for (int i = 0; i < 16; i++)
        {
            var bw = new BlockWeights();
            int expand = expands[i];
            bw.HasExpansion = inCh != expand;
            string prefix = $"mobilenet_v2.layer.{i}";
            if (bw.HasExpansion)
            {
                bw.ExpandW = U(tensors[$"{prefix}.expand_1x1.convolution.weight"].Data);
                (bw.ExpandMean, bw.ExpandVar, bw.ExpandGamma, bw.ExpandBeta) = UBn(tensors, $"{prefix}.expand_1x1");
            }
            bw.DwW = U(tensors[$"{prefix}.conv_3x3.convolution.weight"].Data);
            (bw.DwMean, bw.DwVar, bw.DwGamma, bw.DwBeta) = UBn(tensors, $"{prefix}.conv_3x3");
            bw.ProjW = U(tensors[$"{prefix}.reduce_1x1.convolution.weight"].Data);
            (bw.ProjMean, bw.ProjVar, bw.ProjGamma, bw.ProjBeta) = UBn(tensors, $"{prefix}.reduce_1x1");
            blocks[i] = bw;
            inCh = outs[i];
        }

        headConvW = U(tensors["mobilenet_v2.conv_1x1.convolution.weight"].Data);
        (headBnMean, headBnVar, headBnGamma, headBnBeta) = UBn(tensors, "mobilenet_v2.conv_1x1");

        var cw = tensors["classifier.weight"];
        int oc = cw.Shape[0], ic = cw.Shape[1];
        var wT = new float[ic * oc];
        for (int i = 0; i < oc; i++)
            for (int j = 0; j < ic; j++)
                wT[j * oc + i] = cw.Data[i * ic + j];
        classifierW = U(wT);
        if (tensors.TryGetValue("classifier.bias", out var cb))
            classifierB = U(cb.Data);
        else
            classifierB = runtime.Allocate(oc);
    }

    nint U(float[] data)
    {
        nint buf = runtime.Allocate(data.Length);
        runtime.Upload(buf, data);
        return buf;
    }

    (nint mean, nint var, nint gamma, nint beta) UBn(Dictionary<string, (float[] Data, int[] Shape)> tensors, string prefix)
    {
        float[]? mean = tensors.TryGetValue(prefix + ".normalization.running_mean", out var a) ? a.Data : null;
        float[]? varr = tensors.TryGetValue(prefix + ".normalization.running_var", out var b) ? b.Data : null;
        float[]? gamma = tensors.TryGetValue(prefix + ".normalization.weight", out var c) ? c.Data : null;
        float[]? beta = tensors.TryGetValue(prefix + ".normalization.bias", out var d) ? d.Data : null;
        int ch = (mean?.Length ?? gamma?.Length ?? 1);
        if (mean == null) { mean = new float[ch]; }
        if (varr == null) { varr = new float[ch]; for (int i = 0; i < ch; i++) varr[i] = 1f; }
        if (gamma == null) { gamma = new float[ch]; for (int i = 0; i < ch; i++) gamma[i] = 1f; }
        if (beta == null) { beta = new float[ch]; }
        return (U(mean), U(varr), U(gamma), U(beta));
    }

    static int ConvOut(int size, int kernel, int stride, int pad) => (size + 2 * pad - kernel) / stride + 1;

    nint Conv(nint input, int N, int C, int H, int W, nint weight, int outC, int k, int stride, int pad, int groups, out int oH, out int oW)
    {
        oH = ConvOut(H, k, stride, pad);
        oW = ConvOut(W, k, stride, pad);
        nint output = runtime.Allocate(N * outC * oH * oW);
        SilkLauncher.Conv2d(runtime, input, weight, 0, output, N, C, H, W, outC, k, k, stride, stride, pad, pad, pad, pad, groups, false, oH, oW);
        return output;
    }

    nint BN(nint input, int N, int C, int H, int W, nint mean, nint var, nint gamma, nint beta)
    {
        int HW = H * W;
        nint output = runtime.Allocate(N * C * HW);
        SilkLauncher.BatchNormEval(runtime, input, mean, var, gamma, beta, output, N, C, H, W, 1e-5f, true);
        return output;
    }

    public float[] Forward(float[] inputNchw)
    {
        const int N = 1;
        const int H0 = 224, W0 = 224;
        nint x = runtime.Allocate(inputNchw.Length);
        runtime.Upload(x, inputNchw);

        int oH, oW;
        // Stem first conv 3->32, k3 s2 p1
        nint a = Conv(x, N, 3, H0, W0, stemFirstW, 32, 3, 2, 1, 1, out oH, out oW);
        runtime.Release(x);
        nint b = BN(a, N, 32, oH, oW, stemFirstMean, stemFirstVar, stemFirstGamma, stemFirstBeta);
        runtime.Release(a);
        nint c = runtime.Allocate(N * 32 * oH * oW);
        SilkLauncher.Relu6(runtime, b, c, N * 32 * oH * oW);
        runtime.Release(b);
        int h = oH, w = oW;

        // dw 3x3 groups=32
        nint d = Conv(c, N, 32, h, w, stemDwW, 32, 3, 1, 1, 32, out oH, out oW);
        runtime.Release(c);
        nint e = BN(d, N, 32, oH, oW, stemDwMean, stemDwVar, stemDwGamma, stemDwBeta);
        runtime.Release(d);
        nint f = runtime.Allocate(N * 32 * oH * oW);
        SilkLauncher.Relu6(runtime, e, f, N * 32 * oH * oW);
        runtime.Release(e);
        h = oH; w = oW;

        // pw 1x1 32->16
        nint g = Conv(f, N, 32, h, w, stemPwW, 16, 1, 1, 0, 1, out oH, out oW);
        runtime.Release(f);
        nint hh = BN(g, N, 16, oH, oW, stemPwMean, stemPwVar, stemPwGamma, stemPwBeta);
        runtime.Release(g);
        nint ii = runtime.Allocate(N * 16 * oH * oW);
        SilkLauncher.Relu6(runtime, hh, ii, N * 16 * oH * oW);
        runtime.Release(hh);
        int curC = 16;
        int curH = oH, curW = oW;
        nint curBuf = ii;

        int[] expands = { 96, 144, 144, 192, 192, 192, 384, 384, 384, 384, 576, 576, 576, 960, 960, 960 };
        int[] outs = { 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320 };
        int[] strides = { 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1 };

        for (int bi = 0; bi < 16; bi++)
        {
            var bw = blocks[bi];
            int expand = expands[bi];
            int outCh = outs[bi];
            int stride = strides[bi];
            bool residual = stride == 1 && curC == outCh;

            nint residualBuf = residual ? curBuf : 0; // keep
            // If residual, we must not release curBuf until add; but we still need it. We'll release after add.

            nint cur = curBuf;
            int cC = curC, cH = curH, cW = curW;

            if (bw.HasExpansion)
            {
                nint ex = Conv(cur, N, cC, cH, cW, bw.ExpandW, expand, 1, 1, 0, 1, out oH, out oW);
                nint exb = BN(ex, N, expand, oH, oW, bw.ExpandMean, bw.ExpandVar, bw.ExpandGamma, bw.ExpandBeta);
                runtime.Release(ex);
                nint exr = runtime.Allocate(N * expand * oH * oW);
                SilkLauncher.Relu6(runtime, exb, exr, N * expand * oH * oW);
                runtime.Release(exb);
                if (!residual) runtime.Release(cur);
                cur = exr; cC = expand; cH = oH; cW = oW;
            }
            else
            {
                if (!residual) { /* cur == residual? if residual true we keep */ }
            }

            // depthwise
            nint dw = Conv(cur, N, cC, cH, cW, bw.DwW, cC, 3, stride, 1, cC, out oH, out oW);
            if (bw.HasExpansion) runtime.Release(cur);
            nint dwb = BN(dw, N, cC, oH, oW, bw.DwMean, bw.DwVar, bw.DwGamma, bw.DwBeta);
            runtime.Release(dw);
            nint dwr = runtime.Allocate(N * cC * oH * oW);
            SilkLauncher.Relu6(runtime, dwb, dwr, N * cC * oH * oW);
            runtime.Release(dwb);
            int dH = oH, dW = oW;

            // project
            nint pr = Conv(dwr, N, cC, dH, dW, bw.ProjW, outCh, 1, 1, 0, 1, out oH, out oW);
            runtime.Release(dwr);
            nint prb = BN(pr, N, outCh, oH, oW, bw.ProjMean, bw.ProjVar, bw.ProjGamma, bw.ProjBeta);
            runtime.Release(pr);

            if (residual)
            {
                // shapes must match: outCh==curC and oH==curH etc. prb shape [N,outCh,oH,oW] should equal residualBuf
                int total = N * outCh * oH * oW;
                nint sum = runtime.Allocate(total);
                SilkLauncher.Add(runtime, prb, residualBuf, sum, total);
                runtime.Release(prb);
                runtime.Release(residualBuf);
                curBuf = sum;
            }
            else
            {
                curBuf = prb;
            }
            curC = outCh;
            curH = oH;
            curW = oW;
        }

        // Head conv 1x1 320->1280
        nint hc = Conv(curBuf, N, curC, curH, curW, headConvW, 1280, 1, 1, 0, 1, out oH, out oW);
        runtime.Release(curBuf);
        nint hcb = BN(hc, N, 1280, oH, oW, headBnMean, headBnVar, headBnGamma, headBnBeta);
        runtime.Release(hc);
        nint hcr = runtime.Allocate(N * 1280 * oH * oW);
        SilkLauncher.Relu6(runtime, hcb, hcr, N * 1280 * oH * oW);
        runtime.Release(hcb);
        int hH = oH, hW = oW;

        // Adaptive avg pool -> [N,1280]
        nint pooled = runtime.Allocate(N * 1280);
        SilkLauncher.AdaptiveAvgPool2d(runtime, hcr, pooled, N, 1280, hH, hW);
        runtime.Release(hcr);

        // Classifier: gemm [1,1280] x [1280,numClasses] -> [1,numClasses]
        int totalCls = N * numClasses;
        nint logits = runtime.Allocate(totalCls);
        // Use Gemm: A=pooled [N? Actually pooled is [N,1280] but N=1; aRows=N=1? Need A [aRows=1? or N?]. Gemm C[Arows,Bcols]. A rows = N (1), cols=1280, B cols=numClasses. But pooled has N*1280 contiguous; for N=1 fine.
        SilkLauncher.Gemm(runtime, pooled, classifierW, logits, N, 1280, numClasses);
        runtime.Release(pooled);

        // Add bias
        if (classifierB != 0)
        {
            nint biased = runtime.Allocate(totalCls);
            SilkLauncher.AddBias(runtime, logits, classifierB, biased, N, numClasses);
            runtime.Release(logits);
            logits = biased;
        }

        var result = runtime.Download(logits, totalCls);
        runtime.Release(logits);
        return result;
    }

    public void Dispose()
    {
        // best-effort release
        runtime.Release(stemFirstW); runtime.Release(stemFirstMean); runtime.Release(stemFirstVar); runtime.Release(stemFirstGamma); runtime.Release(stemFirstBeta);
        runtime.Release(stemDwW); runtime.Release(stemDwMean); runtime.Release(stemDwVar); runtime.Release(stemDwGamma); runtime.Release(stemDwBeta);
        runtime.Release(stemPwW); runtime.Release(stemPwMean); runtime.Release(stemPwVar); runtime.Release(stemPwGamma); runtime.Release(stemPwBeta);
        foreach (var bw in blocks)
        {
            if (bw.HasExpansion) { runtime.Release(bw.ExpandW); runtime.Release(bw.ExpandMean); runtime.Release(bw.ExpandVar); runtime.Release(bw.ExpandGamma); runtime.Release(bw.ExpandBeta); }
            runtime.Release(bw.DwW); runtime.Release(bw.DwMean); runtime.Release(bw.DwVar); runtime.Release(bw.DwGamma); runtime.Release(bw.DwBeta);
            runtime.Release(bw.ProjW); runtime.Release(bw.ProjMean); runtime.Release(bw.ProjVar); runtime.Release(bw.ProjGamma); runtime.Release(bw.ProjBeta);
        }
        runtime.Release(headConvW); runtime.Release(headBnMean); runtime.Release(headBnVar); runtime.Release(headBnGamma); runtime.Release(headBnBeta);
        runtime.Release(classifierW); runtime.Release(classifierB);
    }
}
