namespace Nivara.Silk;

public sealed partial class SilkResNet18GpuRunner : IDisposable
{
    readonly SilkRuntime runtime;
    readonly int numClasses;

    // Stem
    readonly nint stemConvW, stemBnMean, stemBnVar, stemBnGamma, stemBnBeta;

    sealed class BlockWeights
    {
        public int InC, OutC, Stride;
        public bool HasDownsample;
        public nint Conv1W, Bn1Mean, Bn1Var, Bn1Gamma, Bn1Beta;
        public nint Conv2W, Bn2Mean, Bn2Var, Bn2Gamma, Bn2Beta;
        public nint DsConvW, DsBnMean, DsBnVar, DsBnGamma, DsBnBeta;
    }

    readonly BlockWeights[] blocks; // 8
    readonly nint fcW, fcB;

    public SilkResNet18GpuRunner(SilkRuntime runtime, Dictionary<string, (float[] Data, int[] Shape)> tensors, int numClasses = 1000)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(tensors);
        this.runtime = runtime;
        this.numClasses = numClasses;

        stemConvW = U(tensors["resnet.embedder.embedder.convolution.weight"].Data);
        var (sm, sv, sg, sb) = UBn(tensors, "resnet.embedder.embedder");
        stemBnMean = sm; stemBnVar = sv; stemBnGamma = sg; stemBnBeta = sb;

        // stages
        int[][] stageChannels = { new[] { 64, 64 }, new[] { 64, 128 }, new[] { 128, 256 }, new[] { 256, 512 } };
        int[] strides = { 1, 2, 2, 2 };
        blocks = new BlockWeights[8];
        int bi = 0;
        for (int s = 0; s < 4; s++)
        {
            int inC = stageChannels[s][0];
            int outC = stageChannels[s][1];
            int stride0 = strides[s];
            for (int l = 0; l < 2; l++)
            {
                int stride = (l == 0) ? stride0 : 1;
                int inChannels = (l == 0) ? inC : outC;
                var bw = new BlockWeights { InC = inChannels, OutC = outC, Stride = stride };
                bw.HasDownsample = (inChannels != outC || stride != 1);
                string prefix = $"resnet.encoder.stages.{s}.layers.{l}";
                bw.Conv1W = U(tensors[$"{prefix}.layer.0.convolution.weight"].Data);
                var b1 = UBn(tensors, $"{prefix}.layer.0");
                bw.Bn1Mean = b1.mean; bw.Bn1Var = b1.var; bw.Bn1Gamma = b1.gamma; bw.Bn1Beta = b1.beta;
                bw.Conv2W = U(tensors[$"{prefix}.layer.1.convolution.weight"].Data);
                var b2 = UBn(tensors, $"{prefix}.layer.1");
                bw.Bn2Mean = b2.mean; bw.Bn2Var = b2.var; bw.Bn2Gamma = b2.gamma; bw.Bn2Beta = b2.beta;
                if (bw.HasDownsample)
                {
                    bw.DsConvW = U(tensors[$"{prefix}.shortcut.convolution.weight"].Data);
                    var ds = UBn(tensors, $"{prefix}.shortcut");
                    bw.DsBnMean = ds.mean; bw.DsBnVar = ds.var; bw.DsBnGamma = ds.gamma; bw.DsBnBeta = ds.beta;
                }
                blocks[bi++] = bw;
            }
        }

        // fc
        var cw = tensors["classifier.1.weight"];
        int oc = cw.Shape[0], ic = cw.Shape[1];
        var wT = new float[ic * oc];
        for (int i = 0; i < oc; i++)
            for (int j = 0; j < ic; j++)
                wT[j * oc + i] = cw.Data[i * ic + j];
        fcW = U(wT);
        if (tensors.TryGetValue("classifier.1.bias", out var cb))
            fcB = U(cb.Data);
        else
            fcB = runtime.Allocate(oc);
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
        if (mean == null) mean = new float[ch];
        if (varr == null) { varr = new float[ch]; for (int i = 0; i < ch; i++) varr[i] = 1f; }
        if (gamma == null) { gamma = new float[ch]; for (int i = 0; i < ch; i++) gamma[i] = 1f; }
        if (beta == null) beta = new float[ch];
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
        nint output = runtime.Allocate(N * C * H * W);
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
        // stem conv 7x7 s2 p3
        nint a = Conv(x, N, 3, H0, W0, stemConvW, 64, 7, 2, 3, 1, out oH, out oW);
        runtime.Release(x);
        nint b = BN(a, N, 64, oH, oW, stemBnMean, stemBnVar, stemBnGamma, stemBnBeta);
        runtime.Release(a);
        nint c = runtime.Allocate(N * 64 * oH * oW);
        SilkLauncher.Relu(runtime, b, c, N * 64 * oH * oW);
        runtime.Release(b);
        int h = oH, w = oW;

        // maxpool 3x3 s2 p1
        int pH = ConvOut(h, 3, 2, 1);
        int pW = ConvOut(w, 3, 2, 1);
        nint p = runtime.Allocate(N * 64 * pH * pW);
        SilkLauncher.MaxPool2d(runtime, c, p, N, 64, h, w, 3, 3, 2, 2, 1, 1, 1, 1, pH, pW);
        runtime.Release(c);
        h = pH; w = pW;

        nint cur = p;
        int curC = 64, curH = h, curW = w;

        foreach (var bw in blocks)
        {
            nint input = cur;
            bool residual = bw.HasDownsample ? false : true; // residual when no downsample
            // conv1
            int o1H, o1W;
            nint c1 = Conv(input, N, bw.InC, curH, curW, bw.Conv1W, bw.OutC, 3, bw.Stride, 1, 1, out o1H, out o1W);
            nint b1 = BN(c1, N, bw.OutC, o1H, o1W, bw.Bn1Mean, bw.Bn1Var, bw.Bn1Gamma, bw.Bn1Beta);
            runtime.Release(c1);
            nint r1 = runtime.Allocate(N * bw.OutC * o1H * o1W);
            SilkLauncher.Relu(runtime, b1, r1, N * bw.OutC * o1H * o1W);
            runtime.Release(b1);

            // conv2
            int o2H, o2W;
            nint c2 = Conv(r1, N, bw.OutC, o1H, o1W, bw.Conv2W, bw.OutC, 3, 1, 1, 1, out o2H, out o2W);
            runtime.Release(r1);
            nint b2 = BN(c2, N, bw.OutC, o2H, o2W, bw.Bn2Mean, bw.Bn2Var, bw.Bn2Gamma, bw.Bn2Beta);
            runtime.Release(c2);

            nint residualBuf;
            if (bw.HasDownsample)
            {
                int dsH, dsW;
                nint dc = Conv(input, N, bw.InC, curH, curW, bw.DsConvW, bw.OutC, 1, bw.Stride, 0, 1, out dsH, out dsW);
                nint db = BN(dc, N, bw.OutC, dsH, dsW, bw.DsBnMean, bw.DsBnVar, bw.DsBnGamma, bw.DsBnBeta);
                runtime.Release(dc);
                residualBuf = db;
                // ensure dims match
                o2H = dsH; o2W = dsW; // align
            }
            else
            {
                residualBuf = input; // alias, do not release separately
            }

            // add
            int total = N * bw.OutC * o2H * o2W;
            nint sum = runtime.Allocate(total);
            SilkLauncher.Add(runtime, b2, residualBuf, sum, total);
            runtime.Release(b2);
            if (bw.HasDownsample) runtime.Release(residualBuf);
            // if not downsample, residualBuf == input; release input now
            else runtime.Release(input);

            nint outBuf = runtime.Allocate(total);
            SilkLauncher.Relu(runtime, sum, outBuf, total);
            runtime.Release(sum);

            cur = outBuf;
            curC = bw.OutC;
            curH = o2H;
            curW = o2W;
        }

        // avg pool
        nint pooled = runtime.Allocate(N * curC);
        SilkLauncher.AdaptiveAvgPool2d(runtime, cur, pooled, N, curC, curH, curW);
        runtime.Release(cur);

        // fc
        int totalCls = N * numClasses;
        nint logits = runtime.Allocate(totalCls);
        SilkLauncher.Gemm(runtime, pooled, fcW, logits, N, curC, numClasses);
        runtime.Release(pooled);

        if (fcB != 0)
        {
            nint biased = runtime.Allocate(totalCls);
            SilkLauncher.AddBias(runtime, logits, fcB, biased, N, numClasses);
            runtime.Release(logits);
            logits = biased;
        }

        var result = runtime.Download(logits, totalCls);
        runtime.Release(logits);
        return result;
    }

    public void Dispose()
    {
        runtime.Release(stemConvW); runtime.Release(stemBnMean); runtime.Release(stemBnVar); runtime.Release(stemBnGamma); runtime.Release(stemBnBeta);
        foreach (var bw in blocks)
        {
            runtime.Release(bw.Conv1W); runtime.Release(bw.Bn1Mean); runtime.Release(bw.Bn1Var); runtime.Release(bw.Bn1Gamma); runtime.Release(bw.Bn1Beta);
            runtime.Release(bw.Conv2W); runtime.Release(bw.Bn2Mean); runtime.Release(bw.Bn2Var); runtime.Release(bw.Bn2Gamma); runtime.Release(bw.Bn2Beta);
            if (bw.HasDownsample) { runtime.Release(bw.DsConvW); runtime.Release(bw.DsBnMean); runtime.Release(bw.DsBnVar); runtime.Release(bw.DsBnGamma); runtime.Release(bw.DsBnBeta); }
        }
        runtime.Release(fcW); runtime.Release(fcB);
    }
}
