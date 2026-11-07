using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Pins the #900 backward kernels against a CPU replay of the same accumulation order.
/// The GEMM NT/TN variants, the reductions and the conv backward kernels only multiply and
/// add in one ascending order under the file-scope <c>FP_CONTRACT OFF</c>, so the replay is
/// byte-identical and the assertions are exact — a bit difference means an order or an index
/// changed, not rounding. <c>adam_update</c> is the exception: fp32 divide and sqrt are not
/// guaranteed correctly rounded without the <c>-cl-fp32-correctly-rounded-divide-sqrt</c>
/// build option (<c>SilkRuntime.BuildProgram</c> passes none), so it is pinned by device-vs-device
/// bitwise determinism plus a tolerance against the host formula.
/// </summary>
[Category("Gpu")]
public sealed class SilkBackwardKernelTests
{
    static SilkRuntime Device()
    {
        try { return new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk backward kernel verification."); throw; }
    }

    static float[] Randoms(int count, uint seed)
    {
        var rng = new Random((int)seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return values;
    }

    static void AssertBits(float[] actual, float[] expected, string label)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length), label);
        for (int i = 0; i < expected.Length; i++)
        {
            int want = BitConverter.SingleToInt32Bits(expected[i]);
            int got = BitConverter.SingleToInt32Bits(actual[i]);
            Assert.That(got, Is.EqualTo(want), $"{label}[{i}]: expected 0x{want:X8}, got 0x{got:X8}");
        }
    }

    [Test]
    public void GemmNt_TransposedB_MatchesCpuReplay_BitForBit()
    {
        // Non-tile-multiple shapes exercise every guard branch; the tile-multiple shape
        // exercises the vload4 path.
        (int m, int k, int n)[] shapes = [(17, 33, 9), (16, 32, 64), (1, 1, 1), (5, 2, 2)];
        using var silk = Device();
        foreach (var (m, k, n) in shapes)
        {
            float[] a = Randoms(m * k, 0x90000001);      // [m, k] row-major
            float[] b = Randoms(n * k, 0x90000002);      // [n, k] row-major (the transposed operand)
            nint sa = silk.Allocate(a.Length);
            nint sb = silk.Allocate(b.Length);
            nint sc = silk.Allocate(m * n);
            silk.Upload(sa, a);
            silk.Upload(sb, b);

            SilkLauncher.GemmNt(silk, sa, sb, sc, m, k, n);
            silk.Finish();
            float[] got = silk.Download(sc, m * n);

            var want = new float[m * n];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float acc = 0f;
                    for (int kk = 0; kk < k; kk++)
                        acc += a[i * k + kk] * b[j * k + kk];
                    want[i * n + j] = acc;
                }

            AssertBits(got, want, $"gemm_row4_nt m={m} k={k} n={n}");
            silk.Release(sa); silk.Release(sb); silk.Release(sc);
        }
    }

    [Test]
    public void GemmTn_TransposedA_MatchesCpuReplay_BitForBit()
    {
        (int m, int k, int n)[] shapes = [(17, 33, 9), (16, 32, 64), (1, 1, 1), (2, 5, 2)];
        using var silk = Device();
        foreach (var (m, k, n) in shapes)
        {
            float[] a = Randoms(k * m, 0x90000011);      // [k, m] row-major (the transposed operand)
            float[] b = Randoms(k * n, 0x90000012);      // [k, n] row-major
            nint sa = silk.Allocate(a.Length);
            nint sb = silk.Allocate(b.Length);
            nint sc = silk.Allocate(m * n);
            silk.Upload(sa, a);
            silk.Upload(sb, b);

            SilkLauncher.GemmTn(silk, sa, sb, sc, m, k, n);
            silk.Finish();
            float[] got = silk.Download(sc, m * n);

            var want = new float[m * n];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float acc = 0f;
                    for (int kk = 0; kk < k; kk++)
                        acc += a[kk * m + i] * b[kk * n + j];
                    want[i * n + j] = acc;
                }

            AssertBits(got, want, $"gemm_row4_tn m={m} k={k} n={n}");
            silk.Release(sa); silk.Release(sb); silk.Release(sc);
        }
    }

    [Test]
    public void ColSum_Scaled_MatchesCpuReplay_BitForBit()
    {
        (int rows, int cols)[] shapes = [(7, 5), (33, 2), (16, 64), (1, 1)];
        using var silk = Device();
        foreach (var (rows, cols) in shapes)
        {
            float[] x = Randoms(rows * cols, 0x90000021);
            nint sx = silk.Allocate(x.Length);
            nint sy = silk.Allocate(cols);
            silk.Upload(sx, x);

            const float scale = 0.25f;
            SilkLauncher.ColSum(silk, sx, sy, rows, cols, scale);
            silk.Finish();
            float[] got = silk.Download(sy, cols);

            var want = new float[cols];
            for (int j = 0; j < cols; j++)
            {
                float acc = 0f;
                for (int r = 0; r < rows; r++)
                    acc += x[r * cols + j];
                want[j] = acc * scale;
            }

            AssertBits(got, want, $"col_sum rows={rows} cols={cols}");
            silk.Release(sx); silk.Release(sy);
        }
    }

    [Test]
    public void ReluBackward_MatchesCpu_BitForBit()
    {
        const int count = 1000;
        using var silk = Device();
        float[] d = Randoms(count, 0x90000031);
        float[] act = Randoms(count, 0x90000032);
        nint sd = silk.Allocate(d.Length);
        nint sa = silk.Allocate(act.Length);
        nint sy = silk.Allocate(count);
        silk.Upload(sd, d);
        silk.Upload(sa, act);

        SilkLauncher.ReluBackward(silk, sd, sa, sy, count);
        silk.Finish();
        float[] got = silk.Download(sy, count);

        var want = new float[count];
        for (int i = 0; i < count; i++)
            want[i] = act[i] > 0.0f ? d[i] : 0.0f;

        AssertBits(got, want, "relu_backward");
        silk.Release(sd); silk.Release(sa); silk.Release(sy);
    }

    [Test]
    public void Fill_ZeroesEveryElement()
    {
        const int count = 513;
        using var silk = Device();
        nint sy = silk.Allocate(count);
        silk.Upload(sy, Randoms(count, 0x90000041));
        SilkLauncher.Fill(silk, sy, 0f, count);
        silk.Finish();

        float[] got = silk.Download(sy, count);
        AssertBits(got, new float[count], "fill");
        silk.Release(sy);
    }

    [Test]
    public void Conv2dBackwardWeight_MatchesCpuReplay_BitForBit()
    {
        // H=1 (the text CNN shape) with stride 1 and stride 2, an H>1 case to pin the NCHW
        // indexing, and a kH=2 case to pin the height windowing.
        (int width, int strideW, int oWidth, int height, int kH, int strideH, int padTop, int oHeight)[] shapes =
            [(7, 1, 7, 1, 1, 1, 0, 1), (8, 2, 4, 1, 1, 1, 0, 1), (9, 1, 9, 2, 1, 1, 0, 2), (6, 1, 6, 3, 2, 1, 1, 3)];
        const int batch = 2, channels = 3, outC = 4, kW = 3, padLeft = 1;
        using var silk = Device();
        foreach (var (width, strideW, oWidth, height, kH, strideH, padTop, oHeight) in shapes)
        {
            float[] dA = Randoms(batch * outC * oHeight * oWidth, 0x90000051);
            float[] x = Randoms(batch * channels * height * width, 0x90000052);
            nint sdA = silk.Allocate(dA.Length);
            nint sx = silk.Allocate(x.Length);
            nint sdW = silk.Allocate(outC * channels * kH * kW);
            silk.Upload(sdA, dA);
            silk.Upload(sx, x);

            SilkLauncher.Conv2dBackwardWeight(silk, sdA, sx, sdW,
                N: batch, C: channels, H: height, W: width, outC: outC, kH: kH, kW: kW,
                strideH: strideH, strideW: strideW, padTop: padTop, padLeft: padLeft, groups: 1,
                oH: oHeight, oW: oWidth);
            silk.Finish();
            float[] got = silk.Download(sdW, outC * channels * kH * kW);

            var want = new float[outC * channels * kH * kW];
            for (int oc = 0; oc < outC; oc++)
                for (int ic = 0; ic < channels; ic++)
                    for (int kh = 0; kh < kH; kh++)
                        for (int kw = 0; kw < kW; kw++)
                        {
                            float acc = 0f;
                            for (int n = 0; n < batch; n++)
                                for (int oh = 0; oh < oHeight; oh++)
                                    for (int ow = 0; ow < oWidth; ow++)
                                    {
                                        int ih = oh * strideH - padTop + kh;
                                        if (ih < 0 || ih >= height) continue;
                                        int iw = ow * strideW - padLeft + kw;
                                        if (iw < 0 || iw >= width) continue;
                                        acc += dA[(n * outC + oc) * oHeight * oWidth + oh * oWidth + ow]
                                             * x[(n * channels + ic) * height * width + ih * width + iw];
                                    }
                            want[(oc * channels + ic) * kH * kW + kh * kW + kw] = acc;
                        }

            AssertBits(got, want, $"conv2d_backward_weight w={width} s={strideW} h={height} kH={kH}");
            silk.Release(sdA); silk.Release(sx); silk.Release(sdW);
        }
    }

    [Test]
    public void Conv2dBackwardInput_Accumulates_MatchesCpuReplay_BitForBit()
    {
        // Two branch launches into one dX (the text CNN's three-branch accumulate), then the
        // stride-2, H>1 and kH>1 single-launch cases.
        const int batch = 2, channels = 3, outC = 4, kW = 3, padLeft = 1;
        using var silk = Device();

        (int width, int strideW, int oWidth, int height, int kH, int strideH, int padTop, int oHeight, int launches)[] shapes =
            [(7, 1, 7, 1, 1, 1, 0, 1, 2), (8, 2, 4, 1, 1, 1, 0, 1, 1), (9, 1, 9, 2, 1, 1, 0, 2, 1), (6, 1, 6, 3, 2, 1, 1, 3, 1)];
        foreach (var (width, strideW, oWidth, height, kH, strideH, padTop, oHeight, launches) in shapes)
        {
            float[] x = Randoms(batch * channels * height * width, 0x90000061);
            var branchWeights = new float[launches][];
            var branchDeltas = new float[launches][];
            nint sdX = silk.Allocate(x.Length);
            nint sdABuf = silk.Allocate(batch * outC * oHeight * oWidth);
            nint swBuf = silk.Allocate(outC * channels * kH * kW);
            for (int l = 0; l < launches; l++)
            {
                branchWeights[l] = Randoms(outC * channels * kH * kW, 0x9000061u + (uint)l);
                branchDeltas[l] = Randoms(batch * outC * oHeight * oWidth, 0x9000071u + (uint)l);
            }

            SilkLauncher.Fill(silk, sdX, 0f, x.Length);
            for (int l = 0; l < launches; l++)
            {
                silk.Upload(sdABuf, branchDeltas[l]);
                silk.Upload(swBuf, branchWeights[l]);
                SilkLauncher.Conv2dBackwardInput(silk, sdABuf, swBuf, sdX,
                    N: batch, C: channels, H: height, W: width, outC: outC, kH: kH, kW: kW,
                    strideH: strideH, strideW: strideW, padTop: padTop, padLeft: padLeft, groups: 1,
                    oH: oHeight, oW: oWidth);
            }
            silk.Finish();
            float[] got = silk.Download(sdX, x.Length);

            var want = new float[x.Length];
            for (int l = 0; l < launches; l++)
                for (int n = 0; n < batch; n++)
                    for (int ic = 0; ic < channels; ic++)
                        for (int ih = 0; ih < height; ih++)
                            for (int iw = 0; iw < width; iw++)
                            {
                                float acc = 0f;
                                for (int oc = 0; oc < outC; oc++)
                                    for (int kh = 0; kh < kH; kh++)
                                    {
                                        int ohRaw = ih + padTop - kh;
                                        if (ohRaw < 0 || ohRaw % strideH != 0) continue;
                                        int oh = ohRaw / strideH;
                                        if (oh >= oHeight) continue;
                                        for (int kw = 0; kw < kW; kw++)
                                        {
                                            int owRaw = iw + padLeft - kw;
                                            if (owRaw < 0 || owRaw % strideW != 0) continue;
                                            int ow = owRaw / strideW;
                                            if (ow >= oWidth) continue;
                                            acc += branchDeltas[l][(n * outC + oc) * oHeight * oWidth + oh * oWidth + ow]
                                                 * branchWeights[l][(oc * channels + ic) * kH * kW + kh * kW + kw];
                                        }
                                    }
                                want[(n * channels + ic) * height * width + ih * width + iw] += acc;
                            }

            AssertBits(got, want, $"conv2d_backward_input w={width} s={strideW} h={height} kH={kH} launches={launches}");
            silk.Release(sdX); silk.Release(sdABuf); silk.Release(swBuf);
        }
    }

    [Test]
    public void Conv2dBackwardBias_MatchesCpuReplay_BitForBit()
    {
        const int batch = 3, outC = 5, oWidth = 11;
        using var silk = Device();
        float[] dA = Randoms(batch * outC * oWidth, 0x90000071);
        nint sdA = silk.Allocate(dA.Length);
        nint sdb = silk.Allocate(outC);
        silk.Upload(sdA, dA);

        SilkLauncher.Conv2dBackwardBias(silk, sdA, sdb, N: batch, outC: outC, oH: 1, oW: oWidth);
        silk.Finish();
        float[] got = silk.Download(sdb, outC);

        var want = new float[outC];
        for (int oc = 0; oc < outC; oc++)
        {
            float acc = 0f;
            for (int n = 0; n < batch; n++)
                for (int i = 0; i < oWidth; i++)
                    acc += dA[(n * outC + oc) * oWidth + i];
            want[oc] = acc;
        }

        AssertBits(got, want, "conv2d_backward_bias");
        silk.Release(sdA); silk.Release(sdb);
    }

    [Test]
    public void AdamUpdate_IsDeterministicOnDevice_AndMatchesHostFormula()
    {
        const int count = 1000;
        const float lr = 0.01f, bc1 = 0.1f, bc2 = 0.001f, eps = 1e-8f;
        using var silk = Device();
        float[] w = Randoms(count, 0x90000081);
        float[] g = Randoms(count, 0x90000082);
        float[] m = Randoms(count, 0x90000083);
        float[] v = Randoms(count, 0x90000084);
        nint sw1 = silk.Allocate(count);
        nint sw2 = silk.Allocate(count);
        nint sg = silk.Allocate(count);
        nint sm1 = silk.Allocate(count);
        nint sv1 = silk.Allocate(count);
        nint sm2 = silk.Allocate(count);
        nint sv2 = silk.Allocate(count);
        silk.Upload(sw1, w); silk.Upload(sw2, w);
        silk.Upload(sg, g);
        silk.Upload(sm1, m); silk.Upload(sv1, v);
        silk.Upload(sm2, m); silk.Upload(sv2, v);

        SilkLauncher.AdamUpdate(silk, sw1, sg, sm1, sv1, lr, bc1, bc2, eps, 0f, count);
        SilkLauncher.AdamUpdate(silk, sw2, sg, sm2, sv2, lr, bc1, bc2, eps, 0f, count);
        silk.Finish();

        float[] run1 = silk.Download(sw1, count);
        float[] run2 = silk.Download(sw2, count);
        for (int i = 0; i < count; i++)
        {
            int a = BitConverter.SingleToInt32Bits(run1[i]);
            int b = BitConverter.SingleToInt32Bits(run2[i]);
            Assert.That(b, Is.EqualTo(a), $"adam run 2 [{i}] differs from run 1: 0x{a:X8} vs 0x{b:X8}");
        }

        // Host replay of the same update text; div/sqrt rounding may differ by a few ulp.
        float worst = 0f;
        for (int i = 0; i < count; i++)
        {
            float m1 = 0.9f * m[i] + 0.1f * g[i];
            float v1 = 0.999f * v[i] + 0.001f * g[i] * g[i];
            float host = w[i] - lr * (m1 / bc1) / (MathF.Sqrt(v1 / bc2) + eps);
            worst = Math.Max(worst, Math.Abs(run1[i] - host));
        }
        Assert.That(worst, Is.LessThan(1e-5f), $"adam worst device-vs-host diff {worst}");

        silk.Release(sw1); silk.Release(sw2); silk.Release(sg);
        silk.Release(sm1); silk.Release(sv1); silk.Release(sm2); silk.Release(sv2);
    }

    [Test]
    public void AdamUpdate_WithWeightDecay_MatchesHostFormula()
    {
        const int count = 1000;
        const float lr = 0.01f, bc1 = 0.1f, bc2 = 0.001f, eps = 1e-8f, wd = 0.01f;
        using var silk = Device();
        float[] w = Randoms(count, 0x90000085);
        float[] g = Randoms(count, 0x90000086);
        float[] m = Randoms(count, 0x90000087);
        float[] v = Randoms(count, 0x90000088);
        nint sw = silk.Allocate(count);
        nint sg = silk.Allocate(count);
        nint sm = silk.Allocate(count);
        nint sv = silk.Allocate(count);
        silk.Upload(sw, w); silk.Upload(sg, g); silk.Upload(sm, m); silk.Upload(sv, v);

        SilkLauncher.AdamUpdate(silk, sw, sg, sm, sv, lr, bc1, bc2, eps, wd, count);
        silk.Finish();
        float[] got = silk.Download(sw, count);

        float worst = 0f;
        for (int i = 0; i < count; i++)
        {
            float m1 = 0.9f * m[i] + 0.1f * g[i];
            float v1 = 0.999f * v[i] + 0.001f * g[i] * g[i];
            float host = w[i] - lr * ((m1 / bc1) / (MathF.Sqrt(v1 / bc2) + eps) + wd * w[i]);
            worst = Math.Max(worst, Math.Abs(got[i] - host));
        }
        Assert.That(worst, Is.LessThan(1e-5f), $"adamw worst device-vs-host diff {worst}");

        silk.Release(sw); silk.Release(sg); silk.Release(sm); silk.Release(sv);
    }

    [Test]
    public void AttentionBackward_MatchesCpuReplay_AndIsDeterministicOnDevice()
    {
        // The forward keeps no softmax, so the backward re-derives it; exp() is not correctly
        // rounded on either side, which is why this follows the adam_update pattern (bit-equal
        // device-vs-device, tolerance against the host replay) instead of AssertBits.
        //
        // The last case (seqLen 1, batch 3) leaves one batch element fully masked, which pins the
        // -INFINITY branch that must write zeros instead of evaluating exp(-inf - -inf).
        (int batch, int seqLen, int heads, int headDim, int band)[] shapes =
            [(2, 5, 3, 4, -1), (1, 4, 2, 3, 1), (3, 1, 2, 2, -1)];
        const float scale = 0.5f;
        using var silk = Device();

        foreach (var (batch, seqLen, numHeads, headDim, band) in shapes)
        {
            int rows = batch * seqLen, D = numHeads * headDim;
            int block = rows * D, total = batch * numHeads * seqLen;

            float[] q = Randoms(block, 0x90000091);
            float[] k = Randoms(block, 0x90000092);
            float[] v = Randoms(block, 0x90000093);
            float[] qkv = [.. q, .. k, .. v];
            float[] dAttn = Randoms(rows * D, 0x90000094);

            // Padding mask: batch element b keeps seqLen - (b % 2) key positions.
            var mask = new float[rows];
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < seqLen - (b % 2); s++)
                    mask[b * seqLen + s] = 1f;

            nint sqkv1 = silk.Allocate(qkv.Length);
            nint sqkv2 = silk.Allocate(qkv.Length);
            nint smask = silk.Allocate(mask.Length);
            nint sdAttn = silk.Allocate(dAttn.Length);
            nint sprobs1 = silk.Allocate(total * seqLen);
            nint sprobs2 = silk.Allocate(total * seqLen);
            nint sdelta1 = silk.Allocate(total);
            nint sdelta2 = silk.Allocate(total);
            nint sdqkv1 = silk.Allocate(qkv.Length);
            nint sdqkv2 = silk.Allocate(qkv.Length);
            silk.Upload(sqkv1, qkv);
            silk.Upload(sqkv2, qkv);
            silk.Upload(smask, mask);
            silk.Upload(sdAttn, dAttn);

            foreach (var (dqkv, probs, delta) in new[] { (sdqkv1, sprobs1, sdelta1), (sdqkv2, sprobs2, sdelta2) })
            {
                SilkLauncher.AttentionBackwardPrep(
                    silk, sqkv1, smask, sdAttn, probs, delta, batch, seqLen, numHeads, headDim, band, scale);
                SilkLauncher.AttentionBackwardDq(
                    silk, sqkv1, probs, delta, sdAttn, dqkv, batch, seqLen, numHeads, headDim, scale);
                SilkLauncher.AttentionBackwardDkv(
                    silk, sqkv1, probs, delta, sdAttn, dqkv, batch, seqLen, numHeads, headDim, scale);
            }
            silk.Finish();

            float[] gotProbs = silk.Download(sprobs1, total * seqLen);
            float[] gotDelta = silk.Download(sdelta1, total);
            float[] gotDqkv = silk.Download(sdqkv1, qkv.Length);

            // Device-vs-device: identical inputs must produce identical bits.
            AssertBits(silk.Download(sprobs2, total * seqLen), gotProbs, "attention probs");
            AssertBits(silk.Download(sdelta2, total), gotDelta, "attention delta");
            AssertBits(silk.Download(sdqkv2, qkv.Length), gotDqkv, "attention dQkv");

            var (wantProbs, wantDelta, wantDqkv) = AttentionBackwardReference(
                qkv, mask, dAttn, batch, seqLen, numHeads, headDim, band, scale);

            float worst = 0f;
            for (int i = 0; i < gotProbs.Length; i++)
                worst = Math.Max(worst, Math.Abs(gotProbs[i] - wantProbs[i]));
            for (int i = 0; i < gotDelta.Length; i++)
                worst = Math.Max(worst, Math.Abs(gotDelta[i] - wantDelta[i]));
            for (int i = 0; i < gotDqkv.Length; i++)
                worst = Math.Max(worst, Math.Abs(gotDqkv[i] - wantDqkv[i]));

            Assert.That(worst, Is.LessThan(1e-5f),
                $"attention backward worst device-vs-host diff {worst} (batch={batch}, seq={seqLen}, heads={numHeads}, headDim={headDim}, band={band})");

            silk.Release(sqkv1); silk.Release(sqkv2); silk.Release(smask); silk.Release(sdAttn);
            silk.Release(sprobs1); silk.Release(sprobs2); silk.Release(sdelta1); silk.Release(sdelta2);
            silk.Release(sdqkv1); silk.Release(sdqkv2);
        }
    }

    [Test]
    public void AttentionBackwardFused_MatchesCpuReplay_AndIsDeterministicOnDevice()
    {
        (int batch, int seqLen, int heads, int headDim, int band)[] shapes =
            [(2, 5, 3, 4, -1), (1, 4, 2, 3, 1), (3, 1, 2, 2, -1)];
        const float scale = 0.5f;
        using var silk = Device();

        foreach (var (batch, seqLen, numHeads, headDim, band) in shapes)
        {
            int rows = batch * seqLen, D = numHeads * headDim;
            int block = rows * D, total = batch * numHeads * seqLen;

            float[] q = Randoms(block, 0x90000091);
            float[] k = Randoms(block, 0x90000092);
            float[] v = Randoms(block, 0x90000093);
            float[] qkv = [.. q, .. k, .. v];
            float[] dAttn = Randoms(rows * D, 0x90000094);

            var mask = new float[rows];
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < seqLen - (b % 2); s++)
                    mask[b * seqLen + s] = 1f;

            nint sqkv1 = silk.Allocate(qkv.Length);
            nint sqkv2 = silk.Allocate(qkv.Length);
            nint smask = silk.Allocate(mask.Length);
            nint sdAttn = silk.Allocate(dAttn.Length);
            nint sprobs1 = silk.Allocate(total * seqLen);
            nint sprobs2 = silk.Allocate(total * seqLen);
            nint sdelta1 = silk.Allocate(total);
            nint sdelta2 = silk.Allocate(total);
            nint sdqkv1 = silk.Allocate(qkv.Length);
            nint sdqkv2 = silk.Allocate(qkv.Length);
            silk.Upload(sqkv1, qkv);
            silk.Upload(sqkv2, qkv);
            silk.Upload(smask, mask);
            silk.Upload(sdAttn, dAttn);

            foreach (var (dqkv, probs, delta) in new[] { (sdqkv1, sprobs1, sdelta1), (sdqkv2, sprobs2, sdelta2) })
            {
                SilkLauncher.AttentionBackwardPrep(
                    silk, sqkv1, smask, sdAttn, probs, delta, batch, seqLen, numHeads, headDim, band, scale);
                SilkLauncher.AttentionBackwardDqDkv(
                    silk, sqkv1, probs, delta, sdAttn, dqkv, batch, seqLen, numHeads, headDim, scale);
            }
            silk.Finish();

            float[] gotProbs = silk.Download(sprobs1, total * seqLen);
            float[] gotDelta = silk.Download(sdelta1, total);
            float[] gotDqkv = silk.Download(sdqkv1, qkv.Length);

            AssertBits(silk.Download(sprobs2, total * seqLen), gotProbs, "attention probs fused");
            AssertBits(silk.Download(sdelta2, total), gotDelta, "attention delta fused");
            AssertBits(silk.Download(sdqkv2, qkv.Length), gotDqkv, "attention dQkv fused");

            var (wantProbs, wantDelta, wantDqkv) = AttentionBackwardReference(
                qkv, mask, dAttn, batch, seqLen, numHeads, headDim, band, scale);

            float worst = 0f;
            for (int i = 0; i < gotProbs.Length; i++)
                worst = Math.Max(worst, Math.Abs(gotProbs[i] - wantProbs[i]));
            for (int i = 0; i < gotDelta.Length; i++)
                worst = Math.Max(worst, Math.Abs(gotDelta[i] - wantDelta[i]));
            for (int i = 0; i < gotDqkv.Length; i++)
                worst = Math.Max(worst, Math.Abs(gotDqkv[i] - wantDqkv[i]));

            Assert.That(worst, Is.LessThan(1e-5f),
                $"attention fused backward worst device-vs-host diff {worst}");

            silk.Release(sqkv1); silk.Release(sqkv2); silk.Release(smask); silk.Release(sdAttn);
            silk.Release(sprobs1); silk.Release(sprobs2); silk.Release(sdelta1); silk.Release(sdelta2);
            silk.Release(sdqkv1); silk.Release(sdqkv2);
        }
    }

    [Test]
    public void Gelu_SplitForm_IsBitIdenticalToTheFusedEpilogue()
    {
        // The training forward needs GELU's pre-activation, so it runs gemm_row4_bias -> gelu
        // instead of gemm_row4_gelu. Both call the same gelu_exact() on the same a+b value, so
        // the split must reproduce the fused store exactly — not within a tolerance.
        (int rows, int k, int n)[] shapes = [(16, 32, 64), (17, 33, 9), (5, 2, 2)];
        using var silk = Device();
        foreach (var (rows, k, n) in shapes)
        {
            float[] a = Randoms(rows * k, 0x900000A1);
            float[] b = Randoms(n * k, 0x900000A2);
            float[] bias = Randoms(n, 0x900000A3);
            nint sa = silk.Allocate(a.Length);
            nint sb = silk.Allocate(b.Length);
            nint sbias = silk.Allocate(bias.Length);
            nint fused = silk.Allocate(rows * n);
            nint split = silk.Allocate(rows * n);
            silk.Upload(sa, a); silk.Upload(sb, b); silk.Upload(sbias, bias);

            SilkLauncher.GemmEpilogue(silk, SilkKernels.GemmGelu, sa, sb, fused, sbias, rows, k, n, null);
            SilkLauncher.GemmEpilogue(silk, SilkKernels.GemmBias, sa, sb, split, sbias, rows, k, n, null);
            SilkLauncher.Gelu(silk, split, split, rows * n);
            silk.Finish();

            AssertBits(silk.Download(split, rows * n), silk.Download(fused, rows * n),
                $"gelu split vs gemm_row4_gelu rows={rows} k={k} n={n}");

            silk.Release(sa); silk.Release(sb); silk.Release(sbias);
            silk.Release(fused); silk.Release(split);
        }
    }

    [Test]
    public void EmbeddingBackwardRows_MatchesCpuReplay_BitForBit()
    {
        (int rows, int embedDim, int vocab)[] shapes = [(7, 5, 4), (1, 1, 2), (33, 8, 6)];
        using var silk = Device();
        foreach (var (rows, embedDim, vocab) in shapes)
        {
            var rng = new Random(0x9000B1);
            int[] ids = new int[rows];
            for (int i = 0; i < rows; i++) ids[i] = rng.Next(vocab);
            float[] dX = Randoms(rows * embedDim, 0x900000B2);
            float[] grad = Randoms(vocab * embedDim, 0x900000B3);

            nint sids = silk.Allocate(rows);
            nint sdX = silk.Allocate(dX.Length);
            nint sgrad = silk.Allocate(grad.Length);
            silk.Upload(sids, ids);
            silk.Upload(sdX, dX);
            silk.Upload(sgrad, grad);

            SilkLauncher.EmbeddingBackwardRows(silk, sids, sdX, sgrad, rows, embedDim);
            silk.Finish();
            float[] got = silk.Download(sgrad, grad.Length);

            var want = (float[])grad.Clone();
            for (int e = 0; e < embedDim; e++)
                for (int r = 0; r < rows; r++)
                    want[ids[r] * embedDim + e] += dX[r * embedDim + e];

            AssertBits(got, want, $"embedding_backward_rows rows={rows} dim={embedDim}");
            silk.Release(sids); silk.Release(sdX); silk.Release(sgrad);
        }
    }

    [Test]
    public void ScatterRows_WritesOnlyTheClsRowsOfAPrezeroedBuffer()
    {
        const int batch = 3, seqLen = 4, embedDim = 5;
        using var silk = Device();
        float[] dX = Randoms(batch * seqLen * embedDim, 0x900000C1);
        float[] dCls = Randoms(batch * embedDim, 0x900000C2);
        nint sdX = silk.Allocate(dX.Length);
        nint sdCls = silk.Allocate(dCls.Length);
        silk.Upload(sdX, dX);
        silk.Upload(sdCls, dCls);

        SilkLauncher.ScatterRows(silk, sdCls, sdX, batch, seqLen, embedDim);
        silk.Finish();
        float[] got = silk.Download(sdX, dX.Length);

        var want = (float[])dX.Clone();
        for (int b = 0; b < batch; b++)
            for (int e = 0; e < embedDim; e++)
                want[(b * seqLen) * embedDim + e] = dCls[b * embedDim + e];

        AssertBits(got, want, "scatter_rows");
        silk.Release(sdX); silk.Release(sdCls);
    }

    [Test]
    public void GemmNtResidual_MatchesNtPlusAdd_BitForBit()
    {
        (int m, int k, int n)[] shapes = [(17, 33, 9), (16, 32, 64), (1, 1, 1), (5, 2, 2)];
        using var silk = Device();
        foreach (var (m, k, n) in shapes)
        {
            float[] a = Randoms(m * k, 0x900001D1);
            float[] b = Randoms(n * k, 0x900001D2);
            float[] residual = Randoms(m * n, 0x900001D3);
            nint sa = silk.Allocate(a.Length);
            nint sb = silk.Allocate(b.Length);
            nint sc = silk.Allocate(m * n);
            nint sr = silk.Allocate(residual.Length);
            silk.Upload(sa, a);
            silk.Upload(sb, b);
            silk.Upload(sr, residual);

            SilkLauncher.GemmNtResidual(silk, sa, sb, sc, sr, m, k, n);
            silk.Finish();
            float[] got = silk.Download(sc, m * n);

            var want = new float[m * n];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float acc = 0f;
                    for (int kk = 0; kk < k; kk++)
                        acc += a[i * k + kk] * b[j * k + kk];
                    want[i * n + j] = acc + residual[i * n + j];
                }

            AssertBits(got, want, $"gemm_row4_nt_residual m={m} k={k} n={n}");
            silk.Release(sa); silk.Release(sb); silk.Release(sc); silk.Release(sr);
        }
    }

    /// <summary>
    /// Replays the three attention-backward kernels on the host in the same accumulation order:
    /// ascending keys for the softmax pass, ascending keys per output component for dQ, and
    /// ascending queries per key for dK/dV. qkv is block-major [Q][K][V], each <c>rows x D</c>.
    /// ascending keys for the softmax pass, ascending keys per output component for dQ, and
    /// ascending queries per key for dK/dV. qkv is block-major [Q][K][V], each <c>rows x D</c>.
    /// </summary>
    static (float[] Probs, float[] Delta, float[] Dqkv) AttentionBackwardReference(
        float[] qkv, float[] mask, float[] dAttn,
        int batch, int seqLen, int numHeads, int headDim, int band, float scale)
    {
        int rows = batch * seqLen, D = numHeads * headDim, block = rows * D;
        int total = batch * numHeads * seqLen;
        var probs = new float[total * seqLen];
        var delta = new float[total];
        var dqkv = new float[qkv.Length];

        static float Dot(float[] a, int aBase, float[] b, int bBase, int n)
        {
            float acc = 0f;
            for (int i = 0; i < n; i++) acc += a[aBase + i] * b[bBase + i];
            return acc;
        }

        bool Keep(int b, int qPos, int j) =>
            mask[b * seqLen + j] >= 0.5f && (band < 0 || Math.Abs(qPos - j) <= band);

        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
                for (int qPos = 0; qPos < seqLen; qPos++)
                {
                    int idx = (b * numHeads + h) * seqLen + qPos;
                    int qRow = b * seqLen + qPos;
                    int qBase = qRow * D + h * headDim;
                    int dAttnBase = qRow * D + h * headDim;
                    int probsBase = idx * seqLen;

                    float mx = float.NegativeInfinity;
                    for (int j = 0; j < seqLen; j++)
                        if (Keep(b, qPos, j))
                            mx = MathF.Max(mx, Dot(qkv, qBase, qkv, block + (b * seqLen + j) * D + h * headDim, headDim) * scale);

                    if (float.IsNegativeInfinity(mx))
                    {
                        delta[idx] = 0f;
                        continue;
                    }

                    float sum = 0f;
                    for (int j = 0; j < seqLen; j++)
                        if (Keep(b, qPos, j))
                            sum += MathF.Exp(Dot(qkv, qBase, qkv, block + (b * seqLen + j) * D + h * headDim, headDim) * scale - mx);

                    float acc = 0f;
                    for (int j = 0; j < seqLen; j++)
                    {
                        float p = Keep(b, qPos, j)
                            ? MathF.Exp(Dot(qkv, qBase, qkv, block + (b * seqLen + j) * D + h * headDim, headDim) * scale - mx) / sum
                            : 0f;
                        probs[probsBase + j] = p;
                        float dotV = Dot(dAttn, dAttnBase, qkv, 2 * block + (b * seqLen + j) * D + h * headDim, headDim);
                        acc += p * dotV;
                    }
                    delta[idx] = acc;
                }

        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
                for (int qPos = 0; qPos < seqLen; qPos++)
                {
                    int idx = (b * numHeads + h) * seqLen + qPos;
                    int qRow = b * seqLen + qPos;
                    int qOutBase = qRow * 3 * D + h * headDim;
                    int dAttnBase = qRow * D + h * headDim;
                    int probsBase = idx * seqLen;

                    for (int d = 0; d < headDim; d++)
                    {
                        float acc = 0f;
                        for (int j = 0; j < seqLen; j++)
                        {
                            float dotV = Dot(dAttn, dAttnBase, qkv, 2 * block + (b * seqLen + j) * D + h * headDim, headDim);
                            float ds = probs[probsBase + j] * (dotV - delta[idx]);
                            acc += ds * qkv[block + (b * seqLen + j) * D + h * headDim + d];
                        }
                        dqkv[qOutBase + d] = acc * scale;
                    }
                }

        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
                for (int jPos = 0; jPos < seqLen; jPos++)
                {
                    int jRow = b * seqLen + jPos;
                    int kOutBase = jRow * 3 * D + D + h * headDim;
                    int vOutBase = jRow * 3 * D + 2 * D + h * headDim;
                    var dk = new float[headDim];
                    var dv = new float[headDim];

                    for (int i = 0; i < seqLen; i++)
                    {
                        int rowIdx = (b * numHeads + h) * seqLen + i;
                        int dAttnBase = (b * seqLen + i) * D + h * headDim;
                        float p = probs[rowIdx * seqLen + jPos];
                        if (p == 0f) continue;
                        float dotV = Dot(dAttn, dAttnBase, qkv, 2 * block + (b * seqLen + jPos) * D + h * headDim, headDim);
                        float ds = p * (dotV - delta[rowIdx]);
                        int qBase = (b * seqLen + i) * D + h * headDim;
                        for (int d = 0; d < headDim; d++)
                        {
                            dv[d] += p * dAttn[dAttnBase + d];
                            dk[d] += ds * qkv[qBase + d];
                        }
                    }

                    for (int d = 0; d < headDim; d++)
                    {
                        dqkv[kOutBase + d] = dk[d] * scale;
                        dqkv[vOutBase + d] = dv[d];
                    }
                }

        return (probs, delta, dqkv);
    }
}
