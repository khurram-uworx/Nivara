using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using Nivara.Samples;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

/// <summary>
/// Parity for the banded-bidirectional-plus-padding attention mask and for the safe-softmax
/// clamp that makes a fully-masked query row well defined - the two constructs introduced with
/// ModernBERT-large.
/// </summary>
/// <remarks>
/// <para>
/// The reference is <c>F.scaled_dot_product_attention</c>, <b>not</b> <c>torch.softmax</c>. SDPA
/// is the path that runs torch's internal <c>_safe_softmax</c>, and the two disagree in exactly
/// the case under test: for a query row whose every key is suppressed the row max is
/// <c>-inf</c>, where plain softmax yields NaN and <c>_safe_softmax</c> yields zeros.
/// <c>GradKernels.SoftmaxSingle</c> clamps to zeros, so only SDPA is a valid reference for it.
/// The pre-existing <c>softmax</c> / <c>attn_*</c> fixtures use benign inputs where the two
/// coincide, which is why no earlier fixture covered a fully-masked row.
/// </para>
/// <para>
/// Geometry is <c>abs(i - j) &lt;= 1</c> intersected with right padding over the last two
/// positions (<c>ModernBertMasks.Build(seqLen: 8, band: 1, validLength: 6)</c>), giving
/// <c>[2,3,3,3,3,2,1,0]</c> visible keys per query row - so row 7 is fully masked.
/// </para>
/// <para>
/// Comparisons use <see cref="TestHelpers.AssertTensorClose"/> rather than
/// <see cref="TestHelpers.AssertTensorEqual"/>: the latter computes
/// <c>diff &gt; threshold</c>, and <c>NaN &gt; x</c> is false, so a NaN reference matched
/// against a finite result would pass silently - which would make "zeros vs NaN" untestable.
/// </para>
/// </remarks>
[TestFixture]
public class BandedAttentionTests
{
    const int SeqLen = 8;
    const int ModelDim = 16;
    const int NumHeads = 4;
    const int HeadDim = ModelDim / NumHeads;
    const int Band = 1;
    const int ValidLength = 6;
    const int FullyMaskedRow = 7;
    const float Scale = 0.5f;

    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void BandedAttention_ModernBertMasksBuild_MatchesPyTorchBandAndPadding()
    {
        var expected = TestHelpers.LoadBin("attn_band_padding_mask.bin");
        var mask = ModernBertMasks.Build<float>(SeqLen, Band, ValidLength);

        Assert.That(mask.Shape, Is.EqualTo(new[] { SeqLen, SeqLen }));
        TestHelpers.AssertTensorClose(expected, RowMajor(mask), label: "band+padding mask");
    }

    [Test]
    public void BandedAttention_VisibleKeyCounts_MatchTheFixtureGeometry()
    {
        var mask = ModernBertMasks.Build<float>(SeqLen, Band, ValidLength);
        var data = RowMajor(mask);
        var visible = new int[SeqLen];
        for (int i = 0; i < SeqLen; i++)
            for (int j = 0; j < SeqLen; j++)
                if (data[i * SeqLen + j] == 0f) visible[i]++;

        Assert.That(visible, Is.EqualTo(new[] { 2, 3, 3, 3, 3, 2, 1, 0 }));
        Assert.That(visible[FullyMaskedRow], Is.Zero, "row 7 must be the fully-masked row");
    }

    [Test]
    public void BandedAttention_NegInfFill_ForwardAndBackward_MatchPyTorch()
    {
        var q = Matrix("attn_band_padding_q.bin", requiresGrad: true);
        var k = Matrix("attn_band_padding_k.bin", requiresGrad: true);
        var v = Matrix("attn_band_padding_v.bin", requiresGrad: true);
        var dout = Matrix("attn_band_padding_dout.bin", requiresGrad: false);
        var mask = ModernBertMasks.Build<float>(SeqLen, Band, ValidLength);

        var output = ReverseGradOperations.MultiHeadAttention(q, k, v, NumHeads, Scale, mask);

        Assert.That(output.Shape, Is.EqualTo(new[] { SeqLen, ModelDim }));
        var actual = TestHelpers.ExtractOutput(output);
        TestHelpers.AssertTensorClose(TestHelpers.LoadBin("attn_band_padding_output.bin"), actual,
            label: "band+padding output");

        output.Backward(dout);

        TestHelpers.AssertTensorClose(TestHelpers.LoadBin("attn_band_padding_dq.bin"), GradArray(q), label: "band+padding dQ");
        TestHelpers.AssertTensorClose(TestHelpers.LoadBin("attn_band_padding_dk.bin"), GradArray(k), label: "band+padding dK");
        TestHelpers.AssertTensorClose(TestHelpers.LoadBin("attn_band_padding_dv.bin"), GradArray(v), label: "band+padding dV");
    }

    [Test]
    public void BandedAttention_FullyMaskedRow_ProducesZeros_NotNaN()
    {
        var q = Matrix("attn_band_padding_q.bin", requiresGrad: false);
        var k = Matrix("attn_band_padding_k.bin", requiresGrad: false);
        var v = Matrix("attn_band_padding_v.bin", requiresGrad: false);
        var mask = ModernBertMasks.Build<float>(SeqLen, Band, ValidLength);

        var actual = TestHelpers.ExtractOutput(
            ReverseGradOperations.MultiHeadAttention(q, k, v, NumHeads, Scale, mask));

        var row = actual.Skip(FullyMaskedRow * ModelDim).Take(ModelDim).ToArray();
        Assert.That(row, Is.All.EqualTo(0f), "a fully-masked query row must be exactly zeros");
        Assert.That(actual.Any(float.IsNaN), Is.False, "no NaN may reach the output");
    }

    [Test]
    public void BandedAttention_FinfoMinFill_Forward_MatchesPyTorch()
    {
        var q = Matrix("attn_band_padding_minfill_q.bin", requiresGrad: false);
        var k = Matrix("attn_band_padding_minfill_k.bin", requiresGrad: false);
        var v = Matrix("attn_band_padding_minfill_v.bin", requiresGrad: false);

        var output = ReverseGradOperations.MultiHeadAttention(q, k, v, NumHeads, Scale,
            FinfoMinMask(SeqLen, Band, ValidLength));

        // Forward agrees on every row, including the fully-masked one: finfo.min + score
        // saturates the score away (the float32 ULP at 3.4e38 is ~2e31), so row 7's scores
        // collapse to a single constant and both sides reduce it to a uniform average of V.
        TestHelpers.AssertTensorClose(TestHelpers.LoadBin("attn_band_padding_minfill_output.bin"),
            TestHelpers.ExtractOutput(output), label: "finfo.min output");
    }

    /// <summary>
    /// The exact derivative under HuggingFace's <c>finfo.min</c> convention, which is
    /// <b>not</b> what the PyTorch <c>minfill</c> backward fixtures contain.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Row 7 is fully masked, and <c>finfo.min</c> saturates: the float32 ULP at 3.4e38 is ~2e31,
    /// so <c>score + finfo.min</c> returns <c>finfo.min</c> for every cell and the row collapses to
    /// one constant. Its softmax is therefore the uniform <c>1/L</c>, and the row's output is
    /// <c>mean(V)</c> — <em>locally constant</em> in <c>q_7</c>. Central finite differences in
    /// float64 return bit-identical losses for that row under both <c>finfo(f64).min</c> and
    /// <c>finfo(f32).min</c>, so <c>dq[7]</c> is exactly zero rather than merely small.
    /// </para>
    /// <para>
    /// The <c>-inf</c> fill is the control that makes this a claim about saturation rather than
    /// about masking: a <em>non</em>-saturating finite fill cancels inside the softmax and leaves
    /// the ordinary unmasked gradient intact, so "fully masked" alone does not imply
    /// <c>dq = 0</c>. <c>finfo.min</c> is in the saturating regime by construction at any
    /// precision, which is why the zero holds for every <c>finfo.min</c> fill.
    /// </para>
    /// <para>
    /// The tolerances here are the ones <see cref="TestHelpers.AssertTensorClose"/> defaults to,
    /// matching every other parity fixture. <c>dq[7]</c> is separately asserted at <b>zero</b>
    /// tolerance against a hard <c>0f</c> rather than a fixture — it is a cleared span, so any
    /// nonzero value there is a structural defect rather than a rounding difference. The
    /// <c>dk</c> and <c>dq</c> rows 0-6 comparisons stay at fixture tolerance because these are
    /// cross-library references: PyTorch and Nivara order the same summation differently, which is
    /// worth ~2e-7 here. The bit-exact claim is Nivara-versus-Nivara and is asserted by
    /// <see cref="BandedAttention_FinfoMinFill_Backward_EqualsNegInfRunExceptForTheUniformVTerm"/>.
    /// </para>
    /// </remarks>
    [Test]
    public void BandedAttention_FinfoMinFill_Backward_EqualsExactDerivative()
    {
        var q = Matrix("attn_band_padding_minfill_q.bin", requiresGrad: true);
        var k = Matrix("attn_band_padding_minfill_k.bin", requiresGrad: true);
        var v = Matrix("attn_band_padding_minfill_v.bin", requiresGrad: true);
        var dout = Matrix("attn_band_padding_minfill_dout.bin", requiresGrad: false);

        var output = ReverseGradOperations.MultiHeadAttention(q, k, v, NumHeads, Scale,
            FinfoMinMask(SeqLen, Band, ValidLength));
        output.Backward(dout);

        var dq = GradArray(q);
        var dk = GradArray(k);
        var dv = GradArray(v);
        int visibleCount = FullyMaskedRow * ModelDim;

        Assert.That(dq.Concat(dk).Concat(dv).Any(float.IsNaN), Is.False, "gradients stay finite");
        Assert.That(dq.Skip(visibleCount).Take(ModelDim), Is.All.EqualTo(0f),
            "dq of a saturated fully-masked row must be exactly zero, not merely small");

        Assert.Multiple(() =>
        {
            TestHelpers.AssertTensorClose(
                TestHelpers.LoadBin("attn_band_padding_dq.bin").Take(visibleCount).ToArray(),
                dq.Take(visibleCount).ToArray(), label: "finfo.min dQ rows 0-6");

            TestHelpers.AssertTensorClose(TestHelpers.LoadBin("attn_band_padding_dk.bin"), dk,
                label: "finfo.min dK");

            TestHelpers.AssertTensorClose(
                GradientsUnder(MaskFill.NegInf).dv.Zip(UniformMaskedRowVTerm(),
                    (a, t) => a + t).ToArray(), dv,
                absTol: 1e-5f, relTol: 1e-5f, label: "finfo.min dV");
        });
    }

    /// <summary>
    /// The bit-exact claim, which the fixture comparison above cannot make: Nivara's
    /// <c>finfo.min</c> backward is bit-for-bit its own <c>-inf</c> backward except for the one
    /// term the V path genuinely keeps. Same inputs, same kernels, same order — so this is a
    /// zero-tolerance structural assertion rather than a tolerance band over a foreign library.
    /// </summary>
    [Test]
    public void BandedAttention_FinfoMinFill_Backward_EqualsNegInfRunExceptForTheUniformVTerm()
    {
        var finfoMin = GradientsUnder(MaskFill.FinfoMin);
        var negInf = GradientsUnder(MaskFill.NegInf);

        Assert.Multiple(() =>
        {
            TestHelpers.AssertTensorClose(negInf.dq, finfoMin.dq, absTol: 0f, relTol: 0f,
                label: "dq is identical across fills: both the -inf and the saturated row give zero");
            TestHelpers.AssertTensorClose(negInf.dk, finfoMin.dk, absTol: 0f, relTol: 0f,
                label: "dk is identical across fills - the cleared dS row adds only zeros");
            // Asserted on the difference rather than the sum, so the tolerance measures only this
            // one term and not the magnitude of dv itself. The residual is the position of the
            // saturated row's term inside MatMul's qLen-term accumulation: dV here is
            // MatMul(pT, dOH), and the term is added at a different point in the sum than when it
            // is added to an already-summed -inf value. It is summation order, not slack in the
            // derivative - which is why this is a difference assertion and the dq/dk comparisons
            // above are not.
            TestHelpers.AssertTensorClose(UniformMaskedRowVTerm(), Subtract(finfoMin.dv, negInf.dv),
                absTol: 1e-6f, relTol: 1e-5f, label: "dv's difference from -inf is exactly the uniform p_7 term");
        });
    }

    /// <summary>
    /// The <c>-inf</c> control on the zero above: its <c>dq</c> is already exactly zero on the
    /// fully-masked row, because <c>-inf</c> suppresses outright and <c>P = 0</c> there. So the
    /// zero-tolerance <c>dq</c> comparison above is a real test rather than two zeros matching.
    /// </summary>
    [Test]
    public void BandedAttention_NegInfFill_Backward_AlsoZeroesTheFullyMaskedRowDq()
    {
        var negInf = GradientsUnder(MaskFill.NegInf);

        Assert.That(negInf.dq.Skip(FullyMaskedRow * ModelDim).Take(ModelDim), Is.All.EqualTo(0f),
            "P = 0 on a -inf-masked row, so its dq is exactly zero too");
    }

    enum MaskFill { NegInf, FinfoMin }

    (float[] dq, float[] dk, float[] dv) GradientsUnder(MaskFill fill)
    {
        var q = Matrix("attn_band_padding_minfill_q.bin", requiresGrad: true);
        var k = Matrix("attn_band_padding_minfill_k.bin", requiresGrad: true);
        var v = Matrix("attn_band_padding_minfill_v.bin", requiresGrad: true);
        var dout = Matrix("attn_band_padding_minfill_dout.bin", requiresGrad: false);
        var mask = fill == MaskFill.NegInf
            ? ModernBertMasks.Build<float>(SeqLen, Band, ValidLength)
            : FinfoMinMask(SeqLen, Band, ValidLength);

        var output = ReverseGradOperations.MultiHeadAttention(q, k, v, NumHeads, Scale, mask);
        output.Backward(dout);

        return (GradArray(q), GradArray(k), GradArray(v));
    }

    /// <summary>
    /// The one term the V path genuinely keeps from a saturated row: <c>p_7 &#183; dout[7]</c>
    /// with <c>p_7 = 1/L</c>, broadcast to every key. <c>dV = MatMul(pT, dOH)</c> sums
    /// <c>p[i,j] * dout[i]</c> over query rows, so the saturated row adds this to each key.
    /// </summary>
    float[] UniformMaskedRowVTerm()
    {
        var doutRow = TestHelpers.LoadBin("attn_band_padding_minfill_dout.bin")
            .Skip(FullyMaskedRow * ModelDim).Take(ModelDim).ToArray();
        var term = new float[SeqLen * ModelDim];

        for (int h = 0; h < NumHeads; h++)
            for (int j = 0; j < SeqLen; j++)
                for (int d = 0; d < HeadDim; d++)
                    term[j * ModelDim + h * HeadDim + d] = doutRow[h * HeadDim + d] / SeqLen;

        return term;
    }

    static float[] Subtract(float[] a, float[] b) =>
        a.Zip(b, (x, y) => x - y).ToArray();

    [Test]
    public void BandedAttention_MaskFillConventions_DivergeOnlyOnTheFullyMaskedRow()
    {
        var q = Matrix("attn_band_padding_q.bin", requiresGrad: false);
        var k = Matrix("attn_band_padding_k.bin", requiresGrad: false);
        var v = Matrix("attn_band_padding_v.bin", requiresGrad: false);

        var negInf = TestHelpers.ExtractOutput(ReverseGradOperations.MultiHeadAttention(
            q, k, v, NumHeads, Scale, ModernBertMasks.Build<float>(SeqLen, Band, ValidLength)));
        var finfoMin = TestHelpers.ExtractOutput(ReverseGradOperations.MultiHeadAttention(
            q, k, v, NumHeads, Scale, FinfoMinMask(SeqLen, Band, ValidLength)));

        for (int row = 0; row < SeqLen; row++)
        {
            var a = negInf.Skip(row * ModelDim).Take(ModelDim);
            var b = finfoMin.Skip(row * ModelDim).Take(ModelDim);
            if (row == FullyMaskedRow) continue;
            TestHelpers.AssertTensorClose(b.ToArray(), a.ToArray(), absTol: 0f, relTol: 0f,
                label: $"row {row} must be bit-identical across mask fills");
        }

        var negInfRow = negInf.Skip(FullyMaskedRow * ModelDim).Take(ModelDim);
        var finfoMinRow = finfoMin.Skip(FullyMaskedRow * ModelDim).Take(ModelDim);
        Assert.That(negInfRow, Is.All.EqualTo(0f), "-inf fill clamps the fully-masked row to zeros");
        Assert.That(finfoMinRow.Any(x => x != 0f), Is.True,
            "finfo.min fill yields a uniform average of V instead of zeros (HuggingFace's convention)");
        Assert.That(finfoMinRow.Any(float.IsNaN), Is.False, "both conventions stay finite");
    }

    [Test]
    public void BandedAttention_NonFiniteMaskCells_PropagateAsNaNRows_MatchingPyTorch()
    {
        var q = Matrix("attn_mask_nonfinite_q.bin", requiresGrad: false);
        var k = Matrix("attn_mask_nonfinite_k.bin", requiresGrad: false);
        var v = Matrix("attn_mask_nonfinite_v.bin", requiresGrad: false);

        var actual = TestHelpers.ExtractOutput(ReverseGradOperations.MultiHeadAttention(
            q, k, v, NumHeads, Scale, Matrix("attn_mask_nonfinite_mask.bin", SeqLen, SeqLen, requiresGrad: false)));

        TestHelpers.AssertTensorClose(TestHelpers.LoadBin("attn_mask_nonfinite_output.bin"), actual,
            label: "non-finite mask cells");

        for (int row = 0; row < SeqLen; row++)
        {
            var slice = actual.Skip(row * ModelDim).Take(ModelDim);
            bool poisoned = row is 0 or 1;
            Assert.That(slice.Any(float.IsNaN), Is.EqualTo(poisoned),
                $"row {row}: +inf/NaN mask cells must poison exactly that row and no other");
        }
    }

    [Test]
    public void AssertTensorClose_DetectsNonFiniteMismatch_ThatAssertTensorEqualCannot()
    {
        // Guards the reason this helper exists. AssertTensorEqual computes
        // `diff > threshold`, and `NaN > x` is false, so a NaN reference compared against a
        // finite result must pass there. If that ever stops being true this test fails and the
        // justification for AssertTensorClose needs revisiting.
        float[] withNaN = { 1f, float.NaN, 3f };
        float[] finite = { 1f, 2f, 3f };

        Assert.DoesNotThrow(() => TestHelpers.AssertTensorEqual(withNaN, finite, label: "legacy"),
            "AssertTensorEqual is expected to miss this mismatch - that is the documented gap");
        var ex = Assert.Throws<AssertionException>(() =>
            TestHelpers.AssertTensorClose(withNaN, finite, label: "NaN vs finite"));
        Assert.That(ex!.Message, Does.Contain("NaN mismatch"));

        Assert.DoesNotThrow(() => TestHelpers.AssertTensorClose(withNaN, withNaN, label: "NaN vs NaN"));
        Assert.Throws<AssertionException>(() =>
            TestHelpers.AssertTensorClose(new[] { float.PositiveInfinity }, new[] { float.MaxValue }));
        Assert.Throws<AssertionException>(() =>
            TestHelpers.AssertTensorClose(new[] { float.NegativeInfinity }, new[] { float.MinValue }));
    }

    static ReverseGradTensor<float> Matrix(string name, bool requiresGrad) =>
        Matrix(name, SeqLen, ModelDim, requiresGrad);

    static ReverseGradTensor<float> Matrix(string name, int rows, int cols, bool requiresGrad)
    {
        var data = TestHelpers.LoadBin(name);
        Assert.That(data.Length, Is.EqualTo(rows * cols), $"Unexpected {name} length {data.Length}");
        return ReverseGradTensor<float>.FromMatrix(data, rows, cols, requiresGrad);
    }

    /// <summary>
    /// HuggingFace's convention: the suppressed constant is <c>torch.finfo(dtype).min</c> rather
    /// than <c>-inf</c>. <see cref="ModernBertMasks"/> deliberately uses <c>-inf</c>, so this
    /// exists only to test the two conventions side by side.
    /// </summary>
    static ReverseGradTensor<float> FinfoMinMask(int seqLen, int band, int validLength)
    {
        var data = new float[seqLen * seqLen];
        for (var i = 0; i < seqLen; i++)
        {
            int first = Math.Max(0, i - band);
            int last = Math.Min(validLength - 1, i + band);
            for (int j = 0; j < seqLen; j++)
                data[i * seqLen + j] = j < first || j > last ? float.MinValue : 0f;
        }
        return ReverseGradTensor<float>.FromMatrix(data, seqLen, seqLen, requiresGrad: false);
    }

    static float[] RowMajor(ReverseGradTensor<float> tensor)
    {
        var result = new float[tensor.Length];
        for (int i = 0; i < tensor.Length; i++) result[i] = tensor[i];
        return result;
    }

    static float[] GradArray(ReverseGradTensor<float> tensor)
    {
        Assert.That(tensor.Grad, Is.Not.Null, $"{nameof(tensor)}.Grad should be populated after backward");
        var grad = tensor.Grad!;
        var result = new float[grad.Length];
        for (int i = 0; i < grad.Length; i++) result[i] = grad[i];
        return result;
    }
}
