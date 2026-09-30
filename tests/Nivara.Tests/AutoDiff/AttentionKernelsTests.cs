using Nivara.AutoDiff.Operations;
using NUnit.Framework;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Tests for <see cref="AttentionKernels{T}.ApplyMask{T}"/>, the additive attention mask that
/// <c>MultiHeadAttention</c> / <c>BatchedMultiHeadAttention</c> apply to the QK^T score buffer.
///
/// <para>
/// The kernel suppresses a cell by <em>assigning</em> <c>-inf</c> rather than summing
/// <c>score + (-inf)</c>. The distinction is only observable when the score is already non-finite:
/// <c>NaN + (-inf) = NaN</c> and <c>(+inf) + (-inf) = NaN</c>, so an additive mask lets a diverged
/// score escape suppression and poison every query row in the frame one layer later.
/// </para>
///
/// <para>
/// The suite is built around a <em>zero-delta</em> contract: for every (mask, score) pair except
/// the two that are the bug — a <c>NaN</c> or <c>+inf</c> score under a <c>-inf</c> mask cell —
/// <c>ApplyMask</c> must produce exactly what <see cref="TensorPrimitives.Add{T}(ReadOnlySpan{T},
/// ReadOnlySpan{T}, Span{T})"/> produces. That is what lets a shared, parity-tested attention path
/// change its suppression rule without touching a single existing fixture.
/// </para>
/// </summary>
[TestFixture]
public class AttentionKernelsTests
{
    static readonly float NegInf = float.NegativeInfinity;
    static readonly float PosInf = float.PositiveInfinity;
    static readonly float NaN = float.NaN;

    [Test]
    public void ApplyMask_SuppressedCell_NaNScoreBecomesNegativeInfinity()
    {
        // The bug. Today this cell is NaN + (-inf) = NaN, so the NaN survives the mask and the
        // row softmax produces a NaN row that poisons every row of the next layer.
        var scores = new[] { 1.0f, NaN, 3.0f };
        var mask = new[] { 0.0f, NegInf, 0.0f };

        AttentionKernels<float>.ApplyMask(scores, mask);

        Assert.That(scores[1], Is.EqualTo(NegInf),
            "a suppressed cell must be assigned -inf so a NaN score cannot escape the mask");
        Assert.That(scores[0], Is.EqualTo(1.0f));
        Assert.That(scores[2], Is.EqualTo(3.0f));
    }

    [Test]
    public void ApplyMask_SuppressedCell_PositiveInfinityScoreBecomesNegativeInfinity()
    {
        // The twin cell: an overflow in a badly-scaled layer gives +inf, and (+inf) + (-inf) = NaN
        // escapes the mask by exactly the same route a NaN does.
        var scores = new[] { PosInf, 2.0f };
        var mask = new[] { NegInf, 0.0f };

        AttentionKernels<float>.ApplyMask(scores, mask);

        Assert.That(scores[0], Is.EqualTo(NegInf));
        Assert.That(scores[1], Is.EqualTo(2.0f));
    }

    [Test]
    public void ApplyMask_NaNScoreInKeptCell_PropagatesNaN()
    {
        // The guard against over-suppression. A NaN score the mask does NOT suppress must stay
        // NaN: a diverged run announcing itself is the most useful diagnostic it produces, and
        // swallowing it would answer a wrong-but-finite question.
        var scores = new[] { NaN, 2.0f };
        var mask = new[] { 0.0f, NegInf };

        AttentionKernels<float>.ApplyMask(scores, mask);

        Assert.That(float.IsNaN(scores[0]), Is.True,
            "an unsuppressed NaN must propagate; suppression is the mask's job, not the kernel's");
        Assert.That(scores[1], Is.EqualTo(NegInf));
    }

    [Test]
    public void ApplyMask_NaNMaskCell_StillPropagatesNaN()
    {
        // `NaN == -inf` is false (IEEE 754 unordered comparison), so a NaN mask cell takes the
        // additive branch. This is what keeps
        // BandedAttention_NonFiniteMaskCells_PropagateAsNaNRows_MatchingPyTorch a parity fixture:
        // PyTorch computes `score + mask`, so a NaN mask cell poisons that row there too.
        var scores = new[] { 1.0f, 2.0f };
        var mask = new[] { NaN, 0.0f };

        AttentionKernels<float>.ApplyMask(scores, mask);

        Assert.That(float.IsNaN(scores[0]), Is.True);
        Assert.That(scores[1], Is.EqualTo(2.0f));
    }

    [Test]
    public void ApplyMask_PositiveInfinityMaskCell_StaysAdditive()
    {
        // The other non-finite fill. +inf is not a suppression signal, it is a bias, so the cell
        // adds and the row's max becomes +inf -> the row softmax yields NaN, matching PyTorch.
        var scores = new[] { 1.0f, 2.0f };
        var mask = new[] { PosInf, 0.0f };

        AttentionKernels<float>.ApplyMask(scores, mask);

        Assert.That(scores[0], Is.EqualTo(PosInf));
        Assert.That(scores[1], Is.EqualTo(2.0f));
    }

    [Test]
    public void ApplyMask_FiniteFillMagnitude_IsPreserved()
    {
        // HuggingFace fills with finfo.min, not -inf, and the magnitude is load-bearing: the
        // float32 ULP at 3.4e38 is ~2e31, so `score + finfo.min` saturates to finfo.min and a
        // fully-masked row collapses to one constant. A select that discarded the magnitude would
        // turn this into a raw softmax over undamped scores and break
        // BandedAttention_FinfoMinFill_Forward_MatchesPyTorch.
        var scores = new[] { 1.0f, -2.5f, 0.0f };
        var mask = new[] { float.MinValue, 0.0f, float.MinValue };

        AttentionKernels<float>.ApplyMask(scores, mask);

        Assert.That(scores[0], Is.EqualTo(float.MinValue), "score + finfo.min saturates to finfo.min");
        Assert.That(scores[1], Is.EqualTo(-2.5f));
        Assert.That(scores[2], Is.EqualTo(float.MinValue));
    }

    [Test]
    public void ApplyMask_OrdinaryMask_MatchesAdditiveAddBitForBit()
    {
        // Every mask builder in the tree (CreateCausalMask, CreateBlockDiagonalMask,
        // CreatePaddingMask, ModernBertMasks.Build, BertModel) emits exactly {0, -inf}. Over that
        // domain ApplyMask must be indistinguishable from the TensorPrimitives.Add it replaced.
        var rng = new Random(0x448);
        var scores = new float[512];
        var mask = new float[512];
        for (int i = 0; i < scores.Length; i++)
        {
            scores[i] = (float)(rng.NextDouble() * 4.0 - 2.0);
            mask[i] = rng.Next(3) == 0 ? NegInf : 0.0f;
        }

        var expected = new float[scores.Length];
        TensorPrimitives.Add(scores, mask, expected);

        AttentionKernels<float>.ApplyMask(scores, mask);

        Assert.That(scores, Is.EqualTo(expected),
            "a {0, -inf} mask must produce bit-identical output to the additive form");
    }

    [Test]
    public void ApplyMask_PerBatchSlice_SuppressesEachBatchIndependently()
    {
        // The batched ops slice a [B, qLen, kvLen] mask per batch element. Batch 0 is fully
        // masked, batch 1 fully open, and batch 2 half masked, so a wrong slice offset cannot
        // hide: any of the three would land the wrong cells.
        const int batch = 3, qLen = 4, kvLen = 5;
        int cellCount = qLen * kvLen;
        var seed = new float[cellCount];
        for (int i = 0; i < cellCount; i++)
            seed[i] = (i % 3) + 0.5f;

        var masks = new float[batch * cellCount];
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < cellCount; i++)
                masks[b * cellCount + i] = b switch
                {
                    0 => NegInf,
                    1 => 0.0f,
                    _ => i % 2 == 0 ? 0.0f : NegInf,
                };

        for (int b = 0; b < batch; b++)
        {
            var scores = (float[])seed.Clone();

            AttentionKernels<float>.ApplyMask(scores, masks.AsSpan(b * cellCount, cellCount));

            switch (b)
            {
                case 0:
                    Assert.That(scores, Is.All.EqualTo(NegInf), $"batch {b} is fully masked");
                    break;
                case 1:
                    Assert.That(scores, Is.EqualTo(seed), $"batch {b} is fully open and must be untouched");
                    break;
                default:
                    for (int i = 0; i < cellCount; i++)
                        Assert.That(scores[i], Is.EqualTo(i % 2 == 0 ? seed[i] : NegInf), $"batch {b} index {i}");
                    break;
            }
        }
    }

    [Test]
    public void ApplyMask_ThenSoftmax_SuppressedNaNCellYieldsZeroProbabilityNotNaN()
    {
        // The end-to-end form of the acceptance criterion: a NaN in a suppressed score position
        // must softmax to zero probability, not to a NaN row.
        var scores = new[] { 1.0f, 2.0f, NaN, 4.0f };
        var mask = new[] { 0.0f, 0.0f, NegInf, 0.0f };
        var actual = new float[scores.Length];

        AttentionKernels<float>.ApplyMask(scores, mask);
        GradKernels.Softmax<float>(scores, actual, scores.Length);

        Assert.That(actual[2], Is.EqualTo(0.0f), "the suppressed NaN cell must have zero probability");
        Assert.That(actual.Any(float.IsNaN), Is.False, "no NaN may reach the softmax output");
        Assert.That(actual.Sum(), Is.EqualTo(1.0f).Within(1e-6));
    }

    [Test]
    public void ApplyMask_ComposedWithRowSoftmax_FullyMaskedRowWithNaNScoreClampsToZeros()
    {
        // Two layers of the same defence. A fully-masked row whose score row is entirely NaN now
        // masks cleanly to -inf, which trips GradKernels' `max == -inf` clamp (the PyTorch
        // _safe_softmax behaviour), so the row is zeros rather than the NaN row that would
        // otherwise escape again on the next layer.
        var scores = new[] { NaN, NaN, NaN, 0.0f, 1.0f, 2.0f };
        var mask = new[] { NegInf, NegInf, NegInf, 0.0f, 0.0f, 0.0f };
        var actual = new float[scores.Length];

        AttentionKernels<float>.ApplyMask(scores, mask);
        GradKernels.Softmax<float>(scores, actual, 3);

        Assert.That(actual.Take(3), Is.All.EqualTo(0.0f),
            "the NaN row must not escape the mask and must clamp to zeros");
        Assert.That(actual.Skip(3).Sum(), Is.EqualTo(1.0f).Within(1e-6),
            "the healthy row is unaffected, so the clamp decision is per-row");
    }

    [Test]
    public void ApplyMask_Double_And_NarrowTypes_BehaveIdentically()
    {
        // ApplyMask is generic over IFloatingPointIeee754<T> (ADR-001's non-nullable domain), so
        // the suppression rule has to hold for float, double, Half and BFloat16 alike.
        AssertSuppressesNaN(double.NaN, double.NegativeInfinity);
        AssertSuppressesNaN(Half.NaN, Half.NegativeInfinity);
        AssertSuppressesNaN(BFloat16.NaN, BFloat16.NegativeInfinity);
    }

    [Test]
    public void ApplyMask_LongerMask_ThrowsRatherThanSilentlyTruncating()
    {
        // TensorPrimitives.Add required exact length equality. The bare loop would fault on a short
        // mask but silently truncate a long one, turning a mis-shaped mask into a wrong answer
        // rather than an exception.
        var scores = new[] { 1.0f, 2.0f };
        var tooLong = new[] { 0.0f, 0.0f, NegInf, NegInf };

        var ex = Assert.Throws<ArgumentException>(() => AttentionKernels<float>.ApplyMask(scores, tooLong));
        Assert.That(ex!.Message, Does.Contain("does not match"));
    }

    [Test]
    public void ApplyMask_ShorterMask_ThrowsRatherThanIndexingOutOfRange()
    {
        var scores = new[] { 1.0f, 2.0f, 3.0f };
        var tooShort = new[] { 0.0f, NegInf };

        var ex = Assert.Throws<ArgumentException>(() => AttentionKernels<float>.ApplyMask(scores, tooShort));
        Assert.That(ex!.Message, Does.Contain("does not match"));
    }

    static void AssertSuppressesNaN<T>(T nan, T negInf)
        where T : struct, IFloatingPointIeee754<T>
    {
        var scores = new[] { T.CreateChecked(1.0), nan };
        var mask = new[] { T.Zero, negInf };

        AttentionKernels<T>.ApplyMask(scores, mask);

        Assert.That(scores[1], Is.EqualTo(negInf), $"{typeof(T).Name}: a NaN score must not escape suppression");
        Assert.That(scores[0], Is.EqualTo(T.One), $"{typeof(T).Name}: a kept cell must be untouched");
    }
}