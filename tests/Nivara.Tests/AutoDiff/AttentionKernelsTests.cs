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
    public void ApplyMask_SuppressedCell_AlreadyNegativeInfinityScoreStaysNegativeInfinity()
    {
        // The last untested row of the zero-delta table: an already-suppressed score under a -inf
        // mask. Additive today gives -inf + (-inf) = -inf, and the select gives -inf, so nothing
        // changed -- which is exactly why it needs a test rather than an argument. A double-add,
        // a saturating combine, or a clamp written against +inf would each pass every other test
        // in this fixture and break only here.
        var scores = new[] { 1.0f, NegInf, 3.0f };
        var mask = new[] { 0.0f, NegInf, NegInf };

        AttentionKernels<float>.ApplyMask(scores, mask);

        Assert.That(scores[1], Is.EqualTo(NegInf), "-inf + (-inf) must stay -inf, not become NaN");
        Assert.That(scores[2], Is.EqualTo(NegInf), "a finite score under -inf is still suppressed");
        Assert.That(scores[0], Is.EqualTo(1.0f));
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
        //
        // Seeded with -0.0f deliberately: for a 0 mask cell, `score + 0.0f` is +0.0f, so returning
        // `score` unchanged would be value-equal but not bit-equal. NUnit's Is.EqualTo on float
        // collections would not notice, so this compares the raw bit patterns -- the assertion the
        // test name claims.
        var rng = new Random(0x448);
        var scores = new float[512];
        var mask = new float[512];
        for (int i = 0; i < scores.Length; i++)
        {
            scores[i] = i % 4 == 0 ? -0.0f : (float)(rng.NextDouble() * 4.0 - 2.0);
            mask[i] = rng.Next(3) == 0 ? NegInf : 0.0f;
        }

        var expected = new float[scores.Length];
        TensorPrimitives.Add(scores, mask, expected);

        AttentionKernels<float>.ApplyMask(scores, mask);

        for (int i = 0; i < scores.Length; i++)
            Assert.That(BitConverter.SingleToInt32Bits(scores[i]), Is.EqualTo(BitConverter.SingleToInt32Bits(expected[i])),
                $"index {i}: a {{0, -inf}} mask must produce a bit-identical result to the additive form");
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

    [Test]
    public void ApplyMask_FinfoMinFill_AnnihilatesEveryScore_AndFlagsTheRow()
    {
        // The saturating fill. score + finfo.min == finfo.min for any score small enough that the
        // float32 ULP at 3.4e38 (~2e31) swallows it, so the row's value no longer reflects the
        // score at all. The row is therefore score-independent and the backward must zero it -
        // which is the whole of #454.
        var scores = new[] { 1.0f, -7.5f, 0.25f, 12f };
        var mask = new[] { float.MinValue, float.MinValue, float.MinValue, float.MinValue };
        var rowFlags = new bool[1];

        AttentionKernels<float>.ApplyMask(scores, mask, rowFlags, 1, 4);

        Assert.Multiple(() =>
        {
            Assert.That(rowFlags, Is.EqualTo(new[] { true }),
                "a fully-saturated row carries no score information and must be flagged");
            Assert.That(scores, Is.All.EqualTo(float.MinValue),
                "every cell saturates to the fill itself");
        });
    }

    [Test]
    public void ApplyMask_OrdinaryMask_FlagsNoRow()
    {
        // The shipped convention must not be flagged. Every mask builder in the tree emits
        // {0, -inf}; a row that keeps any cell open still depends on its scores and must be left
        // alone. If this ever flags, every -inf attention backward in the repo is zeroed.
        var scores = new[] { 1.0f, -7.5f, 0.25f, 12f, 2f, 3f, 4f, 5f };
        var mask = new[] { NegInf, NegInf, 0.0f, NegInf, 0f, 0f, 0f, 0f };
        var rowFlags = new bool[2];

        AttentionKernels<float>.ApplyMask(scores, mask, rowFlags, 2, 4);

        Assert.That(rowFlags, Is.EqualTo(new[] { false, false }),
            "a row with any live cell is still score-dependent");
        Assert.That(scores.Take(4), Is.EqualTo(new[] { NegInf, NegInf, 0.25f, NegInf }));
        Assert.That(scores.Skip(4), Is.EqualTo(new[] { 2f, 3f, 4f, 5f }));
    }

    [Test]
    public void ApplyMask_PartialSaturation_KeepsRowLive_AndLeavesNoFlag()
    {
        // The boundary of the per-row flag, and the reason it is per-row rather than per-cell.
        // A large-but-not-finfo.min fill leaves small scores intact, so this row's softmax is
        // the ordinary one after the additive constant cancels, and its output still depends on
        // q. A row-level flag can only answer "did the fill kill the whole row"; it says yes
        // here is wrong, and #489 tracks the residual per-cell defect this shape leaves behind.
        var scores = new[] { 1e-30f, -1e-30f, 0.0f, 1e-30f };
        var mask = new[] { -1e30f, -1e30f, -1e30f, -1e30f };
        var rowFlags = new bool[1];

        AttentionKernels<float>.ApplyMask(scores, mask, rowFlags, 1, 4);

        Assert.Multiple(() =>
        {
            Assert.That(rowFlags, Is.EqualTo(new[] { false }),
                "a non-saturating fill must not flag the row: the softmax cancels the constant");
            Assert.That(scores[0], Is.EqualTo(-1e30f), "the fill is large enough to dominate the cell");
            Assert.That(scores[0], Is.Not.EqualTo(mask[0]),
                "yet the add is not an absorption - the score changed the stored value");
        });
    }

    [Test]
    public void ApplyMask_NegInfFill_FlagsTheSuppressedRow()
    {
        // -inf needs no special case: the suppressed cells are assigned -inf and count as dead,
        // so a row that is entirely -inf flags exactly as it does under finfo.min. On this path
        // P is already 0, so clearing dS is a no-op - which is what keeps the -inf fixtures
        // bit-for-bit identical after #454.
        var scores = new[] { 1.0f, NegInf, 3.0f, NegInf, NegInf, NegInf, NegInf, NegInf };
        var mask = new[] { 0.0f, NegInf, 0.0f, NegInf, NegInf, NegInf, NegInf, NegInf };
        var rowFlags = new bool[2];

        AttentionKernels<float>.ApplyMask(scores, mask, rowFlags, 2, 4);

        Assert.That(rowFlags, Is.EqualTo(new[] { false, true }));
    }

    [Test]
    public void ApplyMask_NaNMaskCell_CountsAsLiveSoTheRowIsNotFlagged()
    {
        // `NaN == m` is false and `NaN + s` is NaN, so a NaN cell is never an absorption. It
        // poisons the row through the softmax instead (see
        // BandedAttention_NonFiniteMaskCells_PropagateAsNaNRows_MatchingPyTorch), and flagging the
        // row here would silently swallow that signal.
        var scores = new[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var mask = new[] { NaN, NaN, NaN, NaN };
        var rowFlags = new bool[1];

        AttentionKernels<float>.ApplyMask(scores, mask, rowFlags, 1, 4);

        Assert.Multiple(() =>
        {
            Assert.That(rowFlags, Is.EqualTo(new[] { false }));
            Assert.That(scores, Is.All.Matches<float>(float.IsNaN),
                "a NaN mask cell stays additive, so the whole row goes NaN");
        });
    }

    [Test]
    public void ApplyMask_MismatchedRowFlags_ThrowsRatherThanWritingOutOfRange()
    {
        var scores = new[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var mask = new[] { 0.0f, 0.0f, 0.0f, 0.0f };

        var ex = Assert.Throws<ArgumentException>(() =>
            AttentionKernels<float>.ApplyMask(scores, mask, new bool[3], 2, 2));
        Assert.That(ex!.Message, Does.Contain("does not match rows"));
    }

    [Test]
    public void SoftmaxBackwardRows_FlaggedRow_ProducesExactlyZeroDScores()
    {
        // The consume side. A flagged row's dS must be exactly zero, not approximately: the
        // softmax VJP is `P * (dP - dot)`, and with P uniform on a saturated row both terms are
        // non-zero, so this is where the spurious gradient is born.
        var weights = new[] { 0.25f, 0.25f, 0.25f, 0.25f };
        var dS = new[] { 3.0f, -1.0f, 2.0f, 0.5f };

        AttentionKernels<float>.SoftmaxBackwardRows(weights, dS, 1, 4, new[] { true });

        Assert.That(dS, Is.All.EqualTo(0f),
            "a score-independent row contributes no score gradient at all");
    }

    [Test]
    public void SoftmaxBackwardRows_UnflaggedRow_IsUnchangedByTheOverload()
    {
        // The flag must be inert on every row that is not saturated, or the -inf and unmasked
        // attention paths would all change. Pinned to the exact pre-flag values.
        var weights = new[] { 0.25f, 0.25f, 0.25f, 0.25f, 1f, 0f, 0f, 0f };
        var dS = new[] { 3.0f, -1.0f, 2.0f, 0.5f, 4.0f, 0f, 0f, 0f };
        var expected = (float[])dS.Clone();

        var baseline = (float[])dS.Clone();
        AttentionKernels<float>.SoftmaxBackwardRows(weights, baseline, 2, 4);
        AttentionKernels<float>.SoftmaxBackwardRows(weights, dS, 2, 4, new[] { false, false });

        Assert.Multiple(() =>
        {
            Assert.That(dS, Is.EqualTo(expected).AsCollection,
                "passing all-false flags must not perturb any element");
            Assert.That(dS, Is.EqualTo(baseline).AsCollection);
        });
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