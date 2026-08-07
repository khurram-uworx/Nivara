using Nivara.Tensors;
using NUnit.Framework;

namespace Nivara.Tests.Tensors;

/// <summary>
/// Tests for the window-function column primitives (rolling/cumulative/shift)
/// </summary>
/// <remarks>Added as part of issue #135 window functions delivery.</remarks>
[TestFixture]
public class WindowFunctionsTests
{
    // ── Cumulative ──

    [Test]
    public void CumulativeSum_NoNulls_MatchesNaive()
    {
        var column = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 });

        var result = column.CumulativeSum();

        Assert.That(result.Length, Is.EqualTo(4));
        Assert.That(result.HasNulls, Is.False);
        Assert.That(result[0], Is.EqualTo(1));
        Assert.That(result[1], Is.EqualTo(3));
        Assert.That(result[2], Is.EqualTo(6));
        Assert.That(result[3], Is.EqualTo(10));
    }

    [Test]
    public void CumulativeSum_NullsSkipped_NullPositionsStayNull()
    {
        var column = NivaraColumn<int>.CreateFromSpans(new[] { 1, 0, 2, 0, 3 }, new[] { false, true, false, true, false });

        var result = column.CumulativeSum();

        Assert.That(result.HasNulls, Is.True);
        Assert.That(result[0], Is.EqualTo(1));
        Assert.That(result.IsNull(1), Is.True);
        Assert.That(result[2], Is.EqualTo(3));
        Assert.That(result.IsNull(3), Is.True);
        Assert.That(result[4], Is.EqualTo(6));
    }

    [Test]
    public void CumulativeSum_NullHandler_ReplacesNullsAndFillsOutput()
    {
        var column = NivaraColumn<int>.CreateFromSpans(new[] { 1, 0, 2 }, new[] { false, true, false });

        var result = column.CumulativeSum(() => 5);

        Assert.That(result.HasNulls, Is.False);
        Assert.That(result[0], Is.EqualTo(1));
        Assert.That(result[1], Is.EqualTo(6));
        Assert.That(result[2], Is.EqualTo(8));
    }

    [Test]
    public void CumulativeMaxMinProduct_NullCarry_MatchesExpected()
    {
        var column = NivaraColumn<int>.CreateFromSpans(new[] { 3, 0, 1, 0, 2 }, new[] { false, true, false, true, false });

        var max = column.CumulativeMax();
        var min = column.CumulativeMin();
        var product = column.CumulativeProduct();

        Assert.That(max[0], Is.EqualTo(3));
        Assert.That(max.IsNull(1), Is.True);
        Assert.That(max[2], Is.EqualTo(3));
        Assert.That(max[4], Is.EqualTo(3));

        Assert.That(min[0], Is.EqualTo(3));
        Assert.That(min.IsNull(1), Is.True);
        Assert.That(min[2], Is.EqualTo(1));
        Assert.That(min[4], Is.EqualTo(1));

        Assert.That(product[0], Is.EqualTo(3));
        Assert.That(product.IsNull(1), Is.True);
        Assert.That(product[2], Is.EqualTo(3));
        Assert.That(product[4], Is.EqualTo(6));
    }

    [Test]
    public void CumulativeCount_CountsNonNull_NullPositionsStayNull()
    {
        var column = NivaraColumn<int>.CreateFromSpans(new[] { 1, 0, 2, 0, 3, 4 }, new[] { false, true, false, true, false, false });

        var result = column.CumulativeCount();

        Assert.That(result.Length, Is.EqualTo(6));
        Assert.That(result.HasNulls, Is.True);
        Assert.That(result[0], Is.EqualTo(1));
        Assert.That(result.IsNull(1), Is.True);
        Assert.That(result[2], Is.EqualTo(2));
        Assert.That(result.IsNull(3), Is.True);
        Assert.That(result[4], Is.EqualTo(3));
        Assert.That(result[5], Is.EqualTo(4));
    }

    [Test]
    public void Cumulative_EmptyColumn_ReturnsEmpty()
    {
        var column = NivaraColumn<int>.Create(Array.Empty<int>());

        Assert.That(column.CumulativeSum().Length, Is.EqualTo(0));
        Assert.That(column.CumulativeCount().Length, Is.EqualTo(0));
    }

    // ── Rolling ──

    [Test]
    public void RollingSum_FullWindow_NullUntilWindowFilled()
    {
        var column = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });

        var result = column.RollingSum(3);

        Assert.That(result.HasNulls, Is.True);
        Assert.That(result.IsNull(0), Is.True);
        Assert.That(result.IsNull(1), Is.True);
        Assert.That(result[2], Is.EqualTo(6));
        Assert.That(result[3], Is.EqualTo(9));
        Assert.That(result[4], Is.EqualTo(12));
    }

    [Test]
    public void RollingSum_MinPeriods2_NullOnlyUntilTwoValid()
    {
        var column = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });

        var result = column.RollingSum(3, minPeriods: 2);

        Assert.That(result.IsNull(0), Is.True);
        Assert.That(result[1], Is.EqualTo(3));
        Assert.That(result[2], Is.EqualTo(6));
        Assert.That(result[3], Is.EqualTo(9));
        Assert.That(result[4], Is.EqualTo(12));
    }

    [Test]
    public void RollingSum_NullsIgnoredWithinWindow_GatedByMinPeriods()
    {
        var column = NivaraColumn<int>.CreateFromSpans(new[] { 1, 0, 3 }, new[] { false, true, false });

        var gated = column.RollingSum(3, minPeriods: 3);
        var relaxed = column.RollingSum(3, minPeriods: 2);

        Assert.That(gated.IsNull(2), Is.True);
        Assert.That(relaxed[2], Is.EqualTo(4));
        Assert.That(relaxed.HasNulls, Is.True);
    }

    [Test]
    public void RollingSum_NullHandler_CountsReplacementAsValid()
    {
        var column = NivaraColumn<int>.CreateFromSpans(new[] { 1, 0, 3 }, new[] { false, true, false });

        var result = column.RollingSum(3, nullHandler: () => 0);

        Assert.That(result.HasNulls, Is.False);
        Assert.That(result[0], Is.EqualTo(1));
        Assert.That(result[1], Is.EqualTo(1));
        Assert.That(result[2], Is.EqualTo(4));
    }

    [Test]
    public void RollingMean_ReturnsDouble_NullUntilWindowFilled()
    {
        var column = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });

        var result = column.RollingMean(3);

        Assert.That(result, Is.InstanceOf<NivaraColumn<double>>());
        Assert.That(result.HasNulls, Is.True);
        Assert.That(result.IsNull(1), Is.True);
        Assert.That(result[2], Is.EqualTo(2.0).Within(1e-9));
        Assert.That(result[3], Is.EqualTo(3.0).Within(1e-9));
        Assert.That(result[4], Is.EqualTo(4.0).Within(1e-9));
    }

    [Test]
    public void RollingMaxMin_MonotonicDeque_MatchesNaive()
    {
        var column = NivaraColumn<int>.Create(new[] { 3, 1, 4, 1, 5, 9, 2, 6 });

        var max = column.RollingMax(3);
        var min = column.RollingMin(3);

        var expectedMax = new[] { 3, 3, 4, 4, 5, 9, 9, 9 };
        var expectedMin = new[] { 3, 1, 1, 1, 1, 1, 2, 2 };

        for (int i = 0; i < 2; i++)
        {
            Assert.That(max.IsNull(i), Is.True);
            Assert.That(min.IsNull(i), Is.True);
        }

        for (int i = 2; i < column.Length; i++)
        {
            Assert.That(max[i], Is.EqualTo(expectedMax[i]));
            Assert.That(min[i], Is.EqualTo(expectedMin[i]));
        }
    }

    [Test]
    public void Rolling_NullArguments_Throw()
    {
        var column = NivaraColumn<int>.Create(new[] { 1, 2, 3 });

        Assert.Throws<ArgumentOutOfRangeException>(() => column.RollingSum(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => column.RollingSum(3, minPeriods: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => column.RollingSum(3, minPeriods: 4));
    }

    // ── Shift / Lead ──

    [Test]
    public void Shift_PositivePeriods_LagsValues_NullBoundaries()
    {
        var column = NivaraColumn<int>.Create(new[] { 1, 2, 3 });

        var result = column.Shift(1);

        Assert.That(result.HasNulls, Is.True);
        Assert.That(result.IsNull(0), Is.True);
        Assert.That(result[1], Is.EqualTo(1));
        Assert.That(result[2], Is.EqualTo(2));
    }

    [Test]
    public void Shift_NegativePeriods_LeadsValues_NullBoundaries()
    {
        var column = NivaraColumn<int>.Create(new[] { 1, 2, 3 });

        var result = column.Shift(-1);

        Assert.That(result.IsNull(2), Is.True);
        Assert.That(result[0], Is.EqualTo(2));
        Assert.That(result[1], Is.EqualTo(3));
    }

    [Test]
    public void Shift_FillValue_FillsBoundariesOnly()
    {
        var column = NivaraColumn<int>.Create(new[] { 1, 2, 3 });

        var result = column.Shift(1, 0);

        Assert.That(result.HasNulls, Is.False);
        Assert.That(result[0], Is.EqualTo(0));
        Assert.That(result[1], Is.EqualTo(1));
        Assert.That(result[2], Is.EqualTo(2));
    }

    [Test]
    public void Shift_InRangeNullsPreserved_WithFillValue()
    {
        var column = NivaraColumn<int>.CreateFromSpans(new[] { 1, 0, 3 }, new[] { false, true, false });

        var result = column.Shift(1, 0);

        Assert.That(result[0], Is.EqualTo(0));
        Assert.That(result[1], Is.EqualTo(1));
        Assert.That(result.IsNull(2), Is.True);
    }

    [Test]
    public void Lead_EqualsShiftNegative()
    {
        var column = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 });

        var lead = column.Lead(1);
        var shifted = column.Shift(-1);

        Assert.That(lead[0], Is.EqualTo(shifted[0]));
        Assert.That(lead[1], Is.EqualTo(shifted[1]));
        Assert.That(lead.Length, Is.EqualTo(shifted.Length));
    }

    [Test]
    public void Shift_StringColumn_WorksForAnyType()
    {
        var column = NivaraColumn<string>.Create(new[] { "a", "b", "c" });

        var result = column.Shift(1);

        Assert.That(result.HasNulls, Is.True);
        Assert.That(result.IsNull(0), Is.True);
        Assert.That(result[1], Is.EqualTo("a"));
        Assert.That(result[2], Is.EqualTo("b"));
    }

    // ── Property-style: randomized comparison against naive references ──

    [Test]
    public void RollingSum_Int_RandomArrays_MatchesNaive()
    {
        var random = new Random(42);
        int length = 200;
        int[] values = new int[length];
        bool[] mask = new bool[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = random.Next(-50, 51);
            mask[i] = random.Next(4) == 0;
        }
        var column = NivaraColumn<int>.CreateFromSpans(values, mask);

        for (int window = 1; window <= 7; window++)
        {
            for (int minPeriods = 1; minPeriods <= window; minPeriods++)
            {
                var result = column.RollingSum(window, minPeriods);

                int validInWindow = 0;
                int sum = 0;
                for (int i = 0; i < length; i++)
                {
                    if (i - window >= 0 && !mask[i - window])
                    {
                        validInWindow--;
                        sum -= values[i - window];
                    }

                    if (!mask[i])
                    {
                        validInWindow++;
                        sum += values[i];
                    }

                    Assert.That(result.IsNull(i), Is.EqualTo(validInWindow < minPeriods),
                        $"null mismatch at {i} window={window} minPeriods={minPeriods}");

                    if (validInWindow >= minPeriods)
                        Assert.That(result[i], Is.EqualTo(sum),
                            $"value mismatch at {i} window={window} minPeriods={minPeriods}");
                }
            }
        }
    }

    [Test]
    public void CumulativeSum_Int_RandomArrays_MatchesNaive()
    {
        var random = new Random(7);
        int length = 150;
        int[] values = new int[length];
        bool[] mask = new bool[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = random.Next(-20, 21);
            mask[i] = random.Next(3) == 0;
        }
        var column = NivaraColumn<int>.CreateFromSpans(values, mask);

        var result = column.CumulativeSum();

        int running = 0;
        for (int i = 0; i < length; i++)
        {
            if (mask[i])
            {
                Assert.That(result.IsNull(i), Is.True);
                continue;
            }

            running += values[i];
            Assert.That(result[i], Is.EqualTo(running));
        }
    }
}
