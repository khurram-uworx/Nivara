using Nivara.Tensors;
using NUnit.Framework;
using System.Diagnostics;

namespace Nivara.Tests.Tensors;

[TestFixture]
public class MatMulHelperTests
{
    [Test]
    public void PropagateNullMask_NoMasks_ClearsResultMask()
    {
        var resultMask = new[] { true, true, true, true };

        MatMulHelper.PropagateNullMask(
            ReadOnlySpan<bool>.Empty,
            ReadOnlySpan<bool>.Empty,
            resultMask,
            aRows: 2,
            aCols: 2,
            bCols: 2);

        Assert.That(resultMask, Is.EqualTo(new[] { false, false, false, false }));
    }

    [Test]
    public void PropagateNullMask_ANullInRow_NullsWholeResultRow()
    {
        var aMask = new[]
        {
            false, false, false,
            false, true, false
        };
        var resultMask = new bool[4];

        MatMulHelper.PropagateNullMask(
            aMask,
            ReadOnlySpan<bool>.Empty,
            resultMask,
            aRows: 2,
            aCols: 3,
            bCols: 2);

        Assert.That(resultMask, Is.EqualTo(new[] { false, false, true, true }));
    }

    [Test]
    public void PropagateNullMask_BNullInColumn_NullsWholeResultColumn()
    {
        var bMask = new[]
        {
            false, false, true,
            false, false, false
        };
        var resultMask = new bool[6];

        MatMulHelper.PropagateNullMask(
            ReadOnlySpan<bool>.Empty,
            bMask,
            resultMask,
            aRows: 2,
            aCols: 2,
            bCols: 3);

        Assert.That(resultMask, Is.EqualTo(new[] { false, false, true, false, false, true }));
    }

    [Test]
    public void PropagateNullMask_MixedMasks_UsesRowOrColumnSemantics()
    {
        var aMask = new[]
        {
            false, false,
            true, false
        };
        var bMask = new[]
        {
            false, true,
            false, false
        };
        var resultMask = new bool[4];

        MatMulHelper.PropagateNullMask(
            aMask,
            bMask,
            resultMask,
            aRows: 2,
            aCols: 2,
            bCols: 2);

        Assert.That(resultMask, Is.EqualTo(new[] { false, true, true, true }));
    }

    [Test]
    public void Multiply_NoNullMasks_ComputesDenseValuesAndClearsMask()
    {
        var a = new[] { 1f, 2f, 3f, 4f };
        var b = new[] { 5f, 6f, 7f, 8f };
        var result = new float[4];
        var resultMask = new[] { true, true, true, true };

        MatMulHelper.Multiply(
            a,
            ReadOnlySpan<bool>.Empty,
            b,
            ReadOnlySpan<bool>.Empty,
            result,
            resultMask,
            aRows: 2,
            aCols: 2,
            bCols: 2);

        Assert.That(result, Is.EqualTo(new[] { 19f, 22f, 43f, 50f }));
        Assert.That(resultMask, Is.EqualTo(new[] { false, false, false, false }));
    }

    [Test]
    public void PropagateNullMask_PerformanceProbe_IsFasterThanReferenceTripleLoopForSparseMasks()
    {
        const int size = 160;
        var aMask = new bool[size * size];
        var bMask = new bool[size * size];
        var optimized = new bool[size * size];
        var reference = new bool[size * size];

        for (int i = 0; i < size; i += 40)
            aMask[i * size + (i % size)] = true;

        for (int j = 0; j < size; j += 40)
            bMask[(j % size) * size + j] = true;

        MatMulHelper.PropagateNullMask(aMask, bMask, optimized, size, size, size);
        PropagateNullMaskReference(aMask, bMask, reference, size, size, size);
        Assert.That(optimized, Is.EqualTo(reference));

        var optimizedTicks = MeasureBestOfFive(() =>
            MatMulHelper.PropagateNullMask(aMask, bMask, optimized, size, size, size));
        var referenceTicks = MeasureBestOfFive(() =>
            PropagateNullMaskReference(aMask, bMask, reference, size, size, size));

        TestContext.Out.WriteLine($"MatMul mask propagation ticks: optimized={optimizedTicks}, reference={referenceTicks}");
        Assert.That(optimizedTicks, Is.LessThan(referenceTicks));
    }

    static long MeasureBestOfFive(Action action)
    {
        var best = long.MaxValue;

        for (int i = 0; i < 5; i++)
        {
            var sw = Stopwatch.StartNew();
            action();
            sw.Stop();
            best = Math.Min(best, sw.ElapsedTicks);
        }

        return best;
    }

    static void PropagateNullMaskReference(
        ReadOnlySpan<bool> aMask,
        ReadOnlySpan<bool> bMask,
        Span<bool> resultMask,
        int aRows,
        int aCols,
        int bCols)
    {
        bool hasAMask = aMask.Length > 0;
        bool hasBMask = bMask.Length > 0;

        for (int i = 0; i < aRows; i++)
        {
            for (int j = 0; j < bCols; j++)
            {
                bool posNull = false;
                for (int k = 0; k < aCols && !posNull; k++)
                {
                    if ((hasAMask && aMask[i * aCols + k]) ||
                        (hasBMask && bMask[k * bCols + j]))
                        posNull = true;
                }

                resultMask[i * bCols + j] = posNull;
            }
        }
    }
}
