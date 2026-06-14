using Nivara.Tensors;
using NUnit.Framework;
using System.Diagnostics;
using System.Numerics.Tensors;

namespace Nivara.Tests.Tensors;

[TestFixture]
public class TensorsHelperTests
{
    [Test]
    public void PropagateNullMask_NoMasks_ClearsResultMask()
    {
        var resultMask = new[] { true, true, true, true };

        TensorsHelper.PropagateNullMask(
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

        TensorsHelper.PropagateNullMask(
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

        TensorsHelper.PropagateNullMask(
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

        TensorsHelper.PropagateNullMask(
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

        TensorsHelper.Multiply(
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

        TensorsHelper.PropagateNullMask(aMask, bMask, optimized, size, size, size);
        PropagateNullMaskReference(aMask, bMask, reference, size, size, size);
        Assert.That(optimized, Is.EqualTo(reference));

        var optimizedTicks = MeasureBestOfFive(() =>
            TensorsHelper.PropagateNullMask(aMask, bMask, optimized, size, size, size));
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

    #region RowNorms

    [Test]
    public void RowNorms_FloatRowMajor_MatchesTensorPrimitives()
    {
        var rowMajor = new[] { 3f, 4f, 0f, 1f };
        var destination = new float[2];

        TensorsHelper.RowNorms(rowMajor.AsSpan(), destination.AsSpan(), rows: 2, cols: 2);

        Assert.That(destination[0], Is.EqualTo(TensorPrimitives.Norm(new[] { 3f, 4f })).Within(1e-6f));
        Assert.That(destination[1], Is.EqualTo(TensorPrimitives.Norm(new[] { 0f, 1f })).Within(1e-6f));
    }

    [Test]
    public void RowNorms_ZeroRows_DoesNotWriteDestination()
    {
        var destination = new[] { 42f };

        TensorsHelper.RowNorms(ReadOnlySpan<float>.Empty, destination.AsSpan(), rows: 0, cols: 3);

        Assert.That(destination[0], Is.EqualTo(42f));
    }

    [Test]
    public void RowNorms_ZeroColumns_WritesZeroNorms()
    {
        var destination = new[] { 1f, 1f };

        TensorsHelper.RowNorms(ReadOnlySpan<float>.Empty, destination.AsSpan(), rows: 2, cols: 0);

        Assert.That(destination, Is.EqualTo(new[] { 0f, 0f }));
    }

    #endregion

    #region Sigmoid

    [Test]
    public void Sigmoid_Float_MatchesReference()
    {
        var x = new[] { 0f, 1f, -1f, 2f, -2f };
        var dest = new float[5];
        TensorsHelper.Sigmoid(x.AsSpan(), dest.AsSpan());
        for (int i = 0; i < x.Length; i++)
        {
            var expected = 1.0f / (1.0f + MathF.Exp(-x[i]));
            Assert.That(dest[i], Is.EqualTo(expected).Within(1e-6f));
        }
    }

    [Test]
    public void Sigmoid_WithNulls_PropagatesMaskAndZerosOutput()
    {
        var x = new[] { 0f, 1f, -1f };
        var mask = new[] { false, true, false };
        var dest = new float[3];
        var resultMask = new bool[3];
        TensorsHelper.Sigmoid(x.AsSpan(), mask.AsSpan(), dest.AsSpan(), resultMask.AsSpan());
        Assert.That(resultMask, Is.EqualTo(mask));
        Assert.That(dest[1], Is.EqualTo(0f));
        Assert.That(dest[0], Is.EqualTo(0.5f).Within(1e-6f));
    }

    #endregion

    #region Tanh

    [Test]
    public void Tanh_Float_MatchesReference()
    {
        var x = new[] { 0f, 1f, -1f, 2f, -2f };
        var dest = new float[5];
        TensorsHelper.Tanh(x.AsSpan(), dest.AsSpan());
        for (int i = 0; i < x.Length; i++)
            Assert.That(dest[i], Is.EqualTo(MathF.Tanh(x[i])).Within(1e-6f));
    }

    [Test]
    public void Tanh_WithNulls_PropagatesMaskAndZerosOutput()
    {
        var x = new[] { 0f, 1f, -1f };
        var mask = new[] { false, true, false };
        var dest = new float[3];
        var resultMask = new bool[3];
        TensorsHelper.Tanh(x.AsSpan(), mask.AsSpan(), dest.AsSpan(), resultMask.AsSpan());
        Assert.That(resultMask, Is.EqualTo(mask));
        Assert.That(dest[1], Is.EqualTo(0f));
    }

    #endregion

    #region SoftMax

    [Test]
    public void SoftMax_Float_SumToOne()
    {
        var x = new[] { 2f, 1f, 0.1f };
        var dest = new float[3];
        TensorsHelper.SoftMax(x.AsSpan(), dest.AsSpan());
        var sum = TensorPrimitives.Sum(dest.AsSpan());
        Assert.That(sum, Is.EqualTo(1f).Within(1e-5f));
    }

    [Test]
    public void SoftMax_RowWise_SumToOnePerRow()
    {
        var x = new[] { 2f, 1f, 0.1f, 1f, 2f, 3f };
        var dest = new float[6];
        TensorsHelper.SoftMax(x.AsSpan(), dest.AsSpan(), classCount: 3);
        for (int r = 0; r < 2; r++)
        {
            var rowSum = TensorPrimitives.Sum(dest.AsSpan(r * 3, 3));
            Assert.That(rowSum, Is.EqualTo(1f).Within(1e-5f));
        }
    }

    [Test]
    public void SoftMax_WithNulls_PropagatesMask()
    {
        var x = new[] { 2f, 1f, 0.1f };
        var mask = new[] { false, true, false };
        var dest = new float[3];
        var resultMask = new bool[3];
        TensorsHelper.SoftMax(x.AsSpan(), mask.AsSpan(), dest.AsSpan(), resultMask.AsSpan(), classCount: 3);
        Assert.That(resultMask, Is.EqualTo(mask));
        Assert.That(dest[1], Is.EqualTo(0f));
    }

    #endregion

    #region Transpose

    [Test]
    public void Transpose_2x3_CorrectLayout()
    {
        var src = new[] { 1, 2, 3, 4, 5, 6 };
        var dest = new int[6];
        TensorsHelper.Transpose(src.AsSpan(), dest.AsSpan(), rows: 2, cols: 3);
        Assert.That(dest, Is.EqualTo(new[] { 1, 4, 2, 5, 3, 6 }));
    }

    [Test]
    public void Transpose_WithNulls_PropagatesMask()
    {
        var src = new[] { 1f, 2f, 3f, 4f };
        var mask = new[] { false, true, false, false };
        var dest = new float[4];
        var resultMask = new bool[4];
        TensorsHelper.Transpose(src.AsSpan(), mask.AsSpan(), dest.AsSpan(), resultMask.AsSpan(), rows: 2, cols: 2);
        // src[0,1]=2 is null -> after transpose it becomes dst[1,0]=index 2
        Assert.That(dest[2], Is.EqualTo(0f));
        Assert.That(resultMask[1], Is.False);
        Assert.That(resultMask[2], Is.True);
        Assert.That(dest[1], Is.EqualTo(3f)); // src[1,0]=3 is not null
    }

    #endregion
}
