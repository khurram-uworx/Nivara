using Nivara.Tensors;
using NUnit.Framework;
using System.Diagnostics;
using System.Numerics;
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

    #region MultiplyCore

    [Test]
    public void MultiplyCore_Float_MultipleShapes_MatchesReference() => CheckMatMulShapes<float>(1e-4);

    [Test]
    public void MultiplyCore_Double_MultipleShapes_MatchesReference() => CheckMatMulShapes<double>(1e-10);

    [Test]
    public void MultiplyCore_Int_MultipleShapes_MatchesReference() => CheckMatMulShapes<int>(0.0);

    static void CheckMatMulShapes<T>(double tolerance) where T : struct, INumber<T>
    {
        (int Rows, int Cols, int BCols)[] shapes =
        [
            (2, 1, 3),
            (1, 5, 1),
            (3, 7, 11),
            (4, 8, 16),
            (5, 6, 9),
            (12, 20, 24),
            (16, 64, 48),
            (64, 128, 256)
        ];
        foreach (var (rows, cols, bCols) in shapes)
            AssertMatMulMatchesReference<T>(rows, cols, bCols, tolerance);
    }

    static void AssertMatMulMatchesReference<T>(int aRows, int aCols, int bCols, double tolerance)
        where T : struct, INumber<T>
    {
        var rng = new Random(12345 + aRows * 31 + aCols * 7 + bCols);
        var a = new T[aRows * aCols];
        var b = new T[aCols * bCols];
        for (int i = 0; i < a.Length; i++)
            a[i] = FillValue<T>(rng);
        for (int i = 0; i < b.Length; i++)
            b[i] = FillValue<T>(rng);

        var result = new T[aRows * bCols];
        var reference = new T[aRows * bCols];
        TensorsHelper.MultiplyCore(a.AsSpan(), b.AsSpan(), result, aRows, aCols, bCols);
        ReferenceMatMul(a, b, reference, aRows, aCols, bCols);

        for (int i = 0; i < result.Length; i++)
        {
            double diff = Math.Abs(double.CreateChecked(result[i]) - double.CreateChecked(reference[i]));
            double magnitude = Math.Abs(double.CreateChecked(reference[i]));
            Assert.That(diff, Is.LessThanOrEqualTo(tolerance * Math.Max(1.0, magnitude)),
                $"Mismatch at index {i} for {typeof(T).Name} {aRows}x{aCols}@{aCols}x{bCols}: " +
                $"kernel={result[i]}, reference={reference[i]}");
        }
    }

    static T FillValue<T>(Random rng) where T : struct, INumber<T>
        => typeof(T) == typeof(int) ? T.CreateChecked(rng.Next(0, 3)) : T.CreateChecked(rng.NextDouble() * 2 - 1.0);

    static void ReferenceMatMul<T>(T[] a, T[] b, T[] result, int aRows, int aCols, int bCols)
        where T : struct, INumber<T>
    {
        for (int i = 0; i < aRows; i++)
        {
            for (int j = 0; j < bCols; j++)
            {
                T sum = T.Zero;
                for (int k = 0; k < aCols; k++)
                    sum += a[i * aCols + k] * b[k * bCols + j];
                result[i * bCols + j] = sum;
            }
        }
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

    #region Row-slice scoring kernels

    [Test]
    public void RowDot_NoNulls_MatchesPerRowScalar()
    {
        const int rows = 6, cols = 5;
        var rowMajor = FillRowMajor(rows, cols, seed: 1);
        var query = new[] { 0.5f, -0.25f, 0.75f, 0.1f, -0.9f };
        var output = new float[rows];
        var outputMask = new bool[rows];

        TensorsHelper.RowDot(rowMajor, ReadOnlySpan<bool>.Empty, query, ReadOnlySpan<bool>.Empty,
            output, outputMask, rows, cols);

        for (int r = 0; r < rows; r++)
            Assert.That(output[r], Is.EqualTo(TensorPrimitives.Dot(rowMajor.AsSpan(r * cols, cols), query)), $"row {r}");
        Assert.That(outputMask, Is.EqualTo(new bool[rows]));
    }

    [Test]
    public void RowDot_Int_MatchesPerRowScalar()
    {
        const int rows = 4, cols = 4;
        var rowMajor = new[] { 1, 2, 3, 4, 2, 1, 0, 1, 5, 0, 2, 1, 0, 1, 1, 3 };
        var query = new[] { 1, -1, 2, 0 };
        var output = new int[rows];
        var outputMask = new bool[rows];

        TensorsHelper.RowDot(rowMajor, ReadOnlySpan<bool>.Empty, query, ReadOnlySpan<bool>.Empty,
            output, outputMask, rows, cols);

        for (int r = 0; r < rows; r++)
            Assert.That(output[r], Is.EqualTo(TensorPrimitives.Dot(rowMajor.AsSpan(r * cols, cols), query)), $"row {r}");
        Assert.That(outputMask, Is.EqualTo(new bool[rows]));
    }

    [Test]
    public void RowDot_WithRowNulls_MasksOnlyNullRows()
    {
        const int rows = 4, cols = 3;
        var rowMajor = new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f };
        var rowMask = new[]
        {
            false, false, false,
            true, false, false,   // row 1 null
            false, false, false,
            false, true, false    // row 3 null
        };
        var query = new[] { 1f, -1f, 2f };
        var output = new float[rows];
        var outputMask = new bool[rows];

        TensorsHelper.RowDot(rowMajor, rowMask, query, ReadOnlySpan<bool>.Empty, output, outputMask, rows, cols);

        Assert.That(outputMask, Is.EqualTo(new[] { false, true, false, true }));
        Assert.That(output[0], Is.EqualTo(TensorPrimitives.Dot(rowMajor.AsSpan(0, cols), query)));
        Assert.That(output[2], Is.EqualTo(TensorPrimitives.Dot(rowMajor.AsSpan(2 * cols, cols), query)));
        Assert.That(output[1], Is.EqualTo(0f));
        Assert.That(output[3], Is.EqualTo(0f));
    }

    [Test]
    public void RowDot_WithQueryNulls_MasksAllRows()
    {
        const int rows = 3, cols = 3;
        var rowMajor = new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f };
        var query = new[] { 1f, 0f, 1f };
        var queryMask = new[] { false, true, false };

        var output = new float[rows];
        var outputMask = new bool[rows];
        TensorsHelper.RowDot(rowMajor, ReadOnlySpan<bool>.Empty, query, queryMask, output, outputMask, rows, cols);

        Assert.That(outputMask, Is.EqualTo(new[] { true, true, true }));
        Assert.That(output, Is.EqualTo(new[] { 0f, 0f, 0f }));
    }

    [Test]
    public void RowDot_EmptyMasks_TreatedAsNoNulls()
    {
        const int rows = 3, cols = 2;
        var rowMajor = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var query = new[] { 1f, 1f };
        var output = new float[rows];
        var outputMask = new[] { true, true, true };

        TensorsHelper.RowDot(rowMajor, ReadOnlySpan<bool>.Empty, query, ReadOnlySpan<bool>.Empty,
            output, outputMask, rows, cols);

        Assert.That(outputMask, Is.EqualTo(new[] { false, false, false }));
        Assert.That(output, Is.EqualTo(new[] { 3f, 7f, 11f }));
    }

    [Test]
    public void RowDot_QueryLengthMismatch_Throws()
    {
        var rowMajor = new float[6];
        var query = new float[4];
        Assert.That(() => TensorsHelper.RowDot(
                rowMajor, ReadOnlySpan<bool>.Empty, query, ReadOnlySpan<bool>.Empty,
                new float[2], new bool[2], rows: 2, cols: 3),
            Throws.ArgumentException.With.Message.Contains("Query length"));
    }

    [Test]
    public void RowDot_ShortOutput_Throws()
    {
        var rowMajor = new float[6];
        var query = new float[3];
        Assert.That(() => TensorsHelper.RowDot(
                rowMajor, ReadOnlySpan<bool>.Empty, query, ReadOnlySpan<bool>.Empty,
                new float[1], new bool[2], rows: 2, cols: 3),
            Throws.ArgumentException);
    }

    [Test]
    public void RowCosineSimilarity_NoNulls_MatchesPerRowScalar()
    {
        const int rows = 5, cols = 4;
        var rowMajor = FillRowMajor(rows, cols, seed: 7);
        var query = new[] { 0.2f, -0.8f, 0.6f, 0.1f };
        var output = new float[rows];
        var outputMask = new bool[rows];

        TensorsHelper.RowCosineSimilarity(rowMajor, ReadOnlySpan<bool>.Empty, query, ReadOnlySpan<bool>.Empty,
            output, outputMask, rows, cols);

        for (int r = 0; r < rows; r++)
            Assert.That(output[r], Is.EqualTo(TensorPrimitives.CosineSimilarity(rowMajor.AsSpan(r * cols, cols), query)), $"row {r}");
        Assert.That(outputMask, Is.EqualTo(new bool[rows]));
    }

    [Test]
    public void RowCosineSimilarity_WithRowNulls_MasksOnlyNullRows()
    {
        const int rows = 3, cols = 3;
        var rowMajor = new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f };
        var rowMask = new[]
        {
            false, false, false,
            false, true, false,   // row 1 null
            false, false, false
        };
        var query = new[] { 1f, 0f, -1f };
        var output = new float[rows];
        var outputMask = new bool[rows];

        TensorsHelper.RowCosineSimilarity(rowMajor, rowMask, query, ReadOnlySpan<bool>.Empty,
            output, outputMask, rows, cols);

        Assert.That(outputMask, Is.EqualTo(new[] { false, true, false }));
        Assert.That(output[0], Is.EqualTo(TensorPrimitives.CosineSimilarity(rowMajor.AsSpan(0, cols), query)));
        Assert.That(output[2], Is.EqualTo(TensorPrimitives.CosineSimilarity(rowMajor.AsSpan(2 * cols, cols), query)));
        Assert.That(output[1], Is.EqualTo(0f));
    }

    [Test]
    public void RowCosineSimilarity_WithQueryNulls_MasksAllRows()
    {
        const int rows = 2, cols = 3;
        var rowMajor = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var query = new[] { 1f, 1f, 1f };
        var queryMask = new[] { true, false, false };
        var output = new float[rows];
        var outputMask = new bool[rows];

        TensorsHelper.RowCosineSimilarity(rowMajor, ReadOnlySpan<bool>.Empty, query, queryMask,
            output, outputMask, rows, cols);

        Assert.That(outputMask, Is.EqualTo(new[] { true, true }));
        Assert.That(output, Is.EqualTo(new[] { 0f, 0f }));
    }

    [Test]
    public void RowCosineSimilarity_ZeroColumns_Throws()
    {
        var rowMajor = new float[0];
        var query = new float[0];
        Assert.That(() => TensorsHelper.RowCosineSimilarity(
                rowMajor, ReadOnlySpan<bool>.Empty, query, ReadOnlySpan<bool>.Empty,
                new float[1], new bool[1], rows: 1, cols: 0),
            Throws.ArgumentException);
    }

    static float[] FillRowMajor(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1.0);
        return data;
    }

    #endregion
}
