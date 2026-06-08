using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.Extensions.AutoDiff.Operations;

public static class GradOperations
{
    #region Element-wise Operations

    public static ReverseGradTensor<T> Add<T>(ReverseGradTensor<T> a, ReverseGradTensor<T> b) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Cannot add tensors with different lengths: {a.Length} vs {b.Length}");
        }

        var result = a.Data + b.Data;

        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad || b.RequiresGrad, PropagateShape(a, b));

        if (a.RequiresGrad || b.RequiresGrad)
        {
            var gradFn = new OpNode<T>("Add", new object[] { a, b }, typedGradOutput =>
            {
                if (a.RequiresGrad)
                {
                    AccumulateGradient(a, typedGradOutput);
                }
                if (b.RequiresGrad)
                {
                    AccumulateGradient(b, typedGradOutput);
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Subtract<T>(ReverseGradTensor<T> a, ReverseGradTensor<T> b) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Cannot subtract tensors with different lengths: {a.Length} vs {b.Length}");
        }

        var result = SubtractVectorized(a.Data, b.Data);

        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad || b.RequiresGrad, PropagateShape(a, b));

        if (a.RequiresGrad || b.RequiresGrad)
        {
            var gradFn = new OpNode<T>("Subtract", new object[] { a, b }, typedGradOutput =>
            {
                if (a.RequiresGrad)
                {
                    AccumulateGradient(a, typedGradOutput);
                }
                if (b.RequiresGrad)
                {
                    var negatedGrad = NegateVectorized(typedGradOutput);
                    AccumulateGradient(b, negatedGrad);
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Multiply<T>(ReverseGradTensor<T> a, ReverseGradTensor<T> b) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Cannot multiply tensors with different lengths: {a.Length} vs {b.Length}");
        }

        var result = a.Data * b.Data;

        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad || b.RequiresGrad, PropagateShape(a, b));

        if (a.RequiresGrad || b.RequiresGrad)
        {
            var gradFn = new OpNode<T>("Multiply", new object[] { a, b }, typedGradOutput =>
            {
                if (a.RequiresGrad)
                {
                    var aGrad = MultiplyVectorized(typedGradOutput, b.Data);
                    AccumulateGradient(a, aGrad);
                }
                if (b.RequiresGrad)
                {
                    var bGrad = MultiplyVectorized(typedGradOutput, a.Data);
                    AccumulateGradient(b, bGrad);
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Divide<T>(ReverseGradTensor<T> a, ReverseGradTensor<T> b) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Cannot divide tensors with different lengths: {a.Length} vs {b.Length}");
        }

        for (int i = 0; i < b.Length; i++)
        {
            if (!b.IsNull(i) && b[i] == T.Zero)
            {
                throw new DivideByZeroException($"Division by zero at index {i}");
            }
        }

        var result = DivideVectorized(a.Data, b.Data);

        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad || b.RequiresGrad, PropagateShape(a, b));

        if (a.RequiresGrad || b.RequiresGrad)
        {
            var gradFn = new OpNode<T>("Divide", new object[] { a, b }, typedGradOutput =>
            {
                if (a.RequiresGrad)
                {
                    var aGrad = DivideVectorized(typedGradOutput, b.Data);
                    AccumulateGradient(a, aGrad);
                }
                if (b.RequiresGrad)
                {
                    var quotient = DivideVectorized(a.Data, b.Data);
                    var bGradPositive = DivideVectorized(quotient, b.Data);
                    var bGrad = NegateVectorized(bGradPositive);
                    var finalBGrad = MultiplyVectorized(bGrad, typedGradOutput);
                    AccumulateGradient(b, finalBGrad);
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    #endregion

    #region Matrix Operations

    public static ReverseGradTensor<T> MatMul<T>(ReverseGradTensor<T> a, ReverseGradTensor<T> b)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Rank != 2)
            throw new ArgumentException($"Left operand must be a matrix (rank 2), got rank {a.Rank}", nameof(a));
        if (b.Rank != 2)
            throw new ArgumentException($"Right operand must be a matrix (rank 2), got rank {b.Rank}", nameof(b));

        var aRows = a.shape[0];
        var aCols = a.shape[1];
        var bRows = b.shape[0];
        var bCols = b.shape[1];

        if (aCols != bRows)
            throw new ArgumentException(
                $"Matrix dimensions incompatible: a({aRows}x{aCols}) @ b({bRows}x{bCols}). " +
                $"a's column count ({aCols}) must equal b's row count ({bRows}).");

#pragma warning disable CS0618 // internal call to legacy overload
        return MatMul(a, b, aRows, aCols, bCols);
#pragma warning restore CS0618
    }

    [Obsolete("Use the shape-aware MatMul overload instead — reshape your tensors before calling MatMul(a, b)")]
    public static ReverseGradTensor<T> MatMul<T>(ReverseGradTensor<T> a, ReverseGradTensor<T> b, int aRows, int aCols, int bCols)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        var bRows = aCols;

        if (a.Length != aRows * aCols)
        {
            throw new ArgumentException($"Matrix a dimensions don't match: expected {aRows * aCols} elements, got {a.Length}");
        }

        if (b.Length != bRows * bCols)
        {
            throw new ArgumentException($"Matrix b dimensions don't match: expected {bRows * bCols} elements, got {b.Length}");
        }

        var result = MatMulVectorized(a.Data, b.Data, aRows, aCols, bCols);

        var resultShape = new[] { aRows, bCols };
        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad || b.RequiresGrad, resultShape);

        if (a.RequiresGrad || b.RequiresGrad)
        {
            var gradFn = new OpNode<T>("MatMul", new object[] { a, b }, typedGradOutput =>
            {
                if (a.RequiresGrad)
                {
                    var bTransposed = TransposeVectorized(b.Data, bRows, bCols);
                    var aGrad = MatMulVectorized(typedGradOutput, bTransposed, aRows, bCols, bRows);
                    AccumulateGradient(a, aGrad);
                }
                if (b.RequiresGrad)
                {
                    var aTransposed = TransposeVectorized(a.Data, aRows, aCols);
                    var bGrad = MatMulVectorized(aTransposed, typedGradOutput, aCols, aRows, bCols);
                    AccumulateGradient(b, bGrad);
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Transpose<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (a.Rank != 2)
            throw new ArgumentException($"Transpose requires a matrix (rank 2), got rank {a.Rank}", nameof(a));

#pragma warning disable CS0618 // internal call to legacy overload
        return Transpose(a, a.shape[0], a.shape[1]);
#pragma warning restore CS0618
    }

    [Obsolete("Use the shape-aware Transpose overload instead — reshape your tensor before calling Transpose(a)")]
    public static ReverseGradTensor<T> Transpose<T>(ReverseGradTensor<T> a, int rows, int cols) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (a.Length != rows * cols)
        {
            throw new ArgumentException($"Matrix dimensions don't match: expected {rows * cols} elements, got {a.Length}");
        }

        var result = TransposeVectorized(a.Data, rows, cols);

        var resultShape = new[] { cols, rows };
        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad, resultShape);

        if (a.RequiresGrad)
        {
            var gradFn = new OpNode<T>("Transpose", new object[] { a }, typedGradOutput =>
            {
                var aGrad = TransposeVectorized(typedGradOutput, cols, rows);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    #endregion

    #region Reduction Operations

    public static ReverseGradTensor<T> Sum<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (a.Length == 0)
        {
            throw new InvalidOperationException("Cannot compute sum of empty tensor");
        }

        var series = a.ToSeries();
        var sumValue = series.Sum();

        var resultData = NivaraColumn<T>.Create(new T[] { sumValue });
        var resultTensor = new ReverseGradTensor<T>(resultData, a.RequiresGrad, ScalarShape());

        if (a.RequiresGrad)
        {
            var gradFn = new OpNode<T>("Sum", new object[] { a }, typedGradOutput =>
            {
                var aGrad = BroadcastGradient(typedGradOutput, a.Length);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Mean<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (a.Length == 0)
        {
            throw new InvalidOperationException("Cannot compute mean of empty tensor");
        }

        var series = a.ToSeries();
        var meanValue = series.Average();

        var resultData = NivaraColumn<T>.Create(new T[] { meanValue });
        var resultTensor = new ReverseGradTensor<T>(resultData, a.RequiresGrad, ScalarShape());

        if (a.RequiresGrad)
        {
            var gradFn = new OpNode<T>("Mean", new object[] { a }, typedGradOutput =>
            {
                var aGrad = BroadcastGradient(typedGradOutput, a.Length);
                var scaledGrad = DivideByScalar(aGrad, a.Length);
                AccumulateGradient(a, scaledGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    #endregion

    #region Activation Functions

    public static ReverseGradTensor<T> Relu<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = ApplyRelu(a.Data);

        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad, PropagateShape(a));

        if (a.RequiresGrad)
        {
            var gradFn = new OpNode<T>("Relu", new object[] { a }, typedGradOutput =>
            {
                var aGrad = ApplyReluGradient(a.Data, typedGradOutput);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Sigmoid<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = ApplySigmoid(a.Data);

        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad, PropagateShape(a));

        if (a.RequiresGrad)
        {
            var gradFn = new OpNode<T>("Sigmoid", new object[] { a }, typedGradOutput =>
            {
                var aGrad = ApplySigmoidGradient(result, typedGradOutput);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Tanh<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = ApplyTanh(a.Data);

        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad, PropagateShape(a));

        if (a.RequiresGrad)
        {
            var gradFn = new OpNode<T>("Tanh", new object[] { a }, typedGradOutput =>
            {
                var aGrad = ApplyTanhGradient(result, typedGradOutput);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Negate<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = NegateVectorized(a.Data);
        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad, PropagateShape(a));

        if (a.RequiresGrad)
        {
            var gradFn = new OpNode<T>("Negate", new object[] { a }, typedGradOutput =>
            {
                var aGrad = NegateVectorized(typedGradOutput);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Abs<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = ApplyAbs(a.Data);
        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad, PropagateShape(a));

        if (a.RequiresGrad)
        {
            var gradFn = new OpNode<T>("Abs", new object[] { a }, typedGradOutput =>
            {
                var aGrad = ApplyAbsGradient(a.Data, typedGradOutput);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    #endregion

    #region Helper Methods

    private static void AccumulateGradient<T>(ReverseGradTensor<T> tensor, NivaraColumn<T> gradient) where T : struct, INumber<T>
    {
        if (tensor.Grad == null)
        {
            tensor.Grad = gradient;
        }
        else
        {
            tensor.Grad = tensor.Grad + gradient;
        }
    }

    private static NivaraColumn<T> SubtractVectorized<T>(NivaraColumn<T> a, NivaraColumn<T> b) where T : struct, INumber<T>
    {
        int n = a.Length;

        if (!a.HasNulls && !b.HasNulls)
        {
            a.TryGetSpan(out var aSpan);
            b.TryGetSpan(out var bSpan);
            var result = new T[n];
            TensorPrimitives.Subtract(aSpan, bSpan, result);
            return NivaraColumn<T>.Create(result);
        }

        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var aBuf = ArrayPool<T>.Shared.Rent(n);
        var bBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            a.CopyTo(aBuf.AsSpan(0, n), T.Zero);
            b.CopyTo(bBuf.AsSpan(0, n), T.Zero);
            var hasNulls = MergeNullMasks(a, b, nullMask.AsSpan(0, n));
            TensorPrimitives.Subtract(aBuf.AsSpan(0, n), bBuf.AsSpan(0, n), resultBuf.AsSpan(0, n));
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(aBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(bBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> DivideVectorized<T>(NivaraColumn<T> a, NivaraColumn<T> b) where T : struct, INumber<T>
    {
        int n = a.Length;

        if (!a.HasNulls && !b.HasNulls)
        {
            a.TryGetSpan(out var aSpan);
            b.TryGetSpan(out var bSpan);
            var result = new T[n];
            TensorPrimitives.Divide(aSpan, bSpan, result);
            return NivaraColumn<T>.Create(result);
        }

        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var aBuf = ArrayPool<T>.Shared.Rent(n);
        var bBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            a.CopyTo(aBuf.AsSpan(0, n), T.Zero);
            b.CopyTo(bBuf.AsSpan(0, n), T.Zero);
            var hasNulls = MergeNullMasks(a, b, nullMask.AsSpan(0, n));
            TensorPrimitives.Divide(aBuf.AsSpan(0, n), bBuf.AsSpan(0, n), resultBuf.AsSpan(0, n));
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(aBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(bBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> NegateVectorized<T>(NivaraColumn<T> a) where T : struct, INumber<T>
    {
        int n = a.Length;

        if (!a.HasNulls)
        {
            a.TryGetSpan(out var aSpan);
            var result = new T[n];
            TensorPrimitives.Negate(aSpan, result);
            return NivaraColumn<T>.Create(result);
        }

        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var aBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            a.CopyTo(aBuf.AsSpan(0, n), T.Zero);
            a.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));
            TensorPrimitives.Negate(aBuf.AsSpan(0, n), resultBuf.AsSpan(0, n));
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(aBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> MultiplyVectorized<T>(NivaraColumn<T> a, NivaraColumn<T> b) where T : struct, INumber<T>
    {
        return a * b;
    }

    private static NivaraColumn<T> MatMulVectorized<T>(NivaraColumn<T> a, NivaraColumn<T> b, int aRows, int aCols, int bCols)
        where T : struct, INumber<T>
    {
        int n = aRows * bCols;

        if (!a.HasNulls && !b.HasNulls)
        {
            a.TryGetSpan(out var aSpan);
            b.TryGetSpan(out var bSpan);
            var result = new T[n];
            for (int i = 0; i < aRows; i++)
                for (int j = 0; j < bCols; j++)
                {
                    T sum = T.Zero;
                    for (int k = 0; k < aCols; k++)
                        sum += aSpan[i * aCols + k] * bSpan[k * bCols + j];
                    result[i * bCols + j] = sum;
                }
            return NivaraColumn<T>.Create(result);
        }

        var aBuf = ArrayPool<T>.Shared.Rent(a.Length);
        var bBuf = ArrayPool<T>.Shared.Rent(b.Length);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            a.CopyTo(aBuf.AsSpan(0, a.Length), T.Zero);
            b.CopyTo(bBuf.AsSpan(0, b.Length), T.Zero);
            var aHasNulls = a.TryGetNullMask(out var aMask);
            var bHasNulls = b.TryGetNullMask(out var bMask);

            for (int i = 0; i < aRows; i++)
            {
                for (int j = 0; j < bCols; j++)
                {
                    T sum = T.Zero;
                    bool positionNull = false;
                    for (int k = 0; k < aCols; k++)
                    {
                        var aIdx = i * aCols + k;
                        var bIdx = k * bCols + j;

                        var aNull = aHasNulls && aMask[aIdx];
                        var bNull = bHasNulls && bMask[bIdx];
                        if (aNull || bNull)
                        {
                            positionNull = true;
                        }
                        else
                        {
                            sum += aBuf[aIdx] * bBuf[bIdx];
                        }
                    }

                    var resIdx = i * bCols + j;
                    if (positionNull)
                        nullMask[resIdx] = true;
                    else
                        resultBuf[resIdx] = sum;
                }
            }

            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(aBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(bBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> TransposeVectorized<T>(NivaraColumn<T> a, int rows, int cols) where T : struct, INumber<T>
    {
        int n = rows * cols;

        if (!a.HasNulls)
        {
            a.TryGetSpan(out var aSpan);
            var result = new T[n];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[j * rows + i] = aSpan[i * cols + j];
            return NivaraColumn<T>.Create(result);
        }

        var aBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            a.CopyTo(aBuf.AsSpan(0, n), T.Zero);
            a.TryGetNullMask(out var mask);

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                {
                    var srcIdx = i * cols + j;
                    var dstIdx = j * rows + i;
                    if (mask[srcIdx])
                        nullMask[dstIdx] = true;
                    else
                        resultBuf[dstIdx] = aBuf[srcIdx];
                }

            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(aBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> BroadcastGradient<T>(NivaraColumn<T> scalarGrad, int targetLength) where T : struct, INumber<T>
    {
        if (scalarGrad.Length != 1)
            throw new ArgumentException($"Expected scalar gradient with length 1, got {scalarGrad.Length}");

        if (scalarGrad.HasNulls)
        {
            var mask = new bool[targetLength];
            Array.Fill(mask, true);
            return NivaraColumn<T>.CreateFromSpans(new T[targetLength], mask);
        }

        if (scalarGrad.TryGetSpan(out var span))
        {
            var filled = new T[targetLength];
            Array.Fill(filled, span[0]);
            return NivaraColumn<T>.Create(filled);
        }

        var gradValue = scalarGrad[0];
        var rented = ArrayPool<T>.Shared.Rent(targetLength);
        try
        {
            Array.Fill(rented, gradValue, 0, targetLength);
            return NivaraColumn<T>.Create(rented.AsSpan(0, targetLength));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(rented, clearArray: true);
        }
    }

    private static NivaraColumn<T> DivideByScalar<T>(NivaraColumn<T> column, int divisor) where T : struct, INumber<T>
    {
        int n = column.Length;
        T divisorT = ConvertToT<T>(divisor);

        if (!column.HasNulls)
        {
            column.TryGetSpan(out var span);
            var result = new T[n];
            TensorPrimitives.Divide(span, divisorT, result);
            return NivaraColumn<T>.Create(result);
        }

        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            column.CopyTo(resultBuf.AsSpan(0, n), T.Zero);
            column.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));
            TensorPrimitives.Divide(resultBuf.AsSpan(0, n), divisorT, resultBuf.AsSpan(0, n));
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static T ConvertToT<T>(int value) where T : struct, INumber<T>
    {
        var type = typeof(T);

        if (type == typeof(float))
        {
            return (T)(object)(float)value;
        }
        else if (type == typeof(double))
        {
            return (T)(object)(double)value;
        }
        else if (type == typeof(int))
        {
            return (T)(object)value;
        }
        else if (type == typeof(long))
        {
            return (T)(object)(long)value;
        }
        else
        {
            return (T)(object)((dynamic)value)!;
        }
    }

    private static NivaraColumn<T> ApplyRelu<T>(NivaraColumn<T> input) where T : struct, INumber<T>
    {
        int n = input.Length;

        if (!input.HasNulls)
        {
            input.TryGetSpan(out var span);
            var result = new T[n];
            TensorPrimitives.Max(span, T.Zero, result);
            return NivaraColumn<T>.Create(result);
        }

        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            input.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            input.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));
            TensorPrimitives.Max(inputBuf.AsSpan(0, n), T.Zero, resultBuf.AsSpan(0, n));
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyReluGradient<T>(NivaraColumn<T> input, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        int n = input.Length;

        if (!input.HasNulls && !gradOutput.HasNulls)
        {
            input.TryGetSpan(out var inSpan);
            gradOutput.TryGetSpan(out var gradSpan);
            var result = new T[n];
            for (int i = 0; i < n; i++)
                result[i] = inSpan[i] > T.Zero ? gradSpan[i] : T.Zero;
            return NivaraColumn<T>.Create(result);
        }

        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            input.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            var hasNulls = MergeNullMasks(input, gradOutput, nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
                resultBuf[i] = inputBuf[i] > T.Zero ? gradBuf[i] : T.Zero;
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyAbs<T>(NivaraColumn<T> input) where T : struct, INumber<T>
    {
        int n = input.Length;

        if (!input.HasNulls)
        {
            input.TryGetSpan(out var span);
            var result = new T[n];
            TensorPrimitives.Abs(span, result);
            return NivaraColumn<T>.Create(result);
        }

        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            input.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            input.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));
            TensorPrimitives.Abs(inputBuf.AsSpan(0, n), resultBuf.AsSpan(0, n));
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyAbsGradient<T>(NivaraColumn<T> input, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        int n = input.Length;

        if (!input.HasNulls && !gradOutput.HasNulls)
        {
            input.TryGetSpan(out var inSpan);
            gradOutput.TryGetSpan(out var gradSpan);
            var result = new T[n];
            for (int i = 0; i < n; i++)
                result[i] = T.CreateChecked(T.Sign(inSpan[i])) * gradSpan[i];
            return NivaraColumn<T>.Create(result);
        }

        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            input.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            var hasNulls = MergeNullMasks(input, gradOutput, nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
                resultBuf[i] = T.CreateChecked(T.Sign(inputBuf[i])) * gradBuf[i];
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplySigmoid<T>(NivaraColumn<T> input) where T : struct, INumber<T>
    {
        int n = input.Length;

        if (!input.HasNulls)
        {
            input.TryGetSpan(out var span);
            var result = new T[n];
            for (int i = 0; i < n; i++)
            {
                var x = double.CreateChecked(span[i]);
                result[i] = T.CreateChecked(1.0 / (1.0 + Math.Exp(-x)));
            }
            return NivaraColumn<T>.Create(result);
        }

        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            input.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            input.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
            {
                var x = double.CreateChecked(inputBuf[i]);
                resultBuf[i] = T.CreateChecked(1.0 / (1.0 + Math.Exp(-x)));
            }
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplySigmoidGradient<T>(NivaraColumn<T> sigmoidOutput, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        int n = sigmoidOutput.Length;

        if (!sigmoidOutput.HasNulls && !gradOutput.HasNulls)
        {
            sigmoidOutput.TryGetSpan(out var sigSpan);
            gradOutput.TryGetSpan(out var gradSpan);
            var result = new T[n];
            for (int i = 0; i < n; i++)
            {
                var sig = sigSpan[i];
                result[i] = sig * (T.One - sig) * gradSpan[i];
            }
            return NivaraColumn<T>.Create(result);
        }

        var sigBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            sigmoidOutput.CopyTo(sigBuf.AsSpan(0, n), T.Zero);
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            var hasNulls = MergeNullMasks(sigmoidOutput, gradOutput, nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
            {
                var sig = sigBuf[i];
                resultBuf[i] = sig * (T.One - sig) * gradBuf[i];
            }
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(sigBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyTanh<T>(NivaraColumn<T> input) where T : struct, INumber<T>
    {
        int n = input.Length;

        if (!input.HasNulls)
        {
            input.TryGetSpan(out var span);
            var result = new T[n];
            for (int i = 0; i < n; i++)
            {
                var x = double.CreateChecked(span[i]);
                result[i] = T.CreateChecked(Math.Tanh(x));
            }
            return NivaraColumn<T>.Create(result);
        }

        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            input.CopyTo(inputBuf.AsSpan(0, n), T.Zero);
            input.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
            {
                var x = double.CreateChecked(inputBuf[i]);
                resultBuf[i] = T.CreateChecked(Math.Tanh(x));
            }
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyTanhGradient<T>(NivaraColumn<T> tanhOutput, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        int n = tanhOutput.Length;

        if (!tanhOutput.HasNulls && !gradOutput.HasNulls)
        {
            tanhOutput.TryGetSpan(out var tanhSpan);
            gradOutput.TryGetSpan(out var gradSpan);
            var result = new T[n];
            for (int i = 0; i < n; i++)
            {
                var tanh = tanhSpan[i];
                result[i] = (T.One - tanh * tanh) * gradSpan[i];
            }
            return NivaraColumn<T>.Create(result);
        }

        var tanhBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            tanhOutput.CopyTo(tanhBuf.AsSpan(0, n), T.Zero);
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            var hasNulls = MergeNullMasks(tanhOutput, gradOutput, nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
            {
                var tanh = tanhBuf[i];
                resultBuf[i] = (T.One - tanh * tanh) * gradBuf[i];
            }
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(tanhBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static bool MergeNullMasks<T>(NivaraColumn<T> a, NivaraColumn<T> b, Span<bool> destination) where T : struct, INumber<T>
    {
        var aHasNulls = a.TryGetNullMask(out var aMask);
        var bHasNulls = b.TryGetNullMask(out var bMask);

        if (aHasNulls && bHasNulls)
        {
            for (int i = 0; i < destination.Length; i++)
                destination[i] = aMask[i] || bMask[i];
        }
        else if (aHasNulls)
        {
            aMask.CopyTo(destination);
        }
        else if (bHasNulls)
        {
            bMask.CopyTo(destination);
        }

        return aHasNulls || bHasNulls;
    }

    private static int[] PropagateShape<T>(ReverseGradTensor<T> a, ReverseGradTensor<T> b) where T : struct, INumber<T>
    {
        return a.shape;
    }

    private static int[] PropagateShape<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        return a.shape;
    }

    private static int[] ScalarShape()
    {
        return new[] { 1 };
    }

    #endregion
}
