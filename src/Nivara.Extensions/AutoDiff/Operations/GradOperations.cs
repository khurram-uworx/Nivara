using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

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

        return MatMul(a, b, aRows, aCols, bCols);
    }

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

        return Transpose(a, a.shape[0], a.shape[1]);
    }

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
        var result = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        var aBuf = ArrayPool<T>.Shared.Rent(n);
        var bBuf = ArrayPool<T>.Shared.Rent(n);
        bool hasNulls = false;

        try
        {
            for (int i = 0; i < n; i++)
            {
                nullMask[i] = a.IsNull(i) || b.IsNull(i);
                if (nullMask[i]) { hasNulls = true; }
                aBuf[i] = a[i];
                bBuf[i] = b[i];
            }

            if (typeof(T) == typeof(float))
            {
                TensorPrimitives.Subtract(
                    MemoryMarshal.Cast<T, float>(aBuf.AsSpan(0, n)),
                    MemoryMarshal.Cast<T, float>(bBuf.AsSpan(0, n)),
                    MemoryMarshal.Cast<T, float>(result.AsSpan(0, n)));
            }
            else if (typeof(T) == typeof(double))
            {
                TensorPrimitives.Subtract(
                    MemoryMarshal.Cast<T, double>(aBuf.AsSpan(0, n)),
                    MemoryMarshal.Cast<T, double>(bBuf.AsSpan(0, n)),
                    MemoryMarshal.Cast<T, double>(result.AsSpan(0, n)));
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    if (!nullMask[i])
                        result[i] = aBuf[i] - bBuf[i];
                }
            }

            return CreateColumnWithNullMask(result.AsSpan(0, n), nullMask.AsSpan(0, n), hasNulls);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(result, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
            ArrayPool<T>.Shared.Return(aBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(bBuf, clearArray: true);
        }
    }

    private static NivaraColumn<T> DivideVectorized<T>(NivaraColumn<T> a, NivaraColumn<T> b) where T : struct, INumber<T>
    {
        int n = a.Length;
        var result = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        var aBuf = ArrayPool<T>.Shared.Rent(n);
        var bBuf = ArrayPool<T>.Shared.Rent(n);
        bool hasNulls = false;

        try
        {
            for (int i = 0; i < n; i++)
            {
                nullMask[i] = a.IsNull(i) || b.IsNull(i);
                if (nullMask[i]) { hasNulls = true; }
                aBuf[i] = a[i];
                bBuf[i] = b[i];
            }

            if (typeof(T) == typeof(float))
            {
                TensorPrimitives.Divide(
                    MemoryMarshal.Cast<T, float>(aBuf.AsSpan(0, n)),
                    MemoryMarshal.Cast<T, float>(bBuf.AsSpan(0, n)),
                    MemoryMarshal.Cast<T, float>(result.AsSpan(0, n)));
            }
            else if (typeof(T) == typeof(double))
            {
                TensorPrimitives.Divide(
                    MemoryMarshal.Cast<T, double>(aBuf.AsSpan(0, n)),
                    MemoryMarshal.Cast<T, double>(bBuf.AsSpan(0, n)),
                    MemoryMarshal.Cast<T, double>(result.AsSpan(0, n)));
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    if (!nullMask[i])
                        result[i] = aBuf[i] / bBuf[i];
                }
            }

            return CreateColumnWithNullMask(result.AsSpan(0, n), nullMask.AsSpan(0, n), hasNulls);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(result, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
            ArrayPool<T>.Shared.Return(aBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(bBuf, clearArray: true);
        }
    }

    private static NivaraColumn<T> NegateVectorized<T>(NivaraColumn<T> a) where T : struct, INumber<T>
    {
        int n = a.Length;
        var result = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        var aBuf = ArrayPool<T>.Shared.Rent(n);
        bool hasNulls = false;

        try
        {
            for (int i = 0; i < n; i++)
            {
                nullMask[i] = a.IsNull(i);
                if (nullMask[i]) { hasNulls = true; }
                aBuf[i] = a[i];
            }

            if (typeof(T) == typeof(float))
            {
                TensorPrimitives.Negate(
                    MemoryMarshal.Cast<T, float>(aBuf.AsSpan(0, n)),
                    MemoryMarshal.Cast<T, float>(result.AsSpan(0, n)));
            }
            else if (typeof(T) == typeof(double))
            {
                TensorPrimitives.Negate(
                    MemoryMarshal.Cast<T, double>(aBuf.AsSpan(0, n)),
                    MemoryMarshal.Cast<T, double>(result.AsSpan(0, n)));
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    if (!nullMask[i])
                        result[i] = -aBuf[i];
                }
            }

            return CreateColumnWithNullMask(result.AsSpan(0, n), nullMask.AsSpan(0, n), hasNulls);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(result, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
            ArrayPool<T>.Shared.Return(aBuf, clearArray: true);
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
        var result = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        bool hasNulls = false;

        try
        {
            for (int i = 0; i < aRows; i++)
            {
                for (int j = 0; j < bCols; j++)
                {
                    T sum = T.Zero;
                    bool positionNull = false;
                    for (int k = 0; k < aCols; k++)
                    {
                        var aIndex = i * aCols + k;
                        var bIndex = k * bCols + j;

                        if (a.IsNull(aIndex) || b.IsNull(bIndex))
                        {
                            positionNull = true;
                            hasNulls = true;
                        }
                        else
                        {
                            sum += a[aIndex] * b[bIndex];
                        }
                    }

                    var resIndex = i * bCols + j;
                    if (positionNull)
                    {
                        nullMask[resIndex] = true;
                    }
                    else
                    {
                        result[resIndex] = sum;
                    }
                }
            }

            return CreateColumnWithNullMask(result.AsSpan(0, n), nullMask.AsSpan(0, n), hasNulls);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(result, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> TransposeVectorized<T>(NivaraColumn<T> a, int rows, int cols) where T : struct, INumber<T>
    {
        int n = rows * cols;
        var result = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        bool hasNulls = false;

        try
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    var srcIndex = i * cols + j;
                    var dstIndex = j * rows + i;
                    if (a.IsNull(srcIndex))
                    {
                        nullMask[dstIndex] = true;
                        hasNulls = true;
                    }
                    else
                    {
                        result[dstIndex] = a[srcIndex];
                    }
                }
            }

            return CreateColumnWithNullMask(result.AsSpan(0, n), nullMask.AsSpan(0, n), hasNulls);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(result, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> BroadcastGradient<T>(NivaraColumn<T> scalarGrad, int targetLength) where T : struct, INumber<T>
    {
        if (scalarGrad.Length != 1)
        {
            throw new ArgumentException($"Expected scalar gradient with length 1, got {scalarGrad.Length}");
        }

        if (scalarGrad.IsNull(0))
        {
            var nullableData = new T?[targetLength];
            return NivaraColumn<T>.CreateFromNullable(nullableData);
        }

        var gradValue = scalarGrad[0];
        var result = ArrayPool<T>.Shared.Rent(targetLength);
        try
        {
            Array.Fill(result, gradValue, 0, targetLength);
            return NivaraColumn<T>.Create(result.AsSpan(0, targetLength));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(result, clearArray: true);
        }
    }

    private static NivaraColumn<T> DivideByScalar<T>(NivaraColumn<T> column, int divisor) where T : struct, INumber<T>
    {
        int n = column.Length;
        var result = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        bool hasNulls = false;
        try
        {
            T divisorT = ConvertToT<T>(divisor);
            for (int i = 0; i < n; i++)
            {
                nullMask[i] = column.IsNull(i);
                if (nullMask[i])
                {
                    hasNulls = true;
                }
                else
                {
                    result[i] = column[i] / divisorT;
                }
            }
            return CreateColumnWithNullMask(result.AsSpan(0, n), nullMask.AsSpan(0, n), hasNulls);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(result, clearArray: true);
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
        var result = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        bool hasNulls = false;

        try
        {
            for (int i = 0; i < n; i++)
            {
                nullMask[i] = input.IsNull(i);
                if (nullMask[i]) { hasNulls = true; }
                inputBuf[i] = input[i];
            }

            if (typeof(T) == typeof(float))
            {
                var floatSpan = MemoryMarshal.Cast<T, float>(inputBuf.AsSpan(0, n));
                var resultSpan = MemoryMarshal.Cast<T, float>(result.AsSpan(0, n));
                for (int i = 0; i < n; i++)
                    resultSpan[i] = Math.Max(0.0f, floatSpan[i]);
            }
            else if (typeof(T) == typeof(double))
            {
                var doubleSpan = MemoryMarshal.Cast<T, double>(inputBuf.AsSpan(0, n));
                var resultSpan = MemoryMarshal.Cast<T, double>(result.AsSpan(0, n));
                for (int i = 0; i < n; i++)
                    resultSpan[i] = Math.Max(0.0, doubleSpan[i]);
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    if (!nullMask[i])
                        result[i] = inputBuf[i] > T.Zero ? inputBuf[i] : T.Zero;
                }
            }

            return CreateColumnWithNullMask(result.AsSpan(0, n), nullMask.AsSpan(0, n), hasNulls);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(result, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyReluGradient<T>(NivaraColumn<T> input, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        int n = input.Length;
        var result = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        bool hasNulls = false;
        try
        {
            for (int i = 0; i < n; i++)
            {
                nullMask[i] = input.IsNull(i) || gradOutput.IsNull(i);
                if (nullMask[i])
                {
                    hasNulls = true;
                }
                else
                {
                    result[i] = input[i] > T.Zero ? gradOutput[i] : T.Zero;
                }
            }
            return CreateColumnWithNullMask(result.AsSpan(0, n), nullMask.AsSpan(0, n), hasNulls);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(result, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyAbs<T>(NivaraColumn<T> input) where T : struct, INumber<T>
    {
        int n = input.Length;
        var result = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        bool hasNulls = false;

        try
        {
            for (int i = 0; i < n; i++)
            {
                nullMask[i] = input.IsNull(i);
                if (nullMask[i]) { hasNulls = true; }
                inputBuf[i] = input[i];
            }

            if (typeof(T) == typeof(float))
            {
                TensorPrimitives.Abs(
                    MemoryMarshal.Cast<T, float>(inputBuf.AsSpan(0, n)),
                    MemoryMarshal.Cast<T, float>(result.AsSpan(0, n)));
            }
            else if (typeof(T) == typeof(double))
            {
                TensorPrimitives.Abs(
                    MemoryMarshal.Cast<T, double>(inputBuf.AsSpan(0, n)),
                    MemoryMarshal.Cast<T, double>(result.AsSpan(0, n)));
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    if (!nullMask[i])
                        result[i] = T.Abs(inputBuf[i]);
                }
            }

            return CreateColumnWithNullMask(result.AsSpan(0, n), nullMask.AsSpan(0, n), hasNulls);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(result, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyAbsGradient<T>(NivaraColumn<T> input, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        int n = input.Length;
        var result = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        bool hasNulls = false;
        try
        {
            for (int i = 0; i < n; i++)
            {
                nullMask[i] = input.IsNull(i) || gradOutput.IsNull(i);
                if (nullMask[i])
                {
                    hasNulls = true;
                }
                else
                {
                    var sign = T.CreateChecked(T.Sign(input[i]));
                    result[i] = sign * gradOutput[i];
                }
            }
            return CreateColumnWithNullMask(result.AsSpan(0, n), nullMask.AsSpan(0, n), hasNulls);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(result, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplySigmoid<T>(NivaraColumn<T> input) where T : struct, INumber<T>
    {
        int n = input.Length;
        var result = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        bool hasNulls = false;

        try
        {
            for (int i = 0; i < n; i++)
            {
                nullMask[i] = input.IsNull(i);
                if (nullMask[i]) { hasNulls = true; }
                inputBuf[i] = input[i];
            }

            if (typeof(T) == typeof(float))
            {
                var floatSpan = MemoryMarshal.Cast<T, float>(inputBuf.AsSpan(0, n));
                var resultSpan = MemoryMarshal.Cast<T, float>(result.AsSpan(0, n));
                for (int i = 0; i < n; i++)
                    resultSpan[i] = 1.0f / (1.0f + MathF.Exp(-floatSpan[i]));
            }
            else if (typeof(T) == typeof(double))
            {
                var doubleSpan = MemoryMarshal.Cast<T, double>(inputBuf.AsSpan(0, n));
                var resultSpan = MemoryMarshal.Cast<T, double>(result.AsSpan(0, n));
                for (int i = 0; i < n; i++)
                    resultSpan[i] = 1.0 / (1.0 + Math.Exp(-doubleSpan[i]));
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    if (!nullMask[i])
                    {
                        dynamic x = inputBuf[i];
                        result[i] = (T)(object)(1.0 / (1.0 + Math.Exp(-x)))!;
                    }
                }
            }

            return CreateColumnWithNullMask(result.AsSpan(0, n), nullMask.AsSpan(0, n), hasNulls);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(result, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplySigmoidGradient<T>(NivaraColumn<T> sigmoidOutput, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        int n = sigmoidOutput.Length;
        var result = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        bool hasNulls = false;
        try
        {
            for (int i = 0; i < n; i++)
            {
                nullMask[i] = sigmoidOutput.IsNull(i) || gradOutput.IsNull(i);
                if (nullMask[i])
                {
                    hasNulls = true;
                }
                else
                {
                    var sig = sigmoidOutput[i];
                    result[i] = sig * (T.One - sig) * gradOutput[i];
                }
            }
            return CreateColumnWithNullMask(result.AsSpan(0, n), nullMask.AsSpan(0, n), hasNulls);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(result, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyTanh<T>(NivaraColumn<T> input) where T : struct, INumber<T>
    {
        int n = input.Length;
        var result = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        var inputBuf = ArrayPool<T>.Shared.Rent(n);
        bool hasNulls = false;

        try
        {
            for (int i = 0; i < n; i++)
            {
                nullMask[i] = input.IsNull(i);
                if (nullMask[i]) { hasNulls = true; }
                inputBuf[i] = input[i];
            }

            if (typeof(T) == typeof(float))
            {
                var floatSpan = MemoryMarshal.Cast<T, float>(inputBuf.AsSpan(0, n));
                var resultSpan = MemoryMarshal.Cast<T, float>(result.AsSpan(0, n));
                for (int i = 0; i < n; i++)
                    resultSpan[i] = MathF.Tanh(floatSpan[i]);
            }
            else if (typeof(T) == typeof(double))
            {
                var doubleSpan = MemoryMarshal.Cast<T, double>(inputBuf.AsSpan(0, n));
                var resultSpan = MemoryMarshal.Cast<T, double>(result.AsSpan(0, n));
                for (int i = 0; i < n; i++)
                    resultSpan[i] = Math.Tanh(doubleSpan[i]);
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    if (!nullMask[i])
                    {
                        dynamic x = inputBuf[i];
                        result[i] = (T)(object)Math.Tanh(x)!;
                    }
                }
            }

            return CreateColumnWithNullMask(result.AsSpan(0, n), nullMask.AsSpan(0, n), hasNulls);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(result, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
            ArrayPool<T>.Shared.Return(inputBuf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyTanhGradient<T>(NivaraColumn<T> tanhOutput, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        int n = tanhOutput.Length;
        var result = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        bool hasNulls = false;
        try
        {
            for (int i = 0; i < n; i++)
            {
                nullMask[i] = tanhOutput.IsNull(i) || gradOutput.IsNull(i);
                if (nullMask[i])
                {
                    hasNulls = true;
                }
                else
                {
                    var tanh = tanhOutput[i];
                    result[i] = (T.One - tanh * tanh) * gradOutput[i];
                }
            }
            return CreateColumnWithNullMask(result.AsSpan(0, n), nullMask.AsSpan(0, n), hasNulls);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(result, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> CreateColumnWithNullMask<T>(ReadOnlySpan<T> data, ReadOnlySpan<bool> nullMask, bool hasNulls) where T : struct, INumber<T>
    {
        if (!hasNulls)
        {
            return NivaraColumn<T>.Create(data);
        }

        var nullableData = new T?[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            nullableData[i] = nullMask[i] ? null : data[i];
        }

        return NivaraColumn<T>.CreateFromNullable(nullableData);
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
