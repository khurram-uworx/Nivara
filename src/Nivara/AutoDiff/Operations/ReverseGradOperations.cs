using Nivara.AutoDiff.Utilities;
using Nivara.Helpers;
using Nivara.Tensors;
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace Nivara.AutoDiff.Operations;

public static class ReverseGradOperations
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

        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a, b), PropagateShape(a, b));

        if (GradientUtils.ShouldTrackGrad(a, b))
        {
            var gradFn = new OpNode<T>("Add", new object[] { a, b }, (typedGradOutput, sgn) =>
            {
                if (GradientUtils.ShouldTrackGrad(a))
                {
                    AccumulateGradient(a, typedGradOutput, sgn);
                }
                if (b.RequiresGrad)
                {
                    AccumulateGradient(b, typedGradOutput, sgn);
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

        var result = a.Data.Subtract(b.Data);

        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a, b), PropagateShape(a, b));

        if (GradientUtils.ShouldTrackGrad(a, b))
        {
            var gradFn = new OpNode<T>("Subtract", new object[] { a, b }, (typedGradOutput, sgn) =>
            {
                if (GradientUtils.ShouldTrackGrad(a))
                {
                    AccumulateGradient(a, typedGradOutput, sgn);
                }
                if (b.RequiresGrad)
                {
                    var negatedGrad = typedGradOutput.Negate();
                    AccumulateGradient(b, negatedGrad, sgn);
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

        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a, b), PropagateShape(a, b));

        if (GradientUtils.ShouldTrackGrad(a, b))
        {
            var gradFn = new OpNode<T>("Multiply", new object[] { a, b }, (typedGradOutput, sgn) =>
            {
                if (GradientUtils.ShouldTrackGrad(a))
                {
                    var aGrad = typedGradOutput * b.Data;
                    AccumulateGradient(a, aGrad, sgn);
                }
                if (b.RequiresGrad)
                {
                    var bGrad = typedGradOutput * a.Data;
                    AccumulateGradient(b, bGrad, sgn);
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

        var result = a.Data.Divide(b.Data);

        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a, b), PropagateShape(a, b));

        if (GradientUtils.ShouldTrackGrad(a, b))
        {
            var gradFn = new OpNode<T>("Divide", new object[] { a, b }, (typedGradOutput, sgn) =>
            {
                if (GradientUtils.ShouldTrackGrad(a))
                {
                    var aGrad = typedGradOutput.Divide(b.Data);
                    AccumulateGradient(a, aGrad, sgn);
                }
                if (b.RequiresGrad)
                {
                    var quotient = a.Data.Divide(b.Data);
                    var bGradPositive = quotient.Divide(b.Data);
                    var bGrad = bGradPositive.Negate();
                    var finalBGrad = bGrad * typedGradOutput;
                    AccumulateGradient(b, finalBGrad, sgn);
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

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffMatMul",
            a.Length + b.Length,
            a.Data.HasNulls || b.Data.HasNulls,
            () =>
            {
                var result = a.Data.MatMul(b.Data, aRows, aCols, bCols);

                var resultShape = new[] { aRows, bCols };
                var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a, b), resultShape);

                if (GradientUtils.ShouldTrackGrad(a, b))
                {
                    var gradFn = new OpNode<T>("MatMul", new object[] { a, b }, (typedGradOutput, sgn) =>
                    {
                        if (GradientUtils.ShouldTrackGrad(a))
                        {
                            var bTransposed = b.Data.Transpose(bRows, bCols);
                            var aGrad = typedGradOutput.MatMul(bTransposed, aRows, bCols, bRows);
                            AccumulateGradient(a, aGrad, sgn);
                        }
                        if (b.RequiresGrad)
                        {
                            var aTransposed = a.Data.Transpose(aRows, aCols);
                            var bGrad = aTransposed.MatMul(typedGradOutput, aCols, aRows, bCols);
                            AccumulateGradient(b, bGrad, sgn);
                        }
                    });

                    ComputationGraph.AddNode(resultTensor, gradFn);
                }

                return resultTensor;
            },
            AutoDiffDiagnostics.MatrixNote("MatMul", aRows, aCols, bCols));
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

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffTranspose",
            a.Length,
            a.Data.HasNulls,
            () =>
            {
                var result = a.Data.Transpose(rows, cols);

                var resultShape = new[] { cols, rows };
                var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a), resultShape);

                if (GradientUtils.ShouldTrackGrad(a))
                {
                    var gradFn = new OpNode<T>("Transpose", new object[] { a }, (typedGradOutput, sgn) =>
                    {
                        var aGrad = typedGradOutput.Transpose(cols, rows);
                        AccumulateGradient(a, aGrad, sgn);
                    });

                    ComputationGraph.AddNode(resultTensor, gradFn);
                }

                return resultTensor;
            },
            $"AutoDiff=Transpose;Shape={rows}x{cols}->{cols}x{rows}");
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
        var resultTensor = new ReverseGradTensor<T>(resultData, GradientUtils.ShouldTrackGrad(a), ScalarShape());

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Sum", new object[] { a }, (typedGradOutput, sgn) =>
            {
                var aGrad = BroadcastGradient(typedGradOutput, a.Length);
                AccumulateGradient(a, aGrad, sgn);
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
        var resultTensor = new ReverseGradTensor<T>(resultData, GradientUtils.ShouldTrackGrad(a), ScalarShape());

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Mean", new object[] { a }, (typedGradOutput, sgn) =>
            {
                var aGrad = BroadcastGradient(typedGradOutput, a.Length);
                var scaledGrad = aGrad.Divide(T.CreateChecked(a.Length));
                AccumulateGradient(a, scaledGrad, sgn);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// MeanPool: averages values within consecutive groups of <paramref name="poolSize"/>.
    /// Expects a flattened 1D or row-major 2D tensor where each row is [poolSize * embedDim].
    /// The first dimension (batch) is inferred from tensor length / (poolSize * embedDim).
    /// Output shape: [batchSize, embedDim].
    /// Backward: gradients are distributed equally to all positions in each pool window.
    /// </summary>
    public static ReverseGradTensor<T> MeanPool<T>(ReverseGradTensor<T> a, int poolSize, int embedDim)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (poolSize <= 0) throw new ArgumentOutOfRangeException(nameof(poolSize));
        if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));
        if (a.Length == 0) throw new InvalidOperationException("Cannot mean-pool an empty tensor.");
        if (a.Length % (poolSize * embedDim) != 0)
            throw new ArgumentException(
                $"Tensor length {a.Length} is not divisible by poolSize*embedDim = {poolSize * embedDim}.");

        int batchSize = a.Length / (poolSize * embedDim);

        var src = new T[a.Length];
        a.Data.CopyTo(src, default(T)!);

        var resultValues = new T[batchSize * embedDim];
        T tPoolSize = T.CreateChecked(poolSize);

        for (int b = 0; b < batchSize; b++)
        {
            int rowOffset = b * poolSize * embedDim;
            for (int d = 0; d < embedDim; d++)
            {
                T sum = T.Zero;
                for (int l = 0; l < poolSize; l++)
                    sum += src[rowOffset + l * embedDim + d];
                resultValues[b * embedDim + d] = sum / tPoolSize;
            }
        }

        var resultCol = NivaraColumn<T>.Create(resultValues);
        var result = new ReverseGradTensor<T>(resultCol, GradientUtils.ShouldTrackGrad(a), [batchSize, embedDim]);

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("MeanPool", new object[] { a }, (typedGradOutput, sgn) =>
            {
                var gradOut = new T[a.Length];
                var gradSrc = new T[typedGradOutput.Length];
                typedGradOutput.CopyTo(gradSrc, default(T)!);

                for (int b = 0; b < batchSize; b++)
                {
                    int rowOffset = b * poolSize * embedDim;
                    for (int d = 0; d < embedDim; d++)
                    {
                        T gradVal = gradSrc[b * embedDim + d] / tPoolSize;
                        for (int l = 0; l < poolSize; l++)
                            gradOut[rowOffset + l * embedDim + d] = gradVal;
                    }
                }

                var gradCol = NivaraColumn<T>.Create(gradOut);
                AccumulateGradient(a, gradCol, sgn);
            });

            ComputationGraph.AddNode(result, gradFn);
        }

        return result;
    }

    #endregion

    #region Activation Functions

    public static ReverseGradTensor<T> Relu<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffRelu",
            a.Length,
            a.Data.HasNulls,
            () =>
            {
                var result = a.Data.Relu();

                var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a), PropagateShape(a));

                if (GradientUtils.ShouldTrackGrad(a))
                {
                    var gradFn = new OpNode<T>("Relu", new object[] { a }, (typedGradOutput, sgn) =>
                    {
                        var aGrad = a.Data.ReluGradient(typedGradOutput);
                        AccumulateGradient(a, aGrad, sgn);
                    });

                    ComputationGraph.AddNode(resultTensor, gradFn);
                }

                return resultTensor;
            },
            AutoDiffDiagnostics.ShapeNote("Relu", a.Shape));
    }

    public static ReverseGradTensor<T> Sigmoid<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffSigmoid",
            a.Length,
            a.Data.HasNulls,
            () =>
            {
                var result = a.Data.Sigmoid();

                var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a), PropagateShape(a));

                if (GradientUtils.ShouldTrackGrad(a))
                {
                    var gradFn = new OpNode<T>("Sigmoid", new object[] { a }, (typedGradOutput, sgn) =>
                    {
                        var aGrad = result.SigmoidGradient(typedGradOutput);
                        AccumulateGradient(a, aGrad, sgn);
                    });

                    ComputationGraph.AddNode(resultTensor, gradFn);
                }

                return resultTensor;
            },
            AutoDiffDiagnostics.ShapeNote("Sigmoid", a.Shape));
    }

    public static ReverseGradTensor<T> Tanh<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffTanh",
            a.Length,
            a.Data.HasNulls,
            () =>
            {
                var result = a.Data.Tanh();

                var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a), PropagateShape(a));

                if (GradientUtils.ShouldTrackGrad(a))
                {
                    var gradFn = new OpNode<T>("Tanh", new object[] { a }, (typedGradOutput, sgn) =>
                    {
                        var aGrad = result.TanhGradient(typedGradOutput);
                        AccumulateGradient(a, aGrad, sgn);
                    });

                    ComputationGraph.AddNode(resultTensor, gradFn);
                }

                return resultTensor;
            },
            AutoDiffDiagnostics.ShapeNote("Tanh", a.Shape));
    }

    public static ReverseGradTensor<T> Negate<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = a.Data.Negate();
        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a), PropagateShape(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Negate", new object[] { a }, (typedGradOutput, sgn) =>
            {
                var aGrad = typedGradOutput.Negate();
                AccumulateGradient(a, aGrad, sgn);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Abs<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = a.Data.Abs();
        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a), PropagateShape(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Abs", new object[] { a }, (typedGradOutput, sgn) =>
            {
                var aGrad = a.Data.AbsGradient(typedGradOutput);
                AccumulateGradient(a, aGrad, sgn);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Clip<T>(ReverseGradTensor<T> a, T min, T max)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = a.Data.Clamp(min, max);
        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a), PropagateShape(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Clip", new object[] { a, min, max }, (typedGradOutput, sgn) =>
            {
                var aGrad = a.Data.ClipGradient(typedGradOutput, min, max);
                AccumulateGradient(a, aGrad, sgn);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> LeakyRelu<T>(ReverseGradTensor<T> a, T negativeSlope = default)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (negativeSlope == T.Zero)
            negativeSlope = T.CreateChecked(0.01);

        var result = a.Data.LeakyRelu(negativeSlope);
        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a), PropagateShape(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("LeakyRelu", new object[] { a, negativeSlope }, (typedGradOutput, sgn) =>
            {
                var aGrad = a.Data.LeakyReluGradient(typedGradOutput, negativeSlope);
                AccumulateGradient(a, aGrad, sgn);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Exp<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = a.Data.Exp();
        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a), PropagateShape(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Exp", new object[] { a }, (typedGradOutput, sgn) =>
            {
                var aGrad = typedGradOutput * result;
                AccumulateGradient(a, aGrad, sgn);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Log<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = a.Data.Log();
        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a), PropagateShape(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Log", new object[] { a }, (typedGradOutput, sgn) =>
            {
                var aGrad = a.Data.LogGradient(typedGradOutput);
                AccumulateGradient(a, aGrad, sgn);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Pow<T>(ReverseGradTensor<T> a, double exponent) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = ApplyPow(a.Data, exponent);
        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a), PropagateShape(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var savedInput = a.Data;
            var gradFn = new OpNode<T>("Pow", new object[] { a, exponent }, (typedGradOutput, sgn) =>
            {
                var aGrad = ApplyPowGradient(savedInput, typedGradOutput, exponent);
                AccumulateGradient(a, aGrad, sgn);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Extracts a contiguous slice from a 1D or row-vector tensor.
    /// Input shape: [1, n] or [n]; output shape: [1, length] or [length].
    /// Gradient flows back to the original positions in the input.
    /// </summary>
    public static ReverseGradTensor<T> Slice<T>(ReverseGradTensor<T> a, int start, int length)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (start < 0) throw new ArgumentOutOfRangeException(nameof(start));
        if (length <= 0) throw new ArgumentOutOfRangeException(nameof(length));
        if (start + length > a.Length)
            throw new ArgumentException($"Slice ({start}..{start + length}) exceeds tensor length {a.Length}");

        int fullDim = a.shape.Length == 2 ? a.shape[1] : a.Length;
        int batchDim = a.shape.Length == 2 ? a.shape[0] : 1;

        if (start + length > fullDim)
            throw new ArgumentException($"Slice ({start}..{start + length}) exceeds dimension size {fullDim}");

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffSlice",
            length,
            a.Data.HasNulls,
            () =>
            {
                int resultLen = batchDim * length;
                var resultValues = new T[resultLen];
                bool[]? resultNullMask = a.HasNulls ? new bool[resultLen] : null;
                bool anyResultNulls = false;

                var srcData = new T[a.Length];
                a.Data.CopyTo(srcData, default(T)!);

                for (int r = 0; r < batchDim; r++)
                {
                    int srcOffset = r * fullDim + start;
                    int dstOffset = r * length;
                    Array.Copy(srcData, srcOffset, resultValues, dstOffset, length);

                    if (a.HasNulls && resultNullMask != null)
                    {
                        if (a.Data.TryGetNullMask(out var srcNull))
                        {
                            for (int i = 0; i < length; i++)
                            {
                                bool isNull = srcNull[srcOffset + i];
                                resultNullMask[dstOffset + i] = isNull;
                                anyResultNulls |= isNull;
                            }
                        }
                    }
                }

                var resultCol = anyResultNulls
                    ? NivaraColumn<T>.CreateFromSpans(resultValues, resultNullMask!)
                    : NivaraColumn<T>.Create(resultValues);

                var resultShape = batchDim == 1
                    ? new[] { length }
                    : new[] { batchDim, length };

                var result = new ReverseGradTensor<T>(resultCol, a.RequiresGrad, resultShape);

                if (a.RequiresGrad)
                {
                    var savedStart = start;
                    var savedLength = length;
                    var savedFullDim = fullDim;
                    var savedBatchDim = batchDim;
                    var gradFn = new OpNode<T>("Slice", [a], (typedGradOutput, sgn) =>
                    {
                        var gradData = new T[typedGradOutput.Length];
                        typedGradOutput.CopyTo(gradData, default(T)!);

                        var gradResult = new T[a.Length];

                        if (savedBatchDim == 1)
                        {
                            Array.Copy(gradData, 0, gradResult, savedStart, savedLength);
                        }
                        else
                        {
                            for (int r = 0; r < savedBatchDim; r++)
                            {
                                int srcOffset = r * savedLength;
                                int dstOffset = r * savedFullDim + savedStart;
                                Array.Copy(gradData, srcOffset, gradResult, dstOffset, savedLength);
                            }
                        }

                        var gradCol = NivaraColumn<T>.Create(gradResult);
                        AccumulateGradient(a, gradCol, sgn);
                    });

                    ComputationGraph.AddNode(result, gradFn);
                }

                return result;
            },
            $"AutoDiff=Slice;Start={start};Length={length};FullDim={fullDim}");
    }

    public static ReverseGradTensor<T> Concat<T>(ReverseGradTensor<T>[] tensors, int axis = 0)
        where T : struct, INumber<T>
    {
        if (tensors == null || tensors.Length == 0)
            throw new ArgumentException("At least one tensor is required for Concat.", nameof(tensors));
        if (tensors.Length == 1)
            return tensors[0];

        int rank = tensors[0].Rank;
        if (rank < 1 || rank > 2)
            throw new ArgumentException($"Concat supports 1D or 2D tensors, got rank {rank}.");

        for (int i = 1; i < tensors.Length; i++)
        {
            if (tensors[i].Rank != rank)
                throw new ArgumentException(
                    $"All tensors must have the same rank. Tensor 0 has rank {rank}, tensor {i} has rank {tensors[i].Rank}.");

            if (rank == 2 && axis == 1 && tensors[i].shape[0] != tensors[0].shape[0])
                throw new ArgumentException(
                    $"For axis=1 concatenation, all tensors must have the same number of rows. " +
                    $"Tensor 0 has {tensors[0].shape[0]} rows, tensor {i} has {tensors[i].shape[0]} rows.");

            if (rank == 2 && axis == 0 && tensors[i].shape[1] != tensors[0].shape[1])
                throw new ArgumentException(
                    $"For axis=0 concatenation, all tensors must have the same number of columns. " +
                    $"Tensor 0 has {tensors[0].shape[1]} columns, tensor {i} has {tensors[i].shape[1]} columns.");
        }

        // Compute sizes for backward splitting
        int[] inputLengths = new int[tensors.Length];
        for (int i = 0; i < tensors.Length; i++)
            inputLengths[i] = tensors.Length == 1 ? tensors[i].Length : tensors[i].Length;

        bool shouldTrack = false;
        foreach (var t in tensors)
        {
            if (t.RequiresGrad) { shouldTrack = true; break; }
        }

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffConcat",
            tensors.Sum(t => t.Length),
            tensors.Any(t => t.Data.HasNulls),
            () =>
            {
                if (rank == 1)
                {
                    // 1D concatenation: copy data sequentially
                    int totalLen = tensors.Sum(t => t.Length);
                    var resultData = new T[totalLen];
                    int offset = 0;
                    foreach (var t in tensors)
                    {
                        t.Data.CopyTo(resultData.AsSpan(offset, t.Length), default(T)!);
                        offset += t.Length;
                    }

                    var resultCol = NivaraColumn<T>.Create(resultData);
                    var result = new ReverseGradTensor<T>(resultCol, shouldTrack, [totalLen]);

                    if (shouldTrack)
                    {
                        var savedLengths = inputLengths;
                        var gradFn = new OpNode<T>("Concat", tensors, (typedGradOutput, sgn) =>
                        {
                            var fullGrad = new T[typedGradOutput.Length];
                            typedGradOutput.CopyTo(fullGrad.AsSpan(), default(T)!);
                            int gradOffset = 0;
                            for (int i = 0; i < tensors.Length; i++)
                            {
                                if (tensors[i].RequiresGrad)
                                {
                                    var gradSlice = new T[savedLengths[i]];
                                    for (int j = 0; j < savedLengths[i]; j++)
                                        gradSlice[j] = fullGrad[gradOffset + j];

                                    var gradCol = NivaraColumn<T>.Create(gradSlice);
                                    AccumulateGradient(tensors[i], gradCol, sgn);
                                }
                                gradOffset += savedLengths[i];
                            }
                        });

                        ComputationGraph.AddNode(result, gradFn);
                    }

                    return result;
                }
                else // rank == 2
                {
                    int rows = tensors[0].shape[0];
                    int totalCols = tensors.Sum(t => t.shape[1]);

                    var resultData = new T[rows * totalCols];

                    if (axis == 1)
                    {
                        // Column concatenation: place each tensor's columns side by side
                        int colOffset = 0;
                        foreach (var t in tensors)
                        {
                            int tCols = t.shape[1];
                            var srcData = new T[t.Length];
                            t.Data.CopyTo(srcData, default(T)!);
                            for (int r = 0; r < rows; r++)
                            {
                                Array.Copy(srcData, r * tCols, resultData, r * totalCols + colOffset, tCols);
                            }
                            colOffset += tCols;
                        }
                    }
                    else // axis == 0
                    {
                        // Row concatenation: stack tensors vertically
                        int totalRows = tensors.Sum(t => t.shape[0]);
                        int cols = tensors[0].shape[1];
                        resultData = new T[totalRows * cols];
                        int rowOffset = 0;
                        foreach (var t in tensors)
                        {
                            int tRows = t.shape[0];
                            var srcData = new T[t.Length];
                            t.Data.CopyTo(srcData, default(T)!);
                            Array.Copy(srcData, 0, resultData, rowOffset * cols, tRows * cols);
                            rowOffset += tRows;
                        }
                        totalCols = cols;
                    }

                    var resultCol = NivaraColumn<T>.Create(resultData);
                    var resultShape = axis == 0
                        ? new[] { tensors.Sum(t => t.shape[0]), tensors[0].shape[1] }
                        : new[] { tensors[0].shape[0], totalCols };
                    var result = new ReverseGradTensor<T>(resultCol, shouldTrack, resultShape);

                    if (shouldTrack)
                    {
                        // Save shape info for backward (NivaraColumn<T> doesn't have .shape)
                        int outputRows = resultShape[0];
                        int outputCols = resultShape[1];
                        var inputShapes = new int[tensors.Length][];
                        var inputCols = new int[tensors.Length];
                        var inputRows = new int[tensors.Length];
                        for (int i = 0; i < tensors.Length; i++)
                        {
                            inputShapes[i] = tensors[i].shape;
                            inputCols[i] = tensors[i].shape[1];
                            inputRows[i] = tensors[i].shape[0];
                        }

                        var savedTensors = tensors;
                        var savedAxis = axis;
                        var gradFn = new OpNode<T>("Concat", tensors, (typedGradOutput, sgn) =>
                        {
                            if (savedAxis == 1)
                            {
                                // Split along columns — extract column slices per row
                                int colOff = 0;
                                if (typedGradOutput.TryGetSpan(out var srcSpan))
                                {
                                    for (int i = 0; i < savedTensors.Length; i++)
                                    {
                                        if (savedTensors[i].RequiresGrad)
                                        {
                                            int tCols = inputCols[i];
                                            var gradData = new T[outputRows * tCols];
                                            for (int r = 0; r < outputRows; r++)
                                            {
                                                srcSpan.Slice(r * outputCols + colOff, tCols).CopyTo(gradData.AsSpan(r * tCols));
                                            }
                                            var gradCol = NivaraColumn<T>.Create(gradData);
                                            AccumulateGradient(savedTensors[i], gradCol, sgn);
                                        }
                                        colOff += inputCols[i];
                                    }
                                }
                                else
                                {
                                    var srcArr = new T[typedGradOutput.Length];
                                    typedGradOutput.CopyTo(srcArr.AsSpan(), default(T)!);
                                    for (int i = 0; i < savedTensors.Length; i++)
                                    {
                                        if (savedTensors[i].RequiresGrad)
                                        {
                                            int tCols = inputCols[i];
                                            var gradData = new T[outputRows * tCols];
                                            for (int r = 0; r < outputRows; r++)
                                            {
                                                Array.Copy(srcArr, r * outputCols + colOff, gradData, r * tCols, tCols);
                                            }
                                            var gradCol = NivaraColumn<T>.Create(gradData);
                                            AccumulateGradient(savedTensors[i], gradCol, sgn);
                                        }
                                        colOff += inputCols[i];
                                    }
                                }
                            }
                            else // axis == 0
                            {
                                // Split along rows — extract row-contiguous blocks
                                int rowOff = 0;
                                if (typedGradOutput.TryGetSpan(out var srcSpan))
                                {
                                    for (int i = 0; i < savedTensors.Length; i++)
                                    {
                                        if (savedTensors[i].RequiresGrad)
                                        {
                                            int tRows = inputRows[i];
                                            var gradData = new T[tRows * outputCols];
                                            for (int r = 0; r < tRows; r++)
                                            {
                                                srcSpan.Slice((rowOff + r) * outputCols, outputCols).CopyTo(gradData.AsSpan(r * outputCols));
                                            }
                                            var gradCol = NivaraColumn<T>.Create(gradData);
                                            AccumulateGradient(savedTensors[i], gradCol, sgn);
                                        }
                                        rowOff += inputRows[i];
                                    }
                                }
                                else
                                {
                                    var srcFull = new T[typedGradOutput.Length];
                                    typedGradOutput.CopyTo(srcFull.AsSpan(), default(T)!);
                                    for (int i = 0; i < savedTensors.Length; i++)
                                    {
                                        if (savedTensors[i].RequiresGrad)
                                        {
                                            int tRows = inputRows[i];
                                            var gradData = new T[tRows * outputCols];
                                            for (int r = 0; r < tRows; r++)
                                            {
                                                Array.Copy(srcFull, (rowOff + r) * outputCols, gradData, r * outputCols, outputCols);
                                            }
                                            var gradCol = NivaraColumn<T>.Create(gradData);
                                            AccumulateGradient(savedTensors[i], gradCol, sgn);
                                        }
                                        rowOff += inputRows[i];
                                    }
                                }
                            }
                        });

                        ComputationGraph.AddNode(result, gradFn);
                    }

                    return result;
                }
            },
            $"AutoDiff=Concat;Axis={axis};Count={tensors.Length}");
    }

    public static ReverseGradTensor<T> Softmax<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = a.Data.Softmax(a.Rank >= 2 ? a.shape[1] : a.Length);
        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a), PropagateShape(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var savedResult = result;
            var gradFn = new OpNode<T>("Softmax", new object[] { a }, (typedGradOutput, sgn) =>
            {
                var aGrad = savedResult.SoftmaxGradient(typedGradOutput, a.Rank >= 2 ? a.shape[1] : a.Length);
                AccumulateGradient(a, aGrad, sgn);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> LogSoftmax<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = a.Data.LogSoftmax(a.Rank >= 2 ? a.shape[1] : a.Length);
        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a), PropagateShape(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("LogSoftmax", new object[] { a }, (typedGradOutput, sgn) =>
            {
                var aGrad = a.Data.LogSoftmaxGradient(typedGradOutput, a.Rank >= 2 ? a.shape[1] : a.Length);
                AccumulateGradient(a, aGrad, sgn);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> RMSNorm<T>(ReverseGradTensor<T> a, double eps = 1e-5)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffRMSNorm",
            a.Length,
            a.Data.HasNulls,
            () =>
            {
                var result = ApplyRMSNorm(a.Data, eps);
                var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a), PropagateShape(a));

                if (GradientUtils.ShouldTrackGrad(a))
                {
                    var savedInput = a.Data;
                    var savedEps = eps;
                    var gradFn = new OpNode<T>("RMSNorm", new object[] { a, eps }, (typedGradOutput, sgn) =>
                    {
                        var aGrad = ApplyRMSNormGradient(savedInput, typedGradOutput, savedEps);
                        AccumulateGradient(a, aGrad, sgn);
                    });

                    ComputationGraph.AddNode(resultTensor, gradFn);
                }

                return resultTensor;
            },
            AutoDiffDiagnostics.ShapeNote("RMSNorm", a.Shape));
    }

    public static ReverseGradTensor<T> PerRowRMSNorm<T>(ReverseGradTensor<T> a, int rows, int cols, double eps = 1e-5)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffPerRowRMSNorm",
            a.Length,
            a.Data.HasNulls,
            () =>
            {
                var srcData = new T[a.Length];
                a.Data.CopyTo(srcData, default(T)!);
                var resultData = new T[rows * cols];

                if (typeof(T) == typeof(float))
                {
                    var srcFloat = System.Runtime.CompilerServices.Unsafe.As<T[], float[]>(ref srcData);
                    var resFloat = System.Runtime.CompilerServices.Unsafe.As<T[], float[]>(ref resultData);
                    for (int i = 0; i < rows; i++)
                    {
                        int baseIdx = i * cols;
                        var row = srcFloat.AsSpan(baseIdx, cols);
                        var dst = resFloat.AsSpan(baseIdx, cols);
                        float sumSq = TensorPrimitives.Dot(row, row);
                        float rms = MathF.Sqrt(sumSq / cols + (float)eps);
                        TensorPrimitives.Multiply(row, 1.0f / rms, dst);
                    }
                }
                else if (typeof(T) == typeof(double))
                {
                    var srcDouble = System.Runtime.CompilerServices.Unsafe.As<T[], double[]>(ref srcData);
                    var resDouble = System.Runtime.CompilerServices.Unsafe.As<T[], double[]>(ref resultData);
                    for (int i = 0; i < rows; i++)
                    {
                        int baseIdx = i * cols;
                        var row = srcDouble.AsSpan(baseIdx, cols);
                        var dst = resDouble.AsSpan(baseIdx, cols);
                        double sumSq = TensorPrimitives.Dot(row, row);
                        double rms = Math.Sqrt(sumSq / cols + eps);
                        TensorPrimitives.Multiply(row, 1.0 / rms, dst);
                    }
                }
                else
                {
                    for (int i = 0; i < rows; i++)
                    {
                        int baseIdx = i * cols;
                        double sumSq = 0;
                        for (int j = 0; j < cols; j++)
                        {
                            double v = double.CreateChecked(srcData[baseIdx + j]);
                            sumSq += v * v;
                        }
                        double rms = Math.Sqrt(sumSq / cols + eps);
                        double invRms = 1.0 / rms;
                        for (int j = 0; j < cols; j++)
                            resultData[baseIdx + j] = T.CreateChecked(double.CreateChecked(srcData[baseIdx + j]) * invRms);
                    }
                }

                var resultCol = NivaraColumn<T>.Create(resultData);
                var result = new ReverseGradTensor<T>(resultCol, a.RequiresGrad, a.Shape);

                if (a.RequiresGrad)
                {
                    var savedInput = new T[a.Length];
                    a.Data.CopyTo(savedInput, default(T)!);

                    var gradFn = new OpNode<T>("PerRowRMSNorm", [a], (typedGradOutput, sgn) =>
                    {
                        var gradOut = new T[typedGradOutput.Length];
                        typedGradOutput.CopyTo(gradOut.AsSpan(), default(T)!);

                        var gradResult = new T[rows * cols];

                        for (int i = 0; i < rows; i++)
                        {
                            int baseIdx = i * cols;

                            double sumSq = 0;
                            for (int j = 0; j < cols; j++)
                            {
                                double v = double.CreateChecked(savedInput[baseIdx + j]);
                                sumSq += v * v;
                            }
                            double rms = Math.Sqrt(sumSq / cols + eps);
                            double invRms = 1.0 / rms;
                            double rms3 = rms * rms * rms;

                            double sumGradX = 0;
                            for (int j = 0; j < cols; j++)
                            {
                                double g = double.CreateChecked(gradOut[baseIdx + j]);
                                double v = double.CreateChecked(savedInput[baseIdx + j]);
                                sumGradX += g * v;
                            }

                            double scale = sumGradX / (cols * rms3);

                            for (int j = 0; j < cols; j++)
                            {
                                double g = double.CreateChecked(gradOut[baseIdx + j]);
                                double v = double.CreateChecked(savedInput[baseIdx + j]);
                                gradResult[baseIdx + j] = T.CreateChecked(g * invRms - v * scale);
                            }
                        }

                        var gradCol = NivaraColumn<T>.Create(gradResult);
                        AccumulateGradient(a, gradCol, sgn);
                    });

                    ComputationGraph.AddNode(result, gradFn);
                }

                return result;
            },
            AutoDiffDiagnostics.ShapeNote("PerRowRMSNorm", a.Shape));
    }

    public static ReverseGradTensor<T> Dropout<T>(ReverseGradTensor<T> input, double probability, bool isTraining)
        where T : struct, INumber<T>
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (probability < 0.0 || probability >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(probability), "Dropout probability must be in [0, 1).");

        if (!isTraining || probability <= 0.0)
            return input;

        var keepMask = new bool[input.Length];
        var random = Random.Shared;
        for (int i = 0; i < keepMask.Length; i++)
            keepMask[i] = random.NextDouble() >= probability;

        var scale = T.CreateChecked(1.0 / (1.0 - probability));
        return DropoutWithMask(input, keepMask, scale);
    }

    internal static ReverseGradTensor<T> DropoutWithMask<T>(ReverseGradTensor<T> input, ReadOnlySpan<bool> keepMask, T scale)
        where T : struct, INumber<T>
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (keepMask.Length != input.Length)
            throw new ArgumentException($"Dropout mask length ({keepMask.Length}) must match input length ({input.Length})", nameof(keepMask));

        var savedMask = keepMask.ToArray();
        var result = ApplyDropout(input.Data, savedMask, scale);
        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(input), PropagateShape(input));

        if (GradientUtils.ShouldTrackGrad(input))
        {
            var gradFn = new OpNode<T>("Dropout", new object[] { input }, (typedGradOutput, sgn) =>
            {
                var inputGrad = ApplyDropoutGradient(input.Data, typedGradOutput, savedMask, scale);
                AccumulateGradient(input, inputGrad, sgn);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    #endregion

    #region Indexing Operations

    /// <summary>
    /// Looks up sparse feature rows from an embedding matrix and sums them per batch row.
    /// weight shape: [numEmbeddings, embeddingDim], indices shape: [batchSize, maxActiveFeatures].
    /// Each valid index contributes one embedding row to the corresponding output row.
    /// paddingIndex entries are ignored in both forward and backward passes.
    /// </summary>
    public static ReverseGradTensor<T> SparseEmbeddingBag<T>(
        ReverseGradTensor<T> weight,
        ReverseGradTensor<T> indices,
        int paddingIndex = -1)
        where T : struct, INumber<T>
    {
        if (weight == null) throw new ArgumentNullException(nameof(weight));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (weight.Rank != 2)
            throw new ArgumentException("SparseEmbeddingBag weight must be a 2D tensor.", nameof(weight));
        if (indices.Rank != 2)
            throw new ArgumentException("SparseEmbeddingBag indices must be a 2D tensor.", nameof(indices));

        int numEmbeddings = weight.shape[0];
        int embeddingDim = weight.shape[1];
        int batchSize = indices.shape[0];
        int maxActiveFeatures = indices.shape[1];

        var parsedIndices = new int[indices.Length];
        for (int i = 0; i < indices.Length; i++)
        {
            int index = int.CreateChecked(indices.Data[i]);
            if (index != paddingIndex && ((uint)index >= (uint)numEmbeddings))
                throw new ArgumentOutOfRangeException(
                    nameof(indices),
                    $"Index at position {i} is {index}, must be {paddingIndex} or in range [0, {numEmbeddings}).");

            parsedIndices[i] = index;
        }

        bool weightHasNulls = weight.HasNulls;
        var resultValues = new T[batchSize * embeddingDim];
        bool[]? resultNullMask = weightHasNulls ? new bool[resultValues.Length] : null;
        bool anyResultNulls = false;

        if (weightHasNulls)
        {
            for (int batch = 0; batch < batchSize; batch++)
            {
                int indexBase = batch * maxActiveFeatures;
                int outputBase = batch * embeddingDim;

                for (int slot = 0; slot < maxActiveFeatures; slot++)
                {
                    int index = parsedIndices[indexBase + slot];
                    if (index == paddingIndex)
                        continue;

                    int weightBase = index * embeddingDim;
                    for (int dim = 0; dim < embeddingDim; dim++)
                    {
                        int weightOffset = weightBase + dim;
                        int outputOffset = outputBase + dim;

                        if (weight.IsNull(weightOffset))
                        {
                            resultNullMask![outputOffset] = true;
                            anyResultNulls = true;
                            continue;
                        }

                        resultValues[outputOffset] += weight.Data[weightOffset];
                    }
                }
            }
        }
        else
        {
            var weightSpan = weight.Data.AsSpan();
            for (int batch = 0; batch < batchSize; batch++)
            {
                int indexBase = batch * maxActiveFeatures;
                int outputBase = batch * embeddingDim;

                for (int slot = 0; slot < maxActiveFeatures; slot++)
                {
                    int index = parsedIndices[indexBase + slot];
                    if (index == paddingIndex)
                        continue;

                    int weightBase = index * embeddingDim;
                    var src = weightSpan.Slice(weightBase, embeddingDim);
                    var dst = resultValues.AsSpan().Slice(outputBase, embeddingDim);
                    TensorPrimitives.Add(src, dst, dst);
                }
            }
        }

        var resultColumn = anyResultNulls
            ? NivaraColumn<T>.CreateFromSpans(resultValues, resultNullMask!)
            : NivaraColumn<T>.Create(resultValues);

        var result = new ReverseGradTensor<T>(
            resultColumn,
            GradientUtils.ShouldTrackGrad(weight),
            new[] { batchSize, embeddingDim });

        if (GradientUtils.ShouldTrackGrad(weight))
        {
            var savedIndices = parsedIndices;
            var gradFn = new OpNode<T>("SparseEmbeddingBag", new object[] { weight }, (typedGradOutput, stripGradientNulls) =>
            {
                var weightGrad = new T[weight.Length];
                bool gradHasNulls = typedGradOutput.HasNulls;

                if (!gradHasNulls)
                {
                    var gradSpan = typedGradOutput.AsSpan();
                    for (int batch = 0; batch < batchSize; batch++)
                    {
                        int indexBase = batch * maxActiveFeatures;
                        int gradBase = batch * embeddingDim;

                        for (int slot = 0; slot < maxActiveFeatures; slot++)
                        {
                            int index = savedIndices[indexBase + slot];
                            if (index == paddingIndex)
                                continue;

                            int weightBase = index * embeddingDim;
                            var src = gradSpan.Slice(gradBase, embeddingDim);
                            var dst = weightGrad.AsSpan().Slice(weightBase, embeddingDim);
                            TensorPrimitives.Add(src, dst, dst);
                        }
                    }
                }
                else
                {
                    for (int batch = 0; batch < batchSize; batch++)
                    {
                        int indexBase = batch * maxActiveFeatures;
                        int gradBase = batch * embeddingDim;

                        for (int slot = 0; slot < maxActiveFeatures; slot++)
                        {
                            int index = savedIndices[indexBase + slot];
                            if (index == paddingIndex)
                                continue;

                            int weightBase = index * embeddingDim;
                            for (int dim = 0; dim < embeddingDim; dim++)
                            {
                                int gradOffset = gradBase + dim;
                                if (!typedGradOutput.IsNull(gradOffset))
                                    weightGrad[weightBase + dim] += typedGradOutput[gradOffset];
                            }
                        }
                    }
                }

                AccumulateGradient(weight, NivaraColumn<T>.Create(weightGrad), stripGradientNulls);
            });

            ComputationGraph.AddNode(result, gradFn);
        }

        return result;
    }

    /// <summary>
    /// Selects rows from a source tensor by integer index along axis 0.
    /// source shape: [N, ...], indices length: L → result shape: [L, ...].
    /// Backward scatters gradients back to source positions (supports duplicate indices via accumulation).
    /// </summary>
    public static ReverseGradTensor<T> Gather<T>(ReverseGradTensor<T> source, int[] indices, int axis = 0)
        where T : struct, INumber<T>
    {
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (axis != 0) throw new ArgumentOutOfRangeException(nameof(axis), "Only axis 0 is currently supported.");
        if (indices.Length == 0)
            return new ReverseGradTensor<T>(
                NivaraColumn<T>.Create(Array.Empty<T>()),
                requiresGrad: false,
                new[] { 0 });

        int sourceRowCount = source.shape[0];
        int stride = source.Length / sourceRowCount;

        for (int i = 0; i < indices.Length; i++)
        {
            if (indices[i] < 0 || indices[i] >= sourceRowCount)
                throw new ArgumentOutOfRangeException(
                    nameof(indices),
                    $"Index at position {i} is {indices[i]}, must be in range [0, {sourceRowCount}).");
        }

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffGather",
            indices.Length,
            source.Data.HasNulls,
            () =>
            {
                bool sourceHasNulls = source.HasNulls;
                int resultLen = indices.Length * stride;
                bool anyResultNulls = false;
                var resultValues = new T[resultLen];
                bool[]? resultNullMask = sourceHasNulls ? new bool[resultLen] : null;

                var sourceData = new T[source.Length];
                source.Data.CopyTo(sourceData, default(T)!);

                bool[]? sourceNullMask = null;
                if (sourceHasNulls && source.Data.TryGetNullMask(out var srcMask))
                {
                    sourceNullMask = new bool[source.Length];
                    srcMask.CopyTo(sourceNullMask);
                }

                for (int i = 0; i < indices.Length; i++)
                {
                    int srcOffset = indices[i] * stride;
                    int dstOffset = i * stride;
                    Array.Copy(sourceData, srcOffset, resultValues, dstOffset, stride);
                    if (sourceHasNulls)
                    {
                        for (int j = 0; j < stride; j++)
                        {
                            bool isNull = sourceNullMask![srcOffset + j];
                            resultNullMask![dstOffset + j] = isNull;
                            anyResultNulls |= isNull;
                        }
                    }
                }

                var resultCol = anyResultNulls
                    ? NivaraColumn<T>.CreateFromSpans(resultValues, resultNullMask!)
                    : NivaraColumn<T>.Create(resultValues);

                var resultShape = new int[source.shape.Length];
                resultShape[0] = indices.Length;
                for (int d = 1; d < source.shape.Length; d++)
                    resultShape[d] = source.shape[d];

                var result = new ReverseGradTensor<T>(resultCol, GradientUtils.ShouldTrackGrad(source), resultShape);

                if (GradientUtils.ShouldTrackGrad(source))
                {
                    var savedIndices = indices;
                    var savedSourceHasNulls = sourceHasNulls;
                    var gradFn = new OpNode<T>("Gather", new object[] { source }, (typedGradOutput, sgn) =>
                    {
                        var gradBuf = new T[source.Length];
                        var sourceGradNullMask = savedSourceHasNulls ? new bool[source.Length] : null;

                        bool gradHasNulls = typedGradOutput.HasNulls;
                        if (!gradHasNulls && typedGradOutput.TryGetSpan(out var gradSpan))
                        {
                            for (int i = 0; i < savedIndices.Length; i++)
                            {
                                int dstOffset = savedIndices[i] * stride;
                                int srcOffset = i * stride;
                                for (int j = 0; j < stride; j++)
                                    gradBuf[dstOffset + j] += gradSpan[srcOffset + j];
                            }
                        }
                        else
                        {
                            for (int i = 0; i < savedIndices.Length; i++)
                            {
                                int dstOffset = savedIndices[i] * stride;
                                int srcOffset = i * stride;
                                for (int j = 0; j < stride; j++)
                                {
                                    if (!typedGradOutput.IsNull(srcOffset + j))
                                        gradBuf[dstOffset + j] += typedGradOutput[srcOffset + j];
                                }
                            }
                        }

                        if (savedSourceHasNulls && sourceGradNullMask != null)
                        {
                            for (int i = 0; i < savedIndices.Length; i++)
                            {
                                int flatIdx = savedIndices[i] * stride;
                                for (int j = 0; j < stride; j++)
                                {
                                    if (source.Data.IsNull(flatIdx + j))
                                        sourceGradNullMask[flatIdx + j] = true;
                                }
                            }
                            var sourceGrad = NivaraColumn<T>.CreateFromSpans(gradBuf, sourceGradNullMask!);
                            AccumulateGradient(source, sourceGrad, sgn);
                        }
                        else
                        {
                            var sourceGrad = NivaraColumn<T>.Create(gradBuf);
                            AccumulateGradient(source, sourceGrad, sgn);
                        }
                    });

                    ComputationGraph.AddNode(result, gradFn);
                }

                return result;
            },
            $"AutoDiff=Gather;IndicesLength={indices.Length};SourceShape=[{string.Join(", ", source.shape)}]");
    }

    #endregion

    #region VAE Operations

    public static ReverseGradTensor<T> KlDivergence<T>(ReverseGradTensor<T> mean, ReverseGradTensor<T> logVar)
        where T : struct, INumber<T>
    {
        if (mean == null) throw new ArgumentNullException(nameof(mean));
        if (logVar == null) throw new ArgumentNullException(nameof(logVar));

        if (mean.Length != logVar.Length)
            throw new ArgumentException(
                $"mean length ({mean.Length}) must equal logVar length ({logVar.Length})",
                nameof(logVar));

        var klElements = ApplyKlElementWise(mean.Data, logVar.Data);
        var klSum = new NivaraSeries<T>(klElements).Sum();

        var resultData = NivaraColumn<T>.Create(new T[] { klSum });
        var resultTensor = new ReverseGradTensor<T>(resultData, GradientUtils.ShouldTrackGrad(mean, logVar), ScalarShape());

        if (GradientUtils.ShouldTrackGrad(mean, logVar))
        {
            var gradFn = new OpNode<T>("KlDivergence", new object[] { mean, logVar }, (typedGradOutput, sgn) =>
            {
                if (mean.RequiresGrad)
                {
                    var broadcast = BroadcastGradient(typedGradOutput, mean.Length);
                    var dMean = ApplyKlMeanGradient(mean.Data, broadcast);
                    AccumulateGradient(mean, dMean, sgn);
                }
                if (logVar.RequiresGrad)
                {
                    var broadcast = BroadcastGradient(typedGradOutput, logVar.Length);
                    var dLogVar = ApplyKlLogVarGradient(logVar.Data, broadcast);
                    AccumulateGradient(logVar, dLogVar, sgn);
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> SampleNormal<T>(ReverseGradTensor<T> mean, ReverseGradTensor<T> logVar, int? seed = null)
        where T : struct, INumber<T>
    {
        if (mean == null) throw new ArgumentNullException(nameof(mean));
        if (logVar == null) throw new ArgumentNullException(nameof(logVar));

        if (mean.Length != logVar.Length)
            throw new ArgumentException(
                $"mean length ({mean.Length}) must equal logVar length ({logVar.Length})",
                nameof(logVar));

        int n = mean.Length;
        var epsilon = Nivara.Helpers.RandomGeneration.GenerateStandardNormal<T>(n, seed);
        var epsilonCol = NivaraColumn<T>.Create(epsilon.AsSpan());

        var result = ApplySampleNormalForward(mean.Data, logVar.Data, epsilonCol);

        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(mean, logVar), PropagateShape(mean, logVar));

        if (GradientUtils.ShouldTrackGrad(mean, logVar))
        {
            var savedEpsilon = epsilonCol;
            var gradFn = new OpNode<T>("SampleNormal", new object[] { mean, logVar }, (typedGradOutput, sgn) =>
            {
                if (mean.RequiresGrad)
                {
                    AccumulateGradient(mean, typedGradOutput, sgn);
                }
                if (logVar.RequiresGrad)
                {
                    var dLogVar = ApplySampleNormalLogVarGradient(logVar.Data, typedGradOutput, savedEpsilon);
                    AccumulateGradient(logVar, dLogVar, sgn);
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    #endregion

    #region Helper Methods

    internal static void AccumulateGradient<T>(ReverseGradTensor<T> tensor, NivaraColumn<T> gradient, bool stripGradientNulls = true) where T : struct, INumber<T>
    {
        if (tensor.Grad == null)
        {
            tensor.Grad = stripGradientNulls && gradient.HasNulls ? gradient.WithoutNulls() : gradient;
            return;
        }

        if (stripGradientNulls)
        {
            bool existingHasNulls = tensor.Grad.HasNulls;
            bool newHasNulls = gradient.HasNulls;

            if (!existingHasNulls && !newHasNulls)
            {
                int len = tensor.Grad.Length;
                var gradData = new T[len];
                gradient.CopyTo(gradData, default(T)!);
                var result = new T[len];
                tensor.Grad.CopyTo(result.AsSpan(), default(T)!);
                TensorPrimitives.Add(result.AsSpan(), gradData.AsSpan(), result.AsSpan());
                tensor.Grad = NivaraColumn<T>.Create(result);
            }
            else
            {
                var cleanExisting = existingHasNulls ? tensor.Grad.WithoutNulls() : tensor.Grad;
                var cleanG = newHasNulls ? gradient.WithoutNulls() : gradient;
                tensor.Grad = cleanExisting.Add(cleanG);
            }
        }
        else
        {
            tensor.Grad = tensor.Grad.Add(gradient);
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

    private static NivaraColumn<T> ApplyDropout<T>(NivaraColumn<T> input, ReadOnlySpan<bool> keepMask, T scale)
        where T : struct, INumber<T>
    {
        int n = input.Length;
        var resultBuf = ArrayPool<T>.Shared.Rent(n);

        try
        {
            if (!input.HasNulls)
            {
                input.TryGetSpan(out var span);
                for (int i = 0; i < n; i++)
                    resultBuf[i] = keepMask[i] ? span[i] * scale : T.Zero;

                return NivaraColumn<T>.Create(resultBuf.AsSpan(0, n));
            }

            input.CopyTo(resultBuf.AsSpan(0, n), T.Zero);
            input.TryGetNullMask(out var inputMask);
            for (int i = 0; i < n; i++)
            {
                if (!inputMask[i])
                    resultBuf[i] = keepMask[i] ? resultBuf[i] * scale : T.Zero;
            }

            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), inputMask);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyDropoutGradient<T>(
        NivaraColumn<T> input,
        NivaraColumn<T> gradOutput,
        ReadOnlySpan<bool> keepMask,
        T scale)
        where T : struct, INumber<T>
    {
        int n = input.Length;
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);

        try
        {
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            for (int i = 0; i < n; i++)
                resultBuf[i] = keepMask[i] ? gradBuf[i] * scale : T.Zero;

            if (!input.HasNulls && !gradOutput.HasNulls)
                return NivaraColumn<T>.Create(resultBuf.AsSpan(0, n));

            var nullMask = ArrayPool<bool>.Shared.Rent(n);
            try
            {
                NivaraColumnUtility.MergeNullMasks(input, gradOutput, nullMask.AsSpan(0, n));
                return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
            }
            finally
            {
                ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyKlElementWise<T>(NivaraColumn<T> mean, NivaraColumn<T> logVar) where T : struct, INumber<T>
    {
        int n = mean.Length;

        if (!mean.HasNulls && !logVar.HasNulls)
        {
            mean.TryGetSpan(out var mSpan);
            logVar.TryGetSpan(out var lvSpan);
            var result = new T[n];
            if (typeof(T) == typeof(float))
            {
                var m = MemoryMarshal.Cast<T, float>(mSpan);
                var lv = MemoryMarshal.Cast<T, float>(lvSpan);
                var r = MemoryMarshal.Cast<T, float>(result.AsSpan());
                var m2 = new float[n];
                var expLv = new float[n];
                var tmp = new float[n];
                TensorPrimitives.Multiply(m, m, m2);
                TensorPrimitives.Exp(lv, expLv);
                TensorPrimitives.Add(lv, 1.0f, tmp);
                TensorPrimitives.Subtract(tmp, m2, tmp);
                TensorPrimitives.Subtract(tmp, expLv, tmp);
                TensorPrimitives.Multiply(tmp, -0.5f, r);
            }
            else if (typeof(T) == typeof(double))
            {
                var m = MemoryMarshal.Cast<T, double>(mSpan);
                var lv = MemoryMarshal.Cast<T, double>(lvSpan);
                var r = MemoryMarshal.Cast<T, double>(result.AsSpan());
                var m2 = new double[n];
                var expLv = new double[n];
                var tmp = new double[n];
                TensorPrimitives.Multiply(m, m, m2);
                TensorPrimitives.Exp(lv, expLv);
                TensorPrimitives.Add(lv, 1.0, tmp);
                TensorPrimitives.Subtract(tmp, m2, tmp);
                TensorPrimitives.Subtract(tmp, expLv, tmp);
                TensorPrimitives.Multiply(tmp, -0.5, r);
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    var m = double.CreateChecked(mSpan[i]);
                    var lv = double.CreateChecked(lvSpan[i]);
                    result[i] = T.CreateChecked(-0.5 * (1.0 + lv - m * m - Math.Exp(lv)));
                }
            }
            return NivaraColumn<T>.Create(result);
        }

        var meanBuf = ArrayPool<T>.Shared.Rent(n);
        var logVarBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            mean.CopyTo(meanBuf.AsSpan(0, n), T.Zero);
            logVar.CopyTo(logVarBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(mean, logVar, nullMask.AsSpan(0, n));

            for (int i = 0; i < n; i++)
            {
                if (nullMask[i])
                    continue;
                var m = double.CreateChecked(meanBuf[i]);
                var lv = double.CreateChecked(logVarBuf[i]);
                resultBuf[i] = T.CreateChecked(-0.5 * (1.0 + lv - m * m - Math.Exp(lv)));
            }
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(meanBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(logVarBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyKlMeanGradient<T>(NivaraColumn<T> mean, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        // ∂KL/∂μ = μ
        int n = mean.Length;

        if (!mean.HasNulls && !gradOutput.HasNulls)
        {
            mean.TryGetSpan(out var mSpan);
            gradOutput.TryGetSpan(out var gSpan);
            var result = new T[n];
            TensorPrimitives.Multiply(mSpan, gSpan, result);
            return NivaraColumn<T>.Create(result);
        }

        var meanBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            mean.CopyTo(meanBuf.AsSpan(0, n), T.Zero);
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(mean, gradOutput, nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
                resultBuf[i] = meanBuf[i] * gradBuf[i];
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(meanBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyKlLogVarGradient<T>(NivaraColumn<T> logVar, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        // ∂KL/∂logVar = -0.5 * (1 - exp(logVar))
        int n = logVar.Length;

        if (!logVar.HasNulls && !gradOutput.HasNulls)
        {
            logVar.TryGetSpan(out var lvSpan);
            gradOutput.TryGetSpan(out var gSpan);
            var result = new T[n];
            if (typeof(T) == typeof(float))
            {
                var lv = MemoryMarshal.Cast<T, float>(lvSpan);
                var g = MemoryMarshal.Cast<T, float>(gSpan);
                var d = MemoryMarshal.Cast<T, float>(result.AsSpan());
                TensorPrimitives.Exp(lv, d);
                TensorPrimitives.Subtract(1.0f, d, d);
                TensorPrimitives.Multiply(d, g, d);
                TensorPrimitives.Multiply(d, -0.5f, d);
            }
            else if (typeof(T) == typeof(double))
            {
                var lv = MemoryMarshal.Cast<T, double>(lvSpan);
                var g = MemoryMarshal.Cast<T, double>(gSpan);
                var d = MemoryMarshal.Cast<T, double>(result.AsSpan());
                TensorPrimitives.Exp(lv, d);
                TensorPrimitives.Subtract(1.0, d, d);
                TensorPrimitives.Multiply(d, g, d);
                TensorPrimitives.Multiply(d, -0.5, d);
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    var lv = double.CreateChecked(lvSpan[i]);
                    var g = double.CreateChecked(gSpan[i]);
                    result[i] = T.CreateChecked(-0.5 * (1.0 - Math.Exp(lv)) * g);
                }
            }
            return NivaraColumn<T>.Create(result);
        }

        var logVarBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            logVar.CopyTo(logVarBuf.AsSpan(0, n), T.Zero);
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(logVar, gradOutput, nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
            {
                var lv = double.CreateChecked(logVarBuf[i]);
                var g = double.CreateChecked(gradBuf[i]);
                resultBuf[i] = T.CreateChecked(-0.5 * (1.0 - Math.Exp(lv)) * g);
            }
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(logVarBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplySampleNormalForward<T>(NivaraColumn<T> mean, NivaraColumn<T> logVar, NivaraColumn<T> epsilon) where T : struct, INumber<T>
    {
        // z = μ + exp(0.5 * logVar) * ε
        int n = mean.Length;

        if (!mean.HasNulls && !logVar.HasNulls)
        {
            mean.TryGetSpan(out var mSpan);
            logVar.TryGetSpan(out var lvSpan);
            epsilon.TryGetSpan(out var eSpan);
            var result = new T[n];
            if (typeof(T) == typeof(float))
            {
                var m = MemoryMarshal.Cast<T, float>(mSpan);
                var lv = MemoryMarshal.Cast<T, float>(lvSpan);
                var e = MemoryMarshal.Cast<T, float>(eSpan);
                var d = MemoryMarshal.Cast<T, float>(result.AsSpan());
                TensorPrimitives.Multiply(lv, 0.5f, d);
                TensorPrimitives.Exp(d, d);
                TensorPrimitives.Multiply(d, e, d);
                TensorPrimitives.Add(d, m, d);
            }
            else if (typeof(T) == typeof(double))
            {
                var m = MemoryMarshal.Cast<T, double>(mSpan);
                var lv = MemoryMarshal.Cast<T, double>(lvSpan);
                var e = MemoryMarshal.Cast<T, double>(eSpan);
                var d = MemoryMarshal.Cast<T, double>(result.AsSpan());
                TensorPrimitives.Multiply(lv, 0.5, d);
                TensorPrimitives.Exp(d, d);
                TensorPrimitives.Multiply(d, e, d);
                TensorPrimitives.Add(d, m, d);
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    var m = double.CreateChecked(mSpan[i]);
                    var lv = double.CreateChecked(lvSpan[i]);
                    var e = double.CreateChecked(eSpan[i]);
                    result[i] = T.CreateChecked(m + Math.Exp(0.5 * lv) * e);
                }
            }
            return NivaraColumn<T>.Create(result);
        }

        var meanBuf = ArrayPool<T>.Shared.Rent(n);
        var logVarBuf = ArrayPool<T>.Shared.Rent(n);
        var epsBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            mean.CopyTo(meanBuf.AsSpan(0, n), T.Zero);
            logVar.CopyTo(logVarBuf.AsSpan(0, n), T.Zero);
            epsilon.CopyTo(epsBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(mean, logVar, nullMask.AsSpan(0, n));

            for (int i = 0; i < n; i++)
            {
                if (nullMask[i])
                    continue;
                var m = double.CreateChecked(meanBuf[i]);
                var lv = double.CreateChecked(logVarBuf[i]);
                var e = double.CreateChecked(epsBuf[i]);
                resultBuf[i] = T.CreateChecked(m + Math.Exp(0.5 * lv) * e);
            }
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(meanBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(logVarBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(epsBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplySampleNormalLogVarGradient<T>(NivaraColumn<T> logVar, NivaraColumn<T> gradOutput, NivaraColumn<T> epsilon) where T : struct, INumber<T>
    {
        // ∂z/∂logVar = 0.5 * exp(0.5 * logVar) * ε
        int n = logVar.Length;

        if (!logVar.HasNulls && !gradOutput.HasNulls)
        {
            logVar.TryGetSpan(out var lvSpan);
            gradOutput.TryGetSpan(out var gSpan);
            epsilon.TryGetSpan(out var eSpan);
            var result = new T[n];
            if (typeof(T) == typeof(float))
            {
                var lv = MemoryMarshal.Cast<T, float>(lvSpan);
                var g = MemoryMarshal.Cast<T, float>(gSpan);
                var e = MemoryMarshal.Cast<T, float>(eSpan);
                var d = MemoryMarshal.Cast<T, float>(result.AsSpan());
                TensorPrimitives.Multiply(lv, 0.5f, d);
                TensorPrimitives.Exp(d, d);
                TensorPrimitives.Multiply(d, e, d);
                TensorPrimitives.Multiply(d, g, d);
                TensorPrimitives.Multiply(d, 0.5f, d);
            }
            else if (typeof(T) == typeof(double))
            {
                var lv = MemoryMarshal.Cast<T, double>(lvSpan);
                var g = MemoryMarshal.Cast<T, double>(gSpan);
                var e = MemoryMarshal.Cast<T, double>(eSpan);
                var d = MemoryMarshal.Cast<T, double>(result.AsSpan());
                TensorPrimitives.Multiply(lv, 0.5, d);
                TensorPrimitives.Exp(d, d);
                TensorPrimitives.Multiply(d, e, d);
                TensorPrimitives.Multiply(d, g, d);
                TensorPrimitives.Multiply(d, 0.5, d);
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    var lv = double.CreateChecked(lvSpan[i]);
                    var g = double.CreateChecked(gSpan[i]);
                    var e = double.CreateChecked(eSpan[i]);
                    result[i] = T.CreateChecked(0.5 * Math.Exp(0.5 * lv) * e * g);
                }
            }
            return NivaraColumn<T>.Create(result);
        }

        var logVarBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var epsBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            logVar.CopyTo(logVarBuf.AsSpan(0, n), T.Zero);
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            epsilon.CopyTo(epsBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(logVar, gradOutput, nullMask.AsSpan(0, n));

            for (int i = 0; i < n; i++)
            {
                if (nullMask[i])
                    continue;
                var lv = double.CreateChecked(logVarBuf[i]);
                var g = double.CreateChecked(gradBuf[i]);
                var e = double.CreateChecked(epsBuf[i]);
                resultBuf[i] = T.CreateChecked(0.5 * Math.Exp(0.5 * lv) * e * g);
            }
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(logVarBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(epsBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyPow<T>(NivaraColumn<T> input, double exponent) where T : struct, INumber<T>
    {
        int n = input.Length;

        if (!input.HasNulls)
        {
            input.TryGetSpan(out var span);
            var result = new T[n];
            if (typeof(T) == typeof(float))
            {
                var s = MemoryMarshal.Cast<T, float>(span);
                var r = MemoryMarshal.Cast<T, float>(result.AsSpan());
                TensorPrimitives.Pow(s, (float)exponent, r);
            }
            else if (typeof(T) == typeof(double))
            {
                var s = MemoryMarshal.Cast<T, double>(span);
                var r = MemoryMarshal.Cast<T, double>(result.AsSpan());
                TensorPrimitives.Pow(s, exponent, r);
            }
            else
            {
                for (int i = 0; i < n; i++)
                    result[i] = T.CreateChecked(Math.Pow(double.CreateChecked(span[i]), exponent));
            }
            return NivaraColumn<T>.Create(result);
        }

        var buf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            input.CopyTo(buf.AsSpan(0, n), T.Zero);
            input.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
            {
                if (nullMask[i]) continue;
                buf[i] = T.CreateChecked(Math.Pow(double.CreateChecked(buf[i]), exponent));
            }
            return NivaraColumn<T>.CreateFromSpans(buf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyPowGradient<T>(NivaraColumn<T> input, NivaraColumn<T> gradOutput, double exponent) where T : struct, INumber<T>
    {
        int n = input.Length;

        if (!input.HasNulls && !gradOutput.HasNulls)
        {
            input.TryGetSpan(out var inSpan);
            gradOutput.TryGetSpan(out var gSpan);
            var result = new T[n];
            if (typeof(T) == typeof(float))
            {
                var x = MemoryMarshal.Cast<T, float>(inSpan);
                var g = MemoryMarshal.Cast<T, float>(gSpan);
                var r = MemoryMarshal.Cast<T, float>(result.AsSpan());
                TensorPrimitives.Pow(x, (float)(exponent - 1.0), r);
                TensorPrimitives.Multiply(r, (float)exponent, r);
                TensorPrimitives.Multiply(r, g, r);
            }
            else if (typeof(T) == typeof(double))
            {
                var x = MemoryMarshal.Cast<T, double>(inSpan);
                var g = MemoryMarshal.Cast<T, double>(gSpan);
                var r = MemoryMarshal.Cast<T, double>(result.AsSpan());
                TensorPrimitives.Pow(x, exponent - 1.0, r);
                TensorPrimitives.Multiply(r, exponent, r);
                TensorPrimitives.Multiply(r, g, r);
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    var x = double.CreateChecked(inSpan[i]);
                    var g = double.CreateChecked(gSpan[i]);
                    result[i] = T.CreateChecked(exponent * Math.Pow(x, exponent - 1.0) * g);
                }
            }
            return NivaraColumn<T>.Create(result);
        }

        var buf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            input.CopyTo(buf.AsSpan(0, n), T.Zero);
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(input, gradOutput, nullMask.AsSpan(0, n));
            for (int i = 0; i < n; i++)
            {
                if (nullMask[i]) continue;
                resultBuf[i] = T.CreateChecked(
                    exponent * Math.Pow(double.CreateChecked(buf[i]), exponent - 1.0) * double.CreateChecked(gradBuf[i]));
            }
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyRMSNorm<T>(NivaraColumn<T> input, double eps) where T : struct, INumber<T>
    {
        int n = input.Length;

        if (!input.HasNulls)
        {
            input.TryGetSpan(out var span);
            var result = new T[n];
            if (typeof(T) == typeof(float))
            {
                var s = MemoryMarshal.Cast<T, float>(span);
                var r = MemoryMarshal.Cast<T, float>(result.AsSpan());
                double sumSq = 0;
                for (int i = 0; i < n; i++) sumSq += s[i] * s[i];
                double rms = Math.Sqrt(sumSq / n + eps);
                float invRms = (float)(1.0 / rms);
                TensorPrimitives.Multiply(s, invRms, r);
            }
            else if (typeof(T) == typeof(double))
            {
                var s = MemoryMarshal.Cast<T, double>(span);
                var r = MemoryMarshal.Cast<T, double>(result.AsSpan());
                double sumSq = 0;
                for (int i = 0; i < n; i++) sumSq += s[i] * s[i];
                double rms = Math.Sqrt(sumSq / n + eps);
                double invRms = 1.0 / rms;
                TensorPrimitives.Multiply(s, invRms, r);
            }
            else
            {
                double sumSq = 0;
                for (int i = 0; i < n; i++)
                {
                    var x = double.CreateChecked(span[i]);
                    sumSq += x * x;
                }
                double rms = Math.Sqrt(sumSq / n + eps);
                double invRms = 1.0 / rms;
                for (int i = 0; i < n; i++)
                    result[i] = T.CreateChecked(double.CreateChecked(span[i]) * invRms);
            }
            return NivaraColumn<T>.Create(result);
        }

        var buf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            input.CopyTo(buf.AsSpan(0, n), T.Zero);
            input.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));

            double sumSq = 0;
            int validCount = 0;
            for (int i = 0; i < n; i++)
            {
                if (!nullMask[i])
                {
                    var x = double.CreateChecked(buf[i]);
                    sumSq += x * x;
                    validCount++;
                }
            }

            double rms = Math.Sqrt(sumSq / validCount + eps);
            double invRms = 1.0 / rms;

            for (int i = 0; i < n; i++)
            {
                if (!nullMask[i])
                    buf[i] = T.CreateChecked(double.CreateChecked(buf[i]) * invRms);
            }

            return NivaraColumn<T>.CreateFromSpans(buf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyRMSNormGradient<T>(
        NivaraColumn<T> input, NivaraColumn<T> gradOutput, double eps) where T : struct, INumber<T>
    {
        int n = input.Length;

        if (!input.HasNulls && !gradOutput.HasNulls)
        {
            input.TryGetSpan(out var inSpan);
            gradOutput.TryGetSpan(out var gSpan);
            var result = new T[n];

            double sumSq = 0;
            for (int i = 0; i < n; i++)
            {
                var x = double.CreateChecked(inSpan[i]);
                sumSq += x * x;
            }

            double rms = Math.Sqrt(sumSq / n + eps);
            double rms3 = rms * rms * rms;

            double sumGradX = 0;
            for (int i = 0; i < n; i++)
            {
                var x = double.CreateChecked(inSpan[i]);
                var g = double.CreateChecked(gSpan[i]);
                sumGradX += g * x;
            }

            double scale = sumGradX / (n * rms3);

            if (typeof(T) == typeof(float))
            {
                var x = MemoryMarshal.Cast<T, float>(inSpan);
                var g = MemoryMarshal.Cast<T, float>(gSpan);
                var r = MemoryMarshal.Cast<T, float>(result.AsSpan());
                float invRms = (float)(1.0 / rms);
                float s = (float)scale;
                for (int i = 0; i < n; i++)
                    r[i] = g[i] * invRms - x[i] * s;
            }
            else if (typeof(T) == typeof(double))
            {
                var x = MemoryMarshal.Cast<T, double>(inSpan);
                var g = MemoryMarshal.Cast<T, double>(gSpan);
                var r = MemoryMarshal.Cast<T, double>(result.AsSpan());
                double invRms = 1.0 / rms;
                for (int i = 0; i < n; i++)
                    r[i] = g[i] * invRms - x[i] * scale;
            }
            else
            {
                double invRms = 1.0 / rms;
                for (int i = 0; i < n; i++)
                {
                    var x = double.CreateChecked(inSpan[i]);
                    var g = double.CreateChecked(gSpan[i]);
                    result[i] = T.CreateChecked(g * invRms - x * scale);
                }
            }
            return NivaraColumn<T>.Create(result);
        }

        var buf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            input.CopyTo(buf.AsSpan(0, n), T.Zero);
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(input, gradOutput, nullMask.AsSpan(0, n));

            double sumSq = 0;
            int validCount = 0;
            for (int i = 0; i < n; i++)
            {
                if (!nullMask[i])
                {
                    var x = double.CreateChecked(buf[i]);
                    sumSq += x * x;
                    validCount++;
                }
            }

            double rms = Math.Sqrt(sumSq / validCount + eps);
            double rms3 = rms * rms * rms;

            double sumGradX = 0;
            for (int i = 0; i < n; i++)
            {
                if (!nullMask[i])
                {
                    var x = double.CreateChecked(buf[i]);
                    var g = double.CreateChecked(gradBuf[i]);
                    sumGradX += g * x;
                }
            }

            double scale = sumGradX / (validCount * rms3);
            double invRms = 1.0 / rms;

            for (int i = 0; i < n; i++)
            {
                if (!nullMask[i])
                {
                    var x = double.CreateChecked(buf[i]);
                    var g = double.CreateChecked(gradBuf[i]);
                    resultBuf[i] = T.CreateChecked(g * invRms - x * scale);
                }
            }

            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
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

