using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using Nivara.Helpers;
using Nivara.Tensors;
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Operations;

public static class ReverseGradOperations
{
    #region Element-wise Operations

    public static ReverseGradTensor<T> Add<T>(ReverseGradTensor<T> a, ReverseGradTensor<T> b) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
            throw new ArgumentException($"Cannot add tensors with different lengths: {a.Length} vs {b.Length}");

        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();
        var resultArr = new T[a.Length];
        TensorPrimitives.Add(aSpan, bSpan, resultArr);

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a, b));

        if (GradientUtils.ShouldTrackGrad(a, b))
        {
            var gradFn = new OpNode<T>("Add", new object[] { a, b }, (typedGradOutput) =>
            {
                if (GradientUtils.ShouldTrackGrad(a))
                    AccumulateGradient(a, typedGradOutput);
                if (b.RequiresGrad)
                    AccumulateGradient(b, typedGradOutput);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Subtract<T>(ReverseGradTensor<T> a, ReverseGradTensor<T> b) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
            throw new ArgumentException($"Cannot subtract tensors with different lengths: {a.Length} vs {b.Length}");

        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();
        var resultArr = new T[a.Length];
        TensorPrimitives.Subtract(aSpan, bSpan, resultArr);

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a, b));

        if (GradientUtils.ShouldTrackGrad(a, b))
        {
            var gradFn = new OpNode<T>("Subtract", new object[] { a, b }, (typedGradOutput) =>
            {
                if (GradientUtils.ShouldTrackGrad(a))
                    AccumulateGradient(a, typedGradOutput);
                if (b.RequiresGrad)
                {
                    typedGradOutput.TryGetSpan(out var gSpan);
                    TensorPrimitives.Negate(gSpan, resultArr.AsSpan());
                    AccumulateGradient(b, NivaraColumn<T>.Create(resultArr));
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Multiply<T>(ReverseGradTensor<T> a, ReverseGradTensor<T> b) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
            throw new ArgumentException($"Cannot multiply tensors with different lengths: {a.Length} vs {b.Length}");

        var resultArr = new T[a.Length];
        TensorPrimitives.Multiply(a.AsSpan(), b.AsSpan(), resultArr.AsSpan());

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a, b));

        if (GradientUtils.ShouldTrackGrad(a, b))
        {
            var gradFn = new OpNode<T>("Multiply", new object[] { a, b }, (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                if (GradientUtils.ShouldTrackGrad(a))
                {
                    var aGradArr = new T[a.Length];
                    TensorPrimitives.Multiply(gSpan, b.AsSpan(), aGradArr.AsSpan());
                    AccumulateGradient(a, NivaraColumn<T>.Create(aGradArr));
                }
                if (b.RequiresGrad)
                {
                    var bGradArr = new T[a.Length];
                    TensorPrimitives.Multiply(gSpan, a.AsSpan(), bGradArr.AsSpan());
                    AccumulateGradient(b, NivaraColumn<T>.Create(bGradArr));
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Divide<T>(ReverseGradTensor<T> a, ReverseGradTensor<T> b) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
            throw new ArgumentException($"Cannot divide tensors with different lengths: {a.Length} vs {b.Length}");

        var bSpan = b.AsSpan();
        for (int i = 0; i < b.Length; i++)
        {
            if (bSpan[i] == T.Zero)
                throw new DivideByZeroException($"Division by zero at index {i}");
        }

        var resultArr = new T[a.Length];
        TensorPrimitives.Divide(a.AsSpan(), bSpan, resultArr.AsSpan());

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a, b));

        if (GradientUtils.ShouldTrackGrad(a, b))
        {
            var gradFn = new OpNode<T>("Divide", new object[] { a, b }, (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                if (GradientUtils.ShouldTrackGrad(a))
                {
                    var aGradArr = new T[a.Length];
                    TensorPrimitives.Divide(gSpan, b.AsSpan(), aGradArr.AsSpan());
                    AccumulateGradient(a, NivaraColumn<T>.Create(aGradArr));
                }
                if (b.RequiresGrad)
                {
                    var bSpanInner = b.AsSpan();
                    var aSpanInner = a.AsSpan();
                    var quotientArr = new T[a.Length];
                    TensorPrimitives.Divide(aSpanInner, bSpanInner, quotientArr.AsSpan());
                    var bGradPosArr = new T[a.Length];
                    TensorPrimitives.Divide(quotientArr.AsSpan(), bSpanInner, bGradPosArr.AsSpan());
                    var bGradArr = new T[a.Length];
                    TensorPrimitives.Negate(bGradPosArr.AsSpan(), bGradArr.AsSpan());
                    var finalBArr = new T[a.Length];
                    TensorPrimitives.Multiply(bGradArr.AsSpan(), gSpan, finalBArr.AsSpan());
                    AccumulateGradient(b, NivaraColumn<T>.Create(finalBArr));
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    #endregion

    #region Matrix Operations

    public static ReverseGradTensor<T> MatMul<T>(ReverseGradTensor<T> a, ReverseGradTensor<T> b)
        where T : struct, IFloatingPointIeee754<T>
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

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffMatMul",
            a.Length + b.Length,
            
            () =>
            {
                a.Data.TryGetSpan(out var aSpan);
                b.Data.TryGetSpan(out var bSpan);
                var resultArr = new T[aRows * bCols];
                TensorsHelper.MultiplyCore(aSpan, bSpan, resultArr, aRows, aCols, bCols);
                var result = NivaraColumn<T>.Create(resultArr);

                var resultShape = new[] { aRows, bCols };
                var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a, b), resultShape);

                if (GradientUtils.ShouldTrackGrad(a, b))
                {
                    var gradFn = new OpNode<T>("MatMul", new object[] { a, b }, (typedGradOutput) =>
                    {
                        if (GradientUtils.ShouldTrackGrad(a))
                        {
                            b.Data.TryGetSpan(out var bSpan_b);
                            var bTArr = new T[bRows * bCols];
                            TensorsHelper.Transpose(bSpan_b, bTArr.AsSpan(), bRows, bCols);
                            typedGradOutput.TryGetSpan(out var gradSpan);
                            var aGradArr = new T[aRows * aCols];
                            TensorsHelper.MultiplyCore(gradSpan, bTArr.AsSpan(), aGradArr, aRows, bCols, aCols);
                            AccumulateGradient(a, NivaraColumn<T>.Create(aGradArr));
                        }
                        if (b.RequiresGrad)
                        {
                            a.Data.TryGetSpan(out var aSpan_b);
                            var aTArr = new T[aRows * aCols];
                            TensorsHelper.Transpose(aSpan_b, aTArr.AsSpan(), aRows, aCols);
                            typedGradOutput.TryGetSpan(out var gradSpan2);
                            var bGradArr = new T[aCols * bCols];
                            TensorsHelper.MultiplyCore(aTArr.AsSpan(), gradSpan2, bGradArr, aCols, aRows, bCols);
                            AccumulateGradient(b, NivaraColumn<T>.Create(bGradArr));
                        }
                    });

                    ComputationGraph.AddNode(resultTensor, gradFn);
                }

                return resultTensor;
            },
            AutoDiffDiagnostics.MatrixNote("MatMul", aRows, aCols, bCols));
    }

    public static ReverseGradTensor<T> Transpose<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (a.Rank != 2)
            throw new ArgumentException($"Transpose requires a matrix (rank 2), got rank {a.Rank}", nameof(a));

        var rows = a.shape[0];
        var cols = a.shape[1];

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffTranspose",
            a.Length,
            
            () =>
            {
                a.Data.TryGetSpan(out var aSpan);
                var resultArr = new T[rows * cols];
                TensorsHelper.Transpose(aSpan, resultArr.AsSpan(), rows, cols);

                var resultShape = new[] { cols, rows };
                var resultTensor = new ReverseGradTensor<T>(NivaraColumn<T>.Create(resultArr), GradientUtils.ShouldTrackGrad(a), resultShape);

                if (GradientUtils.ShouldTrackGrad(a))
                {
                    var gradFn = new OpNode<T>("Transpose", new object[] { a }, (typedGradOutput) =>
                    {
                        var gArr = new T[cols * rows];
                        typedGradOutput.TryGetSpan(out var gradSpan);
                        TensorsHelper.Transpose(gradSpan, gArr.AsSpan(), cols, rows);
                        AccumulateGradient(a, NivaraColumn<T>.Create(gArr));
                    });

                    ComputationGraph.AddNode(resultTensor, gradFn);
                }

                return resultTensor;
            },
            $"AutoDiff=Transpose;Shape={rows}x{cols}->{cols}x{rows}");
    }

    public static ReverseGradTensor<T> TransposeAxes<T>(ReverseGradTensor<T> a, int axis1, int axis2) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (a.Rank < 2 || a.Rank > 3)
            throw new ArgumentException($"TransposeAxes supports rank 2–3, got rank {a.Rank}", nameof(a));
        if (axis1 < 0 || axis1 >= a.Rank) throw new ArgumentOutOfRangeException(nameof(axis1));
        if (axis2 < 0 || axis2 >= a.Rank) throw new ArgumentOutOfRangeException(nameof(axis2));
        if (axis1 == axis2) throw new ArgumentException("axis1 and axis2 must differ");

        var srcDims = a.shape;
        var dstDims = (int[])srcDims.Clone();
        (dstDims[axis1], dstDims[axis2]) = (dstDims[axis2], dstDims[axis1]);
        string note = $"AutoDiff=TransposeAxes;Shape={string.Join("x", srcDims)}->{string.Join("x", dstDims)}";

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffTransposeAxes",
            a.Length,
            
            () =>
            {
                var srcData = new T[a.Length];
                a.Data.TryGetSpan(out var srcSpan);
                srcSpan.CopyTo(srcData);
                var dstData = new T[a.Length];

                if (a.Rank == 2)
                {
                    int rows = srcDims[0], cols = srcDims[1];
                    for (int r = 0; r < rows; r++)
                        for (int c = 0; c < cols; c++)
                        {
                            int srcIdx = r * cols + c;
                            int dstIdx = c * rows + r;
                            dstData[dstIdx] = srcData[srcIdx];
                        }
                }
                else
                {
                    int d0 = srcDims[0], d1 = srcDims[1], d2 = srcDims[2];
                    int nd1 = dstDims[1], nd2 = dstDims[2];
                    for (int i0 = 0; i0 < d0; i0++)
                        for (int i1 = 0; i1 < d1; i1++)
                            for (int i2 = 0; i2 < d2; i2++)
                            {
                                int srcIdx = i0 * d1 * d2 + i1 * d2 + i2;
                                var indices = new[] { i0, i1, i2 };
                                (indices[axis1], indices[axis2]) = (indices[axis2], indices[axis1]);
                                int dstIdx = indices[0] * nd1 * nd2 + indices[1] * nd2 + indices[2];
                                dstData[dstIdx] = srcData[srcIdx];
                            }
                }

                var resultCol = NivaraColumn<T>.Create(dstData);
                bool shouldTrack = GradientUtils.ShouldTrackGrad(a);
                var resultTensor = new ReverseGradTensor<T>(resultCol, shouldTrack, dstDims);

                if (shouldTrack)
                {
                    int capturedAxis1 = axis1, capturedAxis2 = axis2;
                    int[] capturedDstDims = dstDims;
                    var gradFn = new OpNode<T>("TransposeAxes", new object[] { a }, (typedGradOutput) =>
                    {
                        int gradLen = typedGradOutput.Length;
                        var gSrc = new T[gradLen];
                        typedGradOutput.TryGetSpan(out var gSrcSpan);
                        gSrcSpan.CopyTo(gSrc);
                        var gDst = new T[gradLen];

                        if (capturedDstDims.Length == 2)
                        {
                            int rows = capturedDstDims[0], cols = capturedDstDims[1];
                            for (int r = 0; r < rows; r++)
                                for (int c = 0; c < cols; c++)
                                {
                                    int srcIdx = r * cols + c;
                                    int dstIdx = c * rows + r;
                                    gDst[dstIdx] = gSrc[srcIdx];
                                }
                        }
                        else
                        {
                            int d0 = capturedDstDims[0], d1 = capturedDstDims[1], d2 = capturedDstDims[2];
                            int od1 = srcDims[1], od2 = srcDims[2];
                            for (int i0 = 0; i0 < d0; i0++)
                                for (int i1 = 0; i1 < d1; i1++)
                                    for (int i2 = 0; i2 < d2; i2++)
                                    {
                                        int srcIdx = i0 * d1 * d2 + i1 * d2 + i2;
                                        var indices = new[] { i0, i1, i2 };
                                        (indices[capturedAxis1], indices[capturedAxis2]) = (indices[capturedAxis2], indices[capturedAxis1]);
                                        int dstIdx = indices[0] * od1 * od2 + indices[1] * od2 + indices[2];
                                        gDst[dstIdx] = gSrc[srcIdx];
                                    }
                        }

                        AccumulateGradient(a, NivaraColumn<T>.Create(gDst));
                    });

                    ComputationGraph.AddNode(resultTensor, gradFn);
                }

                return resultTensor;
            }, note);
    }

    #endregion

    #region Reduction Operations

    public static ReverseGradTensor<T> Sum<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
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
            var gradFn = new OpNode<T>("Sum", new object[] { a }, (typedGradOutput) =>
            {
                var aGrad = BroadcastGradient(typedGradOutput, a.Length);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Mean<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
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
            var gradFn = new OpNode<T>("Mean", new object[] { a }, (typedGradOutput) =>
            {
                var aGrad = BroadcastGradient(typedGradOutput, a.Length);
                var scaledGrad = aGrad.Divide(T.CreateChecked(a.Length));
                AccumulateGradient(a, scaledGrad);
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
        where T : struct, IFloatingPointIeee754<T>
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
            var gradFn = new OpNode<T>("MeanPool", new object[] { a }, (typedGradOutput) =>
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
                AccumulateGradient(a, gradCol);
            });

            ComputationGraph.AddNode(result, gradFn);
        }

        return result;
    }

    #endregion

    #region Activation Functions

    public static ReverseGradTensor<T> Relu<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var resultArr = new T[a.Length];
        var aSpan = a.AsSpan();
        for (int i = 0; i < a.Length; i++)
            resultArr[i] = T.Max(aSpan[i], T.Zero);

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Relu", new object[] { a }, (typedGradOutput) =>
            {
                var aSpan = a.AsSpan();
                var gradArr = new T[a.Length];
                for (int i = 0; i < a.Length; i++)
                    gradArr[i] = aSpan[i] > T.Zero ? typedGradOutput[i] : T.Zero;
                AccumulateGradient(a, NivaraColumn<T>.Create(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Gelu<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var aArr = new T[a.Length];
        a.AsSpan().CopyTo(aArr.AsSpan());

        var resultArr = new T[a.Length];
        for (int i = 0; i < a.Length; i++)
        {
            double x = double.CreateChecked(aArr[i]);
            resultArr[i] = T.CreateChecked(0.5 * x * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x))));
        }

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Gelu", new object[] { a }, (typedGradOutput) =>
            {
                var gradArr = new T[a.Length];
                for (int i = 0; i < a.Length; i++)
                {
                    double x = double.CreateChecked(aArr[i]);
                    double cdf = 0.5 * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
                    double pdf = Math.Exp(-0.5 * x * x) / Math.Sqrt(2.0 * Math.PI);
                    double grad = cdf + x * pdf;
                    gradArr[i] = T.CreateChecked(grad * double.CreateChecked(typedGradOutput[i]));
                }
                AccumulateGradient(a, NivaraColumn<T>.Create(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Sigmoid<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var resultArr = new T[a.Length];
        TensorPrimitives.Sigmoid(a.AsSpan(), resultArr.AsSpan());

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Sigmoid", new object[] { a }, (typedGradOutput) =>
            {
                var gradArr = new T[a.Length];
                for (int i = 0; i < a.Length; i++)
                {
                    var s = resultArr[i];
                    gradArr[i] = s * (T.One - s) * typedGradOutput[i];
                }
                AccumulateGradient(a, NivaraColumn<T>.Create(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Tanh<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var resultArr = new T[a.Length];
        TensorPrimitives.Tanh(a.AsSpan(), resultArr.AsSpan());

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Tanh", new object[] { a }, (typedGradOutput) =>
            {
                var gradArr = new T[a.Length];
                for (int i = 0; i < a.Length; i++)
                {
                    var t = resultArr[i];
                    gradArr[i] = (T.One - t * t) * typedGradOutput[i];
                }
                AccumulateGradient(a, NivaraColumn<T>.Create(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Negate<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var resultArr = new T[a.Length];
        TensorPrimitives.Negate(a.AsSpan(), resultArr.AsSpan());

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Negate", new object[] { a }, (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                var gradArr = new T[a.Length];
                TensorPrimitives.Negate(gSpan, gradArr.AsSpan());
                AccumulateGradient(a, NivaraColumn<T>.Create(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Abs<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var resultArr = new T[a.Length];
        TensorPrimitives.Abs(a.AsSpan(), resultArr.AsSpan());

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Abs", new object[] { a }, (typedGradOutput) =>
            {
                var aSpan = a.AsSpan();
                var gradArr = new T[a.Length];
                for (int i = 0; i < a.Length; i++)
                    gradArr[i] = T.CreateChecked(T.Sign(aSpan[i])) * typedGradOutput[i];
                AccumulateGradient(a, NivaraColumn<T>.Create(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Clip<T>(ReverseGradTensor<T> a, T min, T max)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var resultArr = new T[a.Length];
        TensorPrimitives.Clamp(a.AsSpan(), min, max, resultArr.AsSpan());

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Clip", new object[] { a, min, max }, (typedGradOutput) =>
            {
                var aSpan = a.AsSpan();
                var gradArr = new T[a.Length];
                for (int i = 0; i < a.Length; i++)
                    gradArr[i] = aSpan[i] > min && aSpan[i] < max ? typedGradOutput[i] : T.Zero;
                AccumulateGradient(a, NivaraColumn<T>.Create(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> LeakyRelu<T>(ReverseGradTensor<T> a, T negativeSlope = default)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (negativeSlope == T.Zero)
            negativeSlope = T.CreateChecked(0.01);

        var resultArr = new T[a.Length];
        var aSpan = a.AsSpan();
        for (int i = 0; i < a.Length; i++)
            resultArr[i] = aSpan[i] >= T.Zero ? aSpan[i] : aSpan[i] * negativeSlope;

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("LeakyRelu", new object[] { a, negativeSlope }, (typedGradOutput) =>
            {
                var aSpanInner = a.AsSpan();
                var gradArr = new T[a.Length];
                for (int i = 0; i < a.Length; i++)
                    gradArr[i] = aSpanInner[i] >= T.Zero ? typedGradOutput[i] : typedGradOutput[i] * negativeSlope;
                AccumulateGradient(a, NivaraColumn<T>.Create(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Exp<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var resultArr = new T[a.Length];
        TensorPrimitives.Exp(a.AsSpan(), resultArr.AsSpan());

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Exp", new object[] { a }, (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                var gradArr = new T[a.Length];
                TensorPrimitives.Multiply(gSpan, resultArr.AsSpan(), gradArr.AsSpan());
                AccumulateGradient(a, NivaraColumn<T>.Create(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Log<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var resultArr = new T[a.Length];
        TensorPrimitives.Log(a.AsSpan(), resultArr.AsSpan());

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Log", new object[] { a }, (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                var aSpan = a.AsSpan();
                var gradArr = new T[a.Length];
                for (int i = 0; i < a.Length; i++)
                    gradArr[i] = gSpan[i] / aSpan[i];
                AccumulateGradient(a, NivaraColumn<T>.Create(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Pow<T>(ReverseGradTensor<T> a, double exponent) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var aArr = new T[a.Length];
        a.AsSpan().CopyTo(aArr.AsSpan());
        var resultArr = new T[a.Length];
        for (int i = 0; i < a.Length; i++)
            resultArr[i] = T.CreateChecked(Math.Pow(double.CreateChecked(aArr[i]), exponent));

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Pow", new object[] { a, exponent }, (typedGradOutput) =>
            {
                var gradArr = new T[a.Length];
                for (int i = 0; i < a.Length; i++)
                {
                    double x = double.CreateChecked(aArr[i]);
                    double g = double.CreateChecked(typedGradOutput[i]);
                    gradArr[i] = T.CreateChecked(exponent * Math.Pow(x, exponent - 1.0) * g);
                }
                AccumulateGradient(a, NivaraColumn<T>.Create(gradArr));
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
        where T : struct, IFloatingPointIeee754<T>
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
            
            () =>
            {
                int resultLen = batchDim * length;
                var resultValues = new T[resultLen];

                var srcData = new T[a.Length];
                a.Data.CopyTo(srcData, default(T)!);

                for (int r = 0; r < batchDim; r++)
                {
                    int srcOffset = r * fullDim + start;
                    int dstOffset = r * length;
                    Array.Copy(srcData, srcOffset, resultValues, dstOffset, length);
                }

                var resultCol = NivaraColumn<T>.Create(resultValues);

                var resultShape = batchDim == 1
                    ? new[] { length }
                    : new[] { batchDim, length };

                var result = new ReverseGradTensor<T>(resultCol, GradientUtils.ShouldTrackGrad(a), resultShape);

                if (GradientUtils.ShouldTrackGrad(a))
                {
                    var savedStart = start;
                    var savedLength = length;
                    var savedFullDim = fullDim;
                    var savedBatchDim = batchDim;
                    var gradFn = new OpNode<T>("Slice", [a], (typedGradOutput) =>
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
                        AccumulateGradient(a, gradCol);
                    });

                    ComputationGraph.AddNode(result, gradFn);
                }

                return result;
            },
            $"AutoDiff=Slice;Start={start};Length={length};FullDim={fullDim}");
    }

    public static ReverseGradTensor<T> Concat<T>(ReverseGradTensor<T>[] tensors, int axis = 0)
        where T : struct, IFloatingPointIeee754<T>
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

        bool shouldTrack = GradientUtils.ShouldTrackGrad(tensors);

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffConcat",
            tensors.Sum(t => t.Length),
            
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
                        var gradFn = new OpNode<T>("Concat", tensors, (typedGradOutput) =>
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
                                    AccumulateGradient(tensors[i], gradCol);
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
                        var gradFn = new OpNode<T>("Concat", tensors, (typedGradOutput) =>
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
                                            AccumulateGradient(savedTensors[i], gradCol);
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
                                            AccumulateGradient(savedTensors[i], gradCol);
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
                                            AccumulateGradient(savedTensors[i], gradCol);
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
                                            AccumulateGradient(savedTensors[i], gradCol);
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

    public static ReverseGradTensor<T> Softmax<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = a.Data.Softmax(a.Rank >= 2 ? a.shape[1] : a.Length);
        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a), a.shape);

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var savedResult = result;
            var gradFn = new OpNode<T>("Softmax", new object[] { a }, (typedGradOutput) =>
            {
                var aGrad = savedResult.SoftmaxGradient(typedGradOutput, a.Rank >= 2 ? a.shape[1] : a.Length);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> LogSoftmax<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = a.Data.LogSoftmax(a.Rank >= 2 ? a.shape[1] : a.Length);
        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a), a.shape);

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("LogSoftmax", new object[] { a }, (typedGradOutput) =>
            {
                var aGrad = a.Data.LogSoftmaxGradient(typedGradOutput, a.Rank >= 2 ? a.shape[1] : a.Length);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> RMSNorm<T>(ReverseGradTensor<T> a, double eps = 1e-5)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffRMSNorm",
            a.Length,
            
            () =>
            {
                var result = ApplyRMSNorm(a.Data, eps);
                var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a), a.shape);

                if (GradientUtils.ShouldTrackGrad(a))
                {
                    var savedInput = a.Data;
                    var savedEps = eps;
                    var gradFn = new OpNode<T>("RMSNorm", new object[] { a, eps }, (typedGradOutput) =>
                    {
                        var aGrad = ApplyRMSNormGradient(savedInput, typedGradOutput, savedEps);
                        AccumulateGradient(a, aGrad);
                    });

                    ComputationGraph.AddNode(resultTensor, gradFn);
                }

                return resultTensor;
            },
            AutoDiffDiagnostics.ShapeNote("RMSNorm", a.Shape));
    }

    public static ReverseGradTensor<T> PerRowRMSNorm<T>(ReverseGradTensor<T> a, int rows, int cols, double eps = 1e-5)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffPerRowRMSNorm",
            a.Length,
            
            () =>
            {
                var srcData = new T[a.Length];
                a.Data.CopyTo(srcData, default(T)!);
                var resultData = new T[rows * cols];

                RMSNormKernel<T>.PerRowRMSNormForwardKernel(srcData, resultData, rows, cols, eps);

                var resultCol = NivaraColumn<T>.Create(resultData);
                var result = new ReverseGradTensor<T>(resultCol, GradientUtils.ShouldTrackGrad(a), a.Shape);

                if (GradientUtils.ShouldTrackGrad(a))
                {
                    var savedInput = new T[a.Length];
                    a.Data.CopyTo(savedInput, default(T)!);

                    var gradFn = new OpNode<T>("PerRowRMSNorm", [a], (typedGradOutput) =>
                    {
                        var gradOut = new T[typedGradOutput.Length];
                        typedGradOutput.CopyTo(gradOut.AsSpan(), default(T)!);

                        var gradResult = new T[rows * cols];

                        RMSNormKernel<T>.PerRowRMSNormBackwardKernel(
                            savedInput, gradOut, gradResult, rows, cols, eps);

                        var gradCol = NivaraColumn<T>.Create(gradResult);
                        AccumulateGradient(a, gradCol);
                    });

                    ComputationGraph.AddNode(result, gradFn);
                }

                return result;
            },
            AutoDiffDiagnostics.ShapeNote("PerRowRMSNorm", a.Shape));
    }

    public static ReverseGradTensor<T> Dropout<T>(ReverseGradTensor<T> input, double probability, bool isTraining)
        where T : struct, IFloatingPointIeee754<T>
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
        where T : struct, IFloatingPointIeee754<T>
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (keepMask.Length != input.Length)
            throw new ArgumentException($"Dropout mask length ({keepMask.Length}) must match input length ({input.Length})", nameof(keepMask));

        var savedMask = keepMask.ToArray();
        var result = ApplyDropout(input.Data, savedMask, scale);
        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(input), input.shape);

        if (GradientUtils.ShouldTrackGrad(input))
        {
            var gradFn = new OpNode<T>("Dropout", new object[] { input }, (typedGradOutput) =>
            {
                var inputGrad = ApplyDropoutGradient(input.Data, typedGradOutput, savedMask, scale);
                AccumulateGradient(input, inputGrad);
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
        where T : struct, IFloatingPointIeee754<T>
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

        var resultValues = new T[batchSize * embeddingDim];
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

        var resultColumn = NivaraColumn<T>.Create(resultValues);

        var result = new ReverseGradTensor<T>(
            resultColumn,
            GradientUtils.ShouldTrackGrad(weight),
            new[] { batchSize, embeddingDim });

        if (GradientUtils.ShouldTrackGrad(weight))
        {
            var savedIndices = parsedIndices;
            var gradFn = new OpNode<T>("SparseEmbeddingBag", new object[] { weight }, (typedGradOutput) =>
            {
                var weightGrad = new T[weight.Length];
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

                AccumulateGradient(weight, NivaraColumn<T>.Create(weightGrad));
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
        where T : struct, IFloatingPointIeee754<T>
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
            
            () =>
            {
                int resultLen = indices.Length * stride;
                var resultValues = new T[resultLen];

                if (source.Data.TryGetSpan(out var span))
                {
                    for (int i = 0; i < indices.Length; i++)
                    {
                        int srcOffset = indices[i] * stride;
                        int dstOffset = i * stride;
                        span.Slice(srcOffset, stride).CopyTo(resultValues.AsSpan(dstOffset, stride));
                    }
                }
                else
                {
                    for (int i = 0; i < indices.Length; i++)
                    {
                        int srcOffset = indices[i] * stride;
                        int dstOffset = i * stride;
                        for (int j = 0; j < stride; j++)
                            resultValues[dstOffset + j] = source.Data[srcOffset + j];
                    }
                }

                var resultCol = NivaraColumn<T>.Create(resultValues);

                var resultShape = new int[source.shape.Length];
                resultShape[0] = indices.Length;
                for (int d = 1; d < source.shape.Length; d++)
                    resultShape[d] = source.shape[d];

                var result = new ReverseGradTensor<T>(resultCol, GradientUtils.ShouldTrackGrad(source), resultShape);

                if (GradientUtils.ShouldTrackGrad(source))
                {
                    var savedIndices = indices;
                    var gradFn = new OpNode<T>("Gather", new object[] { source }, (typedGradOutput) =>
                    {
                        var gradBuf = ArrayPool<T>.Shared.Rent(source.Length);
                        Array.Clear(gradBuf, 0, source.Length);

                        try
                        {
                            typedGradOutput.TryGetSpan(out var gradSpan);
                            for (int i = 0; i < savedIndices.Length; i++)
                            {
                                int dstOffset = savedIndices[i] * stride;
                                int srcOffset = i * stride;
                                for (int j = 0; j < stride; j++)
                                    gradBuf[dstOffset + j] += gradSpan[srcOffset + j];
                            }

                            var sourceGrad = NivaraColumn<T>.Create(gradBuf.AsSpan(0, source.Length));
                            AccumulateGradient(source, sourceGrad);
                        }
                        finally
                        {
                            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
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
        where T : struct, IFloatingPointIeee754<T>
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
            var gradFn = new OpNode<T>("KlDivergence", new object[] { mean, logVar }, (typedGradOutput) =>
            {
                if (mean.RequiresGrad)
                {
                    var broadcast = BroadcastGradient(typedGradOutput, mean.Length);
                    var dMean = ApplyKlMeanGradient(mean.Data, broadcast);
                    AccumulateGradient(mean, dMean);
                }
                if (logVar.RequiresGrad)
                {
                    var broadcast = BroadcastGradient(typedGradOutput, logVar.Length);
                    var dLogVar = ApplyKlLogVarGradient(logVar.Data, broadcast);
                    AccumulateGradient(logVar, dLogVar);
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> SampleNormal<T>(ReverseGradTensor<T> mean, ReverseGradTensor<T> logVar, int? seed = null)
        where T : struct, IFloatingPointIeee754<T>
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

        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(mean, logVar), mean.shape);

        if (GradientUtils.ShouldTrackGrad(mean, logVar))
        {
            var savedEpsilon = epsilonCol;
            var gradFn = new OpNode<T>("SampleNormal", new object[] { mean, logVar }, (typedGradOutput) =>
            {
                if (mean.RequiresGrad)
                {
                    AccumulateGradient(mean, typedGradOutput);
                }
                if (logVar.RequiresGrad)
                {
                    var dLogVar = ApplySampleNormalLogVarGradient(logVar.Data, typedGradOutput, savedEpsilon);
                    AccumulateGradient(logVar, dLogVar);
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    #endregion

    #region Helper Methods

    internal static void AccumulateGradient<T>(ReverseGradTensor<T> tensor, NivaraColumn<T> gradient) where T : struct, IFloatingPointIeee754<T>
    {
        if (tensor.Grad == null)
        {
            tensor.Grad = gradient;
            return;
        }

        int len = tensor.Grad.Length;
        var resultBuf = ArrayPool<T>.Shared.Rent(len);
        var gradBuf = ArrayPool<T>.Shared.Rent(len);
        try
        {
            tensor.Grad.CopyTo(resultBuf.AsSpan(0, len), default(T)!);
            gradient.CopyTo(gradBuf.AsSpan(0, len), default(T)!);
            TensorPrimitives.Add(resultBuf.AsSpan(0, len), gradBuf.AsSpan(0, len), resultBuf.AsSpan(0, len));
            tensor.Grad = NivaraColumn<T>.Create(resultBuf.AsSpan(0, len));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
        }
    }

    private static NivaraColumn<T> BroadcastGradient<T>(NivaraColumn<T> scalarGrad, int targetLength) where T : struct, IFloatingPointIeee754<T>
    {
        if (scalarGrad.Length != 1)
            throw new ArgumentException($"Expected scalar gradient with length 1, got {scalarGrad.Length}");

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
        where T : struct, IFloatingPointIeee754<T>
    {
        int n = input.Length;
        input.TryGetSpan(out var span);
        var result = new T[n];
        for (int i = 0; i < n; i++)
            result[i] = keepMask[i] ? span[i] * scale : T.Zero;

        return NivaraColumn<T>.Create(result);
    }

    private static NivaraColumn<T> ApplyDropoutGradient<T>(
        NivaraColumn<T> input,
        NivaraColumn<T> gradOutput,
        ReadOnlySpan<bool> keepMask,
        T scale)
        where T : struct, IFloatingPointIeee754<T>
    {
        int n = input.Length;
        gradOutput.TryGetSpan(out var gSpan);
        var result = new T[n];
        for (int i = 0; i < n; i++)
            result[i] = keepMask[i] ? gSpan[i] * scale : T.Zero;

        return NivaraColumn<T>.Create(result);
    }

    private static NivaraColumn<T> ApplyKlElementWise<T>(NivaraColumn<T> mean, NivaraColumn<T> logVar) where T : struct, IFloatingPointIeee754<T>
    {
        int n = mean.Length;
        mean.TryGetSpan(out var mSpan);
        logVar.TryGetSpan(out var lvSpan);
        var result = new T[n];
        var m2 = new T[n];
        var expLv = new T[n];
        var tmp = new T[n];
        TensorPrimitives.Multiply(mSpan, mSpan, m2);
        TensorPrimitives.Exp(lvSpan, expLv);
        TensorPrimitives.Add(lvSpan, T.One, tmp);
        TensorPrimitives.Subtract(tmp, m2, tmp);
        TensorPrimitives.Subtract(tmp, expLv, tmp);
        TensorPrimitives.Multiply(tmp, T.CreateChecked(-0.5), result);
        return NivaraColumn<T>.Create(result);
    }

    private static NivaraColumn<T> ApplyKlMeanGradient<T>(NivaraColumn<T> mean, NivaraColumn<T> gradOutput) where T : struct, IFloatingPointIeee754<T>
    {
        int n = mean.Length;
        mean.TryGetSpan(out var mSpan);
        gradOutput.TryGetSpan(out var gSpan);
        var result = new T[n];
        TensorPrimitives.Multiply(mSpan, gSpan, result);
        return NivaraColumn<T>.Create(result);
    }

    private static NivaraColumn<T> ApplyKlLogVarGradient<T>(NivaraColumn<T> logVar, NivaraColumn<T> gradOutput) where T : struct, IFloatingPointIeee754<T>
    {
        int n = logVar.Length;
        logVar.TryGetSpan(out var lvSpan);
        gradOutput.TryGetSpan(out var gSpan);
        var result = new T[n];
        TensorPrimitives.Exp(lvSpan, result);
        TensorPrimitives.Multiply(result, T.CreateChecked(-1), result);
        TensorPrimitives.Add(result, T.One, result);
        TensorPrimitives.Multiply(result, gSpan, result);
        TensorPrimitives.Multiply(result, T.CreateChecked(-0.5), result);
        return NivaraColumn<T>.Create(result);
    }

    private static NivaraColumn<T> ApplySampleNormalForward<T>(NivaraColumn<T> mean, NivaraColumn<T> logVar, NivaraColumn<T> epsilon) where T : struct, IFloatingPointIeee754<T>
    {
        int n = mean.Length;
        mean.TryGetSpan(out var mSpan);
        logVar.TryGetSpan(out var lvSpan);
        epsilon.TryGetSpan(out var eSpan);
        var result = new T[n];
        TensorPrimitives.Multiply(lvSpan, T.CreateChecked(0.5), result);
        TensorPrimitives.Exp(result, result);
        TensorPrimitives.Multiply(result, eSpan, result);
        TensorPrimitives.Add(result, mSpan, result);
        return NivaraColumn<T>.Create(result);
    }

    private static NivaraColumn<T> ApplySampleNormalLogVarGradient<T>(NivaraColumn<T> logVar, NivaraColumn<T> gradOutput, NivaraColumn<T> epsilon) where T : struct, IFloatingPointIeee754<T>
    {
        int n = logVar.Length;
        logVar.TryGetSpan(out var lvSpan);
        gradOutput.TryGetSpan(out var gSpan);
        epsilon.TryGetSpan(out var eSpan);
        var result = new T[n];
        TensorPrimitives.Multiply(lvSpan, T.CreateChecked(0.5), result);
        TensorPrimitives.Exp(result, result);
        TensorPrimitives.Multiply(result, eSpan, result);
        TensorPrimitives.Multiply(result, gSpan, result);
        TensorPrimitives.Multiply(result, T.CreateChecked(0.5), result);
        return NivaraColumn<T>.Create(result);
    }

    private static NivaraColumn<T> ApplyPow<T>(NivaraColumn<T> input, double exponent) where T : struct, IFloatingPointIeee754<T>
    {
        int n = input.Length;
        input.TryGetSpan(out var span);
        var result = new T[n];
        TensorPrimitives.Pow(span, T.CreateChecked(exponent), result);
        return NivaraColumn<T>.Create(result);
    }

    private static NivaraColumn<T> ApplyPowGradient<T>(NivaraColumn<T> input, NivaraColumn<T> gradOutput, double exponent) where T : struct, IFloatingPointIeee754<T>
    {
        int n = input.Length;
        input.TryGetSpan(out var inSpan);
        gradOutput.TryGetSpan(out var gSpan);
        var result = new T[n];
        TensorPrimitives.Pow(inSpan, T.CreateChecked(exponent - 1.0), result);
        TensorPrimitives.Multiply(result, T.CreateChecked(exponent), result);
        TensorPrimitives.Multiply(result, gSpan, result);
        return NivaraColumn<T>.Create(result);
    }

    private static NivaraColumn<T> ApplyRMSNorm<T>(NivaraColumn<T> input, double eps) where T : struct, IFloatingPointIeee754<T>
    {
        int n = input.Length;
        input.TryGetSpan(out var span);
        var result = new T[n];
        double sumSq = double.CreateChecked(TensorPrimitives.SumOfSquares(span));
        double rms = Math.Sqrt(sumSq / n + eps);
        T invRms = T.CreateChecked(1.0 / rms);
        TensorPrimitives.Multiply(span, invRms, result);
        return NivaraColumn<T>.Create(result);
    }

    private static NivaraColumn<T> ApplyRMSNormGradient<T>(
        NivaraColumn<T> input, NivaraColumn<T> gradOutput, double eps) where T : struct, IFloatingPointIeee754<T>
    {
        int n = input.Length;
        input.TryGetSpan(out var inSpan);
        gradOutput.TryGetSpan(out var gSpan);
        var result = new T[n];

        double sumSq = double.CreateChecked(TensorPrimitives.SumOfSquares(inSpan));
        double rms = Math.Sqrt(sumSq / n + eps);
        double rms3 = rms * rms * rms;

        T sumGradX = TensorPrimitives.Dot(gSpan, inSpan);
        T scale = sumGradX / T.CreateChecked(n * rms3);
        T invRms = T.CreateChecked(1.0 / rms);

        for (int i = 0; i < n; i++)
            result[i] = gSpan[i] * invRms - inSpan[i] * scale;

        return NivaraColumn<T>.Create(result);
    }

    public static ReverseGradTensor<T> BroadcastMultiply<T>(ReverseGradTensor<T> input, ReverseGradTensor<T> scale) where T : struct, IFloatingPointIeee754<T>
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (scale == null) throw new ArgumentNullException(nameof(scale));
        if (input.Rank < 2) throw new ArgumentException($"Input must be at least 2D [batch, channels, ...], got {input.Rank}D");
        if (scale.Rank != 1) throw new ArgumentException($"Scale must be 1D [channels], got {scale.Rank}D");

        int c = input.shape[1];
        if (scale.Length != c)
            throw new ArgumentException($"Scale length ({scale.Length}) must match input channel dimension ({c})");

        var inputData = new T[input.Length];
        input.Data.CopyTo(inputData, T.Zero);
        var scaleData = new T[c];
        scale.Data.CopyTo(scaleData, T.Zero);

        int channelStride = 1;
        for (int d = 2; d < input.Rank; d++) channelStride *= input.shape[d];

        var outputData = new T[input.Length];
        for (int idx = 0; idx < input.Length; idx++)
        {
            int ch = (idx / channelStride) % c;
            outputData[idx] = inputData[idx] * scaleData[ch];
        }

        var result = NivaraColumn<T>.Create(outputData);
        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(input, scale), input.shape);

        if (GradientUtils.ShouldTrackGrad(input, scale))
        {
            var gradFn = new OpNode<T>("BroadcastMultiply", new object[] { input, scale }, (typedGradOutput) =>
            {
                var gradData = new T[input.Length];
                typedGradOutput.CopyTo(gradData, T.Zero);

                if (GradientUtils.ShouldTrackGrad(input))
                {
                    var inputGrad = new T[input.Length];
                    for (int idx = 0; idx < input.Length; idx++)
                    {
                        int ch = (idx / channelStride) % c;
                        inputGrad[idx] = gradData[idx] * scaleData[ch];
                    }
                    AccumulateGradient(input, NivaraColumn<T>.Create(inputGrad));
                }
                if (scale.RequiresGrad)
                {
                    var scaleGrad = new T[c];
                    for (int idx = 0; idx < input.Length; idx++)
                    {
                        int ch = (idx / channelStride) % c;
                        scaleGrad[ch] += gradData[idx] * inputData[idx];
                    }
                    AccumulateGradient(scale, NivaraColumn<T>.Create(scaleGrad));
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> BroadcastAdd<T>(ReverseGradTensor<T> input, ReverseGradTensor<T> bias) where T : struct, IFloatingPointIeee754<T>
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (bias == null) throw new ArgumentNullException(nameof(bias));
        if (input.Rank < 2) throw new ArgumentException($"Input must be at least 2D [batch, channels, ...], got {input.Rank}D");
        if (bias.Rank != 1) throw new ArgumentException($"Bias must be 1D [channels], got {bias.Rank}D");

        int c = input.shape[1];
        if (bias.Length != c)
            throw new ArgumentException($"Bias length ({bias.Length}) must match input channel dimension ({c})");

        var inputData = new T[input.Length];
        input.Data.CopyTo(inputData, T.Zero);
        var biasData = new T[c];
        bias.Data.CopyTo(biasData, T.Zero);

        int channelStride = 1;
        for (int d = 2; d < input.Rank; d++) channelStride *= input.shape[d];

        var outputData = new T[input.Length];
        for (int idx = 0; idx < input.Length; idx++)
        {
            int ch = (idx / channelStride) % c;
            outputData[idx] = inputData[idx] + biasData[ch];
        }

        var result = NivaraColumn<T>.Create(outputData);
        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(input, bias), input.shape);

        if (GradientUtils.ShouldTrackGrad(input, bias))
        {
            var gradFn = new OpNode<T>("BroadcastAdd", new object[] { input, bias }, (typedGradOutput) =>
            {
                var gradData = new T[input.Length];
                typedGradOutput.CopyTo(gradData, T.Zero);

                if (GradientUtils.ShouldTrackGrad(input))
                {
                    AccumulateGradient(input, typedGradOutput);
                }
                if (bias.RequiresGrad)
                {
                    var biasGrad = new T[c];
                    for (int idx = 0; idx < input.Length; idx++)
                    {
                        int ch = (idx / channelStride) % c;
                        biasGrad[ch] += gradData[idx];
                    }
                    AccumulateGradient(bias, NivaraColumn<T>.Create(biasGrad));
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Gets the underlying data span. ADR-001 guarantees no nulls in AutoDiff,
    /// so TryGetSpan always succeeds.
    /// </summary>
    internal static ReadOnlySpan<T> AsSpan<T>(this ReverseGradTensor<T> t) where T : struct, IFloatingPointIeee754<T>
    {
        t.Data.TryGetSpan(out var span);
        return span;
    }

    /// <summary>
    /// Creates a result tensor from a raw T[] array, reusing the source tensor's shape.
    /// </summary>
    static ReverseGradTensor<T> ResultTensor<T>(T[] data, ReverseGradTensor<T> shapeSrc, bool requiresGrad)
        where T : struct, IFloatingPointIeee754<T>
    {
        return new ReverseGradTensor<T>(NivaraColumn<T>.Create(data), requiresGrad, shapeSrc.shape);
    }

    private static int[] ScalarShape()
    {
        return new[] { 1 };
    }

    #endregion
}

