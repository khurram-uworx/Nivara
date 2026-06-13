using Nivara.AutoDiff.Extensions;
using Nivara.Tensors;
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace Nivara.AutoDiff.Operations;

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

        int n = a.Length;
        a.Data.TryGetSpan(out var aSpan);
        b.Data.TryGetSpan(out var bSpan);
        var buf = ArrayPool<T>.Shared.Rent(n);
        NivaraColumn<T> result;
        try
        {
            TensorPrimitives.Add(aSpan, bSpan, buf.AsSpan(0, n));
            result = NivaraColumn<T>.Create(buf.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }

        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad || b.RequiresGrad, PropagateShape(a, b));

        if (a.RequiresGrad || b.RequiresGrad)
        {
            var gradFn = new OpNode<T>("Add", new object[] { a, b }, (typedGradOutput, _) =>
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
            var gradFn = new OpNode<T>("Subtract", new object[] { a, b }, (typedGradOutput, _) =>
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

        int n = a.Length;
        a.Data.TryGetSpan(out var aSpan);
        b.Data.TryGetSpan(out var bSpan);
        var buf = ArrayPool<T>.Shared.Rent(n);
        NivaraColumn<T> result;
        try
        {
            TensorPrimitives.Multiply(aSpan, bSpan, buf.AsSpan(0, n));
            result = NivaraColumn<T>.Create(buf.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }

        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad || b.RequiresGrad, PropagateShape(a, b));

        if (a.RequiresGrad || b.RequiresGrad)
        {
            var gradFn = new OpNode<T>("Multiply", new object[] { a, b }, (typedGradOutput, _) =>
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
            var gradFn = new OpNode<T>("Divide", new object[] { a, b }, (typedGradOutput, _) =>
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
            var gradFn = new OpNode<T>("MatMul", new object[] { a, b }, (typedGradOutput, _) =>
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
            var gradFn = new OpNode<T>("Transpose", new object[] { a }, (typedGradOutput, _) =>
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
            var gradFn = new OpNode<T>("Sum", new object[] { a }, (typedGradOutput, _) =>
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
            var gradFn = new OpNode<T>("Mean", new object[] { a }, (typedGradOutput, _) =>
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
            var gradFn = new OpNode<T>("Relu", new object[] { a }, (typedGradOutput, _) =>
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
            var gradFn = new OpNode<T>("Sigmoid", new object[] { a }, (typedGradOutput, _) =>
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
            var gradFn = new OpNode<T>("Tanh", new object[] { a }, (typedGradOutput, _) =>
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
            var gradFn = new OpNode<T>("Negate", new object[] { a }, (typedGradOutput, _) =>
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
            var gradFn = new OpNode<T>("Abs", new object[] { a }, (typedGradOutput, _) =>
            {
                var aGrad = ApplyAbsGradient(a.Data, typedGradOutput);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Clip<T>(ReverseGradTensor<T> a, T min, T max)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = ApplyClip(a.Data, min, max);
        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad, PropagateShape(a));

        if (a.RequiresGrad)
        {
            var gradFn = new OpNode<T>("Clip", new object[] { a, min, max }, (typedGradOutput, _) =>
            {
                var aGrad = ApplyClipGradient(a.Data, typedGradOutput, min, max);
                AccumulateGradient(a, aGrad);
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

        var result = ApplyLeakyRelu(a.Data, negativeSlope);
        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad, PropagateShape(a));

        if (a.RequiresGrad)
        {
            var gradFn = new OpNode<T>("LeakyRelu", new object[] { a, negativeSlope }, (typedGradOutput, _) =>
            {
                var aGrad = ApplyLeakyReluGradient(a.Data, typedGradOutput, negativeSlope);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Exp<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = ApplyExp(a.Data);
        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad, PropagateShape(a));

        if (a.RequiresGrad)
        {
            var gradFn = new OpNode<T>("Exp", new object[] { a }, (typedGradOutput, _) =>
            {
                var aGrad = MultiplyVectorized(typedGradOutput, result);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Log<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = ApplyLog(a.Data);
        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad, PropagateShape(a));

        if (a.RequiresGrad)
        {
            var gradFn = new OpNode<T>("Log", new object[] { a }, (typedGradOutput, _) =>
            {
                var aGrad = ApplyLogGradient(a.Data, typedGradOutput);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> Softmax<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = ApplySoftmax(a.Data, a.Rank >= 2 ? a.shape[1] : a.Length);
        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad, PropagateShape(a));

        if (a.RequiresGrad)
        {
            var savedResult = result;
            var gradFn = new OpNode<T>("Softmax", new object[] { a }, (typedGradOutput, _) =>
            {
                var aGrad = ApplySoftmaxGradient(savedResult, typedGradOutput, a.Rank >= 2 ? a.shape[1] : a.Length);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    public static ReverseGradTensor<T> LogSoftmax<T>(ReverseGradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var result = ApplyLogSoftmax(a.Data, a.Rank >= 2 ? a.shape[1] : a.Length);
        var resultTensor = new ReverseGradTensor<T>(result, a.RequiresGrad, PropagateShape(a));

        if (a.RequiresGrad)
        {
            var gradFn = new OpNode<T>("LogSoftmax", new object[] { a }, (typedGradOutput, _) =>
            {
                var aGrad = ApplyLogSoftmaxGradient(a.Data, typedGradOutput, a.Rank >= 2 ? a.shape[1] : a.Length);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
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
        var resultTensor = new ReverseGradTensor<T>(result, input.RequiresGrad, PropagateShape(input));

        if (input.RequiresGrad)
        {
            var gradFn = new OpNode<T>("Dropout", new object[] { input }, (typedGradOutput, _) =>
            {
                var inputGrad = ApplyDropoutGradient(input.Data, typedGradOutput, savedMask, scale);
                AccumulateGradient(input, inputGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
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
        var resultTensor = new ReverseGradTensor<T>(resultData, mean.RequiresGrad || logVar.RequiresGrad, ScalarShape());

        if (mean.RequiresGrad || logVar.RequiresGrad)
        {
            var gradFn = new OpNode<T>("KlDivergence", new object[] { mean, logVar }, (typedGradOutput, _) =>
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
        where T : struct, INumber<T>
    {
        if (mean == null) throw new ArgumentNullException(nameof(mean));
        if (logVar == null) throw new ArgumentNullException(nameof(logVar));

        if (mean.Length != logVar.Length)
            throw new ArgumentException(
                $"mean length ({mean.Length}) must equal logVar length ({logVar.Length})",
                nameof(logVar));

        int n = mean.Length;
        var epsilonBuf = ArrayPool<T>.Shared.Rent(n);
        NivaraColumn<T> epsilonCol;
        try
        {
            var epsilon = epsilonBuf.AsSpan(0, n);
            GenerateStandardNormal(epsilon, seed);
            epsilonCol = NivaraColumn<T>.Create(epsilon);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(epsilonBuf, clearArray: true);
        }

        var result = ApplySampleNormalForward(mean.Data, logVar.Data, epsilonCol);

        var resultTensor = new ReverseGradTensor<T>(result, mean.RequiresGrad || logVar.RequiresGrad, PropagateShape(mean, logVar));

        if (mean.RequiresGrad || logVar.RequiresGrad)
        {
            var savedEpsilon = epsilonCol;
            var gradFn = new OpNode<T>("SampleNormal", new object[] { mean, logVar }, (typedGradOutput, _) =>
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

    private static void AccumulateGradient<T>(ReverseGradTensor<T> tensor, NivaraColumn<T> gradient) where T : struct, INumber<T>
    {
        if (tensor.Grad == null)
            tensor.Grad = gradient;
        else
            tensor.Grad = tensor.Grad.Add(gradient);
    }

    private static NivaraColumn<T> SubtractVectorized<T>(NivaraColumn<T> a, NivaraColumn<T> b) where T : struct, INumber<T>
        => a.Apply(b, static (l, r, d) => TensorPrimitives.Subtract(l, r, d));

    private static NivaraColumn<T> DivideVectorized<T>(NivaraColumn<T> a, NivaraColumn<T> b) where T : struct, INumber<T>
        => a.Apply(b, static (l, r, d) => TensorPrimitives.Divide(l, r, d));

    private static NivaraColumn<T> NegateVectorized<T>(NivaraColumn<T> a) where T : struct, INumber<T>
        => a.Apply(static (s, d) => TensorPrimitives.Negate(s, d));

    private static NivaraColumn<T> MultiplyVectorized<T>(NivaraColumn<T> a, NivaraColumn<T> b) where T : struct, INumber<T>
        => a.Apply(b, static (l, r, d) => TensorPrimitives.Multiply(l, r, d));

    private static NivaraColumn<T> MatMulVectorized<T>(NivaraColumn<T> a, NivaraColumn<T> b, int aRows, int aCols, int bCols)
        where T : struct, INumber<T>
    {
        int n = aRows * bCols;
        a.TryGetSpan(out var aSpan);
        b.TryGetSpan(out var bSpan);
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            MatMulHelper.MultiplyCore(aSpan, bSpan, buf, aRows, aCols, bCols);
            return NivaraColumn<T>.Create(buf.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> TransposeVectorized<T>(NivaraColumn<T> a, int rows, int cols) where T : struct, INumber<T>
    {
        var aSpan = a.AsSpan();
        int n = rows * cols;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[j * rows + i] = aSpan[i * cols + j];
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> BroadcastGradient<T>(NivaraColumn<T> scalarGrad, int targetLength) where T : struct, INumber<T>
    {
        if (scalarGrad.Length != 1)
            throw new ArgumentException($"Expected scalar gradient with length 1, got {scalarGrad.Length}");

        var span = scalarGrad.AsSpan();
        var buf = ArrayPool<T>.Shared.Rent(targetLength);
        try
        {
            var filled = buf.AsSpan(0, targetLength);
            filled.Fill(span[0]);
            return NivaraColumn<T>.Create(filled);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> DivideByScalar<T>(NivaraColumn<T> column, int divisor) where T : struct, INumber<T>
    {
        T divisorT = ConvertToT<T>(divisor);
        return column.Apply(divisorT, static (s, v, d) => TensorPrimitives.Divide(s, v, d));
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
        return input.Apply(T.Zero, static (s, threshold, d) => TensorPrimitives.Max(s, threshold, d));
    }

    private static NivaraColumn<T> ApplyDropout<T>(NivaraColumn<T> input, ReadOnlySpan<bool> keepMask, T scale)
        where T : struct, INumber<T>
    {
        input.TryGetSpan(out var span);
        int n = input.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            for (int i = 0; i < n; i++)
                result[i] = keepMask[i] ? span[i] * scale : T.Zero;
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyDropoutGradient<T>(
        NivaraColumn<T> input,
        NivaraColumn<T> gradOutput,
        ReadOnlySpan<bool> keepMask,
        T scale)
        where T : struct, INumber<T>
    {
        gradOutput.TryGetSpan(out var gradSpan);
        int n = input.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            for (int i = 0; i < n; i++)
                result[i] = keepMask[i] ? gradSpan[i] * scale : T.Zero;
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyReluGradient<T>(NivaraColumn<T> input, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        input.TryGetSpan(out var inSpan);
        gradOutput.TryGetSpan(out var gradSpan);
        int n = input.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            for (int i = 0; i < n; i++)
                result[i] = inSpan[i] > T.Zero ? gradSpan[i] : T.Zero;
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyAbs<T>(NivaraColumn<T> input) where T : struct, INumber<T>
        => input.Apply(static (s, d) => TensorPrimitives.Abs(s, d));

    private static NivaraColumn<T> ApplyAbsGradient<T>(NivaraColumn<T> input, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        input.TryGetSpan(out var inSpan);
        gradOutput.TryGetSpan(out var gradSpan);
        int n = input.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            for (int i = 0; i < n; i++)
                result[i] = T.CreateChecked(T.Sign(inSpan[i])) * gradSpan[i];
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyClip<T>(NivaraColumn<T> input, T min, T max) where T : struct, INumber<T>
        => input.Apply(min, max, static (s, mn, mx, d) => TensorPrimitives.Clamp(s, mn, mx, d));

    private static NivaraColumn<T> ApplyClipGradient<T>(NivaraColumn<T> input, NivaraColumn<T> gradOutput, T min, T max) where T : struct, INumber<T>
    {
        input.TryGetSpan(out var inSpan);
        gradOutput.TryGetSpan(out var gradSpan);
        int n = input.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            for (int i = 0; i < n; i++)
                result[i] = (inSpan[i] >= min && inSpan[i] <= max) ? gradSpan[i] : T.Zero;
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyLeakyRelu<T>(NivaraColumn<T> input, T negativeSlope) where T : struct, INumber<T>
    {
        input.TryGetSpan(out var span);
        int n = input.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            for (int i = 0; i < n; i++)
                result[i] = span[i] > T.Zero ? span[i] : negativeSlope * span[i];
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyLeakyReluGradient<T>(NivaraColumn<T> input, NivaraColumn<T> gradOutput, T negativeSlope) where T : struct, INumber<T>
    {
        input.TryGetSpan(out var inSpan);
        gradOutput.TryGetSpan(out var gradSpan);
        int n = input.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            for (int i = 0; i < n; i++)
                result[i] = inSpan[i] > T.Zero ? gradSpan[i] : negativeSlope * gradSpan[i];
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyExp<T>(NivaraColumn<T> input) where T : struct, INumber<T>
    {
        input.TryGetSpan(out var span);
        int n = input.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            if (typeof(T) == typeof(float))
                TensorPrimitives.Exp(MemoryMarshal.Cast<T, float>(span), MemoryMarshal.Cast<T, float>(result));
            else if (typeof(T) == typeof(double))
                TensorPrimitives.Exp(MemoryMarshal.Cast<T, double>(span), MemoryMarshal.Cast<T, double>(result));
            else
                for (int i = 0; i < n; i++)
                    result[i] = T.CreateChecked(Math.Exp(double.CreateChecked(span[i])));
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyLog<T>(NivaraColumn<T> input) where T : struct, INumber<T>
    {
        input.TryGetSpan(out var span);
        int n = input.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            if (typeof(T) == typeof(float))
                TensorPrimitives.Log(MemoryMarshal.Cast<T, float>(span), MemoryMarshal.Cast<T, float>(result));
            else if (typeof(T) == typeof(double))
                TensorPrimitives.Log(MemoryMarshal.Cast<T, double>(span), MemoryMarshal.Cast<T, double>(result));
            else
                for (int i = 0; i < n; i++)
                    result[i] = T.CreateChecked(Math.Log(double.CreateChecked(span[i])));
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyLogGradient<T>(NivaraColumn<T> input, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        input.TryGetSpan(out var inSpan);
        gradOutput.TryGetSpan(out var gradSpan);
        int n = input.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            for (int i = 0; i < n; i++)
                result[i] = gradSpan[i] / inSpan[i];
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplySigmoid<T>(NivaraColumn<T> input) where T : struct, INumber<T>
    {
        input.TryGetSpan(out var span);
        int n = input.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            if (typeof(T) == typeof(float))
            {
                var s = MemoryMarshal.Cast<T, float>(span);
                var d = MemoryMarshal.Cast<T, float>(result);
                TensorPrimitives.Negate(s, d);
                TensorPrimitives.Exp(d, d);
                TensorPrimitives.Add(d, 1.0f, d);
                TensorPrimitives.Divide(1.0f, d, d);
            }
            else if (typeof(T) == typeof(double))
            {
                var s = MemoryMarshal.Cast<T, double>(span);
                var d = MemoryMarshal.Cast<T, double>(result);
                TensorPrimitives.Negate(s, d);
                TensorPrimitives.Exp(d, d);
                TensorPrimitives.Add(d, 1.0, d);
                TensorPrimitives.Divide(1.0, d, d);
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    var x = double.CreateChecked(span[i]);
                    result[i] = T.CreateChecked(1.0 / (1.0 + Math.Exp(-x)));
                }
            }
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplySigmoidGradient<T>(NivaraColumn<T> sigmoidOutput, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        sigmoidOutput.TryGetSpan(out var sigSpan);
        gradOutput.TryGetSpan(out var gradSpan);
        int n = sigmoidOutput.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            for (int i = 0; i < n; i++)
            {
                var sig = sigSpan[i];
                result[i] = sig * (T.One - sig) * gradSpan[i];
            }
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyTanh<T>(NivaraColumn<T> input) where T : struct, INumber<T>
    {
        input.TryGetSpan(out var span);
        int n = input.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            for (int i = 0; i < n; i++)
            {
                var x = double.CreateChecked(span[i]);
                result[i] = T.CreateChecked(Math.Tanh(x));
            }
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyTanhGradient<T>(NivaraColumn<T> tanhOutput, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        tanhOutput.TryGetSpan(out var tanhSpan);
        gradOutput.TryGetSpan(out var gradSpan);
        int n = tanhOutput.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            for (int i = 0; i < n; i++)
            {
                var tanh = tanhSpan[i];
                result[i] = (T.One - tanh * tanh) * gradSpan[i];
            }
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplySoftmax<T>(NivaraColumn<T> input, int classCount) where T : struct, INumber<T>
    {
        input.TryGetSpan(out var span);
        int n = input.Length;
        int rows = classCount > 0 ? n / classCount : n;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);

            if (typeof(T) == typeof(float))
            {
                var s = MemoryMarshal.Cast<T, float>(span);
                var d = MemoryMarshal.Cast<T, float>(result);
                for (int r = 0; r < rows; r++)
                {
                    int start = r * classCount;
                    var rowSpan = s.Slice(start, classCount);
                    var rowDst = d.Slice(start, classCount);
                    float max = float.NegativeInfinity;
                    for (int c = 0; c < classCount; c++)
                        if (rowSpan[c] > max) max = rowSpan[c];
                    TensorPrimitives.Subtract(rowSpan, max, rowDst);
                    TensorPrimitives.Exp(rowDst, rowDst);
                    TensorPrimitives.Divide(rowDst, TensorPrimitives.Sum(rowDst), rowDst);
                }
            }
            else if (typeof(T) == typeof(double))
            {
                var s = MemoryMarshal.Cast<T, double>(span);
                var d = MemoryMarshal.Cast<T, double>(result);
                for (int r = 0; r < rows; r++)
                {
                    int start = r * classCount;
                    var rowSpan = s.Slice(start, classCount);
                    var rowDst = d.Slice(start, classCount);
                    double max = double.NegativeInfinity;
                    for (int c = 0; c < classCount; c++)
                        if (rowSpan[c] > max) max = rowSpan[c];
                    TensorPrimitives.Subtract(rowSpan, max, rowDst);
                    TensorPrimitives.Exp(rowDst, rowDst);
                    TensorPrimitives.Divide(rowDst, TensorPrimitives.Sum(rowDst), rowDst);
                }
            }
            else
            {
                for (int r = 0; r < rows; r++)
                {
                    int rowStart = r * classCount;
                    double max = double.NegativeInfinity;
                    for (int c = 0; c < classCount; c++)
                    {
                        var val = double.CreateChecked(span[rowStart + c]);
                        if (val > max) max = val;
                    }
                    double sum = 0.0;
                    for (int c = 0; c < classCount; c++)
                    {
                        var exp = Math.Exp(double.CreateChecked(span[rowStart + c]) - max);
                        result[rowStart + c] = T.CreateChecked(exp);
                        sum += exp;
                    }
                    for (int c = 0; c < classCount; c++)
                        result[rowStart + c] = T.CreateChecked(double.CreateChecked(result[rowStart + c]) / sum);
                }
            }
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplySoftmaxGradient<T>(NivaraColumn<T> softmaxOutput, NivaraColumn<T> gradOutput, int classCount) where T : struct, INumber<T>
    {
        softmaxOutput.TryGetSpan(out var softSpan);
        gradOutput.TryGetSpan(out var gradSpan);
        int n = softmaxOutput.Length;
        int rows = classCount > 0 ? n / classCount : n;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            for (int r = 0; r < rows; r++)
            {
                int rowStart = r * classCount;
                double dot = 0.0;
                for (int c = 0; c < classCount; c++)
                    dot += double.CreateChecked(softSpan[rowStart + c]) * double.CreateChecked(gradSpan[rowStart + c]);
                for (int c = 0; c < classCount; c++)
                {
                    var s = double.CreateChecked(softSpan[rowStart + c]);
                    var dy = double.CreateChecked(gradSpan[rowStart + c]);
                    result[rowStart + c] = T.CreateChecked(s * (dy - dot));
                }
            }
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyLogSoftmax<T>(NivaraColumn<T> input, int classCount) where T : struct, INumber<T>
    {
        input.TryGetSpan(out var span);
        int n = input.Length;
        int rows = classCount > 0 ? n / classCount : n;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            for (int r = 0; r < rows; r++)
            {
                int rowStart = r * classCount;
                double max = double.NegativeInfinity;
                for (int c = 0; c < classCount; c++)
                {
                    var val = double.CreateChecked(span[rowStart + c]);
                    if (val > max) max = val;
                }
                double sum = 0.0;
                for (int c = 0; c < classCount; c++)
                    sum += Math.Exp(double.CreateChecked(span[rowStart + c]) - max);
                var logSum = Math.Log(sum);
                for (int c = 0; c < classCount; c++)
                    result[rowStart + c] = T.CreateChecked(double.CreateChecked(span[rowStart + c]) - max - logSum);
            }
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyLogSoftmaxGradient<T>(NivaraColumn<T> input, NivaraColumn<T> gradOutput, int classCount) where T : struct, INumber<T>
    {
        var softmax = ApplySoftmax(input, classCount);
        softmax.TryGetSpan(out var softSpan);
        gradOutput.TryGetSpan(out var gradSpan);
        int n = input.Length;
        int rows = classCount > 0 ? n / classCount : n;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            for (int r = 0; r < rows; r++)
            {
                int rowStart = r * classCount;
                double sumGrad = 0.0;
                for (int c = 0; c < classCount; c++)
                    sumGrad += double.CreateChecked(gradSpan[rowStart + c]);
                for (int c = 0; c < classCount; c++)
                {
                    var dy = double.CreateChecked(gradSpan[rowStart + c]);
                    var s = double.CreateChecked(softSpan[rowStart + c]);
                    result[rowStart + c] = T.CreateChecked(dy - s * sumGrad);
                }
            }
            return NivaraColumn<T>.Create(result);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyKlElementWise<T>(NivaraColumn<T> mean, NivaraColumn<T> logVar) where T : struct, INumber<T>
    {
        mean.TryGetSpan(out var mSpan);
        logVar.TryGetSpan(out var lvSpan);
        int n = mean.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            if (typeof(T) == typeof(float))
            {
                var m = MemoryMarshal.Cast<T, float>(mSpan);
                var lv = MemoryMarshal.Cast<T, float>(lvSpan);
                var r = MemoryMarshal.Cast<T, float>(result);
                var m2 = ArrayPool<float>.Shared.Rent(n);
                var expLv = ArrayPool<float>.Shared.Rent(n);
                var tmp = ArrayPool<float>.Shared.Rent(n);
                try
                {
                    TensorPrimitives.Multiply(m, m, m2.AsSpan(0, n));
                    TensorPrimitives.Exp(lv, expLv.AsSpan(0, n));
                    TensorPrimitives.Add(lv, 1.0f, tmp.AsSpan(0, n));
                    TensorPrimitives.Subtract(tmp.AsSpan(0, n), m2.AsSpan(0, n), tmp.AsSpan(0, n));
                    TensorPrimitives.Subtract(tmp.AsSpan(0, n), expLv.AsSpan(0, n), tmp.AsSpan(0, n));
                    TensorPrimitives.Multiply(tmp.AsSpan(0, n), -0.5f, r);
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(m2, clearArray: true);
                    ArrayPool<float>.Shared.Return(expLv, clearArray: true);
                    ArrayPool<float>.Shared.Return(tmp, clearArray: true);
                }
            }
            else if (typeof(T) == typeof(double))
            {
                var m = MemoryMarshal.Cast<T, double>(mSpan);
                var lv = MemoryMarshal.Cast<T, double>(lvSpan);
                var r = MemoryMarshal.Cast<T, double>(result);
                var m2 = ArrayPool<double>.Shared.Rent(n);
                var expLv = ArrayPool<double>.Shared.Rent(n);
                var tmp = ArrayPool<double>.Shared.Rent(n);
                try
                {
                    TensorPrimitives.Multiply(m, m, m2.AsSpan(0, n));
                    TensorPrimitives.Exp(lv, expLv.AsSpan(0, n));
                    TensorPrimitives.Add(lv, 1.0, tmp.AsSpan(0, n));
                    TensorPrimitives.Subtract(tmp.AsSpan(0, n), m2.AsSpan(0, n), tmp.AsSpan(0, n));
                    TensorPrimitives.Subtract(tmp.AsSpan(0, n), expLv.AsSpan(0, n), tmp.AsSpan(0, n));
                    TensorPrimitives.Multiply(tmp.AsSpan(0, n), -0.5, r);
                }
                finally
                {
                    ArrayPool<double>.Shared.Return(m2, clearArray: true);
                    ArrayPool<double>.Shared.Return(expLv, clearArray: true);
                    ArrayPool<double>.Shared.Return(tmp, clearArray: true);
                }
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
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyKlMeanGradient<T>(NivaraColumn<T> mean, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
        => mean.Apply(gradOutput, static (m, g, d) => TensorPrimitives.Multiply(m, g, d));

    private static NivaraColumn<T> ApplyKlLogVarGradient<T>(NivaraColumn<T> logVar, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        logVar.TryGetSpan(out var lvSpan);
        gradOutput.TryGetSpan(out var gSpan);
        int n = logVar.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            if (typeof(T) == typeof(float))
            {
                var lv = MemoryMarshal.Cast<T, float>(lvSpan);
                var g = MemoryMarshal.Cast<T, float>(gSpan);
                var d = MemoryMarshal.Cast<T, float>(result);
                TensorPrimitives.Exp(lv, d);
                TensorPrimitives.Subtract(1.0f, d, d);
                TensorPrimitives.Multiply(d, g, d);
                TensorPrimitives.Multiply(d, -0.5f, d);
            }
            else if (typeof(T) == typeof(double))
            {
                var lv = MemoryMarshal.Cast<T, double>(lvSpan);
                var g = MemoryMarshal.Cast<T, double>(gSpan);
                var d = MemoryMarshal.Cast<T, double>(result);
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
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplySampleNormalForward<T>(NivaraColumn<T> mean, NivaraColumn<T> logVar, NivaraColumn<T> epsilon) where T : struct, INumber<T>
    {
        mean.TryGetSpan(out var mSpan);
        logVar.TryGetSpan(out var lvSpan);
        epsilon.TryGetSpan(out var eSpan);
        int n = mean.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            if (typeof(T) == typeof(float))
            {
                var m = MemoryMarshal.Cast<T, float>(mSpan);
                var lv = MemoryMarshal.Cast<T, float>(lvSpan);
                var e = MemoryMarshal.Cast<T, float>(eSpan);
                var d = MemoryMarshal.Cast<T, float>(result);
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
                var d = MemoryMarshal.Cast<T, double>(result);
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
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplySampleNormalLogVarGradient<T>(NivaraColumn<T> logVar, NivaraColumn<T> gradOutput, NivaraColumn<T> epsilon) where T : struct, INumber<T>
    {
        logVar.TryGetSpan(out var lvSpan);
        gradOutput.TryGetSpan(out var gSpan);
        epsilon.TryGetSpan(out var eSpan);
        int n = logVar.Length;
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);
            if (typeof(T) == typeof(float))
            {
                var lv = MemoryMarshal.Cast<T, float>(lvSpan);
                var g = MemoryMarshal.Cast<T, float>(gSpan);
                var e = MemoryMarshal.Cast<T, float>(eSpan);
                var d = MemoryMarshal.Cast<T, float>(result);
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
                var d = MemoryMarshal.Cast<T, double>(result);
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
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
    }

    private static void GenerateStandardNormal<T>(Span<T> result, int? seed) where T : struct, INumber<T>
    {
        var rng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        for (int i = 0; i < result.Length; i++)
        {
            double u1, u2;
            u1 = rng.NextDouble();
            u2 = rng.NextDouble();
            var z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            result[i] = T.CreateChecked(z);
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
