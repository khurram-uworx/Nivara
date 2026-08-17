using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Operations;

/// <summary>
/// Reverse-mode automatic differentiation operations. Each method computes the forward result
/// and, when gradients are being tracked (inside a <see cref="GradientUtils.Grad"/> scope),
/// records an <see cref="OpNode{T}"/> whose VJP accumulates gradients back to the inputs.
/// All kernels are span-based and use <see cref="TensorPrimitives"/> where applicable.
/// </summary>
public static class ReverseGradOperations
{
    #region Element-wise Operations

    /// <summary>
    /// Adds two tensors element-wise: result[i] = a[i] + b[i]. Requires equal lengths.
    /// </summary>
    /// <param name="a">The left operand</param>
    /// <param name="b">The right operand</param>
    /// <returns>A new tensor holding the element-wise sum</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null</exception>
    /// <exception cref="ArgumentException">Thrown when the operand lengths differ</exception>
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
            var gradFn = new OpNode<T>("Add", [a, b], (typedGradOutput) =>
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

    /// <summary>
    /// Adds a bias vector to each row of a matrix: result[i,j] = a[i,j] + bias[j].
    /// Avoids the Ones+MatMul broadcast used for linear layer biases.
    /// </summary>
    public static ReverseGradTensor<T> AddBias<T>(ReverseGradTensor<T> a, ReverseGradTensor<T> bias)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (bias == null) throw new ArgumentNullException(nameof(bias));

        if (a.Rank != 2)
            throw new ArgumentException($"AddBias expects a matrix (rank 2) input, got rank {a.Rank}", nameof(a));

        var rows = a.shape[0];
        var cols = a.shape[1];
        if (bias.Length != cols)
            throw new ArgumentException($"AddBias bias length ({bias.Length}) must equal the input column count ({cols}).");

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffAddBias",
            a.Length + bias.Length,

            () =>
            {
                var aSpan = a.AsSpan();
                var biasSpan = bias.AsSpan();
                var resultArr = new T[a.Length];
                for (int r = 0; r < rows; r++)
                {
                    var resultRow = resultArr.AsSpan(r * cols, cols);
                    TensorPrimitives.Add(aSpan.Slice(r * cols, cols), biasSpan, resultRow);
                }

                var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a, bias));

                if (GradientUtils.ShouldTrackGrad(a, bias))
                {
                    var gradFn = new OpNode<T>("AddBias", [a, bias], (typedGradOutput) =>
                    {
                        typedGradOutput.TryGetSpan(out var gSpan);
                        if (GradientUtils.ShouldTrackGrad(a))
                            AccumulateGradient(a, typedGradOutput);
                        if (bias.RequiresGrad)
                        {
                            var biasGrad = new T[cols];
                            for (int r = 0; r < rows; r++)
                                TensorPrimitives.Add(biasGrad, gSpan.Slice(r * cols, cols), biasGrad);
                            AccumulateGradient(bias, NivaraColumn<T>.CreateFromOwnedArray(biasGrad));
                        }
                    });

                    ComputationGraph.AddNode(resultTensor, gradFn);
                }

                return resultTensor;
            },
            AutoDiffDiagnostics.MatrixNote("AddBias", rows, cols, cols));
    }

    /// <summary>
    /// Subtracts two tensors element-wise: result[i] = a[i] - b[i]. Requires equal lengths.
    /// </summary>
    /// <param name="a">The left operand</param>
    /// <param name="b">The right operand</param>
    /// <returns>A new tensor holding the element-wise difference</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null</exception>
    /// <exception cref="ArgumentException">Thrown when the operand lengths differ</exception>
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
            var gradFn = new OpNode<T>("Subtract", [a, b], (typedGradOutput) =>
            {
                if (GradientUtils.ShouldTrackGrad(a))
                    AccumulateGradient(a, typedGradOutput);
                if (b.RequiresGrad)
                {
                    typedGradOutput.TryGetSpan(out var gSpan);
                    var bGradArr = new T[a.Length];
                    TensorPrimitives.Negate(gSpan, bGradArr.AsSpan());
                    AccumulateGradient(b, NivaraColumn<T>.CreateFromOwnedArray(bGradArr));
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Multiplies two tensors element-wise: result[i] = a[i] * b[i]. Requires equal lengths.
    /// </summary>
    /// <param name="a">The left operand</param>
    /// <param name="b">The right operand</param>
    /// <returns>A new tensor holding the element-wise product</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null</exception>
    /// <exception cref="ArgumentException">Thrown when the operand lengths differ</exception>
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
            var gradFn = new OpNode<T>("Multiply", [a, b], (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                if (GradientUtils.ShouldTrackGrad(a))
                {
                    var aGradArr = new T[a.Length];
                    TensorPrimitives.Multiply(gSpan, b.AsSpan(), aGradArr.AsSpan());
                    AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(aGradArr));
                }
                if (b.RequiresGrad)
                {
                    var bGradArr = new T[a.Length];
                    TensorPrimitives.Multiply(gSpan, a.AsSpan(), bGradArr.AsSpan());
                    AccumulateGradient(b, NivaraColumn<T>.CreateFromOwnedArray(bGradArr));
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Divides two tensors element-wise: result[i] = a[i] / b[i]. Requires equal lengths and
    /// rejects zero divisors.
    /// </summary>
    /// <param name="a">The left operand (dividend)</param>
    /// <param name="b">The right operand (divisor)</param>
    /// <returns>A new tensor holding the element-wise quotient</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null</exception>
    /// <exception cref="ArgumentException">Thrown when the operand lengths differ</exception>
    /// <exception cref="DivideByZeroException">Thrown when the divisor contains a zero</exception>
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
            var gradFn = new OpNode<T>("Divide", [a, b], (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                if (GradientUtils.ShouldTrackGrad(a))
                {
                    var aGradArr = new T[a.Length];
                    TensorPrimitives.Divide(gSpan, b.AsSpan(), aGradArr.AsSpan());
                    AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(aGradArr));
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
                    AccumulateGradient(b, NivaraColumn<T>.CreateFromOwnedArray(finalBArr));
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Divides a tensor by a scalar: result[i] = a[i] / scalar.
    /// Unlike <see cref="Divide{T}(ReverseGradTensor{T}, ReverseGradTensor{T})"/>, no divisor
    /// tensor is constructed (the divisor is a plain scalar with no gradient).
    /// Backward scales the gradient by 1/scalar.
    /// </summary>
    public static ReverseGradTensor<T> DivideScalar<T>(ReverseGradTensor<T> a, T scalar) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (scalar == T.Zero)
            throw new DivideByZeroException($"Division by zero with scalar divisor");

        var resultArr = new T[a.Length];
        TensorPrimitives.Divide(a.AsSpan(), scalar, resultArr);

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("DivideScalar", [a], (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                var gradArr = new T[a.Length];
                TensorPrimitives.Divide(gSpan, scalar, gradArr);
                AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    #endregion

    #region Matrix Operations

    /// <summary>
    /// Computes the matrix product of two rank-2 tensors: result = a @ b. Requires
    /// a's column count to equal b's row count.
    /// </summary>
    /// <param name="a">The left matrix operand</param>
    /// <param name="b">The right matrix operand</param>
    /// <returns>A new [aRows, bCols] tensor holding the matrix product</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null</exception>
    /// <exception cref="ArgumentException">Thrown when an operand is not rank 2 or dimensions are incompatible</exception>
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
                GradKernels.MatMul(aSpan, bSpan, resultArr, aRows, aCols, bCols);
                var result = NivaraColumn<T>.CreateFromOwnedArray(resultArr);

                var resultShape = new[] { aRows, bCols };
                var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a, b), resultShape);

                if (GradientUtils.ShouldTrackGrad(a, b))
                {
                    var gradFn = new OpNode<T>("MatMul", [a, b], (typedGradOutput) =>
                    {
                        var bTArr = ArrayPool<T>.Shared.Rent(Math.Max(bRows * bCols, 1));
                        var aTArr = ArrayPool<T>.Shared.Rent(Math.Max(aRows * aCols, 1));
                        try
                        {
                            if (GradientUtils.ShouldTrackGrad(a))
                            {
                                b.Data.TryGetSpan(out var bSpan_b);
                                GradKernels.Transpose(bSpan_b, bTArr.AsSpan(0, bRows * bCols), bRows, bCols);
                                typedGradOutput.TryGetSpan(out var gradSpan);
                                var aGradArr = new T[aRows * aCols];
                                GradKernels.MatMul(gradSpan, bTArr.AsSpan(0, bRows * bCols), aGradArr, aRows, bCols, aCols);
                                AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(aGradArr));
                            }
                            if (b.RequiresGrad)
                            {
                                a.Data.TryGetSpan(out var aSpan_b);
                                GradKernels.Transpose(aSpan_b, aTArr.AsSpan(0, aRows * aCols), aRows, aCols);
                                typedGradOutput.TryGetSpan(out var gradSpan2);
                                var bGradArr = new T[aCols * bCols];
                                GradKernels.MatMul(aTArr.AsSpan(0, aRows * aCols), gradSpan2, bGradArr, aCols, aRows, bCols);
                                AccumulateGradient(b, NivaraColumn<T>.CreateFromOwnedArray(bGradArr));
                            }
                        }
                        finally
                        {
                            ArrayPool<T>.Shared.Return(bTArr, clearArray: true);
                            ArrayPool<T>.Shared.Return(aTArr, clearArray: true);
                        }
                    });

                    ComputationGraph.AddNode(resultTensor, gradFn);
                }

                return resultTensor;
            },
            AutoDiffDiagnostics.MatrixNote("MatMul", aRows, aCols, bCols));
    }

    /// <summary>
    /// Matmul where <paramref name="b"/> is already in the transposed-B
    /// layout the core kernel consumes (b is [bCols, aCols] row-major — i.e.
    /// the raw weight of a linear layer). Computes a @ b^T with zero
    /// transposes on either operand. Outside grad scope this is a plain
    /// inference kernel with no graph node; inside
    /// <see cref="GradientUtils.Grad"/> scope it records a single
    /// "MatMulTransposedB" node whose VJP produces dA = g @ b and
    /// dB = g^T @ a (one transpose of the output gradient, never of the
    /// weights), so training Linear no longer double-transposes weights.
    /// </summary>
    public static ReverseGradTensor<T> MatMulTransposedB<T>(ReverseGradTensor<T> a, ReverseGradTensor<T> b)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rank != 2 || b.Rank != 2)
            throw new ArgumentException("MatMulTransposedB requires rank-2 operands.");

        var aRows = a.shape[0];
        var aCols = a.shape[1];
        var bCols = b.shape[0];
        if (b.shape[1] != aCols)
            throw new ArgumentException(
                $"MatMulTransposedB dimension mismatch: a is {aRows}x{aCols}, b is {bCols}x{b.shape[1]}. " +
                $"b's column count ({b.shape[1]}) must equal a's column count ({aCols}).");

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffMatMulTransposedB",
            a.Length + b.Length,

            () =>
            {
                a.Data.TryGetSpan(out var aSpan);
                b.Data.TryGetSpan(out var bSpan);
                var resultArr = new T[aRows * bCols];
                GradKernels.MatMulTransposedB(aSpan, bSpan, resultArr, aRows, aCols, bCols);

                var shouldTrack = GradientUtils.ShouldTrackGrad(a, b);
                var resultTensor = new ReverseGradTensor<T>(
                    NivaraColumn<T>.CreateFromOwnedArray(resultArr),
                    shouldTrack,
                    new[] { aRows, bCols });

                if (shouldTrack)
                {
                    var gradFn = new OpNode<T>("MatMulTransposedB", [a, b], (typedGradOutput) =>
                    {
                        if (GradientUtils.ShouldTrackGrad(a))
                        {
                            b.Data.TryGetSpan(out var bSpan_b);
                            typedGradOutput.TryGetSpan(out var gradSpan);
                            var aGradArr = new T[aRows * aCols];
                            GradKernels.MatMul(gradSpan, bSpan_b, aGradArr, aRows, bCols, aCols);
                            AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(aGradArr));
                        }

                        if (GradientUtils.ShouldTrackGrad(b))
                        {
                            a.Data.TryGetSpan(out var aSpan_b);
                            typedGradOutput.TryGetSpan(out var gradSpan2);
                            var gTArr = ArrayPool<T>.Shared.Rent(Math.Max(bCols * aRows, 1));
                            try
                            {
                                GradKernels.Transpose(gradSpan2, gTArr.AsSpan(0, bCols * aRows), aRows, bCols);
                                var bGradArr = new T[bCols * aCols];
                                GradKernels.MatMul(gTArr.AsSpan(0, bCols * aRows), aSpan_b, bGradArr, bCols, aRows, aCols);
                                AccumulateGradient(b, NivaraColumn<T>.CreateFromOwnedArray(bGradArr));
                            }
                            finally
                            {
                                ArrayPool<T>.Shared.Return(gTArr, clearArray: true);
                            }
                        }
                    });

                    ComputationGraph.AddNode(resultTensor, gradFn);
                }

                return resultTensor;
            },
            AutoDiffDiagnostics.MatrixNote("MatMulTransposedB", aRows, aCols, bCols));
    }

    /// <summary>
    /// Fused multi-head scaled dot-product attention as a single graph node.
    ///
    /// Computes, per head, softmax(scale * (Q_h @ K_h^T) + mask) @ V_h using
    /// SIMD <see cref="GradKernels.MatMulTransposedB{T}"/> QK^T row kernels and a
    /// transpose-free path (the transposed-B layout the kernel consumes
    /// equals K's raw layout). The attention scale and the additive
    /// padding/causal mask fold into the per-row score softmax over a single
    /// [qLen, kvLen] pass — no separate Scale/Mask/Add nodes. In training one
    /// backward VJP produces dQ, dK, dV, replacing the multi-node
    /// Slice/Transpose/MatMul/Multiply/Add/Softmax/Concat chain.
    ///
    /// Query is [qLen, D]; key/value are [kvLen, D]; output is [qLen, D] with
    /// D = numHeads * headDim. <paramref name="mask"/> is an optional
    /// [qLen, kvLen] additive mask — use <c>T.NegativeInfinity</c> to suppress
    /// positions (padding/causal).
    /// </summary>
    public static ReverseGradTensor<T> MultiHeadAttention<T>(
        ReverseGradTensor<T> query,
        ReverseGradTensor<T> key,
        ReverseGradTensor<T> value,
        int numHeads,
        T scale,
        ReverseGradTensor<T>? mask = null)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (query == null) throw new ArgumentNullException(nameof(query));
        if (key == null) throw new ArgumentNullException(nameof(key));
        if (value == null) throw new ArgumentNullException(nameof(value));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (query.Rank != 2)
            throw new ArgumentException($"Query must be a matrix (rank 2), got rank {query.Rank}", nameof(query));
        if (key.Rank != 2)
            throw new ArgumentException($"Key must be a matrix (rank 2), got rank {key.Rank}", nameof(key));
        if (value.Rank != 2)
            throw new ArgumentException($"Value must be a matrix (rank 2), got rank {value.Rank}", nameof(value));

        int qLen = query.shape[0];
        int kvLen = key.shape[0];
        int D = query.shape[1];
        if (D % numHeads != 0)
            throw new ArgumentException($"Query width ({D}) must be divisible by numHeads ({numHeads}).", nameof(query));
        int headDim = D / numHeads;
        if (key.shape[1] != D || value.shape[1] != D)
            throw new ArgumentException($"Key/Value width must equal query width ({D}).");
        if (key.shape[0] != value.shape[0])
            throw new ArgumentException($"Key and Value row counts must match (got {key.shape[0]} vs {value.shape[0]}).");
        if (mask != null && (mask.Rank != 2 || mask.shape[0] != qLen || mask.shape[1] != kvLen))
            throw new ArgumentException($"Mask must be a {qLen}x{kvLen} additive matrix.");

        bool shouldTrack = GradientUtils.ShouldTrackGrad(query, key, value);
        int scoreLen = qLen * kvLen;
        int qHeadsLen = numHeads * qLen * headDim;
        int kvHeadsLen = numHeads * kvLen * headDim;

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffMultiHeadAttention",
            query.Length + key.Length + value.Length,

            () =>
            {
                var qSpan = query.AsSpan();
                var kSpan = key.AsSpan();
                var vSpan = value.AsSpan();
                var maskSpan = mask != null ? mask.AsSpan() : ReadOnlySpan<T>.Empty;

                var qHeads = ArrayPool<T>.Shared.Rent(Math.Max(qHeadsLen, 1));
                var kHeads = ArrayPool<T>.Shared.Rent(Math.Max(kvHeadsLen, 1));
                var vHeads = ArrayPool<T>.Shared.Rent(Math.Max(kvHeadsLen, 1));
                var scores = ArrayPool<T>.Shared.Rent(Math.Max(scoreLen, 1));
                var outHead = ArrayPool<T>.Shared.Rent(Math.Max(qLen * headDim, 1));
                try
                {
                    AttentionKernels<T>.PackHeads(qSpan, qHeads.AsSpan(0, qHeadsLen), qLen, numHeads, headDim);
                    AttentionKernels<T>.PackHeads(kSpan, kHeads.AsSpan(0, kvHeadsLen), kvLen, numHeads, headDim);
                    AttentionKernels<T>.PackHeads(vSpan, vHeads.AsSpan(0, kvHeadsLen), kvLen, numHeads, headDim);

                    var output = new T[qLen * D];
                    T[]? savedWeights = shouldTrack ? new T[numHeads * scoreLen] : null;

                    for (int h = 0; h < numHeads; h++)
                    {
                        var qh = qHeads.AsSpan(h * qLen * headDim, qLen * headDim);
                        var kh = kHeads.AsSpan(h * kvLen * headDim, kvLen * headDim);
                        var vh = vHeads.AsSpan(h * kvLen * headDim, kvLen * headDim);

                        GradKernels.MatMulTransposedB(qh, kh, scores, qLen, headDim, kvLen);
                        var scoresSpan = scores.AsSpan(0, scoreLen);
                        TensorPrimitives.Multiply(scoresSpan, scale, scoresSpan);
                        if (!maskSpan.IsEmpty)
                            TensorPrimitives.Add(scoresSpan, maskSpan, scoresSpan);

                        AttentionKernels<T>.SoftmaxRows(scoresSpan, qLen, kvLen);

                        if (savedWeights != null)
                            scoresSpan.CopyTo(savedWeights.AsSpan(h * scoreLen, scoreLen));

                        GradKernels.MatMul(scores, vh, outHead, qLen, kvLen, headDim);
                        AttentionKernels<T>.ScatterHead(
                            outHead.AsSpan(0, qLen * headDim), output.AsSpan(), qLen, D, h, headDim);
                    }

                    var result = new ReverseGradTensor<T>(
                        NivaraColumn<T>.CreateFromOwnedArray(output), shouldTrack, new[] { qLen, D });

                    if (shouldTrack)
                    {
                        var weights = savedWeights!;
                        var inputs = mask != null
                            ? new ReverseGradTensor<T>[] { query, key, value, mask! }
                            : new ReverseGradTensor<T>[] { query, key, value };

                        var gradFn = new OpNode<T>("MultiHeadAttention", inputs, (typedGradOutput) =>
                        {
                            var dQ = new T[qLen * D];
                            var dK = new T[kvLen * D];
                            var dV = new T[kvLen * D];

                            var qHead = ArrayPool<T>.Shared.Rent(Math.Max(qLen * headDim, 1));
                            var kHead = ArrayPool<T>.Shared.Rent(Math.Max(kvLen * headDim, 1));
                            var vHead = ArrayPool<T>.Shared.Rent(Math.Max(kvLen * headDim, 1));
                            var dOHead = ArrayPool<T>.Shared.Rent(Math.Max(qLen * headDim, 1));
                            var dQHead = ArrayPool<T>.Shared.Rent(Math.Max(qLen * headDim, 1));
                            var dKHead = ArrayPool<T>.Shared.Rent(Math.Max(kvLen * headDim, 1));
                            var dVHead = ArrayPool<T>.Shared.Rent(Math.Max(kvLen * headDim, 1));
                            var dP = ArrayPool<T>.Shared.Rent(Math.Max(scoreLen, 1));
                            var dPT = ArrayPool<T>.Shared.Rent(Math.Max(scoreLen, 1));
                            var pT = ArrayPool<T>.Shared.Rent(Math.Max(scoreLen, 1));
                            try
                            {
                                typedGradOutput.TryGetSpan(out var gradSpan);
                                var bQSpan = query.AsSpan();
                                var bKSpan = key.AsSpan();
                                var bVSpan = value.AsSpan();

                                for (int h = 0; h < numHeads; h++)
                                {
                                    var qhSpan = qHead.AsSpan(0, qLen * headDim);
                                    var khSpan = kHead.AsSpan(0, kvLen * headDim);
                                    var vhSpan = vHead.AsSpan(0, kvLen * headDim);
                                    AttentionKernels<T>.GatherHead(bQSpan, qhSpan, qLen, D, h, headDim);
                                    AttentionKernels<T>.GatherHead(bKSpan, khSpan, kvLen, D, h, headDim);
                                    AttentionKernels<T>.GatherHead(bVSpan, vhSpan, kvLen, D, h, headDim);

                                    var dOH = dOHead.AsSpan(0, qLen * headDim);
                                    AttentionKernels<T>.GatherHead(gradSpan, dOH, qLen, D, h, headDim);

                                    var pHead = weights.AsSpan(h * scoreLen, scoreLen);
                                    var dPSpan = dP.AsSpan(0, scoreLen);

                                    GradKernels.MatMulTransposedB(dOH, vhSpan, dP, qLen, headDim, kvLen);
                                    AttentionKernels<T>.SoftmaxBackwardRows(pHead, dPSpan, qLen, kvLen);
                                    TensorPrimitives.Multiply(dPSpan, scale, dPSpan);

                                    var dQH = dQHead.AsSpan(0, qLen * headDim);
                                    var dKH = dKHead.AsSpan(0, kvLen * headDim);
                                    var dVH = dVHead.AsSpan(0, kvLen * headDim);

                                    GradKernels.MatMul(dP, khSpan, dQHead, qLen, kvLen, headDim);
                                    GradKernels.Transpose(dPSpan, dPT.AsSpan(0, scoreLen), qLen, kvLen);
                                    GradKernels.MatMul(dPT, qhSpan, dKHead, kvLen, qLen, headDim);
                                    GradKernels.Transpose(pHead, pT.AsSpan(0, scoreLen), qLen, kvLen);
                                    GradKernels.MatMul(pT, dOH, dVHead, kvLen, qLen, headDim);

                                    AttentionKernels<T>.ScatterHead(dQH, dQ.AsSpan(), qLen, D, h, headDim);
                                    AttentionKernels<T>.ScatterHead(dKH, dK.AsSpan(), kvLen, D, h, headDim);
                                    AttentionKernels<T>.ScatterHead(dVH, dV.AsSpan(), kvLen, D, h, headDim);
                                }

                                if (query.RequiresGrad)
                                    AccumulateGradient(query, NivaraColumn<T>.CreateFromOwnedArray(dQ));
                                if (key.RequiresGrad)
                                    AccumulateGradient(key, NivaraColumn<T>.CreateFromOwnedArray(dK));
                                if (value.RequiresGrad)
                                    AccumulateGradient(value, NivaraColumn<T>.CreateFromOwnedArray(dV));
                            }
                            finally
                            {
                                ArrayPool<T>.Shared.Return(qHead, clearArray: true);
                                ArrayPool<T>.Shared.Return(kHead, clearArray: true);
                                ArrayPool<T>.Shared.Return(vHead, clearArray: true);
                                ArrayPool<T>.Shared.Return(dOHead, clearArray: true);
                                ArrayPool<T>.Shared.Return(dQHead, clearArray: true);
                                ArrayPool<T>.Shared.Return(dKHead, clearArray: true);
                                ArrayPool<T>.Shared.Return(dVHead, clearArray: true);
                                ArrayPool<T>.Shared.Return(dP, clearArray: true);
                                ArrayPool<T>.Shared.Return(dPT, clearArray: true);
                                ArrayPool<T>.Shared.Return(pT, clearArray: true);
                            }
                        });

                        ComputationGraph.AddNode(result, gradFn);
                    }

                    return result;
                }
                finally
                {
                    ArrayPool<T>.Shared.Return(qHeads, clearArray: true);
                    ArrayPool<T>.Shared.Return(kHeads, clearArray: true);
                    ArrayPool<T>.Shared.Return(vHeads, clearArray: true);
                    ArrayPool<T>.Shared.Return(scores, clearArray: true);
                    ArrayPool<T>.Shared.Return(outHead, clearArray: true);
                }
            },
            AutoDiffDiagnostics.ShapeNote("MultiHeadAttention", [qLen, D]));
    }

    /// <summary>
    /// Batched fused multi-head scaled dot-product attention as a single graph
    /// node. The batch overload of <see cref="MultiHeadAttention{T}"/>: query is
    /// [B, qLen, D], key/value are [B, kvLen, D], output is [B, qLen, D] with
    /// D = numHeads * headDim. Each batch element runs the identical per-head
    /// fused kernel (pack heads, QK^T via the transposed-B layout, scale/mask,
    /// softmax rows, scores @ V, scatter) so the batch never flattens into
    /// cross-sequence scores. The batch dimension loops internally and
    /// parallelizes over B when the total work justifies it
    /// (<see cref="ShouldParallelizeBatch"/>); the single-sequence op is
    /// untouched and bit-identical.
    ///
    /// <paramref name="mask"/> is an optional [B, qLen, kvLen] additive mask —
    /// use <c>T.NegativeInfinity</c> to suppress positions (padding/causal),
    /// applied per batch element.
    /// </summary>
    public static ReverseGradTensor<T> BatchedMultiHeadAttention<T>(
        ReverseGradTensor<T> query,
        ReverseGradTensor<T> key,
        ReverseGradTensor<T> value,
        int numHeads,
        T scale,
        ReverseGradTensor<T>? mask = null)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (query == null) throw new ArgumentNullException(nameof(query));
        if (key == null) throw new ArgumentNullException(nameof(key));
        if (value == null) throw new ArgumentNullException(nameof(value));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (query.Rank != 3)
            throw new ArgumentException($"Query must be [B, L, D] (rank 3), got rank {query.Rank}", nameof(query));
        if (key.Rank != 3)
            throw new ArgumentException($"Key must be [B, L, D] (rank 3), got rank {key.Rank}", nameof(key));
        if (value.Rank != 3)
            throw new ArgumentException($"Value must be [B, L, D] (rank 3), got rank {value.Rank}", nameof(value));

        int batch = query.shape[0];
        int qLen = query.shape[1];
        int kvLen = key.shape[1];
        int D = query.shape[2];
        if (D % numHeads != 0)
            throw new ArgumentException($"Query width ({D}) must be divisible by numHeads ({numHeads}).", nameof(query));
        int headDim = D / numHeads;
        if (key.shape[2] != D || value.shape[2] != D)
            throw new ArgumentException($"Key/Value width must equal query width ({D}).");
        if (key.shape[0] != batch || value.shape[0] != batch)
            throw new ArgumentException($"Key/Value batch size must equal query batch size ({batch}).");
        if (key.shape[1] != value.shape[1])
            throw new ArgumentException($"Key and Value sequence lengths must match (got {key.shape[1]} vs {value.shape[1]}).");
        if (mask != null && (mask.Rank != 3 || mask.shape[0] != batch || mask.shape[1] != qLen || mask.shape[2] != kvLen))
            throw new ArgumentException($"Mask must be a {batch}x{qLen}x{kvLen} additive tensor.");

        bool shouldTrack = GradientUtils.ShouldTrackGrad(query, key, value);
        int scoreLen = qLen * kvLen;
        int qHeadsLen = numHeads * qLen * headDim;
        int kvHeadsLen = numHeads * kvLen * headDim;

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffBatchedMultiHeadAttention",
            query.Length + key.Length + value.Length,

            () =>
            {
                var qSpan = query.AsSpan();
                var kSpan = key.AsSpan();
                var vSpan = value.AsSpan();

                var qHeads = ArrayPool<T>.Shared.Rent(Math.Max(batch * qHeadsLen, 1));
                var kHeads = ArrayPool<T>.Shared.Rent(Math.Max(batch * kvHeadsLen, 1));
                var vHeads = ArrayPool<T>.Shared.Rent(Math.Max(batch * kvHeadsLen, 1));
                try
                {
                    int qRowLen = qLen * D;
                    int kvRowLen = kvLen * D;
                    for (int b = 0; b < batch; b++)
                    {
                        AttentionKernels<T>.PackHeads(
                            qSpan.Slice(b * qRowLen, qRowLen),
                            qHeads.AsSpan(b * qHeadsLen, qHeadsLen), qLen, numHeads, headDim);
                        AttentionKernels<T>.PackHeads(
                            kSpan.Slice(b * kvRowLen, kvRowLen),
                            kHeads.AsSpan(b * kvHeadsLen, kvHeadsLen), kvLen, numHeads, headDim);
                        AttentionKernels<T>.PackHeads(
                            vSpan.Slice(b * kvRowLen, kvRowLen),
                            vHeads.AsSpan(b * kvHeadsLen, kvHeadsLen), kvLen, numHeads, headDim);
                    }

                    var output = new T[batch * qLen * D];
                    T[]? savedWeights = shouldTrack ? new T[batch * numHeads * scoreLen] : null;

                    bool parallel = ShouldParallelizeBatch(batch, (long)qLen * kvLen * headDim * 2);
                    if (parallel)
                        Parallel.For(0, batch, b => ProcessBatchForward(b, mask));
                    else
                        for (int b = 0; b < batch; b++)
                            ProcessBatchForward(b, mask);

                    void ProcessBatchForward(int b, ReverseGradTensor<T>? maskTensor)
                    {
                        var maskSpan = maskTensor != null ? maskTensor.AsSpan() : ReadOnlySpan<T>.Empty;
                        var scores = ArrayPool<T>.Shared.Rent(Math.Max(scoreLen, 1));
                        var outHead = ArrayPool<T>.Shared.Rent(Math.Max(qLen * headDim, 1));
                        try
                        {
                            var qhBatch = qHeads.AsSpan(b * qHeadsLen, qHeadsLen);
                            var kvhBatch = kHeads.AsSpan(b * kvHeadsLen, kvHeadsLen);
                            var vBatch = vHeads.AsSpan(b * kvHeadsLen, kvHeadsLen);
                            int outOff = b * qLen * D;
                            int wOff = b * numHeads * scoreLen;

                            for (int h = 0; h < numHeads; h++)
                            {
                                var qh = qhBatch.Slice(h * qLen * headDim, qLen * headDim);
                                var kh = kvhBatch.Slice(h * kvLen * headDim, kvLen * headDim);
                                var vh = vBatch.Slice(h * kvLen * headDim, kvLen * headDim);

                                GradKernels.MatMulTransposedB(qh, kh, scores, qLen, headDim, kvLen);
                                var scoresSpan = scores.AsSpan(0, scoreLen);
                                TensorPrimitives.Multiply(scoresSpan, scale, scoresSpan);
                                if (!maskSpan.IsEmpty)
                                    TensorPrimitives.Add(scoresSpan, maskSpan.Slice(b * scoreLen, scoreLen), scoresSpan);

                                AttentionKernels<T>.SoftmaxRows(scoresSpan, qLen, kvLen);

                                if (savedWeights != null)
                                    scoresSpan.CopyTo(savedWeights.AsSpan(wOff + h * scoreLen, scoreLen));

                                GradKernels.MatMul(scores, vh, outHead, qLen, kvLen, headDim);
                                AttentionKernels<T>.ScatterHead(
                                    outHead.AsSpan(0, qLen * headDim), output.AsSpan(outOff, qLen * D), qLen, D, h, headDim);
                            }
                        }
                        finally
                        {
                            ArrayPool<T>.Shared.Return(scores, clearArray: true);
                            ArrayPool<T>.Shared.Return(outHead, clearArray: true);
                        }
                    }

                    var result = new ReverseGradTensor<T>(
                        NivaraColumn<T>.CreateFromOwnedArray(output), shouldTrack, new[] { batch, qLen, D });

                    if (shouldTrack)
                    {
                        var weights = savedWeights!;
                        var inputs = mask != null
                            ? new ReverseGradTensor<T>[] { query, key, value, mask! }
                            : new ReverseGradTensor<T>[] { query, key, value };

                        var gradFn = new OpNode<T>("BatchedMultiHeadAttention", inputs, (typedGradOutput) =>
                        {
                            var dQ = new T[batch * qLen * D];
                            var dK = new T[batch * kvLen * D];
                            var dV = new T[batch * kvLen * D];

                            typedGradOutput.TryGetSpan(out var gradSpan);

                            bool gradParallel = ShouldParallelizeBatch(batch, (long)qLen * kvLen * headDim * 2);
                            if (gradParallel)
                                Parallel.For(0, batch, b => ProcessBatchGrad(b, query, key, value, typedGradOutput));
                            else
                                for (int b = 0; b < batch; b++)
                                    ProcessBatchGrad(b, query, key, value, typedGradOutput);

                            if (query.RequiresGrad)
                                AccumulateGradient(query, NivaraColumn<T>.CreateFromOwnedArray(dQ));
                            if (key.RequiresGrad)
                                AccumulateGradient(key, NivaraColumn<T>.CreateFromOwnedArray(dK));
                            if (value.RequiresGrad)
                                AccumulateGradient(value, NivaraColumn<T>.CreateFromOwnedArray(dV));

                            void ProcessBatchGrad(
                                int b,
                                ReverseGradTensor<T> qTensor,
                                ReverseGradTensor<T> kTensor,
                                ReverseGradTensor<T> vTensor,
                                NivaraColumn<T> gradColumn)
                            {
                                var bQSpan = qTensor.AsSpan();
                                var bKSpan = kTensor.AsSpan();
                                var bVSpan = vTensor.AsSpan();
                                gradColumn.TryGetSpan(out var bGradSpan);
                                var qHead = ArrayPool<T>.Shared.Rent(Math.Max(qLen * headDim, 1));
                                var kHead = ArrayPool<T>.Shared.Rent(Math.Max(kvLen * headDim, 1));
                                var vHead = ArrayPool<T>.Shared.Rent(Math.Max(kvLen * headDim, 1));
                                var dOHead = ArrayPool<T>.Shared.Rent(Math.Max(qLen * headDim, 1));
                                var dQHead = ArrayPool<T>.Shared.Rent(Math.Max(qLen * headDim, 1));
                                var dKHead = ArrayPool<T>.Shared.Rent(Math.Max(kvLen * headDim, 1));
                                var dVHead = ArrayPool<T>.Shared.Rent(Math.Max(kvLen * headDim, 1));
                                var dP = ArrayPool<T>.Shared.Rent(Math.Max(scoreLen, 1));
                                var dPT = ArrayPool<T>.Shared.Rent(Math.Max(scoreLen, 1));
                                var pT = ArrayPool<T>.Shared.Rent(Math.Max(scoreLen, 1));
                                try
                                {
                                    bQSpan = bQSpan.Slice(b * qLen * D, qLen * D);
                                    bKSpan = bKSpan.Slice(b * kvLen * D, kvLen * D);
                                    bVSpan = bVSpan.Slice(b * kvLen * D, kvLen * D);
                                    bGradSpan = bGradSpan.Slice(b * qLen * D, qLen * D);
                                    int dQOff = b * qLen * D;
                                    int dKOff = b * kvLen * D;
                                    int dVOff = b * kvLen * D;
                                    int wOff = b * numHeads * scoreLen;

                                    for (int h = 0; h < numHeads; h++)
                                    {
                                        var qhSpan = qHead.AsSpan(0, qLen * headDim);
                                        var khSpan = kHead.AsSpan(0, kvLen * headDim);
                                        var vhSpan = vHead.AsSpan(0, kvLen * headDim);
                                        AttentionKernels<T>.GatherHead(bQSpan, qhSpan, qLen, D, h, headDim);
                                        AttentionKernels<T>.GatherHead(bKSpan, khSpan, kvLen, D, h, headDim);
                                        AttentionKernels<T>.GatherHead(bVSpan, vhSpan, kvLen, D, h, headDim);

                                        var dOH = dOHead.AsSpan(0, qLen * headDim);
                                        AttentionKernels<T>.GatherHead(bGradSpan, dOH, qLen, D, h, headDim);

                                        var pHead = weights.AsSpan(wOff + h * scoreLen, scoreLen);
                                        var dPSpan = dP.AsSpan(0, scoreLen);

                                        GradKernels.MatMulTransposedB(dOH, vhSpan, dP, qLen, headDim, kvLen);
                                        AttentionKernels<T>.SoftmaxBackwardRows(pHead, dPSpan, qLen, kvLen);
                                        TensorPrimitives.Multiply(dPSpan, scale, dPSpan);

                                        var dQH = dQHead.AsSpan(0, qLen * headDim);
                                        var dKH = dKHead.AsSpan(0, kvLen * headDim);
                                        var dVH = dVHead.AsSpan(0, kvLen * headDim);

                                        GradKernels.MatMul(dP, khSpan, dQHead, qLen, kvLen, headDim);
                                        GradKernels.Transpose(dPSpan, dPT.AsSpan(0, scoreLen), qLen, kvLen);
                                        GradKernels.MatMul(dPT, qhSpan, dKHead, kvLen, qLen, headDim);
                                        GradKernels.Transpose(pHead, pT.AsSpan(0, scoreLen), qLen, kvLen);
                                        GradKernels.MatMul(pT, dOH, dVHead, kvLen, qLen, headDim);

                                        AttentionKernels<T>.ScatterHead(dQH, dQ.AsSpan(dQOff, qLen * D), qLen, D, h, headDim);
                                        AttentionKernels<T>.ScatterHead(dKH, dK.AsSpan(dKOff, kvLen * D), kvLen, D, h, headDim);
                                        AttentionKernels<T>.ScatterHead(dVH, dV.AsSpan(dVOff, kvLen * D), kvLen, D, h, headDim);
                                    }
                                }
                                finally
                                {
                                    ArrayPool<T>.Shared.Return(qHead, clearArray: true);
                                    ArrayPool<T>.Shared.Return(kHead, clearArray: true);
                                    ArrayPool<T>.Shared.Return(vHead, clearArray: true);
                                    ArrayPool<T>.Shared.Return(dOHead, clearArray: true);
                                    ArrayPool<T>.Shared.Return(dQHead, clearArray: true);
                                    ArrayPool<T>.Shared.Return(dKHead, clearArray: true);
                                    ArrayPool<T>.Shared.Return(dVHead, clearArray: true);
                                    ArrayPool<T>.Shared.Return(dP, clearArray: true);
                                    ArrayPool<T>.Shared.Return(dPT, clearArray: true);
                                    ArrayPool<T>.Shared.Return(pT, clearArray: true);
                                }
                            }
                        });

                        ComputationGraph.AddNode(result, gradFn);
                    }

                    return result;
                }
                finally
                {
                    ArrayPool<T>.Shared.Return(qHeads, clearArray: true);
                    ArrayPool<T>.Shared.Return(kHeads, clearArray: true);
                    ArrayPool<T>.Shared.Return(vHeads, clearArray: true);
                }
            },
            AutoDiffDiagnostics.ShapeNote("BatchedMultiHeadAttention", [batch, qLen, D]));
    }

    static bool ShouldParallelizeBatch(int batch, long workPerBatch)
        => batch >= 4 && batch * workPerBatch >= 2L << 20;

    /// <summary>
    /// Transposes a rank-2 tensor, producing shape [cols, rows].
    /// </summary>
    /// <param name="a">The matrix to transpose</param>
    /// <returns>A new tensor holding the transposed matrix</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    /// <exception cref="ArgumentException">Thrown when the input is not rank 2</exception>
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
                GradKernels.Transpose(aSpan, resultArr.AsSpan(), rows, cols);

                var resultShape = new[] { cols, rows };
                var resultTensor = new ReverseGradTensor<T>(NivaraColumn<T>.CreateFromOwnedArray(resultArr), GradientUtils.ShouldTrackGrad(a), resultShape);

                if (GradientUtils.ShouldTrackGrad(a))
                {
                    var gradFn = new OpNode<T>("Transpose", [a], (typedGradOutput) =>
                    {
                        var gArr = new T[cols * rows];
                        typedGradOutput.TryGetSpan(out var gradSpan);
                        GradKernels.Transpose(gradSpan, gArr.AsSpan(), cols, rows);
                        AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(gArr));
                    });

                    ComputationGraph.AddNode(resultTensor, gradFn);
                }

                return resultTensor;
            },
            $"AutoDiff=Transpose;Shape={rows}x{cols}->{cols}x{rows}");
    }

    /// <summary>
    /// Swaps two axes of a rank-2 or rank-3 tensor.
    /// </summary>
    /// <param name="a">The tensor to permute</param>
    /// <param name="axis1">The first axis to swap</param>
    /// <param name="axis2">The second axis to swap</param>
    /// <returns>A new tensor with the two axes swapped</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    /// <exception cref="ArgumentException">Thrown when the input is not rank 2-3 or the axes coincide</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when an axis is out of range</exception>
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

                var resultCol = NivaraColumn<T>.CreateFromOwnedArray(dstData);
                bool shouldTrack = GradientUtils.ShouldTrackGrad(a);
                var resultTensor = new ReverseGradTensor<T>(resultCol, shouldTrack, dstDims);

                if (shouldTrack)
                {
                    int capturedAxis1 = axis1, capturedAxis2 = axis2;
                    int[] capturedDstDims = dstDims;
                    var gradFn = new OpNode<T>("TransposeAxes", [a], (typedGradOutput) =>
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

                        AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(gDst));
                    });

                    ComputationGraph.AddNode(resultTensor, gradFn);
                }

                return resultTensor;
            }, note);
    }

    #endregion

    #region Reduction Operations

    /// <summary>
    /// Reduces a tensor to a scalar sum.
    /// </summary>
    /// <param name="a">The tensor to reduce</param>
    /// <returns>A scalar (length-1) tensor holding the sum</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when the tensor is empty</exception>
    public static ReverseGradTensor<T> Sum<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (a.Length == 0)
        {
            throw new InvalidOperationException("Cannot compute sum of empty tensor");
        }

        var sumValue = TensorPrimitives.Sum(a.AsSpan());

        var resultData = NivaraColumn<T>.CreateFromOwnedArray(new T[] { sumValue });
        var resultTensor = new ReverseGradTensor<T>(resultData, GradientUtils.ShouldTrackGrad(a), ScalarShape());

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Sum", [a], (typedGradOutput) =>
            {
                var aGrad = GradOperationKernels.BroadcastGradient(typedGradOutput, a.Length);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Reduces a tensor to a scalar mean.
    /// </summary>
    /// <param name="a">The tensor to reduce</param>
    /// <returns>A scalar (length-1) tensor holding the mean</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when the tensor is empty</exception>
    public static ReverseGradTensor<T> Mean<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (a.Length == 0)
        {
            throw new InvalidOperationException("Cannot compute mean of empty tensor");
        }

        var meanValue = TensorPrimitives.Sum(a.AsSpan()) / T.CreateChecked(a.Length);

        var resultData = NivaraColumn<T>.CreateFromOwnedArray(new T[] { meanValue });
        var resultTensor = new ReverseGradTensor<T>(resultData, GradientUtils.ShouldTrackGrad(a), ScalarShape());

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Mean", [a], (typedGradOutput) =>
            {
                var aGrad = GradOperationKernels.BroadcastGradient(typedGradOutput, a.Length);
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

        var resultCol = NivaraColumn<T>.CreateFromOwnedArray(resultValues);
        var result = new ReverseGradTensor<T>(resultCol, GradientUtils.ShouldTrackGrad(a), [batchSize, embedDim]);

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("MeanPool", [a], (typedGradOutput) =>
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

                var gradCol = NivaraColumn<T>.CreateFromOwnedArray(gradOut);
                AccumulateGradient(a, gradCol);
            });

            ComputationGraph.AddNode(result, gradFn);
        }

        return result;
    }

    #endregion

    #region Activation Functions

    /// <summary>
    /// Applies the rectified linear unit: result[i] = max(0, a[i]).
    /// </summary>
    /// <param name="a">The input tensor</param>
    /// <returns>A new tensor with ReLU applied element-wise</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    public static ReverseGradTensor<T> Relu<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var resultArr = new T[a.Length];
        GradKernels.Relu(a.AsSpan(), resultArr);

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Relu", [a], (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                var gradArr = new T[a.Length];
                GradKernels.ReluGradient(a.AsSpan(), gSpan, gradArr);
                AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Applies the Gaussian error linear unit (tanh approximation).
    /// </summary>
    /// <param name="a">The input tensor</param>
    /// <returns>A new tensor with GELU applied element-wise</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    public static ReverseGradTensor<T> Gelu<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        bool trackGrad = GradientUtils.ShouldTrackGrad(a);

        var resultArr = new T[a.Length];
        GradKernels.Gelu(a.AsSpan(), resultArr);

        var resultTensor = ResultTensor(resultArr, a, trackGrad);

        if (trackGrad)
        {
            var gradFn = new OpNode<T>("Gelu", [a], (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                var gradArr = new T[a.Length];
                GradKernels.GeluGradient(a.AsSpan(), gSpan, gradArr);
                AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Applies the exact Gaussian error linear unit using the error function.
    /// </summary>
    /// <param name="a">The input tensor</param>
    /// <returns>A new tensor with exact GELU applied element-wise</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    public static ReverseGradTensor<T> GeluExact<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        bool trackGrad = GradientUtils.ShouldTrackGrad(a);

        var resultArr = new T[a.Length];
        GradKernels.GeluExact(a.AsSpan(), resultArr);

        var resultTensor = ResultTensor(resultArr, a, trackGrad);

        if (trackGrad)
        {
            var aArr = new T[a.Length];
            a.AsSpan().CopyTo(aArr.AsSpan());

            var gradFn = new OpNode<T>("GeluExact", [a], (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gradSpan);
                var gradArr = new T[a.Length];
                GradKernels.GeluExactGradient(aArr, gradSpan, gradArr);
                AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Applies the logistic sigmoid element-wise: result[i] = 1 / (1 + exp(-a[i])).
    /// </summary>
    /// <param name="a">The input tensor</param>
    /// <returns>A new tensor with sigmoid applied element-wise</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    public static ReverseGradTensor<T> Sigmoid<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var resultArr = new T[a.Length];
        GradKernels.Sigmoid(a.AsSpan(), resultArr);

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Sigmoid", [a], (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                var gradArr = new T[a.Length];
                GradKernels.SigmoidGradient(resultArr, gSpan, gradArr);
                AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Applies the hyperbolic tangent element-wise.
    /// </summary>
    /// <param name="a">The input tensor</param>
    /// <returns>A new tensor with tanh applied element-wise</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    public static ReverseGradTensor<T> Tanh<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var resultArr = new T[a.Length];
        GradKernels.Tanh(a.AsSpan(), resultArr);

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Tanh", [a], (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                var gradArr = new T[a.Length];
                GradKernels.TanhGradient(resultArr, gSpan, gradArr);
                AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Negates a tensor element-wise: result[i] = -a[i].
    /// </summary>
    /// <param name="a">The input tensor</param>
    /// <returns>A new tensor with all elements negated</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    public static ReverseGradTensor<T> Negate<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var resultArr = new T[a.Length];
        GradKernels.Negate(a.AsSpan(), resultArr);

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Negate", [a], (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                var gradArr = new T[a.Length];
                GradKernels.Negate(gSpan, gradArr);
                AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Applies the absolute value element-wise.
    /// </summary>
    /// <param name="a">The input tensor</param>
    /// <returns>A new tensor with absolute values</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    public static ReverseGradTensor<T> Abs<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var resultArr = new T[a.Length];
        GradKernels.Abs(a.AsSpan(), resultArr);

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Abs", [a], (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                var gradArr = new T[a.Length];
                GradKernels.AbsGradient(a.AsSpan(), gSpan, gradArr);
                AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Clamps a tensor to [min, max] element-wise. Gradients are zero outside the range.
    /// </summary>
    /// <param name="a">The input tensor</param>
    /// <param name="min">The lower bound</param>
    /// <param name="max">The upper bound</param>
    /// <returns>A new tensor with elements clamped to the range</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    public static ReverseGradTensor<T> Clip<T>(ReverseGradTensor<T> a, T min, T max)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var resultArr = new T[a.Length];
        GradKernels.Clamp(a.AsSpan(), min, max, resultArr);

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Clip", [a], (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                var gradArr = new T[a.Length];
                GradKernels.ClipGradient(a.AsSpan(), gSpan, min, max, gradArr);
                AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Applies leaky ReLU element-wise: result[i] = a[i] for a[i] &gt;= 0, else negativeSlope * a[i].
    /// A zero slope defaults to 0.01.
    /// </summary>
    /// <param name="a">The input tensor</param>
    /// <param name="negativeSlope">The slope for negative inputs; zero is treated as 0.01</param>
    /// <returns>A new tensor with leaky ReLU applied</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    public static ReverseGradTensor<T> LeakyRelu<T>(ReverseGradTensor<T> a, T negativeSlope = default)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (negativeSlope == T.Zero)
            negativeSlope = T.CreateChecked(0.01);

        var resultArr = new T[a.Length];
        GradKernels.LeakyRelu(a.AsSpan(), negativeSlope, resultArr);

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("LeakyRelu", [a], (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                var gradArr = new T[a.Length];
                GradKernels.LeakyReluGradient(a.AsSpan(), gSpan, negativeSlope, gradArr);
                AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Applies the exponential function element-wise.
    /// </summary>
    /// <param name="a">The input tensor</param>
    /// <returns>A new tensor with exp applied element-wise</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    public static ReverseGradTensor<T> Exp<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var resultArr = new T[a.Length];
        GradKernels.Exp(a.AsSpan(), resultArr);

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Exp", [a], (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                var gradArr = new T[a.Length];
                TensorPrimitives.Multiply(gSpan, resultArr.AsSpan(), gradArr.AsSpan());
                AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Applies the natural logarithm element-wise.
    /// </summary>
    /// <param name="a">The input tensor</param>
    /// <returns>A new tensor with log applied element-wise</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    public static ReverseGradTensor<T> Log<T>(ReverseGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var resultArr = new T[a.Length];
        GradKernels.Log(a.AsSpan(), resultArr);

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Log", [a], (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                var gradArr = new T[a.Length];
                GradKernels.LogGradient(a.AsSpan(), gSpan, gradArr);
                AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(gradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Raises a tensor to a scalar power element-wise: result[i] = a[i]^exponent.
    /// </summary>
    /// <param name="a">The input tensor</param>
    /// <param name="exponent">The scalar exponent</param>
    /// <returns>A new tensor with each element raised to the power</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    public static ReverseGradTensor<T> Pow<T>(ReverseGradTensor<T> a, double exponent) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var resultTensor = new ReverseGradTensor<T>(
            GradOperationKernels.ApplyPow(a.Data, exponent),
            GradientUtils.ShouldTrackGrad(a),
            a.shape);

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Pow", [a], (typedGradOutput) =>
            {
                AccumulateGradient(a, GradOperationKernels.ApplyPowGradient(a.Data, typedGradOutput, exponent));
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

                var resultCol = NivaraColumn<T>.CreateFromOwnedArray(resultValues);

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

                        var gradCol = NivaraColumn<T>.CreateFromOwnedArray(gradResult);
                        AccumulateGradient(a, gradCol);
                    });

                    ComputationGraph.AddNode(result, gradFn);
                }

                return result;
            },
            $"AutoDiff=Slice;Start={start};Length={length};FullDim={fullDim}");
    }

    /// <summary>
    /// Concatenates 1D or 2D tensors along an axis. For 2D input, axis 0 joins rows and
    /// axis 1 joins columns.
    /// </summary>
    /// <param name="tensors">The tensors to concatenate (at least one)</param>
    /// <param name="axis">The axis to concatenate along (0 or 1 for 2D input)</param>
    /// <returns>A new tensor holding the concatenated data</returns>
    /// <exception cref="ArgumentException">Thrown when no tensors are given, ranks differ, or the non-concatenation dimensions mismatch</exception>
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

                    var resultCol = NivaraColumn<T>.CreateFromOwnedArray(resultData);
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

                                    var gradCol = NivaraColumn<T>.CreateFromOwnedArray(gradSlice);
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

                    var resultCol = NivaraColumn<T>.CreateFromOwnedArray(resultData);
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
                                            var gradCol = NivaraColumn<T>.CreateFromOwnedArray(gradData);
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
                                            var gradCol = NivaraColumn<T>.CreateFromOwnedArray(gradData);
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
                                            var gradCol = NivaraColumn<T>.CreateFromOwnedArray(gradData);
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
                                            var gradCol = NivaraColumn<T>.CreateFromOwnedArray(gradData);
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

    /// <summary>
    /// Applies softmax along a dimension, producing a probability distribution per slice.
    /// A negative <paramref name="dim"/> is resolved relative to the tensor rank.
    /// </summary>
    /// <param name="a">The input tensor</param>
    /// <param name="dim">The dimension to normalize (default -1, the last dimension)</param>
    /// <returns>A new tensor with softmax applied</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the dimension is out of range</exception>
    public static ReverseGradTensor<T> Softmax<T>(ReverseGradTensor<T> a, int dim = -1) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        ResolveDim(a, dim, "Softmax", out int outer, out int classCount, out int inner);
        var resultArr = new T[a.Length];
        if (inner == 1)
        {
            GradKernels.Softmax(a.AsSpan(), resultArr, classCount);
        }
        else
        {
            GradKernels.SoftmaxDim(a.AsSpan(), resultArr, outer, classCount, inner);
        }

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("Softmax", [a], (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                var aGradArr = new T[a.Length];
                if (inner == 1)
                {
                    GradKernels.SoftmaxGradient(resultArr, gSpan, aGradArr, classCount);
                }
                else
                {
                    GradKernels.SoftmaxDimGradient(resultArr, gSpan, aGradArr, outer, classCount, inner);
                }

                AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(aGradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Applies log-softmax along a dimension, producing log probabilities per slice.
    /// A negative <paramref name="dim"/> is resolved relative to the tensor rank.
    /// </summary>
    /// <param name="a">The input tensor</param>
    /// <param name="dim">The dimension to normalize (default -1, the last dimension)</param>
    /// <returns>A new tensor with log-softmax applied</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the dimension is out of range</exception>
    public static ReverseGradTensor<T> LogSoftmax<T>(ReverseGradTensor<T> a, int dim = -1) where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        ResolveDim(a, dim, "LogSoftmax", out int outer, out int classCount, out int inner);
        var resultArr = new T[a.Length];
        if (inner == 1)
        {
            GradKernels.LogSoftmax(a.AsSpan(), resultArr, classCount);
        }
        else
        {
            GradKernels.LogSoftmaxDim(a.AsSpan(), resultArr, outer, classCount, inner);
        }

        var resultTensor = ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a));

        if (GradientUtils.ShouldTrackGrad(a))
        {
            var gradFn = new OpNode<T>("LogSoftmax", [a], (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gSpan);
                var aGradArr = new T[a.Length];
                if (inner == 1)
                {
                    GradKernels.LogSoftmaxGradient(a.AsSpan(), gSpan, aGradArr, classCount);
                }
                else
                {
                    GradKernels.LogSoftmaxDimGradient(a.AsSpan(), gSpan, aGradArr, outer, classCount, inner);
                }

                AccumulateGradient(a, NivaraColumn<T>.CreateFromOwnedArray(aGradArr));
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Resolves a (possibly negative) softmax dimension into a strided layout:
    /// <c>outer</c> blocks, <c>classCount</c> elements per slice spaced <c>inner</c>
    /// apart. When <c>inner == 1</c> the layout is row-major contiguous.
    /// </summary>
    static void ResolveDim<T>(ReverseGradTensor<T> a, int dim, string operationName, out int outer, out int classCount, out int inner)
        where T : struct, IFloatingPointIeee754<T>
    {
        int rank = a.Rank;
        int originalDim = dim;
        if (dim < 0)
            dim += rank;
        if (dim < 0 || dim >= rank)
            throw new ArgumentOutOfRangeException(
                nameof(dim),
                $"{operationName} dim {originalDim} is out of range for a {rank}D tensor.");

        outer = 1;
        for (int i = 0; i < dim; i++)
            outer *= a.shape[i];

        classCount = a.shape[dim];

        inner = 1;
        for (int i = dim + 1; i < rank; i++)
            inner *= a.shape[i];
    }

    /// <summary>
    /// Normalizes the whole tensor by its root-mean-square, scaled by 1/sqrt(eps + mean(x^2)).
    /// </summary>
    /// <param name="a">The input tensor</param>
    /// <param name="eps">Small constant added to the denominator for numerical stability</param>
    /// <returns>A new tensor with RMS normalization applied</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
    public static ReverseGradTensor<T> RMSNorm<T>(ReverseGradTensor<T> a, double eps = 1e-5)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffRMSNorm",
            a.Length,

            () =>
            {
                var result = GradOperationKernels.ApplyRMSNorm(a.Data, eps);
                var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(a), a.shape);

                if (GradientUtils.ShouldTrackGrad(a))
                {
                    var savedInput = a.Data;
                    var savedEps = eps;
                    var gradFn = new OpNode<T>("RMSNorm", [a], (typedGradOutput) =>
                    {
                        var aGrad = GradOperationKernels.ApplyRMSNormGradient(savedInput, typedGradOutput, savedEps);
                        AccumulateGradient(a, aGrad);
                    });

                    ComputationGraph.AddNode(resultTensor, gradFn);
                }

                return resultTensor;
            },
            AutoDiffDiagnostics.ShapeNote("RMSNorm", a.Shape));
    }

    /// <summary>
    /// Applies RMS normalization independently to each row of a [rows, cols] tensor.
    /// </summary>
    /// <param name="a">The input tensor</param>
    /// <param name="rows">The number of rows</param>
    /// <param name="cols">The number of columns</param>
    /// <param name="eps">Small constant added to each row's denominator</param>
    /// <returns>A new tensor with per-row RMS normalization applied</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is null</exception>
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

                var resultCol = NivaraColumn<T>.CreateFromOwnedArray(resultData);
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

                        var gradCol = NivaraColumn<T>.CreateFromOwnedArray(gradResult);
                        AccumulateGradient(a, gradCol);
                    });

                    ComputationGraph.AddNode(result, gradFn);
                }

                return result;
            },
            AutoDiffDiagnostics.ShapeNote("PerRowRMSNorm", a.Shape));
    }

    /// <summary>
    /// Applies dropout: in training, elements are zeroed with the given probability and the
    /// survivors are scaled by 1/(1-p). In evaluation (or with zero probability), the input
    /// is returned unchanged.
    /// </summary>
    /// <param name="input">The input tensor</param>
    /// <param name="probability">Dropout probability in [0, 1)</param>
    /// <param name="isTraining">Whether to apply dropout; false returns the input unchanged</param>
    /// <returns>The dropout-masked tensor (the same instance when not applied)</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is null</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the probability is outside [0, 1)</exception>
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
        var result = GradOperationKernels.ApplyDropout(input.Data, savedMask, scale);
        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(input), input.shape);

        if (GradientUtils.ShouldTrackGrad(input))
        {
            var gradFn = new OpNode<T>("Dropout", [input], (typedGradOutput) =>
            {
                var inputGrad = GradOperationKernels.ApplyDropoutGradient(input.Data, typedGradOutput, savedMask, scale);
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

        var resultColumn = NivaraColumn<T>.CreateFromOwnedArray(resultValues);

        var result = new ReverseGradTensor<T>(
            resultColumn,
            GradientUtils.ShouldTrackGrad(weight),
            new[] { batchSize, embeddingDim });

        if (GradientUtils.ShouldTrackGrad(weight))
        {
            var savedIndices = parsedIndices;
            var gradFn = new OpNode<T>("SparseEmbeddingBag", [weight], (typedGradOutput) =>
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

                AccumulateGradient(weight, NivaraColumn<T>.CreateFromOwnedArray(weightGrad));
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

                var resultCol = NivaraColumn<T>.CreateFromOwnedArray(resultValues);

                var resultShape = new int[source.shape.Length];
                resultShape[0] = indices.Length;
                for (int d = 1; d < source.shape.Length; d++)
                    resultShape[d] = source.shape[d];

                var result = new ReverseGradTensor<T>(resultCol, GradientUtils.ShouldTrackGrad(source), resultShape);

                if (GradientUtils.ShouldTrackGrad(source))
                {
                    var savedIndices = indices;
                    var gradFn = new OpNode<T>("Gather", [source], (typedGradOutput) =>
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

    /// <summary>
    /// Computes the Kullback-Leibler divergence of the reparameterized Gaussian
    /// N(mean, exp(logVar)) from a standard normal, summed to a scalar.
    /// </summary>
    /// <param name="mean">The mean tensor</param>
    /// <param name="logVar">The log-variance tensor (same length as <paramref name="mean"/>)</param>
    /// <returns>A scalar (length-1) tensor holding the summed KL divergence</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null</exception>
    /// <exception cref="ArgumentException">Thrown when the operand lengths differ</exception>
    public static ReverseGradTensor<T> KlDivergence<T>(ReverseGradTensor<T> mean, ReverseGradTensor<T> logVar)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (mean == null) throw new ArgumentNullException(nameof(mean));
        if (logVar == null) throw new ArgumentNullException(nameof(logVar));

        if (mean.Length != logVar.Length)
            throw new ArgumentException(
                $"mean length ({mean.Length}) must equal logVar length ({logVar.Length})",
                nameof(logVar));

        var klElements = GradOperationKernels.ApplyKlElementWise(mean.Data, logVar.Data);
        klElements.TryGetSpan(out var klSpan);
        var klSum = TensorPrimitives.Sum(klSpan);

        var resultData = NivaraColumn<T>.CreateFromOwnedArray(new T[] { klSum });
        var resultTensor = new ReverseGradTensor<T>(resultData, GradientUtils.ShouldTrackGrad(mean, logVar), ScalarShape());

        if (GradientUtils.ShouldTrackGrad(mean, logVar))
        {
            var gradFn = new OpNode<T>("KlDivergence", [mean, logVar], (typedGradOutput) =>
            {
                if (mean.RequiresGrad)
                {
                    var broadcast = GradOperationKernels.BroadcastGradient(typedGradOutput, mean.Length);
                    var dMean = GradOperationKernels.ApplyKlMeanGradient(mean.Data, broadcast);
                    AccumulateGradient(mean, dMean);
                }
                if (logVar.RequiresGrad)
                {
                    var broadcast = GradOperationKernels.BroadcastGradient(typedGradOutput, logVar.Length);
                    var dLogVar = GradOperationKernels.ApplyKlLogVarGradient(logVar.Data, broadcast);
                    AccumulateGradient(logVar, dLogVar);
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Samples from a Gaussian using the reparameterization trick:
    /// result = mean + exp(logVar / 2) * epsilon, where epsilon ~ N(0, 1).
    /// </summary>
    /// <param name="mean">The mean tensor</param>
    /// <param name="logVar">The log-variance tensor (same length as <paramref name="mean"/>)</param>
    /// <param name="seed">Optional RNG seed for deterministic sampling</param>
    /// <returns>A new tensor with the sampled values</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null</exception>
    /// <exception cref="ArgumentException">Thrown when the operand lengths differ</exception>
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
        var epsilonCol = NivaraColumn<T>.CreateFromOwnedArray(epsilon);

        var result = GradOperationKernels.ApplySampleNormalForward(mean.Data, logVar.Data, epsilonCol);

        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(mean, logVar), mean.shape);

        if (GradientUtils.ShouldTrackGrad(mean, logVar))
        {
            var savedEpsilon = epsilonCol;
            var gradFn = new OpNode<T>("SampleNormal", [mean, logVar], (typedGradOutput) =>
            {
                if (mean.RequiresGrad)
                {
                    AccumulateGradient(mean, typedGradOutput);
                }
                if (logVar.RequiresGrad)
                {
                    var dLogVar = GradOperationKernels.ApplySampleNormalLogVarGradient(logVar.Data, typedGradOutput, savedEpsilon);
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

    /// <summary>
    /// Broadcasts a per-channel scale across a 2D+ input: result[i, c, ...] = input[i, c, ...] * scale[c].
    /// </summary>
    /// <param name="input">The input tensor (rank 2 or higher)</param>
    /// <param name="scale">A 1D [channels] tensor matching the input's second dimension</param>
    /// <returns>A new tensor with each channel scaled</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null</exception>
    /// <exception cref="ArgumentException">Thrown when the input is not 2D+, the scale is not 1D, or the channel counts differ</exception>
    public static ReverseGradTensor<T> BroadcastMultiply<T>(ReverseGradTensor<T> input, ReverseGradTensor<T> scale) where T : struct, IFloatingPointIeee754<T>
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (scale == null) throw new ArgumentNullException(nameof(scale));
        if (input.Rank < 2) throw new ArgumentException($"Input must be at least 2D [batch, channels, ...], got {input.Rank}D");
        if (scale.Rank != 1) throw new ArgumentException($"Scale must be 1D [channels], got {scale.Rank}D");

        int c = input.shape[1];
        if (scale.Length != c)
            throw new ArgumentException($"Scale length ({scale.Length}) must match input channel dimension ({c})");

        input.Data.TryGetSpan(out var inputSpan);
        scale.Data.TryGetSpan(out var scaleSpan);

        var inputArr = inputSpan.ToArray();
        var scaleArr = scaleSpan.ToArray();

        int channelStride = 1;
        for (int d = 2; d < input.Rank; d++) channelStride *= input.shape[d];

        var outputData = new T[input.Length];
        for (int b = 0; b < input.shape[0]; b++)
            for (int ch = 0; ch < c; ch++)
            {
                int offset = (b * c + ch) * channelStride;
                TensorPrimitives.Multiply(inputArr.AsSpan(offset, channelStride), scaleArr[ch], outputData.AsSpan(offset, channelStride));
            }

        var result = NivaraColumn<T>.CreateFromOwnedArray(outputData);
        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(input, scale), input.shape);

        if (GradientUtils.ShouldTrackGrad(input, scale))
        {
            var gradFn = new OpNode<T>("BroadcastMultiply", [input, scale], (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gradSpan);

                if (GradientUtils.ShouldTrackGrad(input))
                {
                    var inputGrad = new T[input.Length];
                    for (int b = 0; b < input.shape[0]; b++)
                        for (int ch = 0; ch < c; ch++)
                        {
                            int offset = (b * c + ch) * channelStride;
                            TensorPrimitives.Multiply(gradSpan.Slice(offset, channelStride), scaleArr[ch], inputGrad.AsSpan(offset, channelStride));
                        }
                    AccumulateGradient(input, NivaraColumn<T>.CreateFromOwnedArray(inputGrad));
                }
                if (scale.RequiresGrad)
                {
                    var scaleGrad = new T[c];
                    for (int b = 0; b < input.shape[0]; b++)
                        for (int ch = 0; ch < c; ch++)
                        {
                            int offset = (b * c + ch) * channelStride;
                            scaleGrad[ch] += TensorPrimitives.Dot(gradSpan.Slice(offset, channelStride), inputArr.AsSpan(offset, channelStride));
                        }
                    AccumulateGradient(scale, NivaraColumn<T>.CreateFromOwnedArray(scaleGrad));
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Broadcasts a per-channel bias across a 2D+ input: result[i, c, ...] = input[i, c, ...] + bias[c].
    /// </summary>
    /// <param name="input">The input tensor (rank 2 or higher)</param>
    /// <param name="bias">A 1D [channels] tensor matching the input's second dimension</param>
    /// <returns>A new tensor with each channel offset by the bias</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null</exception>
    /// <exception cref="ArgumentException">Thrown when the input is not 2D+, the bias is not 1D, or the channel counts differ</exception>
    public static ReverseGradTensor<T> BroadcastAdd<T>(ReverseGradTensor<T> input, ReverseGradTensor<T> bias) where T : struct, IFloatingPointIeee754<T>
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (bias == null) throw new ArgumentNullException(nameof(bias));
        if (input.Rank < 2) throw new ArgumentException($"Input must be at least 2D [batch, channels, ...], got {input.Rank}D");
        if (bias.Rank != 1) throw new ArgumentException($"Bias must be 1D [channels], got {bias.Rank}D");

        int c = input.shape[1];
        if (bias.Length != c)
            throw new ArgumentException($"Bias length ({bias.Length}) must match input channel dimension ({c})");

        input.Data.TryGetSpan(out var inputSpan);
        bias.Data.TryGetSpan(out var biasSpan);

        int channelStride = 1;
        for (int d = 2; d < input.Rank; d++) channelStride *= input.shape[d];

        var outputData = new T[input.Length];
        for (int b = 0; b < input.shape[0]; b++)
            for (int ch = 0; ch < c; ch++)
            {
                int offset = (b * c + ch) * channelStride;
                TensorPrimitives.Add(inputSpan.Slice(offset, channelStride), biasSpan[ch], outputData.AsSpan(offset, channelStride));
            }

        var result = NivaraColumn<T>.CreateFromOwnedArray(outputData);
        var resultTensor = new ReverseGradTensor<T>(result, GradientUtils.ShouldTrackGrad(input, bias), input.shape);

        if (GradientUtils.ShouldTrackGrad(input, bias))
        {
            var gradFn = new OpNode<T>("BroadcastAdd", [input, bias], (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gradSpan);

                if (GradientUtils.ShouldTrackGrad(input))
                {
                    AccumulateGradient(input, typedGradOutput);
                }
                if (bias.RequiresGrad)
                {
                    var biasGrad = new T[c];
                    for (int b = 0; b < input.shape[0]; b++)
                        for (int ch = 0; ch < c; ch++)
                        {
                            int offset = (b * c + ch) * channelStride;
                            biasGrad[ch] += TensorPrimitives.Sum(gradSpan.Slice(offset, channelStride));
                        }
                    AccumulateGradient(bias, NivaraColumn<T>.CreateFromOwnedArray(biasGrad));
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Gets the underlying data span. Delegates to <see cref="GradTensor{T}.AsSpan"/>,
    /// which enforces the ADR-001 non-nullable boundary.
    /// </summary>
    internal static ReadOnlySpan<T> AsSpan<T>(this ReverseGradTensor<T> t) where T : struct, IFloatingPointIeee754<T>
    {
        return t.AsSpan();
    }

    /// <summary>
    /// Creates a result tensor from a raw T[] array, reusing the source tensor's shape.
    /// </summary>
    static ReverseGradTensor<T> ResultTensor<T>(T[] data, ReverseGradTensor<T> shapeSrc, bool requiresGrad)
        where T : struct, IFloatingPointIeee754<T>
    {
        return new ReverseGradTensor<T>(NivaraColumn<T>.CreateFromOwnedArray(data), requiresGrad, shapeSrc.shape);
    }

    private static int[] ScalarShape()
    {
        return new[] { 1 };
    }

    #endregion
}

