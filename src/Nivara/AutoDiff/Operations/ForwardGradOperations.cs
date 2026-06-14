using Nivara.Helpers;
using Nivara.Tensors;
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace Nivara.AutoDiff.Operations;

/// <summary>
/// Forward-mode automatic differentiation operations.
/// Each method computes the primal (forward value) and the tangent (directional derivative
/// via JVP — Jacobian-Vector Product) for a seeded tangent direction.
/// Mirrors <see cref="ReverseGradOperations"/> in structure and convention.
/// </summary>
public static class ForwardGradOperations
{
    #region Element-wise Operations

    /// <summary>
    /// Adds two tensors element-wise.
    /// JVP: t_out = t_a + t_b
    /// </summary>
    public static ForwardGradTensor<T> Add<T>(ForwardGradTensor<T> a, ForwardGradTensor<T> b)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Cannot add tensors with different lengths: {a.Length} vs {b.Length}");
        }

        var primal = a.Data + b.Data;
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent || b.RequiresTangent)
        {
            if (a.Tangent == null)
                tangent = b.Tangent;
            else if (b.Tangent == null)
                tangent = a.Tangent;
            else
                tangent = a.Tangent + b.Tangent;
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a, b));
    }

    /// <summary>
    /// Subtracts two tensors element-wise.
    /// JVP: t_out = t_a - t_b
    /// </summary>
    public static ForwardGradTensor<T> Subtract<T>(ForwardGradTensor<T> a, ForwardGradTensor<T> b)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Cannot subtract tensors with different lengths: {a.Length} vs {b.Length}");
        }

        var primal = a.Data.Subtract(b.Data);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent || b.RequiresTangent)
        {
            if (a.Tangent == null)
                tangent = b.Tangent?.Negate();
            else if (b.Tangent == null)
                tangent = a.Tangent;
            else
                tangent = a.Tangent.Subtract(b.Tangent);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a, b));
    }

    /// <summary>
    /// Multiplies two tensors element-wise.
    /// JVP: t_out = t_a * b + a * t_b
    /// </summary>
    public static ForwardGradTensor<T> Multiply<T>(ForwardGradTensor<T> a, ForwardGradTensor<T> b)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Cannot multiply tensors with different lengths: {a.Length} vs {b.Length}");
        }

        var primal = a.Data * b.Data;
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent || b.RequiresTangent)
        {
            if (a.Tangent == null)
                tangent = a.Data * b.Tangent!;
            else if (b.Tangent == null)
                tangent = a.Tangent * b.Data;
            else
                tangent = a.Tangent * b.Data + a.Data * b.Tangent;
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a, b));
    }

    /// <summary>
    /// Divides two tensors element-wise.
    /// JVP: t_out = (t_a - result * t_b) / b
    /// </summary>
    public static ForwardGradTensor<T> Divide<T>(ForwardGradTensor<T> a, ForwardGradTensor<T> b)
        where T : struct, INumber<T>
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

        var primal = a.Data.Divide(b.Data);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent || b.RequiresTangent)
        {
            if (a.Tangent == null)
                tangent = primal.Negate().Multiply(b.Tangent!).Divide(b.Data);
            else if (b.Tangent == null)
                tangent = a.Tangent.Divide(b.Data);
            else
                tangent = a.Tangent.Subtract(primal.Multiply(b.Tangent)).Divide(b.Data);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a, b));
    }

    #endregion

    #region Matrix Operations

    /// <summary>
    /// Multiplies two matrices: result = a @ b.
    /// JVP: t_out = t_a @ B + A @ t_b
    /// </summary>
    public static ForwardGradTensor<T> MatMul<T>(ForwardGradTensor<T> a, ForwardGradTensor<T> b)
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

        var primal = a.Data.MatMul(b.Data, aRows, aCols, bCols);
        var resultShape = new[] { aRows, bCols };

        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent || b.RequiresTangent)
        {
            var aTan = a.Tangent;
            var bTan = b.Tangent;

            if (aTan != null && bTan != null)
            {
                var tAB = aTan.MatMul(b.Data, aRows, aCols, bCols);
                var aT_B = a.Data.MatMul(bTan, aRows, aCols, bCols);
                tangent = tAB + aT_B;
            }
            else if (aTan != null)
            {
                tangent = aTan.MatMul(b.Data, aRows, aCols, bCols);
            }
            else if (bTan != null)
            {
                tangent = a.Data.MatMul(bTan, aRows, aCols, bCols);
            }
        }

        return new ForwardGradTensor<T>(primal, tangent, resultShape);
    }

    /// <summary>
    /// Transposes a matrix.
    /// JVP: t_out = Transpose(t_a)
    /// </summary>
    public static ForwardGradTensor<T> Transpose<T>(ForwardGradTensor<T> a)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (a.Rank != 2)
            throw new ArgumentException($"Transpose requires a matrix (rank 2), got rank {a.Rank}", nameof(a));

        var rows = a.shape[0];
        var cols = a.shape[1];
        var primal = a.Data.Transpose(rows, cols);
        var resultShape = new[] { cols, rows };

        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            tangent = a.Tangent.Transpose(rows, cols);
        }

        return new ForwardGradTensor<T>(primal, tangent, resultShape);
    }

    #endregion

    #region Reduction Operations

    /// <summary>
    /// Computes the sum of all elements.
    /// JVP: t_out = sum(t_a)  (scalar)
    /// </summary>
    public static ForwardGradTensor<T> Sum<T>(ForwardGradTensor<T> a)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (a.Length == 0)
        {
            throw new InvalidOperationException("Cannot compute sum of empty tensor");
        }

        var series = a.ToSeries();
        var sumValue = series.Sum();
        var resultData = NivaraColumn<T>.Create(new T[] { sumValue });

        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            var tanSum = new NivaraSeries<T>(a.Tangent).Sum();
            tangent = NivaraColumn<T>.Create(new T[] { tanSum });
        }

        return new ForwardGradTensor<T>(resultData, tangent, ScalarShape());
    }

    /// <summary>
    /// Computes the mean (average) of all elements.
    /// JVP: t_out = sum(t_a) / n  (scalar)
    /// </summary>
    public static ForwardGradTensor<T> Mean<T>(ForwardGradTensor<T> a)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (a.Length == 0)
        {
            throw new InvalidOperationException("Cannot compute mean of empty tensor");
        }

        var series = a.ToSeries();
        var meanValue = series.Average();
        var resultData = NivaraColumn<T>.Create(new T[] { meanValue });

        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            var tanSum = new NivaraSeries<T>(a.Tangent).Sum();
            var tanMean = tanSum / T.CreateChecked(a.Length);
            tangent = NivaraColumn<T>.Create(new T[] { tanMean });
        }

        return new ForwardGradTensor<T>(resultData, tangent, ScalarShape());
    }

    #endregion

    #region Activation Functions

    /// <summary>
    /// Applies the ReLU activation: max(0, x).
    /// JVP: t_out = (a > 0) ? t_a : 0
    /// </summary>
    public static ForwardGradTensor<T> Relu<T>(ForwardGradTensor<T> a)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var primal = a.Data.Relu();
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            tangent = a.Data.ReluGradient(a.Tangent);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies the Sigmoid activation: σ(x) = 1 / (1 + e⁻ˣ).
    /// JVP: t_out = σ(a) * (1 - σ(a)) * t_a = result * (1 - result) * t_a
    /// </summary>
    public static ForwardGradTensor<T> Sigmoid<T>(ForwardGradTensor<T> a)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var primal = a.Data.Sigmoid();
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            tangent = primal.SigmoidGradient(a.Tangent);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies the Tanh activation.
    /// JVP: t_out = (1 - tanh²(a)) * t_a = (1 - result²) * t_a
    /// </summary>
    public static ForwardGradTensor<T> Tanh<T>(ForwardGradTensor<T> a)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var primal = a.Data.Tanh();
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            tangent = primal.TanhGradient(a.Tangent);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Negates all elements.
    /// JVP: t_out = -t_a
    /// </summary>
    public static ForwardGradTensor<T> Negate<T>(ForwardGradTensor<T> a)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var primal = a.Data.Negate();
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            tangent = a.Tangent.Negate();
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Computes element-wise absolute value.
    /// JVP: t_out = sign(a) * t_a
    /// </summary>
    public static ForwardGradTensor<T> Abs<T>(ForwardGradTensor<T> a)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var primal = a.Data.Abs();
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            tangent = a.Data.AbsGradient(a.Tangent);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Clips values to [min, max].
    /// JVP: t_out = (a in [min, max]) ? t_a : 0
    /// </summary>
    public static ForwardGradTensor<T> Clip<T>(ForwardGradTensor<T> a, T min, T max)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var primal = a.Data.Clamp(min, max);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            tangent = a.Data.ClipGradient(a.Tangent, min, max);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies LeakyReLU activation: x if x > 0, else αx.
    /// JVP: t_out = (a > 0) ? t_a : α * t_a
    /// </summary>
    public static ForwardGradTensor<T> LeakyRelu<T>(ForwardGradTensor<T> a, T negativeSlope = default)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (negativeSlope == T.Zero)
            negativeSlope = T.CreateChecked(0.01);

        var primal = a.Data.LeakyRelu(negativeSlope);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            tangent = a.Data.LeakyReluGradient(a.Tangent, negativeSlope);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Computes element-wise exponential.
    /// JVP: t_out = e^a * t_a = result * t_a
    /// </summary>
    public static ForwardGradTensor<T> Exp<T>(ForwardGradTensor<T> a)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var primal = a.Data.Exp();
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            tangent = primal * a.Tangent;
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Computes element-wise natural logarithm.
    /// JVP: t_out = t_a / a
    /// </summary>
    public static ForwardGradTensor<T> Log<T>(ForwardGradTensor<T> a)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var primal = a.Data.Log();
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            tangent = a.Data.LogGradient(a.Tangent);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies the Softmax function along the last dimension.
    /// JVP: s ⊙ (t_a - Σ(s * t_a)) where s = softmax(a)
    /// The Jacobian is symmetric, so SoftmaxGradient(result, t_a, dim) computes the JVP.
    /// </summary>
    public static ForwardGradTensor<T> Softmax<T>(ForwardGradTensor<T> a)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var classCount = a.Rank >= 2 ? a.shape[1] : a.Length;
        var primal = a.Data.Softmax(classCount);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            tangent = primal.SoftmaxGradient(a.Tangent, classCount);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies the LogSoftmax function along the last dimension.
    /// JVP: t_a - Σ(s * t_a) where s = softmax(a)
    /// </summary>
    public static ForwardGradTensor<T> LogSoftmax<T>(ForwardGradTensor<T> a)
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var classCount = a.Rank >= 2 ? a.shape[1] : a.Length;
        var primal = a.Data.LogSoftmax(classCount);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            tangent = a.Data.LogSoftmaxGradient(a.Tangent, classCount);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies dropout during training. In eval mode (isTraining=false) returns the input unchanged.
    /// JVP: mask * t_a * scale  (same mask used in forward)
    /// </summary>
    public static ForwardGradTensor<T> Dropout<T>(ForwardGradTensor<T> input, double probability, bool isTraining)
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

    /// <summary>
    /// Applies dropout with a pre-generated mask.
    /// JVP: same mask applied to tangent with scaling.
    /// </summary>
    internal static ForwardGradTensor<T> DropoutWithMask<T>(ForwardGradTensor<T> input, ReadOnlySpan<bool> keepMask, T scale)
        where T : struct, INumber<T>
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (keepMask.Length != input.Length)
            throw new ArgumentException(
                $"Dropout mask length ({keepMask.Length}) must match input length ({input.Length})",
                nameof(keepMask));

        var savedMask = keepMask.ToArray();
        var primal = ApplyDropout(input.Data, savedMask, scale);
        NivaraColumn<T>? tangent = null;
        if (input.RequiresTangent && input.Tangent != null)
        {
            tangent = ApplyDropoutTangent(input.Data, input.Tangent, savedMask, scale);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(input));
    }

    #endregion

    #region VAE Operations

    /// <summary>
    /// Computes the KL divergence between a diagonal Gaussian and N(0, 1).
    /// KL = -0.5 * sum(1 + logVar - mean² - exp(logVar))
    /// Output is a scalar (sum of per-element KL values).
    ///
    /// JVP: sum(mean * t_mean) + sum(0.5 * (exp(logVar) - 1) * t_logVar)
    /// </summary>
    public static ForwardGradTensor<T> KlDivergence<T>(ForwardGradTensor<T> mean, ForwardGradTensor<T> logVar)
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

        NivaraColumn<T>? tangent = null;
        if (mean.RequiresTangent || logVar.RequiresTangent)
        {
            var tanValue = T.Zero;

            if (mean.RequiresTangent && mean.Tangent != null)
            {
                var dMean = mean.Data.Multiply(mean.Tangent);
                tanValue += new NivaraSeries<T>(dMean).Sum();
            }

            if (logVar.RequiresTangent && logVar.Tangent != null)
            {
                var expLv = logVar.Data.Exp();
                var half = T.CreateChecked(0.5);
                var dLogVar = expLv.Multiply(logVar.Tangent).Subtract(logVar.Tangent).Multiply(half);
                tanValue += new NivaraSeries<T>(dLogVar).Sum();
            }

            tangent = NivaraColumn<T>.Create(new T[] { tanValue });
        }

        return new ForwardGradTensor<T>(resultData, tangent, ScalarShape());
    }

    /// <summary>
    /// Reparameterized sampling from a diagonal Gaussian: z = mean + exp(0.5 * logVar) * ε.
    /// JVP: t_z = t_mean + 0.5 * exp(0.5 * logVar) * ε * t_logVar
    /// </summary>
    public static ForwardGradTensor<T> SampleNormal<T>(ForwardGradTensor<T> mean, ForwardGradTensor<T> logVar, int? seed = null)
        where T : struct, INumber<T>
    {
        if (mean == null) throw new ArgumentNullException(nameof(mean));
        if (logVar == null) throw new ArgumentNullException(nameof(logVar));

        if (mean.Length != logVar.Length)
            throw new ArgumentException(
                $"mean length ({mean.Length}) must equal logVar length ({logVar.Length})",
                nameof(logVar));

        int n = mean.Length;
        var epsilon = RandomGeneration.GenerateStandardNormal<T>(n, seed);
        var epsilonCol = NivaraColumn<T>.Create(epsilon.AsSpan());
        var primal = ApplySampleNormalForward(mean.Data, logVar.Data, epsilonCol);

        NivaraColumn<T>? tangent = null;
        if (mean.RequiresTangent || logVar.RequiresTangent)
        {
            if (mean.Tangent != null && logVar.Tangent != null)
            {
                var dLogVar = ApplySampleNormalLogVarTangent(logVar.Data, logVar.Tangent, epsilonCol);
                tangent = mean.Tangent + dLogVar;
            }
            else if (mean.Tangent != null)
            {
                tangent = mean.Tangent;
            }
            else if (logVar.Tangent != null)
            {
                tangent = ApplySampleNormalLogVarTangent(logVar.Data, logVar.Tangent, epsilonCol);
            }
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(mean, logVar));
    }

    #endregion

    #region Helper Methods

    private static int[] PropagateShape<T>(ForwardGradTensor<T> a, ForwardGradTensor<T> b) where T : struct, INumber<T>
    {
        return a.shape;
    }

    private static int[] PropagateShape<T>(ForwardGradTensor<T> a) where T : struct, INumber<T>
    {
        return a.shape;
    }

    private static int[] ScalarShape()
    {
        return new[] { 1 };
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

    private static NivaraColumn<T> ApplyDropoutTangent<T>(
        NivaraColumn<T> input,
        NivaraColumn<T> tangent,
        ReadOnlySpan<bool> keepMask,
        T scale)
        where T : struct, INumber<T>
    {
        int n = input.Length;
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);

        try
        {
            tangent.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            for (int i = 0; i < n; i++)
                resultBuf[i] = keepMask[i] ? gradBuf[i] * scale : T.Zero;

            if (!input.HasNulls && !tangent.HasNulls)
                return NivaraColumn<T>.Create(resultBuf.AsSpan(0, n));

            var nullMask = ArrayPool<bool>.Shared.Rent(n);
            try
            {
                NivaraColumnUtility.MergeNullMasks(input, tangent, nullMask.AsSpan(0, n));
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

    private static NivaraColumn<T> ApplyKlElementWise<T>(NivaraColumn<T> mean, NivaraColumn<T> logVar)
        where T : struct, INumber<T>
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

    private static NivaraColumn<T> ApplySampleNormalForward<T>(NivaraColumn<T> mean, NivaraColumn<T> logVar, NivaraColumn<T> epsilon)
        where T : struct, INumber<T>
    {
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

    private static NivaraColumn<T> ApplySampleNormalLogVarTangent<T>(
        NivaraColumn<T> logVar,
        NivaraColumn<T> tangent,
        NivaraColumn<T> epsilon)
        where T : struct, INumber<T>
    {
        // ∂z/∂logVar = 0.5 * exp(0.5 * logVar) * ε
        // JVP contribution: 0.5 * exp(0.5 * logVar) * ε * t_logVar
        int n = logVar.Length;

        if (!logVar.HasNulls && !tangent.HasNulls)
        {
            logVar.TryGetSpan(out var lvSpan);
            tangent.TryGetSpan(out var gSpan);
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
                    var t = double.CreateChecked(gSpan[i]);
                    var e = double.CreateChecked(eSpan[i]);
                    result[i] = T.CreateChecked(0.5 * Math.Exp(0.5 * lv) * e * t);
                }
            }
            return NivaraColumn<T>.Create(result);
        }

        var logVarBuf = ArrayPool<T>.Shared.Rent(n);
        var tanBuf = ArrayPool<T>.Shared.Rent(n);
        var epsBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            logVar.CopyTo(logVarBuf.AsSpan(0, n), T.Zero);
            tangent.CopyTo(tanBuf.AsSpan(0, n), T.Zero);
            epsilon.CopyTo(epsBuf.AsSpan(0, n), T.Zero);
            NivaraColumnUtility.MergeNullMasks(logVar, tangent, nullMask.AsSpan(0, n));

            for (int i = 0; i < n; i++)
            {
                if (nullMask[i])
                    continue;
                var lv = double.CreateChecked(logVarBuf[i]);
                var t = double.CreateChecked(tanBuf[i]);
                var e = double.CreateChecked(epsBuf[i]);
                resultBuf[i] = T.CreateChecked(0.5 * Math.Exp(0.5 * lv) * e * t);
            }
            return NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(logVarBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(tanBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(epsBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    #endregion
}
