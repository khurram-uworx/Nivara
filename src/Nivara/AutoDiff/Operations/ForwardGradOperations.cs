using Nivara.Helpers;
using Nivara.Tensors;
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;

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
        where T : struct, IFloatingPointIeee754<T>
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
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Cannot subtract tensors with different lengths: {a.Length} vs {b.Length}");
        }

        var primalArr = new T[a.Length];
        a.Data.TryGetSpan(out var aSpan);
        b.Data.TryGetSpan(out var bSpan);
        TensorPrimitives.Subtract(aSpan, bSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent || b.RequiresTangent)
        {
            if (a.Tangent == null)
            {
                b.Tangent!.TryGetSpan(out var bTanSpan);
                var tanArr = new T[a.Length];
                TensorPrimitives.Negate(bTanSpan, tanArr);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
            else if (b.Tangent == null)
                tangent = a.Tangent;
            else
            {
                a.Tangent.TryGetSpan(out var aTanSpan);
                b.Tangent.TryGetSpan(out var bTanSpan);
                var tanArr = new T[a.Length];
                TensorPrimitives.Subtract(aTanSpan, bTanSpan, tanArr);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a, b));
    }

    /// <summary>
    /// Multiplies two tensors element-wise.
    /// JVP: t_out = t_a * b + a * t_b
    /// </summary>
    public static ForwardGradTensor<T> Multiply<T>(ForwardGradTensor<T> a, ForwardGradTensor<T> b)
        where T : struct, IFloatingPointIeee754<T>
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
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Cannot divide tensors with different lengths: {a.Length} vs {b.Length}");
        }

        for (int i = 0; i < b.Length; i++)
        {
            if (b[i] == T.Zero)
            {
                throw new DivideByZeroException($"Division by zero at index {i}");
            }
        }

        a.Data.TryGetSpan(out var aSpan);
        b.Data.TryGetSpan(out var bSpan);
        var primalArr = new T[a.Length];
        TensorPrimitives.Divide(aSpan, bSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent || b.RequiresTangent)
        {
            if (a.Tangent == null)
            {
                b.Tangent!.TryGetSpan(out var bTanSpan);
                var numArr = new T[a.Length];
                TensorPrimitives.Multiply(primalArr, bTanSpan, numArr);
                TensorPrimitives.Negate(numArr, numArr);
                var tanArr = new T[a.Length];
                TensorPrimitives.Divide(numArr, bSpan, tanArr);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
            else if (b.Tangent == null)
            {
                a.Tangent.TryGetSpan(out var aTanSpan);
                var tanArr = new T[a.Length];
                TensorPrimitives.Divide(aTanSpan, bSpan, tanArr);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
            else
            {
                a.Tangent.TryGetSpan(out var aTanSpan);
                b.Tangent.TryGetSpan(out var bTanSpan);
                var numArr = new T[a.Length];
                TensorPrimitives.Multiply(primalArr, bTanSpan, numArr);
                TensorPrimitives.Subtract(aTanSpan, numArr, numArr);
                var tanArr = new T[a.Length];
                TensorPrimitives.Divide(numArr, bSpan, tanArr);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
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

        a.Data.TryGetSpan(out var aSpan);
        b.Data.TryGetSpan(out var bSpan);
        var primalArr = new T[aRows * bCols];
        GradKernels.MatMul(aSpan, bSpan, primalArr, aRows, aCols, bCols);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        var resultShape = new[] { aRows, bCols };

        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent || b.RequiresTangent)
        {
            var aTan = a.Tangent;
            var bTan = b.Tangent;

            if (aTan != null && bTan != null)
            {
                aTan.TryGetSpan(out var aTanSpan);
                var tAB = new T[aRows * bCols];
                GradKernels.MatMul(aTanSpan, bSpan, tAB, aRows, aCols, bCols);
                bTan.TryGetSpan(out var bTanSpan);
                var aT_B = new T[aRows * bCols];
                GradKernels.MatMul(aSpan, bTanSpan, aT_B, aRows, aCols, bCols);
                var tanArr = new T[aRows * bCols];
                TensorPrimitives.Add(tAB, aT_B, tanArr);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
            else if (aTan != null)
            {
                aTan.TryGetSpan(out var aTanSpan);
                var tanArr = new T[aRows * bCols];
                GradKernels.MatMul(aTanSpan, bSpan, tanArr, aRows, aCols, bCols);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
            else if (bTan != null)
            {
                bTan.TryGetSpan(out var bTanSpan);
                var tanArr = new T[aRows * bCols];
                GradKernels.MatMul(aSpan, bTanSpan, tanArr, aRows, aCols, bCols);
                tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
            }
        }

        return new ForwardGradTensor<T>(primal, tangent, resultShape);
    }

    /// <summary>
    /// Transposes a matrix.
    /// JVP: t_out = Transpose(t_a)
    /// </summary>
    public static ForwardGradTensor<T> Transpose<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (a.Rank != 2)
            throw new ArgumentException($"Transpose requires a matrix (rank 2), got rank {a.Rank}", nameof(a));

        var rows = a.shape[0];
        var cols = a.shape[1];
        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Transpose(aSpan, primalArr, rows, cols);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        var resultShape = new[] { cols, rows };

        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.Transpose(aTanSpan, tanArr, rows, cols);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
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
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (a.Length == 0)
        {
            throw new InvalidOperationException("Cannot compute sum of empty tensor");
        }

        var sumValue = a.Data.Sum();
        var resultData = NivaraColumn<T>.Create(new T[] { sumValue });

        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            var tanSum = a.Tangent!.Sum();
            tangent = NivaraColumn<T>.Create(new T[] { tanSum });
        }

        return new ForwardGradTensor<T>(resultData, tangent, ScalarShape());
    }

    /// <summary>
    /// Computes the mean (average) of all elements.
    /// JVP: t_out = sum(t_a) / n  (scalar)
    /// </summary>
    public static ForwardGradTensor<T> Mean<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
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
            var tanSum = a.Tangent!.Sum();
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
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Relu(aSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.ReluGradient(aSpan, aTanSpan, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    public static ForwardGradTensor<T> Gelu<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Gelu(aSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.GeluGradient(aSpan, aTanSpan, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies the Sigmoid activation: σ(x) = 1 / (1 + e⁻ˣ).
    /// JVP: t_out = σ(a) * (1 - σ(a)) * t_a = result * (1 - result) * t_a
    /// </summary>
    public static ForwardGradTensor<T> Sigmoid<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Sigmoid(aSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.SigmoidGradient(primalArr, aTanSpan, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies the Tanh activation.
    /// JVP: t_out = (1 - tanh²(a)) * t_a = (1 - result²) * t_a
    /// </summary>
    public static ForwardGradTensor<T> Tanh<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Tanh(aSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.TanhGradient(primalArr, aTanSpan, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Negates all elements.
    /// JVP: t_out = -t_a
    /// </summary>
    public static ForwardGradTensor<T> Negate<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Negate(aSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.Negate(aTanSpan, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Computes element-wise absolute value.
    /// JVP: t_out = sign(a) * t_a
    /// </summary>
    public static ForwardGradTensor<T> Abs<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Abs(aSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.AbsGradient(aSpan, aTanSpan, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Clips values to [min, max].
    /// JVP: t_out = (a in [min, max]) ? t_a : 0
    /// </summary>
    public static ForwardGradTensor<T> Clip<T>(ForwardGradTensor<T> a, T min, T max)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Clamp(aSpan, min, max, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.ClipGradient(aSpan, aTanSpan, min, max, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies LeakyReLU activation: x if x > 0, else αx.
    /// JVP: t_out = (a > 0) ? t_a : α * t_a
    /// </summary>
    public static ForwardGradTensor<T> LeakyRelu<T>(ForwardGradTensor<T> a, T negativeSlope = default)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (negativeSlope == T.Zero)
            negativeSlope = T.CreateChecked(0.01);

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.LeakyRelu(aSpan, negativeSlope, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.LeakyReluGradient(aSpan, aTanSpan, negativeSlope, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Computes element-wise exponential.
    /// JVP: t_out = e^a * t_a = result * t_a
    /// </summary>
    public static ForwardGradTensor<T> Exp<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Exp(aSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            TensorPrimitives.Multiply(primalArr, aTanSpan, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Computes element-wise natural logarithm.
    /// JVP: t_out = t_a / a
    /// </summary>
    public static ForwardGradTensor<T> Log<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Log(aSpan, primalArr);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.LogGradient(aSpan, aTanSpan, tanArr);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies the Softmax function along the last dimension.
    /// JVP: s ⊙ (t_a - Σ(s * t_a)) where s = softmax(a)
    /// The Jacobian is symmetric, so SoftmaxGradient(result, t_a, dim) computes the JVP.
    /// </summary>
    public static ForwardGradTensor<T> Softmax<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var classCount = a.Rank >= 2 ? a.shape[1] : a.Length;
        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.Softmax(aSpan, primalArr, classCount);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.SoftmaxGradient(primalArr, aTanSpan, tanArr, classCount);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies the LogSoftmax function along the last dimension.
    /// JVP: t_a - Σ(s * t_a) where s = softmax(a)
    /// </summary>
    public static ForwardGradTensor<T> LogSoftmax<T>(ForwardGradTensor<T> a)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        var classCount = a.Rank >= 2 ? a.shape[1] : a.Length;
        a.Data.TryGetSpan(out var aSpan);
        var primalArr = new T[a.Length];
        GradKernels.LogSoftmax(aSpan, primalArr, classCount);
        var primal = NivaraColumn<T>.CreateFromOwnedArray(primalArr);
        NivaraColumn<T>? tangent = null;
        if (a.RequiresTangent && a.Tangent != null)
        {
            a.Tangent.TryGetSpan(out var aTanSpan);
            var tanArr = new T[a.Length];
            GradKernels.LogSoftmaxGradient(aSpan, aTanSpan, tanArr, classCount);
            tangent = NivaraColumn<T>.CreateFromOwnedArray(tanArr);
        }

        return new ForwardGradTensor<T>(primal, tangent, PropagateShape(a));
    }

    /// <summary>
    /// Applies dropout during training. In eval mode (isTraining=false) returns the input unchanged.
    /// JVP: mask * t_a * scale  (same mask used in forward)
    /// </summary>
    public static ForwardGradTensor<T> Dropout<T>(ForwardGradTensor<T> input, double probability, bool isTraining)
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

    /// <summary>
    /// Applies dropout with a pre-generated mask.
    /// JVP: same mask applied to tangent with scaling.
    /// </summary>
    internal static ForwardGradTensor<T> DropoutWithMask<T>(ForwardGradTensor<T> input, ReadOnlySpan<bool> keepMask, T scale)
        where T : struct, IFloatingPointIeee754<T>
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
        where T : struct, IFloatingPointIeee754<T>
    {
        if (mean == null) throw new ArgumentNullException(nameof(mean));
        if (logVar == null) throw new ArgumentNullException(nameof(logVar));

        if (mean.Length != logVar.Length)
            throw new ArgumentException(
                $"mean length ({mean.Length}) must equal logVar length ({logVar.Length})",
                nameof(logVar));

        var klElements = ApplyKlElementWise(mean.Data, logVar.Data);
        var klSum = klElements.Sum();
        var resultData = NivaraColumn<T>.Create(new T[] { klSum });

        NivaraColumn<T>? tangent = null;
        if (mean.RequiresTangent || logVar.RequiresTangent)
        {
            var tanValue = T.Zero;

            if (mean.RequiresTangent && mean.Tangent != null)
            {
                mean.Data.TryGetSpan(out var mSpan);
                mean.Tangent.TryGetSpan(out var mTanSpan);
                var dMeanArr = new T[mean.Length];
                TensorPrimitives.Multiply(mSpan, mTanSpan, dMeanArr);
                tanValue += TensorPrimitives.Sum(dMeanArr);
            }

            if (logVar.RequiresTangent && logVar.Tangent != null)
            {
                logVar.Data.TryGetSpan(out var lvSpan);
                logVar.Tangent.TryGetSpan(out var lvTanSpan);
                var expLvArr = new T[logVar.Length];
                TensorPrimitives.Exp(lvSpan, expLvArr);
                var dLogVarArr = new T[logVar.Length];
                TensorPrimitives.Multiply(expLvArr, lvTanSpan, dLogVarArr);
                TensorPrimitives.Subtract(dLogVarArr, lvTanSpan, dLogVarArr);
                TensorPrimitives.Multiply(dLogVarArr, T.CreateChecked(0.5), dLogVarArr);
                tanValue += TensorPrimitives.Sum(dLogVarArr);
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
        where T : struct, IFloatingPointIeee754<T>
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

    private static int[] PropagateShape<T>(ForwardGradTensor<T> a, ForwardGradTensor<T> b) where T : struct, IFloatingPointIeee754<T>
    {
        return a.shape;
    }

    private static int[] PropagateShape<T>(ForwardGradTensor<T> a) where T : struct, IFloatingPointIeee754<T>
    {
        return a.shape;
    }

    private static int[] ScalarShape()
    {
        return new[] { 1 };
    }

    private static NivaraColumn<T> ApplyDropout<T>(NivaraColumn<T> input, ReadOnlySpan<bool> keepMask, T scale)
        where T : struct, IFloatingPointIeee754<T>
    {
        int n = input.Length;
        var resultBuf = ArrayPool<T>.Shared.Rent(n);

        try
        {
            input.TryGetSpan(out var span);
            for (int i = 0; i < n; i++)
                resultBuf[i] = keepMask[i] ? span[i] * scale : T.Zero;

            return NivaraColumn<T>.Create(resultBuf.AsSpan(0, n));
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
        where T : struct, IFloatingPointIeee754<T>
    {
        int n = input.Length;
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);

        try
        {
            tangent.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            for (int i = 0; i < n; i++)
                resultBuf[i] = keepMask[i] ? gradBuf[i] * scale : T.Zero;

            return NivaraColumn<T>.Create(resultBuf.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
        }
    }

    private static NivaraColumn<T> ApplyKlElementWise<T>(NivaraColumn<T> mean, NivaraColumn<T> logVar)
        where T : struct, IFloatingPointIeee754<T>
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

    private static NivaraColumn<T> ApplySampleNormalForward<T>(NivaraColumn<T> mean, NivaraColumn<T> logVar, NivaraColumn<T> epsilon)
        where T : struct, IFloatingPointIeee754<T>
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

    private static NivaraColumn<T> ApplySampleNormalLogVarTangent<T>(
        NivaraColumn<T> logVar,
        NivaraColumn<T> tangent,
        NivaraColumn<T> epsilon)
        where T : struct, IFloatingPointIeee754<T>
    {
        int n = logVar.Length;
        logVar.TryGetSpan(out var lvSpan);
        tangent.TryGetSpan(out var gSpan);
        epsilon.TryGetSpan(out var eSpan);
        var result = new T[n];
        TensorPrimitives.Multiply(lvSpan, T.CreateChecked(0.5), result);
        TensorPrimitives.Exp(result, result);
        TensorPrimitives.Multiply(result, eSpan, result);
        TensorPrimitives.Multiply(result, gSpan, result);
        TensorPrimitives.Multiply(result, T.CreateChecked(0.5), result);
        return NivaraColumn<T>.Create(result);
    }

    #endregion
}
