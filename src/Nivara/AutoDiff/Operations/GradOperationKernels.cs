using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Operations;

/// <summary>
/// Shared column-level kernels used by both <see cref="ForwardGradOperations"/> and
/// <see cref="ReverseGradOperations"/>. Reverse-mode VJP kernels and forward-mode JVP
/// tangents are often the same math (element-wise chains over <see cref="TensorPrimitives"/>);
/// hosting them here avoids duplicating the implementations in both operations classes.
/// </summary>
internal static class GradOperationKernels
{
    /// <summary>
    /// Dropout forward: result[i] = keepMask[i] ? input[i] * scale : 0.
    /// </summary>
    public static NivaraColumn<T> ApplyDropout<T>(NivaraColumn<T> input, ReadOnlySpan<bool> keepMask, T scale)
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

    /// <summary>
    /// Dropout gradient (reverse VJP) and tangent (forward JVP): the same transform,
    /// result[i] = keepMask[i] ? gradOutput[i] * scale : 0.
    /// </summary>
    public static NivaraColumn<T> ApplyDropoutGradient<T>(
        NivaraColumn<T> input,
        NivaraColumn<T> gradOutput,
        ReadOnlySpan<bool> keepMask,
        T scale)
        where T : struct, IFloatingPointIeee754<T>
    {
        int n = input.Length;
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);

        try
        {
            gradOutput.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
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

    /// <summary>
    /// KL(mean, logVar) element-wise: -0.5 * (1 + logVar - mean^2 - exp(logVar)).
    /// </summary>
    public static NivaraColumn<T> ApplyKlElementWise<T>(NivaraColumn<T> mean, NivaraColumn<T> logVar)
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
        return NivaraColumn<T>.CreateFromOwnedArray(result);
    }

    /// <summary>
    /// KL dKL/dmean = mean.
    /// </summary>
    public static NivaraColumn<T> ApplyKlMeanGradient<T>(NivaraColumn<T> mean, NivaraColumn<T> gradOutput)
        where T : struct, IFloatingPointIeee754<T>
    {
        int n = mean.Length;
        mean.TryGetSpan(out var mSpan);
        gradOutput.TryGetSpan(out var gSpan);
        var result = new T[n];
        TensorPrimitives.Multiply(mSpan, gSpan, result);
        return NivaraColumn<T>.CreateFromOwnedArray(result);
    }

    /// <summary>
    /// KL dKL/dlogVar = -0.5 * (1 - exp(logVar)).
    /// </summary>
    public static NivaraColumn<T> ApplyKlLogVarGradient<T>(NivaraColumn<T> logVar, NivaraColumn<T> gradOutput)
        where T : struct, IFloatingPointIeee754<T>
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
        return NivaraColumn<T>.CreateFromOwnedArray(result);
    }

    /// <summary>
    /// Reparameterized sample: mean + exp(0.5 * logVar) * epsilon.
    /// </summary>
    public static NivaraColumn<T> ApplySampleNormalForward<T>(NivaraColumn<T> mean, NivaraColumn<T> logVar, NivaraColumn<T> epsilon)
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
        return NivaraColumn<T>.CreateFromOwnedArray(result);
    }

    /// <summary>
    /// SampleNormal gradient (reverse VJP) and tangent (forward JVP) w.r.t. logVar:
    /// 0.5 * exp(0.5 * logVar) * epsilon * gradOutput.
    /// </summary>
    public static NivaraColumn<T> ApplySampleNormalLogVarGradient<T>(NivaraColumn<T> logVar, NivaraColumn<T> gradOutput, NivaraColumn<T> epsilon)
        where T : struct, IFloatingPointIeee754<T>
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
        return NivaraColumn<T>.CreateFromOwnedArray(result);
    }

    /// <summary>
    /// Pow forward: input^exponent.
    /// </summary>
    public static NivaraColumn<T> ApplyPow<T>(NivaraColumn<T> input, double exponent)
        where T : struct, IFloatingPointIeee754<T>
    {
        int n = input.Length;
        input.TryGetSpan(out var span);
        var result = new T[n];
        TensorPrimitives.Pow(span, T.CreateChecked(exponent), result);
        return NivaraColumn<T>.CreateFromOwnedArray(result);
    }

    /// <summary>
    /// Pow gradient (reverse VJP) and tangent (forward JVP): exponent * input^(exponent-1).
    /// </summary>
    public static NivaraColumn<T> ApplyPowGradient<T>(NivaraColumn<T> input, NivaraColumn<T> gradOutput, double exponent)
        where T : struct, IFloatingPointIeee754<T>
    {
        int n = input.Length;
        input.TryGetSpan(out var inSpan);
        gradOutput.TryGetSpan(out var gSpan);
        var result = new T[n];
        TensorPrimitives.Pow(inSpan, T.CreateChecked(exponent - 1.0), result);
        TensorPrimitives.Multiply(result, T.CreateChecked(exponent), result);
        TensorPrimitives.Multiply(result, gSpan, result);
        return NivaraColumn<T>.CreateFromOwnedArray(result);
    }

    /// <summary>
    /// RMSNorm forward: input / sqrt(mean(input^2) + eps).
    /// </summary>
    public static NivaraColumn<T> ApplyRMSNorm<T>(NivaraColumn<T> input, double eps)
        where T : struct, IFloatingPointIeee754<T>
    {
        int n = input.Length;
        input.TryGetSpan(out var span);
        var result = new T[n];
        double sumSq = double.CreateChecked(TensorPrimitives.SumOfSquares(span));
        double rms = Math.Sqrt(sumSq / n + eps);
        T invRms = T.CreateChecked(1.0 / rms);
        TensorPrimitives.Multiply(span, invRms, result);
        return NivaraColumn<T>.CreateFromOwnedArray(result);
    }

    /// <summary>
    /// RMSNorm gradient (reverse VJP) and tangent (forward JVP). The RMSNorm Jacobian is
    /// symmetric, so the same kernel serves both directions.
    /// </summary>
    public static NivaraColumn<T> ApplyRMSNormGradient<T>(
        NivaraColumn<T> input, NivaraColumn<T> gradOutput, double eps)
        where T : struct, IFloatingPointIeee754<T>
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

        TensorPrimitives.Multiply(gSpan, invRms, result);
        T negScale = -scale;
        TensorPrimitives.MultiplyAdd(inSpan, negScale, result, result);

        return NivaraColumn<T>.CreateFromOwnedArray(result);
    }

    /// <summary>
    /// Broadcasts a length-1 scalar gradient to a column of <paramref name="targetLength"/>.
    /// </summary>
    public static NivaraColumn<T> BroadcastGradient<T>(NivaraColumn<T> scalarGrad, int targetLength)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (scalarGrad.Length != 1)
            throw new ArgumentException($"Expected scalar gradient with length 1, got {scalarGrad.Length}");

        scalarGrad.TryGetSpan(out var span);
        var filled = new T[targetLength];
        Array.Fill(filled, span[0]);
        return NivaraColumn<T>.CreateFromOwnedArray(filled);
    }
}
