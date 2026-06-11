using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Optimizer;

public static class SgdOptimizer
{
    public static ReverseGradTensor<T> SgdUpdate<T>(ReverseGradTensor<T> parameter, T learningRate) where T : struct, INumber<T>
    {
        if (parameter == null)
            throw new ArgumentNullException(nameof(parameter));

        if (parameter.Grad == null)
            throw new InvalidOperationException("Parameter has no gradient computed. Call Backward() first.");

        if (learningRate <= T.Zero)
            throw new ArgumentException("Learning rate must be positive", nameof(learningRate));

        var grad = parameter.Grad;
        var data = parameter.Data;
        int n = data.Length;

        if (!data.HasNulls && !grad.HasNulls)
        {
            data.TryGetSpan(out var dataSpan);
            grad.TryGetSpan(out var gradSpan);
            var result = new T[n];
            TensorPrimitives.Multiply(gradSpan, learningRate, result);
            TensorPrimitives.Subtract(dataSpan, result, result);
            return new ReverseGradTensor<T>(NivaraColumn<T>.Create(result), requiresGrad: false, parameter.shape);
        }

        var dataBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            data.CopyTo(dataBuf.AsSpan(0, n), T.Zero);
            grad.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            var dataHasNulls = data.TryGetNullMask(out var dataMask);

            TensorPrimitives.Multiply(gradBuf.AsSpan(0, n), learningRate, gradBuf.AsSpan(0, n));
            TensorPrimitives.Subtract(dataBuf.AsSpan(0, n), gradBuf.AsSpan(0, n), dataBuf.AsSpan(0, n));

            nullMask.AsSpan(0, n).Clear();
            for (int i = 0; i < n; i++)
                if (dataHasNulls && dataMask[i])
                    nullMask[i] = true;

            var resultColumn = NivaraColumn<T>.CreateFromSpans(dataBuf.AsSpan(0, n), nullMask.AsSpan(0, n));
            return new ReverseGradTensor<T>(resultColumn, requiresGrad: false, parameter.shape);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(dataBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }
}
