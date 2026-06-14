using Nivara.Helpers;
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Optimizer;

public sealed class SGD<T> : Optimizer<T> where T : struct, INumber<T>
{
    static ReverseGradTensor<T> stepNoMomentum(ReverseGradTensor<T> tensor, T lr, T wd)
    {
        var result = SgdUpdate(tensor, lr, wd);
        return new ReverseGradTensor<T>(result.Data, requiresGrad: true, tensor.shape);
    }

    static ReverseGradTensor<T> stepWithMomentum(ReverseGradTensor<T> tensor, T[] velocity, T lr, T wd)
    {
        var data = tensor.Data;
        var grad = tensor.Grad!;
        int n = data.Length;
        var momentumT = T.CreateChecked(0.9);

        if (!data.HasNulls && !grad.HasNulls)
        {
            data.TryGetSpan(out var dataSpan);
            grad.TryGetSpan(out var gradSpan);
            var result = new T[n];

            for (int i = 0; i < n; i++)
                velocity[i] = wd != T.Zero
                    ? momentumT * velocity[i] + lr * (wd * dataSpan[i] + gradSpan[i])
                    : momentumT * velocity[i] + lr * gradSpan[i];

            for (int i = 0; i < n; i++)
                result[i] = dataSpan[i] - velocity[i];

            return new ReverseGradTensor<T>(NivaraColumn<T>.Create(result), requiresGrad: true, tensor.shape);
        }

        var dataBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            data.CopyTo(dataBuf.AsSpan(0, n), T.Zero);
            grad.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            var hasNulls = NivaraColumnUtility.MergeNullMasks(data, grad, nullMask.AsSpan(0, n));

            for (int i = 0; i < n; i++)
            {
                if (nullMask[i])
                {
                    velocity[i] = T.Zero;
                    resultBuf[i] = dataBuf[i];
                }
                else
                {
                    velocity[i] = wd != T.Zero
                        ? momentumT * velocity[i] + lr * (wd * dataBuf[i] + gradBuf[i])
                        : momentumT * velocity[i] + lr * gradBuf[i];
                    resultBuf[i] = dataBuf[i] - velocity[i];
                }
            }

            var resultColumn = hasNulls
                ? NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n))
                : NivaraColumn<T>.Create(resultBuf.AsSpan(0, n));

            return new ReverseGradTensor<T>(resultColumn, requiresGrad: true, tensor.shape);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(dataBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    public static ReverseGradTensor<T> SgdUpdate(ReverseGradTensor<T> tensor, T learningRate, T weightDecay = default)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (tensor.Grad == null)
            throw new InvalidOperationException("Parameter has no gradient computed. Call Backward() first.");

        if (learningRate <= T.Zero)
            throw new ArgumentException("Learning rate must be positive", nameof(learningRate));

        var data = tensor.Data;
        var grad = tensor.Grad!;
        int n = data.Length;

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffSgdUpdate",
            n,
            data.HasNulls || grad.HasNulls,
            () => ApplySgdUpdate(tensor, learningRate, weightDecay, data, grad, n),
            $"AutoDiff=SgdUpdate;Shape=[{string.Join(", ", tensor.Shape)}];WeightDecay={weightDecay != T.Zero}");
    }

    static ReverseGradTensor<T> ApplySgdUpdate(
        ReverseGradTensor<T> tensor,
        T learningRate,
        T weightDecay,
        NivaraColumn<T> data,
        NivaraColumn<T> grad,
        int n)
    {
        if (!data.HasNulls && !grad.HasNulls)
        {
            data.TryGetSpan(out var dataSpan);
            grad.TryGetSpan(out var gradSpan);
            var result = new T[n];

            if (weightDecay != T.Zero)
            {
                TensorPrimitives.Multiply(dataSpan, weightDecay, result);
                TensorPrimitives.Add(gradSpan, result, result);
                TensorPrimitives.Multiply(result, learningRate, result);
                TensorPrimitives.Subtract(dataSpan, result, result);
            }
            else
            {
                TensorPrimitives.Multiply(gradSpan, learningRate, result);
                TensorPrimitives.Subtract(dataSpan, result, result);
            }

            return new ReverseGradTensor<T>(NivaraColumn<T>.Create(result), requiresGrad: false, tensor.shape);
        }

        var dataBuf = ArrayPool<T>.Shared.Rent(n);
        var gradBuf = ArrayPool<T>.Shared.Rent(n);
        var resultBuf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            data.CopyTo(dataBuf.AsSpan(0, n), T.Zero);
            grad.CopyTo(gradBuf.AsSpan(0, n), T.Zero);
            var hasNulls = NivaraColumnUtility.MergeNullMasks(data, grad, nullMask.AsSpan(0, n));

            if (weightDecay != T.Zero)
            {
                TensorPrimitives.Multiply(dataBuf.AsSpan(0, n), weightDecay, resultBuf.AsSpan(0, n));
                TensorPrimitives.Add(gradBuf.AsSpan(0, n), resultBuf.AsSpan(0, n), resultBuf.AsSpan(0, n));
                TensorPrimitives.Multiply(resultBuf.AsSpan(0, n), learningRate, resultBuf.AsSpan(0, n));
                TensorPrimitives.Subtract(dataBuf.AsSpan(0, n), resultBuf.AsSpan(0, n), resultBuf.AsSpan(0, n));
            }
            else
            {
                TensorPrimitives.Multiply(gradBuf.AsSpan(0, n), learningRate, resultBuf.AsSpan(0, n));
                TensorPrimitives.Subtract(dataBuf.AsSpan(0, n), resultBuf.AsSpan(0, n), resultBuf.AsSpan(0, n));
            }

            var resultColumn = hasNulls
                ? NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), nullMask.AsSpan(0, n))
                : NivaraColumn<T>.Create(resultBuf.AsSpan(0, n));

            return new ReverseGradTensor<T>(resultColumn, requiresGrad: false, tensor.shape);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(dataBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(gradBuf, clearArray: true);
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
        }
    }

    readonly double momentum;
    readonly List<T[]> velocityBuffers = [];

    public SGD(T learningRate, double momentum = 0.0)
        : base(learningRate)
    {
        if (momentum < 0.0 || momentum >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(momentum), "Momentum must be in [0, 1).");

        this.momentum = momentum;
    }

    void ensureVelocityBuffer(int idx, int size)
    {
        while (idx >= velocityBuffers.Count)
            velocityBuffers.Add(ArrayPool<T>.Shared.Rent(size));
    }

    public override void Step()
    {
        var velIdx = 0;

        foreach (var group in ParameterGroups)
        {
            var lr = group.LearningRate;
            var wd = group.WeightDecay;

            foreach (var param in group.Parameters)
            {
                var tensor = param.Tensor;
                if (tensor.Grad == null || !tensor.RequiresGrad)
                    continue;

                if (momentum > 0.0)
                {
                    ensureVelocityBuffer(velIdx, tensor.Length);
                    var newTensor = stepWithMomentum(tensor, velocityBuffers[velIdx], lr, wd);
                    param.Tensor = newTensor;
                    velIdx++;
                }
                else
                {
                    var newTensor = stepNoMomentum(tensor, lr, wd);
                    param.Tensor = newTensor;
                }
            }
        }
    }

    protected override void DisposeManaged()
    {
        foreach (var buf in velocityBuffers)
        {
            if (buf != null)
                ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
        velocityBuffers.Clear();
    }
}
