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

        data.TryGetSpan(out var dataSpan);
        grad.TryGetSpan(out var gradSpan);
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);

            for (int i = 0; i < n; i++)
                velocity[i] = wd != T.Zero
                    ? momentumT * velocity[i] + lr * (wd * dataSpan[i] + gradSpan[i])
                    : momentumT * velocity[i] + lr * gradSpan[i];

            for (int i = 0; i < n; i++)
                result[i] = dataSpan[i] - velocity[i];

            return new ReverseGradTensor<T>(NivaraColumn<T>.Create(result), requiresGrad: true, tensor.shape);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
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

        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            data.TryGetSpan(out var dataSpan);
            grad.TryGetSpan(out var gradSpan);
            var result = buf.AsSpan(0, n);

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
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
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
