using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Optimizer;

/// <summary>
/// Stochastic gradient descent optimizer with optional momentum and per-group weight decay.
/// Writes updates in place on the parameter tensors (no <c>requiresGrad</c> replacement),
/// and tracks momentum in pooled velocity buffers.
/// </summary>
public sealed class SGD<T> : Optimizer<T> where T : struct, IFloatingPointIeee754<T>
{
    static void stepNoMomentumInPlace(NivaraColumn<T> data, NivaraColumn<T> grad, Span<T> writable, T lr, T wd)
    {
        data.TryGetSpan(out var dataSpan);
        grad.TryGetSpan(out var gradSpan);
        int n = data.Length;

        for (int i = 0; i < n; i++)
            writable[i] = wd != T.Zero
                ? dataSpan[i] - lr * (wd * dataSpan[i] + gradSpan[i])
                : dataSpan[i] - lr * gradSpan[i];
    }

    static void stepWithMomentumInPlace(NivaraColumn<T> data, NivaraColumn<T> grad, Span<T> writable, T[] velocity, T lr, T wd)
    {
        data.TryGetSpan(out var dataSpan);
        grad.TryGetSpan(out var gradSpan);
        int n = data.Length;
        var momentumT = T.CreateChecked(0.9);

        for (int i = 0; i < n; i++)
            velocity[i] = wd != T.Zero
                ? momentumT * velocity[i] + lr * (wd * dataSpan[i] + gradSpan[i])
                : momentumT * velocity[i] + lr * gradSpan[i];

        for (int i = 0; i < n; i++)
            writable[i] = dataSpan[i] - velocity[i];
    }

    /// <summary>
    /// Functional SGD update for a single tensor outside the module system. Computes the
    /// update into a new <see cref="ReverseGradTensor{T}"/> (with
    /// <see cref="ReverseGradTensor{T}.RequiresGrad"/> = false); no momentum is tracked.
    /// </summary>
    /// <param name="tensor">The tensor to update</param>
    /// <param name="learningRate">The learning rate (must be positive)</param>
    /// <param name="weightDecay">The L2 weight decay</param>
    /// <returns>A new tensor holding the updated values</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="tensor"/> is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when the tensor has no gradient computed</exception>
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

        return new ReverseGradTensor<T>(NivaraColumn<T>.CreateFromOwnedArray(result), requiresGrad: false, tensor.shape);
    }

    readonly double momentum;
    readonly List<T[]> velocityBuffers = [];

    /// <summary>
    /// Creates an SGD optimizer with an optional momentum coefficient.
    /// </summary>
    /// <param name="learningRate">The default learning rate (must be positive)</param>
    /// <param name="momentum">Momentum coefficient in <c>[0, 1)</c>; zero disables momentum</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when momentum is outside <c>[0, 1)</c></exception>
    public SGD(T learningRate, double momentum = 0.0)
        : base(learningRate)
    {
        if (momentum < 0.0 || momentum >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(momentum), "Momentum must be in [0, 1).");

        this.momentum = momentum;
    }

    private void ensureBuffer(int idx, int size)
    {
        while (idx >= velocityBuffers.Count)
        {
            var buf = ArrayPool<T>.Shared.Rent(size);
            buf.AsSpan(0, size).Clear();
            velocityBuffers.Add(buf);
        }
    }

    /// <summary>
    /// Applies one SGD step to every registered parameter that has a computed gradient,
    /// updating the parameter tensors in place and touching each parameter.
    /// </summary>
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
                    ensureBuffer(velIdx, tensor.Length);
                    stepWithMomentumInPlace(tensor.Data, tensor.Grad!, tensor.Data.AsWritableSpan(), velocityBuffers[velIdx], lr, wd);
                    velIdx++;
                }
                else
                {
                    stepNoMomentumInPlace(tensor.Data, tensor.Grad!, tensor.Data.AsWritableSpan(), lr, wd);
                }
                param.Touch();
            }
        }
    }

    /// <summary>
    /// Saves the momentum velocity buffers keyed by index (e.g. <c>velocity_0</c>).
    /// </summary>
    /// <returns>A state dictionary for <see cref="LoadStateDict"/></returns>
    public override Dictionary<string, T[]> StateDict()
    {
        var state = new Dictionary<string, T[]>();
        for (int i = 0; i < velocityBuffers.Count; i++)
        {
            var copy = new T[velocityBuffers[i].Length];
            velocityBuffers[i].AsSpan(0, copy.Length).CopyTo(copy);
            state[$"velocity_{i}"] = copy;
        }
        return state;
    }

    /// <summary>
    /// Restores momentum velocity buffers saved by <see cref="StateDict"/>.
    /// </summary>
    /// <param name="state">The state dictionary to load from</param>
    public override void LoadStateDict(Dictionary<string, T[]> state)
    {
        int i = 0;
        while (state.TryGetValue($"velocity_{i}", out var buf))
        {
            ensureBuffer(i, buf.Length);
            buf.AsSpan(0, Math.Min(buf.Length, velocityBuffers[i].Length)).CopyTo(velocityBuffers[i]);
            i++;
        }
    }

    /// <summary>
    /// Returns pooled velocity buffers to the shared array pool.
    /// </summary>
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
