using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace Nivara.AutoDiff.Optimizer;

/// <summary>
/// AdamW optimizer: Adam with decoupled weight decay applied directly to the parameters
/// rather than through the gradient. Uses SIMD-accelerated <see cref="TensorPrimitives"/>
/// chains for float/double/Half, and tracks the exponential-average state in pooled buffers.
/// </summary>
public sealed class AdamW<T> : Optimizer<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly double beta1;
    readonly double beta2;
    readonly double eps;

    readonly List<T[]> expAvgBuffers = [];
    readonly List<T[]> expAvgSqBuffers = [];
    int step;

    /// <summary>
    /// Creates an AdamW optimizer with a default learning rate of 0.001.
    /// </summary>
    /// <param name="beta1">First-moment exponential decay rate</param>
    /// <param name="beta2">Second-moment exponential decay rate</param>
    /// <param name="eps">Small constant added to the denominator for numerical stability</param>
    public AdamW(double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8)
        : this(T.CreateChecked(0.001), beta1, beta2, eps)
    {
    }

    /// <summary>
    /// Creates an AdamW optimizer with an explicit learning rate.
    /// </summary>
    /// <param name="learningRate">The default learning rate (must be positive)</param>
    /// <param name="beta1">First-moment exponential decay rate</param>
    /// <param name="beta2">Second-moment exponential decay rate</param>
    /// <param name="eps">Small constant added to the denominator for numerical stability</param>
    public AdamW(T learningRate, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8)
        : base(learningRate)
    {
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
    }

    void applyAdamW(
        ReverseGradTensor<T> tensor, T[] expAvg, T[] expAvgSq, T lr, T wd, T biasCorr1, T biasCorr2)
    {
        var data = tensor.Data;
        var grad = tensor.Grad!;
        int n = data.Length;
        data.TryGetSpan(out var dataSpan);
        grad.TryGetSpan(out var gradSpan);

        ApplyAdamWToSpan(
            data.AsWritableSpan(), dataSpan, gradSpan, expAvg, expAvgSq, n,
            lr, wd, biasCorr1, biasCorr2,
            T.CreateChecked(beta1), T.CreateChecked(beta2), T.CreateChecked(eps));
    }

    static void ApplyAdamWToSpan(
        Span<T> writable, ReadOnlySpan<T> dataSpan, ReadOnlySpan<T> gradSpan,
        T[] expAvg, T[] expAvgSq, int n,
        T lr, T wd, T biasCorr1, T biasCorr2, T beta1T, T beta2T, T epsT)
    {
        if (typeof(T) == typeof(float))
        {
            ApplyAdamW_Kernel_Float(
                MemoryMarshal.Cast<T, float>(expAvg.AsSpan())[..n],
                MemoryMarshal.Cast<T, float>(expAvgSq.AsSpan())[..n],
                MemoryMarshal.Cast<T, float>(writable),
                MemoryMarshal.Cast<T, float>(dataSpan),
                MemoryMarshal.Cast<T, float>(gradSpan),
                n, float.CreateChecked(lr), float.CreateChecked(wd),
                float.CreateChecked(biasCorr1), float.CreateChecked(biasCorr2),
                float.CreateChecked(beta1T), float.CreateChecked(beta2T), float.CreateChecked(epsT));
            return;
        }

        if (typeof(T) == typeof(double))
        {
            ApplyAdamW_Kernel_Double(
                MemoryMarshal.Cast<T, double>(expAvg.AsSpan())[..n],
                MemoryMarshal.Cast<T, double>(expAvgSq.AsSpan())[..n],
                MemoryMarshal.Cast<T, double>(writable),
                MemoryMarshal.Cast<T, double>(dataSpan),
                MemoryMarshal.Cast<T, double>(gradSpan),
                n, double.CreateChecked(lr), double.CreateChecked(wd),
                double.CreateChecked(biasCorr1), double.CreateChecked(biasCorr2),
                double.CreateChecked(beta1T), double.CreateChecked(beta2T), double.CreateChecked(epsT));
            return;
        }

        if (typeof(T) == typeof(Half))
        {
            ApplyAdamW_Kernel_Half(
                MemoryMarshal.Cast<T, Half>(expAvg.AsSpan())[..n],
                MemoryMarshal.Cast<T, Half>(expAvgSq.AsSpan())[..n],
                MemoryMarshal.Cast<T, Half>(writable),
                MemoryMarshal.Cast<T, Half>(dataSpan),
                MemoryMarshal.Cast<T, Half>(gradSpan),
                n, Half.CreateChecked(lr), Half.CreateChecked(wd),
                Half.CreateChecked(biasCorr1), Half.CreateChecked(biasCorr2),
                Half.CreateChecked(beta1T), Half.CreateChecked(beta2T), Half.CreateChecked(epsT));
            return;
        }

        for (int i = 0; i < n; i++)
        {
            expAvg[i] = beta1T * expAvg[i] + (T.One - beta1T) * gradSpan[i];
            expAvgSq[i] = beta2T * expAvgSq[i] + (T.One - beta2T) * gradSpan[i] * gradSpan[i];
        }

        for (int i = 0; i < n; i++)
        {
            var mHat = expAvg[i] / biasCorr1;
            var vHat = expAvgSq[i] / biasCorr2;
            var denom = T.CreateChecked(Math.Sqrt(double.CreateChecked(vHat))) + epsT;
            writable[i] = wd != T.Zero
                ? dataSpan[i] - lr * mHat / denom - lr * wd * dataSpan[i]
                : dataSpan[i] - lr * mHat / denom;
        }
    }

    /// <summary>
    /// Functional AdamW update for a single tensor outside the module system.
    /// Computes the bias-corrected update with decoupled weight decay into a new
    /// <see cref="ReverseGradTensor{T}"/> (with <see cref="ReverseGradTensor{T}.RequiresGrad"/>
    /// = false) while mutating the caller-owned <paramref name="expAvg"/>/<paramref name="expAvgSq"/>
    /// state buffers in place, so consecutive calls accumulate momentum across steps.
    /// </summary>
    /// <param name="step">Current training step, 1-based; drives bias correction.</param>
    public static ReverseGradTensor<T> AdamWUpdate(
        ReverseGradTensor<T> tensor, T learningRate, T[] expAvg, T[] expAvgSq, int step,
        double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, T weightDecay = default)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
        if (expAvg == null)
            throw new ArgumentNullException(nameof(expAvg));
        if (expAvgSq == null)
            throw new ArgumentNullException(nameof(expAvgSq));

        if (tensor.Grad == null)
            throw new InvalidOperationException("Parameter has no gradient computed. Call Backward() first.");

        if (learningRate <= T.Zero)
            throw new ArgumentException("Learning rate must be positive", nameof(learningRate));
        if (step < 1)
            throw new ArgumentOutOfRangeException(nameof(step), "Step must be >= 1");

        int n = tensor.Length;
        if (expAvg.Length < n || expAvgSq.Length < n)
            throw new ArgumentException($"State buffers must be at least {n} elements", nameof(expAvg));

        var biasCorr1 = T.CreateChecked(1.0 - Math.Pow(beta1, step));
        var biasCorr2 = T.CreateChecked(1.0 - Math.Pow(beta2, step));

        return AutoDiffDiagnostics.Measure<T, ReverseGradTensor<T>>(
            "AutoDiffAdamWUpdate",
            n,

            () => ApplyAdamWUpdateResult(
                tensor, expAvg, expAvgSq, n,
                learningRate, weightDecay, biasCorr1, biasCorr2,
                T.CreateChecked(beta1), T.CreateChecked(beta2), T.CreateChecked(eps)),
            $"AutoDiff=AdamWUpdate;Shape=[{string.Join(", ", tensor.Shape)}];WeightDecay={weightDecay != T.Zero};Step={step}");
    }

    static ReverseGradTensor<T> ApplyAdamWUpdateResult(
        ReverseGradTensor<T> tensor, T[] expAvg, T[] expAvgSq, int n,
        T lr, T wd, T biasCorr1, T biasCorr2, T beta1T, T beta2T, T epsT)
    {
        var data = tensor.Data;
        var grad = tensor.Grad!;
        data.TryGetSpan(out var dataSpan);
        grad.TryGetSpan(out var gradSpan);

        var result = new T[n];
        ApplyAdamWToSpan(
            result, dataSpan, gradSpan, expAvg, expAvgSq, n,
            lr, wd, biasCorr1, biasCorr2, beta1T, beta2T, epsT);
        return new ReverseGradTensor<T>(NivaraColumn<T>.CreateFromOwnedArray(result), requiresGrad: false, tensor.shape);
    }

    static void ApplyAdamW_Kernel_Float(
        Span<float> expAvg, Span<float> expAvgSq, Span<float> writable,
        ReadOnlySpan<float> dataSpan, ReadOnlySpan<float> gradSpan,
        int n, float lr, float wd, float biasCorr1, float biasCorr2,
        float beta1T, float beta2T, float epsT)
    {
        float oneMinusBeta1 = 1f - beta1T;
        float oneMinusBeta2 = 1f - beta2T;
        var tempArr = ArrayPool<float>.Shared.Rent(n);
        var updateArr = ArrayPool<float>.Shared.Rent(n);
        try
        {
            var temp = tempArr.AsSpan(0, n);
            var update = updateArr.AsSpan(0, n);

            TensorPrimitives.Multiply(gradSpan, gradSpan, temp);
            TensorPrimitives.Multiply(temp, oneMinusBeta2, temp);
            TensorPrimitives.MultiplyAdd(expAvgSq, beta2T, temp, expAvgSq);

            TensorPrimitives.Multiply(gradSpan, oneMinusBeta1, temp);
            TensorPrimitives.MultiplyAdd(expAvg, beta1T, temp, expAvg);

            TensorPrimitives.Divide(expAvg, biasCorr1, update);
            TensorPrimitives.Divide(expAvgSq, biasCorr2, temp);
            TensorPrimitives.Sqrt(temp, temp);
            TensorPrimitives.Add(temp, epsT, temp);
            TensorPrimitives.Divide(update, temp, update);
            TensorPrimitives.Multiply(update, lr, update);

            if (wd != 0f)
            {
                TensorPrimitives.Multiply(dataSpan, lr * wd, temp);
                TensorPrimitives.Add(update, temp, update);
            }

            for (int i = 0; i < n; i++)
                writable[i] = dataSpan[i] - update[i];
        }
        finally
        {
            ArrayPool<float>.Shared.Return(tempArr);
            ArrayPool<float>.Shared.Return(updateArr);
        }
    }

    static void ApplyAdamW_Kernel_Double(
        Span<double> expAvg, Span<double> expAvgSq, Span<double> writable,
        ReadOnlySpan<double> dataSpan, ReadOnlySpan<double> gradSpan,
        int n, double lr, double wd, double biasCorr1, double biasCorr2,
        double beta1T, double beta2T, double epsT)
    {
        double oneMinusBeta1 = 1.0 - beta1T;
        double oneMinusBeta2 = 1.0 - beta2T;
        var tempArr = ArrayPool<double>.Shared.Rent(n);
        var updateArr = ArrayPool<double>.Shared.Rent(n);
        try
        {
            var temp = tempArr.AsSpan(0, n);
            var update = updateArr.AsSpan(0, n);

            TensorPrimitives.Multiply(gradSpan, gradSpan, temp);
            TensorPrimitives.Multiply(temp, oneMinusBeta2, temp);
            TensorPrimitives.MultiplyAdd(expAvgSq, beta2T, temp, expAvgSq);

            TensorPrimitives.Multiply(gradSpan, oneMinusBeta1, temp);
            TensorPrimitives.MultiplyAdd(expAvg, beta1T, temp, expAvg);

            TensorPrimitives.Divide(expAvg, biasCorr1, update);
            TensorPrimitives.Divide(expAvgSq, biasCorr2, temp);
            TensorPrimitives.Sqrt(temp, temp);
            TensorPrimitives.Add(temp, epsT, temp);
            TensorPrimitives.Divide(update, temp, update);
            TensorPrimitives.Multiply(update, lr, update);

            if (wd != 0.0)
            {
                TensorPrimitives.Multiply(dataSpan, lr * wd, temp);
                TensorPrimitives.Add(update, temp, update);
            }

            for (int i = 0; i < n; i++)
                writable[i] = dataSpan[i] - update[i];
        }
        finally
        {
            ArrayPool<double>.Shared.Return(tempArr);
            ArrayPool<double>.Shared.Return(updateArr);
        }
    }

    static void ApplyAdamW_Kernel_Half(
        Span<Half> expAvg, Span<Half> expAvgSq, Span<Half> writable,
        ReadOnlySpan<Half> dataSpan, ReadOnlySpan<Half> gradSpan,
        int n, Half lr, Half wd, Half biasCorr1, Half biasCorr2,
        Half beta1T, Half beta2T, Half epsT)
    {
        Half oneMinusBeta1 = (Half)(1f - (float)beta1T);
        Half oneMinusBeta2 = (Half)(1f - (float)beta2T);
        var tempArr = ArrayPool<Half>.Shared.Rent(n);
        var updateArr = ArrayPool<Half>.Shared.Rent(n);
        try
        {
            var temp = tempArr.AsSpan(0, n);
            var update = updateArr.AsSpan(0, n);

            TensorPrimitives.Multiply(gradSpan, gradSpan, temp);
            TensorPrimitives.Multiply(temp, oneMinusBeta2, temp);
            TensorPrimitives.MultiplyAdd(expAvgSq, beta2T, temp, expAvgSq);

            TensorPrimitives.Multiply(gradSpan, oneMinusBeta1, temp);
            TensorPrimitives.MultiplyAdd(expAvg, beta1T, temp, expAvg);

            TensorPrimitives.Divide(expAvg, biasCorr1, update);
            TensorPrimitives.Divide(expAvgSq, biasCorr2, temp);
            TensorPrimitives.Sqrt(temp, temp);
            TensorPrimitives.Add(temp, epsT, temp);
            TensorPrimitives.Divide(update, temp, update);
            TensorPrimitives.Multiply(update, lr, update);

            if ((float)wd != 0f)
            {
                TensorPrimitives.Multiply(dataSpan, (Half)((float)lr * (float)wd), temp);
                TensorPrimitives.Add(update, temp, update);
            }

            for (int i = 0; i < n; i++)
                writable[i] = (Half)((float)dataSpan[i] - (float)update[i]);
        }
        finally
        {
            ArrayPool<Half>.Shared.Return(tempArr);
            ArrayPool<Half>.Shared.Return(updateArr);
        }
    }

    private void ensureBuffer(int idx, int size)
    {
        while (idx >= expAvgBuffers.Count)
        {
            var expAvg = ArrayPool<T>.Shared.Rent(size);
            expAvg.AsSpan(0, size).Clear();
            expAvgBuffers.Add(expAvg);

            var expAvgSq = ArrayPool<T>.Shared.Rent(size);
            expAvgSq.AsSpan(0, size).Clear();
            expAvgSqBuffers.Add(expAvgSq);
        }
    }

    /// <summary>
    /// Applies one AdamW step to every registered parameter that has a computed gradient,
    /// updating the parameter tensors in place, touching each parameter, and advancing the
    /// internal step counter used for bias correction.
    /// </summary>
    public override void Step()
    {
        step++;
        var bufIdx = 0;

        var biasCorr1 = T.CreateChecked(1.0 - Math.Pow(beta1, step));
        var biasCorr2 = T.CreateChecked(1.0 - Math.Pow(beta2, step));

        foreach (var group in ParameterGroups)
        {
            var lr = group.LearningRate;
            var wd = group.WeightDecay;

            foreach (var param in group.Parameters)
            {
                var tensor = param.Tensor;
                if (tensor.Grad == null || !tensor.RequiresGrad)
                    continue;

                ensureBuffer(bufIdx, tensor.Length);
                applyAdamW(tensor, expAvgBuffers[bufIdx], expAvgSqBuffers[bufIdx], lr, wd, biasCorr1, biasCorr2);
                param.Touch();
                bufIdx++;
            }
        }
    }

    /// <summary>
    /// Saves the step counter and the exponential-average state buffers keyed by index
    /// (e.g. <c>step</c>, <c>expAvg_0</c>, <c>expAvgSq_0</c>).
    /// </summary>
    /// <returns>A state dictionary for <see cref="LoadStateDict"/></returns>
    public override Dictionary<string, T[]> StateDict()
    {
        var state = new Dictionary<string, T[]> { ["step"] = [T.CreateChecked(step)] };
        for (int i = 0; i < expAvgBuffers.Count; i++)
        {
            var copy = new T[expAvgBuffers[i].Length];
            expAvgBuffers[i].AsSpan(0, copy.Length).CopyTo(copy);
            state[$"expAvg_{i}"] = copy;
        }
        for (int i = 0; i < expAvgSqBuffers.Count; i++)
        {
            var copy = new T[expAvgSqBuffers[i].Length];
            expAvgSqBuffers[i].AsSpan(0, copy.Length).CopyTo(copy);
            state[$"expAvgSq_{i}"] = copy;
        }
        return state;
    }

    /// <summary>
    /// Restores the step counter and exponential-average state buffers saved by
    /// <see cref="StateDict"/>.
    /// </summary>
    /// <param name="state">The state dictionary to load from</param>
    public override void LoadStateDict(Dictionary<string, T[]> state)
    {
        if (state.TryGetValue("step", out var stepVal))
            step = int.CreateChecked(stepVal[0]);

        int i = 0;
        while (state.TryGetValue($"expAvg_{i}", out var buf))
        {
            ensureBuffer(i, buf.Length);
            buf.AsSpan(0, Math.Min(buf.Length, expAvgBuffers[i].Length)).CopyTo(expAvgBuffers[i]);

            if (state.TryGetValue($"expAvgSq_{i}", out var sqBuf))
                sqBuf.AsSpan(0, Math.Min(sqBuf.Length, expAvgSqBuffers[i].Length)).CopyTo(expAvgSqBuffers[i]);

            i++;
        }
    }

    /// <summary>
    /// Returns pooled exponential-average buffers to the shared array pool.
    /// </summary>
    protected override void DisposeManaged()
    {
        foreach (var buf in expAvgBuffers.Concat(expAvgSqBuffers))
        {
            if (buf != null)
                ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
        expAvgBuffers.Clear();
        expAvgSqBuffers.Clear();
    }
}
