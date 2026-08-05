using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace Nivara.AutoDiff.Optimizer;

public sealed class Adam<T> : Optimizer<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly double beta1;
    readonly double beta2;
    readonly double eps;

    readonly List<T[]> expAvgBuffers = [];
    readonly List<T[]> expAvgSqBuffers = [];
    int step;

    public Adam(double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8)
        : this(T.CreateChecked(0.001), beta1, beta2, eps)
    {
    }

    public Adam(T learningRate, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8)
        : base(learningRate)
    {
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
    }

    void ensureBuffer(int idx, int size)
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

    void applyAdam(
        ReverseGradTensor<T> tensor, T[] expAvg, T[] expAvgSq, T lr, T wd, T biasCorr1, T biasCorr2)
    {
        var data = tensor.Data;
        var grad = tensor.Grad!;
        int n = data.Length;
        var beta1T = T.CreateChecked(beta1);
        var beta2T = T.CreateChecked(beta2);
        var epsT = T.CreateChecked(eps);

        data.TryGetSpan(out var dataSpan);
        grad.TryGetSpan(out var gradSpan);
        var writable = data.AsWritableSpan();

        if (typeof(T) == typeof(float))
        {
            ApplyAdam_Kernel_Float(
                MemoryMarshal.Cast<T, float>(expAvg.AsSpan())[..n],
                MemoryMarshal.Cast<T, float>(expAvgSq.AsSpan())[..n],
                MemoryMarshal.Cast<T, float>(writable),
                MemoryMarshal.Cast<T, float>(dataSpan),
                MemoryMarshal.Cast<T, float>(gradSpan),
                n, (float)(object)lr!, (float)(object)wd!,
                (float)(object)biasCorr1!, (float)(object)biasCorr2!,
                (float)(object)beta1T!, (float)(object)beta2T!, (float)(object)epsT!);
            return;
        }

        if (typeof(T) == typeof(double))
        {
            ApplyAdam_Kernel_Double(
                MemoryMarshal.Cast<T, double>(expAvg.AsSpan())[..n],
                MemoryMarshal.Cast<T, double>(expAvgSq.AsSpan())[..n],
                MemoryMarshal.Cast<T, double>(writable),
                MemoryMarshal.Cast<T, double>(dataSpan),
                MemoryMarshal.Cast<T, double>(gradSpan),
                n, (double)(object)lr!, (double)(object)wd!,
                (double)(object)biasCorr1!, (double)(object)biasCorr2!,
                (double)(object)beta1T!, (double)(object)beta2T!, (double)(object)epsT!);
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

    static void ApplyAdam_Kernel_Float(
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

    static void ApplyAdam_Kernel_Double(
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
                applyAdam(tensor, expAvgBuffers[bufIdx], expAvgSqBuffers[bufIdx], lr, wd, biasCorr1, biasCorr2);
                param.Touch();
                bufIdx++;
            }
        }
    }

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

    public override void LoadStateDict(Dictionary<string, T[]> state)
    {
        if (state.TryGetValue("step", out var stepVal))
            step = int.CreateChecked(stepVal[0]);

        for (int i = 0; i < expAvgBuffers.Count; i++)
        {
            if (state.TryGetValue($"expAvg_{i}", out var buf))
                buf.AsSpan(0, Math.Min(buf.Length, expAvgBuffers[i].Length)).CopyTo(expAvgBuffers[i]);
            if (state.TryGetValue($"expAvgSq_{i}", out var sqBuf))
                sqBuf.AsSpan(0, Math.Min(sqBuf.Length, expAvgSqBuffers[i].Length)).CopyTo(expAvgSqBuffers[i]);
        }
    }

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
