using System.Buffers;
using System.Numerics;

namespace Nivara.AutoDiff.Optimizer;

public sealed class Adam<T> : Optimizer<T> where T : struct, INumber<T>
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
            expAvgBuffers.Add(ArrayPool<T>.Shared.Rent(size));
            expAvgSqBuffers.Add(ArrayPool<T>.Shared.Rent(size));
        }
    }

    ReverseGradTensor<T> applyAdam(
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
        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            var result = buf.AsSpan(0, n);

            for (int i = 0; i < n; i++)
            {
                expAvg[i] = beta1T * expAvg[i] + (T.One - beta1T) * gradSpan[i];
                expAvgSq[i] = beta2T * expAvgSq[i] + (T.One - beta2T) * gradSpan[i] * gradSpan[i];
            }

            if (wd != T.Zero)
            {
                for (int i = 0; i < n; i++)
                {
                    var mHat = expAvg[i] / biasCorr1;
                    var vHat = expAvgSq[i] / biasCorr2;
                    var denom = T.CreateChecked(Math.Sqrt(double.CreateChecked(vHat))) + epsT;
                    result[i] = dataSpan[i] - lr * mHat / denom - lr * wd * dataSpan[i];
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    var mHat = expAvg[i] / biasCorr1;
                    var vHat = expAvgSq[i] / biasCorr2;
                    var denom = T.CreateChecked(Math.Sqrt(double.CreateChecked(vHat))) + epsT;
                    result[i] = dataSpan[i] - lr * mHat / denom;
                }
            }

            return new ReverseGradTensor<T>(NivaraColumn<T>.Create(result), requiresGrad: true, tensor.shape);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
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
                var newTensor = applyAdam(tensor, expAvgBuffers[bufIdx], expAvgSqBuffers[bufIdx], lr, wd, biasCorr1, biasCorr2);
                param.Tensor = newTensor;
                bufIdx++;
            }
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
