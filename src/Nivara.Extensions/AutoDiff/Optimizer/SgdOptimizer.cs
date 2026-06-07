using System.Numerics;

namespace Nivara.Extensions.AutoDiff.Optimizer;

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
        var resultData = new T[n];

        for (int i = 0; i < n; i++)
        {
            if (!grad.IsNull(i))
            {
                resultData[i] = data[i] - learningRate * grad[i];
            }
            else
            {
                resultData[i] = data[i];
            }
        }

        var resultColumn = NivaraColumn<T>.Create(resultData);
        return new ReverseGradTensor<T>(resultColumn, requiresGrad: false);
    }
}
