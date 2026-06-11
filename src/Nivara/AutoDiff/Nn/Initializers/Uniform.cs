using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

public static class Uniform
{
    public static void Init<T>(Dictionary<string, ReverseGradTensor<T>> parameters, T? lower = null, T? upper = null)
        where T : struct, INumber<T>
    {
        if (parameters == null) throw new ArgumentNullException(nameof(parameters));

        var lo = lower ?? -T.One;
        var hi = upper ?? T.One;
        var range = hi - lo;
        var random = Random.Shared;

        foreach (var kvp in parameters)
        {
            var tensor = kvp.Value;
            var n = tensor.Length;
            var data = new T[n];
            for (int i = 0; i < n; i++)
                data[i] = T.CreateChecked(random.NextDouble()) * range + lo;

            var column = NivaraColumn<T>.Create(data);
            var newTensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, tensor.Shape);
            parameters[kvp.Key] = newTensor;
        }
    }
}
