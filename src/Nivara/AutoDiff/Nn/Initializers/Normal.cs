using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

public static class Normal
{
    public static void Init<T>(Dictionary<string, ReverseGradTensor<T>> parameters, T? mean = null, T? std = null)
        where T : struct, INumber<T>
    {
        if (parameters == null) throw new ArgumentNullException(nameof(parameters));

        var mu = mean ?? T.Zero;
        var sigma = std ?? T.One;

        foreach (var kvp in parameters)
        {
            var tensor = kvp.Value;
            var n = tensor.Length;
            var data = new T[n];
            for (int i = 0; i < n; i++)
            {
                var u1 = Random.Shared.NextDouble();
                var u2 = Random.Shared.NextDouble();
                var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                data[i] = T.CreateChecked(normal) * sigma + mu;
            }

            var column = NivaraColumn<T>.Create(data);
            var newTensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, tensor.Shape);
            parameters[kvp.Key] = newTensor;
        }
    }
}
