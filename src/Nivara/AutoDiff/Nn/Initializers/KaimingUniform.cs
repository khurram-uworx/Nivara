using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

public static class KaimingUniform
{
    public static void Init<T>(Dictionary<string, ReverseGradTensor<T>> parameters)
        where T : struct, INumber<T>
    {
        if (parameters == null) throw new ArgumentNullException(nameof(parameters));

        foreach (var kvp in parameters)
        {
            var tensor = kvp.Value;
            var shape = tensor.Shape;
            if (shape.Length < 2) continue;

            var fanIn = shape.Length > 1 ? shape[1] : shape[0];
            var bound = T.CreateChecked(Math.Sqrt(6.0 / fanIn));
            var random = Random.Shared;
            var n = tensor.Length;
            var data = new T[n];

            for (int i = 0; i < n; i++)
                data[i] = T.CreateChecked(random.NextDouble() * 2.0 - 1.0) * bound;

            var column = NivaraColumn<T>.Create(data);
            var newTensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, shape);
            parameters[kvp.Key] = newTensor;
        }
    }
}
