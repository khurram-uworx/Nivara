using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public static class Initializers
{
    public static void KaimingUniform<T>(Parameter<T> parameter)
        where T : struct, INumber<T>
    {
        var tensor = parameter.Tensor;
        var shape = tensor.Shape;
        if (shape.Length < 2) return;

        var fanIn = shape[1];
        var bound = T.CreateChecked(Math.Sqrt(6.0 / fanIn));
        var random = Random.Shared;
        var n = tensor.Length;
        var data = new T[n];

        for (int i = 0; i < n; i++)
            data[i] = T.CreateChecked(random.NextDouble() * 2.0 - 1.0) * bound;

        var column = NivaraColumn<T>.Create(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, shape);
    }

    public static void KaimingNormal<T>(Parameter<T> parameter)
        where T : struct, INumber<T>
    {
        var tensor = parameter.Tensor;
        var shape = tensor.Shape;
        if (shape.Length < 2) return;

        var fanIn = shape[1];
        var std = T.CreateChecked(Math.Sqrt(2.0 / fanIn));
        var n = tensor.Length;
        var data = new T[n];

        for (int i = 0; i < n; i++)
        {
            var u1 = Random.Shared.NextDouble();
            var u2 = Random.Shared.NextDouble();
            var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            data[i] = T.CreateChecked(normal) * std;
        }

        var column = NivaraColumn<T>.Create(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, shape);
    }

    public static void XavierUniform<T>(Parameter<T> parameter)
        where T : struct, INumber<T>
    {
        var tensor = parameter.Tensor;
        var shape = tensor.Shape;
        if (shape.Length < 2) return;

        var fanIn = shape[1];
        var fanOut = shape[0];
        var bound = T.CreateChecked(Math.Sqrt(6.0 / (fanIn + fanOut)));
        var random = Random.Shared;
        var n = tensor.Length;
        var data = new T[n];

        for (int i = 0; i < n; i++)
            data[i] = T.CreateChecked(random.NextDouble() * 2.0 - 1.0) * bound;

        var column = NivaraColumn<T>.Create(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, shape);
    }

    public static void XavierNormal<T>(Parameter<T> parameter)
        where T : struct, INumber<T>
    {
        var tensor = parameter.Tensor;
        var shape = tensor.Shape;
        if (shape.Length < 2) return;

        var fanIn = shape[1];
        var fanOut = shape[0];
        var std = T.CreateChecked(Math.Sqrt(2.0 / (fanIn + fanOut)));
        var n = tensor.Length;
        var data = new T[n];

        for (int i = 0; i < n; i++)
        {
            var u1 = Random.Shared.NextDouble();
            var u2 = Random.Shared.NextDouble();
            var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            data[i] = T.CreateChecked(normal) * std;
        }

        var column = NivaraColumn<T>.Create(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, shape);
    }

    public static void PyTorchDefault<T>(Parameter<T> parameter)
        where T : struct, INumber<T>
    {
        var tensor = parameter.Tensor;
        var shape = tensor.Shape;
        if (shape.Length < 2) return;

        var fanIn = shape[1];
        var bound = T.CreateChecked(1.0 / Math.Sqrt(fanIn));
        var random = Random.Shared;
        var n = tensor.Length;
        var data = new T[n];

        for (int i = 0; i < n; i++)
            data[i] = T.CreateChecked(random.NextDouble() * 2.0 - 1.0) * bound;

        var column = NivaraColumn<T>.Create(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, shape);
    }

    public static void Uniform<T>(Parameter<T> parameter, T lower, T upper)
        where T : struct, INumber<T>
    {
        var tensor = parameter.Tensor;
        var range = upper - lower;
        var random = Random.Shared;
        var n = tensor.Length;
        var data = new T[n];

        for (int i = 0; i < n; i++)
            data[i] = T.CreateChecked(random.NextDouble()) * range + lower;

        var column = NivaraColumn<T>.Create(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, tensor.Shape);
    }

    public static void Normal<T>(Parameter<T> parameter, T mean, T std)
        where T : struct, INumber<T>
    {
        var tensor = parameter.Tensor;
        var n = tensor.Length;
        var data = new T[n];

        for (int i = 0; i < n; i++)
        {
            var u1 = Random.Shared.NextDouble();
            var u2 = Random.Shared.NextDouble();
            var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            data[i] = T.CreateChecked(normal) * std + mean;
        }

        var column = NivaraColumn<T>.Create(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, tensor.Shape);
    }

    public static void KaimingUniform<T>(Dictionary<string, ReverseGradTensor<T>> parameters)
        where T : struct, INumber<T>
    {
        if (parameters == null) throw new ArgumentNullException(nameof(parameters));

        foreach (var kvp in parameters)
        {
            var tensor = kvp.Value;
            var shape = tensor.Shape;
            if (shape.Length < 2) continue;

            var fanIn = shape[1];
            var bound = T.CreateChecked(Math.Sqrt(6.0 / fanIn));
            var random = Random.Shared;
            var n = tensor.Length;
            var data = new T[n];

            for (int i = 0; i < n; i++)
                data[i] = T.CreateChecked(random.NextDouble() * 2.0 - 1.0) * bound;

            var column = NivaraColumn<T>.Create(data);
            parameters[kvp.Key] = new ReverseGradTensor<T>(column, tensor.RequiresGrad, shape);
        }
    }

    public static void XavierUniform<T>(Dictionary<string, ReverseGradTensor<T>> parameters)
        where T : struct, INumber<T>
    {
        if (parameters == null) throw new ArgumentNullException(nameof(parameters));

        foreach (var kvp in parameters)
        {
            var tensor = kvp.Value;
            var shape = tensor.Shape;
            if (shape.Length < 2) continue;

            var fanIn = shape[1];
            var fanOut = shape[0];
            var bound = T.CreateChecked(Math.Sqrt(6.0 / (fanIn + fanOut)));
            var random = Random.Shared;
            var n = tensor.Length;
            var data = new T[n];

            for (int i = 0; i < n; i++)
                data[i] = T.CreateChecked(random.NextDouble() * 2.0 - 1.0) * bound;

            var column = NivaraColumn<T>.Create(data);
            parameters[kvp.Key] = new ReverseGradTensor<T>(column, tensor.RequiresGrad, shape);
        }
    }

    public static void Normal<T>(Dictionary<string, ReverseGradTensor<T>> parameters, T? mean = null, T? std = null)
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
            parameters[kvp.Key] = new ReverseGradTensor<T>(column, tensor.RequiresGrad, tensor.Shape);
        }
    }

    public static void Uniform<T>(Dictionary<string, ReverseGradTensor<T>> parameters, T? lower = null, T? upper = null)
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
            parameters[kvp.Key] = new ReverseGradTensor<T>(column, tensor.RequiresGrad, tensor.Shape);
        }
    }
}
