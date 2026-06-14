using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public abstract class Module<T> : IDisposable where T : struct, INumber<T>
{
    readonly List<Module<T>> modules = [];
    readonly List<Parameter<T>> parameters = [];
    bool disposed;

    public bool IsTraining { get; private set; } = true;

    public abstract ReverseGradTensor<T> Forward(ReverseGradTensor<T> input);

    public virtual ReverseGradTensor<T> Forward(ReverseGradTensor<T> input1, ReverseGradTensor<T> input2)
    {
        throw new NotSupportedException($"{GetType().Name} does not support multi-input Forward.");
    }

    public void Train()
    {
        IsTraining = true;
        foreach (var module in modules)
            module.Train();
    }

    public void Eval()
    {
        IsTraining = false;
        foreach (var module in modules)
            module.Eval();
    }

    public void RegisterModules(params Module<T>[] modules)
    {
        ArgumentNullException.ThrowIfNull(modules);
        foreach (var module in modules)
        {
            if (module != null)
                this.modules.Add(module);
        }
    }

    public void RegisterParameters(params Parameter<T>[] parameters)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        foreach (var param in parameters)
        {
            if (param != null)
                this.parameters.Add(param);
        }
    }

    public Dictionary<string, ReverseGradTensor<T>> Parameters()
    {
        return Parameters("");
    }

    internal Dictionary<string, ReverseGradTensor<T>> Parameters(string prefix)
    {
        var result = new Dictionary<string, ReverseGradTensor<T>>();

        foreach (var param in parameters)
            result[prefix + param.Name] = param.Tensor;

        for (int i = 0; i < modules.Count; i++)
        {
            var childParams = modules[i].Parameters(prefix + $"Module_{i}.");
            foreach (var kvp in childParams)
                result[kvp.Key] = kvp.Value;
        }

        return result;
    }

    public Dictionary<string, Parameter<T>> GetParameters()
    {
        return GetParameters("");
    }

    internal Dictionary<string, Parameter<T>> GetParameters(string prefix)
    {
        var result = new Dictionary<string, Parameter<T>>();

        foreach (var param in parameters)
            result[prefix + param.Name] = param;

        for (int i = 0; i < modules.Count; i++)
        {
            var childParams = modules[i].GetParameters(prefix + $"Module_{i}.");
            foreach (var kvp in childParams)
                result[kvp.Key] = kvp.Value;
        }

        return result;
    }

    public IReadOnlyList<Module<T>> NamedModules() => modules.AsReadOnly();

    public Dictionary<string, ReverseGradTensor<T>> StateDict()
    {
        var state = new Dictionary<string, ReverseGradTensor<T>>();

        foreach (var (name, tensor) in Parameters())
            state[name] = CloneTensor(tensor);

        return state;
    }

    public void LoadStateDict(
        IReadOnlyDictionary<string, ReverseGradTensor<T>> stateDict,
        bool strict = false)
    {
        ArgumentNullException.ThrowIfNull(stateDict);

        var modelParameters = GetParameters();

        foreach (var (name, source) in stateDict)
        {
            ArgumentNullException.ThrowIfNull(source);

            if (!modelParameters.TryGetValue(name, out var parameter))
                throw new InvalidOperationException(
                    $"Parameter '{name}' not found in model. " +
                    $"Available parameters: [{string.Join(", ", modelParameters.Keys)}]");

            ValidateShape(name, source.Shape, parameter.Shape);
            parameter.Tensor = CloneTensor(source, parameter.Tensor.RequiresGrad);
        }

        if (strict)
        {
            var missing = modelParameters.Keys
                .Where(name => !stateDict.ContainsKey(name))
                .ToArray();

            if (missing.Length > 0)
                throw new InvalidOperationException(
                    $"State dictionary is missing model parameters: [{string.Join(", ", missing)}]");
        }
    }

    internal static ReverseGradTensor<T> CloneTensor(
        ReverseGradTensor<T> tensor,
        bool? requiresGrad = null)
    {
        ArgumentNullException.ThrowIfNull(tensor);

        var values = new T[tensor.Length];
        tensor.Data.CopyTo(values, T.Zero);

        NivaraColumn<T> data;
        if (tensor.Data.HasNulls && tensor.Data.TryGetNullMask(out var mask))
        {
            var nullMask = new bool[tensor.Length];
            mask.CopyTo(nullMask);
            data = NivaraColumn<T>.CreateFromSpans(values, nullMask);
        }
        else
        {
            data = NivaraColumn<T>.Create(values);
        }

        return new ReverseGradTensor<T>(
            data,
            requiresGrad ?? tensor.RequiresGrad,
            tensor.Shape);
    }

    static void ValidateShape(string name, int[] sourceShape, int[] targetShape)
    {
        if (sourceShape.Length != targetShape.Length)
            throw new InvalidOperationException(
                $"Parameter '{name}' shape rank mismatch: " +
                $"state has {sourceShape.Length}D, model has {targetShape.Length}D.");

        for (int i = 0; i < sourceShape.Length; i++)
        {
            if (sourceShape[i] != targetShape[i])
                throw new InvalidOperationException(
                    $"Parameter '{name}' shape mismatch: " +
                    $"state has [{string.Join(", ", sourceShape)}], " +
                    $"model has [{string.Join(", ", targetShape)}].");
        }
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (disposed) return;
        if (disposing)
        {
            foreach (var module in modules)
                module.Dispose();

            foreach (var parameter in parameters)
                parameter.Dispose();
        }
        disposed = true;
    }
}
