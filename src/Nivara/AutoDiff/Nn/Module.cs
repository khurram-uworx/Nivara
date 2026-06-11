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
        if (modules == null) throw new ArgumentNullException(nameof(modules));
        foreach (var module in modules)
        {
            if (module != null)
                this.modules.Add(module);
        }
    }

    public void RegisterParameters(params Parameter<T>[] parameters)
    {
        if (parameters == null) throw new ArgumentNullException(nameof(parameters));
        foreach (var param in parameters)
        {
            if (param != null)
                this.parameters.Add(param);
        }
    }

    public Dictionary<string, ReverseGradTensor<T>> Parameters()
    {
        var result = new Dictionary<string, ReverseGradTensor<T>>();

        foreach (var param in parameters)
            result[param.Name] = param.Tensor;

        foreach (var module in modules)
        {
            var childParams = module.Parameters();
            foreach (var kvp in childParams)
                result[kvp.Key] = kvp.Value;
        }

        return result;
    }

    public Dictionary<string, Parameter<T>> GetParameters()
    {
        var result = new Dictionary<string, Parameter<T>>();

        foreach (var param in parameters)
            result[param.Name] = param;

        foreach (var module in modules)
        {
            var childParams = module.GetParameters();
            foreach (var kvp in childParams)
                result[kvp.Key] = kvp.Value;
        }

        return result;
    }

    public IReadOnlyList<Module<T>> NamedModules() => modules.AsReadOnly();

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
        }
        disposed = true;
    }
}
