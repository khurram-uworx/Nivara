using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Base class for neural-network modules. A module owns child modules and parameters
/// registered via <see cref="RegisterModules"/> / <see cref="RegisterParameters"/> and
/// exposes them as flat, dotted-path state dictionaries for training and serialization.
/// </summary>
public abstract class Module<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
{
    readonly List<Module<T>> modules = [];
    readonly List<Parameter<T>> parameters = [];
    bool disposed;

    /// <summary>
    /// Gets whether the module is in training mode. Stochastic layers such as
    /// <see cref="Dropout{T}"/> behave differently in training vs evaluation mode.
    /// True by default; switch via <see cref="Train"/> / <see cref="Eval"/>.
    /// </summary>
    public bool IsTraining { get; private set; } = true;

    /// <summary>
    /// Runs the forward pass for this module. When gradient tracking is enabled the
    /// computation graph is recorded so a subsequent <c>Backward()</c> can differentiate it.
    /// </summary>
    /// <param name="input">The input tensor</param>
    /// <returns>The output tensor</returns>
    public abstract ReverseGradTensor<T> Forward(ReverseGradTensor<T> input);

    /// <summary>Puts this module and all registered child modules into training mode.</summary>
    public void Train()
    {
        IsTraining = true;
        foreach (var module in modules)
            module.Train();
    }

    /// <summary>Puts this module and all registered child modules into evaluation mode.</summary>
    public void Eval()
    {
        IsTraining = false;
        foreach (var module in modules)
            module.Eval();
    }

    /// <summary>
    /// Registers child modules so their parameters are included in <see cref="Parameters()"/>,
    /// <see cref="GetParameters()"/>, <see cref="StateDict"/>, and <see cref="Train"/> / <see cref="Eval"/>.
    /// </summary>
    /// <param name="modules">The child modules to register</param>
    public void RegisterModules(params Module<T>[] modules)
    {
        ArgumentNullException.ThrowIfNull(modules);
        foreach (var module in modules)
            this.modules.Add(module);
    }

    /// <summary>
    /// Registers parameters so they are included in <see cref="Parameters()"/>,
    /// <see cref="GetParameters()"/>, and <see cref="StateDict"/>.
    /// </summary>
    /// <param name="parameters">The parameters to register</param>
    public void RegisterParameters(params Parameter<T>[] parameters)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        foreach (var param in parameters)
            this.parameters.Add(param);
    }

    /// <summary>
    /// Returns a flat dictionary of all parameter tensors keyed by dotted path
    /// (e.g. <c>Weight</c>, <c>Module_0.Weight</c>).
    /// </summary>
    /// <returns>A flat name-to-tensor dictionary</returns>
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

    /// <summary>
    /// Returns a flat dictionary of all <see cref="Parameter{T}"/> objects keyed by dotted path
    /// (e.g. <c>Weight</c>, <c>Module_0.Weight</c>).
    /// </summary>
    /// <returns>A flat name-to-parameter dictionary</returns>
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

    /// <summary>Gets the read-only list of registered child modules.</summary>
    public IReadOnlyList<Module<T>> NamedModules() => modules.AsReadOnly();

    /// <summary>
    /// Returns a deep copy of all parameter tensors keyed by dotted path, suitable for
    /// serialization or for loading into another module of the same architecture.
    /// </summary>
    /// <returns>A state dictionary of parameter tensors</returns>
    public virtual Dictionary<string, ReverseGradTensor<T>> StateDict()
    {
        var state = new Dictionary<string, ReverseGradTensor<T>>();

        foreach (var (name, tensor) in Parameters())
            state[name] = CloneTensor(tensor);

        return state;
    }

    /// <summary>
    /// Restores parameter values from a state dictionary produced by <see cref="StateDict"/>.
    /// Throws if a dictionary key does not exist in the model; when <paramref name="strict"/> is
    /// true it also throws if the model has parameters absent from the dictionary.
    /// </summary>
    /// <param name="stateDict">The state dictionary to load from</param>
    /// <param name="strict">Whether missing model parameters should throw</param>
    public virtual void LoadStateDict(
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

        var data = NivaraColumn<T>.Create(values);

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

    /// <summary>
    /// Returns the index of the largest logit within one row of a <c>[row, numClasses]</c> tensor.
    /// </summary>
    /// <param name="logits">The logits tensor</param>
    /// <param name="row">The row index to scan</param>
    /// <param name="numClasses">The number of classes per row</param>
    /// <returns>The class index with the maximum logit in the given row</returns>
    protected static int ArgMax(ReverseGradTensor<T> logits, int row, int numClasses)
    {
        int bestClass = 0;
        T bestVal = logits.Data[row * numClasses];
        for (int c = 1; c < numClasses; c++)
        {
            T val = logits.Data[row * numClasses + c];
            if (val > bestVal) { bestVal = val; bestClass = c; }
        }
        return bestClass;
    }

    /// <summary>
    /// Disposes this module and all registered child modules and parameters.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases managed resources held by this module and its children.
    /// </summary>
    /// <param name="disposing">True when called from <see cref="Dispose()"/>; false from a finalizer</param>
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
