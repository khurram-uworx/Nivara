using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Container that runs a sequence of modules in order, feeding each module's output
/// into the next module's input.
/// </summary>
public sealed class Sequential<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly List<Module<T>> modules = [];

    /// <summary>Gets the registered modules in execution order.</summary>
    public IReadOnlyList<Module<T>> Modules => modules.AsReadOnly();

    /// <summary>
    /// Creates a sequential container from the given modules, executed in the provided order.
    /// </summary>
    /// <param name="modules">The modules to run in sequence</param>
    public Sequential(params Module<T>[] modules)
    {
        ArgumentNullException.ThrowIfNull(modules);
        foreach (var m in modules)
        {
            ArgumentNullException.ThrowIfNull(m);
            this.modules.Add(m);
            RegisterModules(m);
        }
    }

    /// <summary>
    /// Runs each module in order and returns the final output.
    /// </summary>
    /// <param name="input">The input tensor</param>
    /// <returns>The output of the last module</returns>
    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        var current = input;
        foreach (var module in modules)
            current = module.Forward(current);
        return current;
    }

    /// <summary>
    /// Adds a module to the end of the sequence.
    /// </summary>
    /// <param name="module">The module to append</param>
    public void Append(Module<T> module)
    {
        if (module == null) throw new ArgumentNullException(nameof(module));
        modules.Add(module);
        RegisterModules(module);
    }
}
