using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class Sequential<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly List<Module<T>> modules = [];

    public IReadOnlyList<Module<T>> Modules => modules.AsReadOnly();

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

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        var current = input;
        foreach (var module in modules)
            current = module.Forward(current);
        return current;
    }

    public void Append(Module<T> module)
    {
        if (module == null) throw new ArgumentNullException(nameof(module));
        modules.Add(module);
        RegisterModules(module);
    }
}
