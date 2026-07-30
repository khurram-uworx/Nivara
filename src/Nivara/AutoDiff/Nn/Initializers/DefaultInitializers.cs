using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

public static class DefaultInitializers
{
    public static IInitializer<T> Weight<T>() where T : struct, IFloatingPointIeee754<T>
        => KaimingUniformInitializer<T>.Instance;

    public static IInitializer<T>? Bias<T>() where T : struct, IFloatingPointIeee754<T>
        => null;
}
