using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

public interface IInitializer<T> where T : struct, IFloatingPointIeee754<T>
{
    void Initialize(Parameter<T> parameter);
}
