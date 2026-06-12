using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

public interface IInitializer<T> where T : struct, INumber<T>
{
    void Initialize(Parameter<T> parameter);
}
