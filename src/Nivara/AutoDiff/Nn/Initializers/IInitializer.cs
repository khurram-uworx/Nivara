using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

/// <summary>Strategy for initializing a parameter's weights.</summary>
public interface IInitializer<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>
    /// Replaces the parameter's tensor values with initialized ones, preserving its
    /// requires-grad flag and shape.
    /// </summary>
    /// <param name="parameter">The parameter to initialize</param>
    void Initialize(Parameter<T> parameter);
}
