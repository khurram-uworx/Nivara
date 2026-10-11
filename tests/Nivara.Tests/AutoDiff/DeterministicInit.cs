using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Initializers;
using System.Numerics;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// A deterministic weight initializer for tests: fills the parameter with
/// <c>0.05 * (i + 1)</c> for each element <c>i</c>, preserving the tensor's
/// requires-grad flag and shape. Used wherever a test asserts on a training
/// trajectory, so results do not depend on the unseeded <see cref="Random.Shared"/>.
/// </summary>
internal sealed class DeterministicInit<T> : IInitializer<T> where T : struct, IFloatingPointIeee754<T>
{
    public void Initialize(Parameter<T> parameter)
    {
        var tensor = parameter.Tensor;
        var data = new T[tensor.Length];
        for (int i = 0; i < data.Length; i++)
            data[i] = T.CreateChecked(0.05) * T.CreateChecked(i + 1);
        var column = NivaraColumn<T>.CreateFromOwnedArray(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, tensor.Shape);
    }
}