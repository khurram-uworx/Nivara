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
/// <remarks>
/// The values grow linearly with the element index, so the ramp is only safe for shallow
/// stacks. Applied to the six stacked layers of <c>VAE&lt;T&gt;</c> it drives
/// <c>exp(0.5 * logVar)</c> to +Inf and the loss to NaN. For a multi-layer network use a
/// seedable, fan-in-scaled initializer (e.g. <c>KaimingUniformInitializer&lt;T&gt;(seed: ...)</c>).
/// </remarks>
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