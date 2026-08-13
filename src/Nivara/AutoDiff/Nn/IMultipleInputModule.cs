using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public interface IMultipleInputModule<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
{
    ReverseGradTensor<T> Forward(ReverseGradTensor<T> input1, ReverseGradTensor<T> input2);
}
