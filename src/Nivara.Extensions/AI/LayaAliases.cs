using System.Numerics;

namespace Nivara.AI;

public sealed class LayaHeadLayer<T> : LayaHeadLayer<T, T> where T : struct, IFloatingPointIeee754<T>
{
    public LayaHeadLayer(int hiddenSize, int intermediateSize, int numHeads, float eps)
        : base(hiddenSize, intermediateSize, numHeads, eps) { }
}

public sealed class LayaDecisionHead<T> : LayaDecisionHead<T, T> where T : struct, IFloatingPointIeee754<T>
{
    public LayaDecisionHead(int hiddenSize, int headLayerCount = 2, int actClassCount = 2, float eps = 1e-5f)
        : base(hiddenSize, headLayerCount, actClassCount, eps) { }
}
