using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;

namespace Nivara.Tests.Docs.Preambles.NnModels;

/// <summary>
/// Compile-time stand-in for the <c>Module&lt;float&gt;</c> subclasses the AutoDiff snippets declare in
/// their own <c>mode: File</c> block. Mirrors that declaration member for member so the training-loop
/// snippets bind against the same surface - <c>Forward</c> is overridden rather than stubbed out,
/// because a snippet that passes the tensor through a <c>Linear</c> layer must have that call checked.
/// </summary>
public class LinearModel : Module<float>
{
    Linear<float> L1;

    public LinearModel()
    {
        L1 = new Linear<float>(3, 1);
        RegisterModules(L1);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> x)
        => L1.Forward(x);
}

/// <summary>Compile-time stand-in for the Act 8 fraud classifier, mirroring its declared shape.</summary>
public class FraudNet : Module<float>
{
    Linear<float> L1, L2, L3;

    public FraudNet()
    {
        L1 = new Linear<float>(8, 64);
        L2 = new Linear<float>(64, 32);
        L3 = new Linear<float>(32, 1);
        RegisterModules(L1, L2, L3);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> x)
    {
        var h = ReverseGradOperations.Relu(L1.Forward(x));
        h = ReverseGradOperations.Relu(L2.Forward(h));
        return L3.Forward(h);
    }
}
