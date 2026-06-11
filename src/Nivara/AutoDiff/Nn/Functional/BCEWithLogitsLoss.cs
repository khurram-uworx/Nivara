using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn.Functional;

public sealed class BCEWithLogitsLoss<T> where T : struct, INumber<T>
{
    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> logits, ReverseGradTensor<T> targets)
    {
        if (logits == null) throw new ArgumentNullException(nameof(logits));
        if (targets == null) throw new ArgumentNullException(nameof(targets));

        var one = new ReverseGradTensor<T>(NivaraColumn<T>.Create([T.One]),
            requiresGrad: false, logits.shape);

        var maxX = GradOperations.LeakyRelu(logits);
        var negAbs = GradOperations.Negate(GradOperations.Abs(logits));
        var negExpNegAbs = GradOperations.Exp(negAbs);
        var log1pExp = GradOperations.Log(GradOperations.Add(one, negExpNegAbs));

        var loss = GradOperations.Subtract(
            GradOperations.Subtract(maxX, GradOperations.Multiply(logits, targets)),
            log1pExp);
        return GradOperations.Sum(loss);
    }
}
