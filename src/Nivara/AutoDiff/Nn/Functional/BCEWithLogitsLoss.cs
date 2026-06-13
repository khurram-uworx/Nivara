using Nivara.AutoDiff.Operations;
using System.Buffers;
using System.Numerics;

namespace Nivara.AutoDiff.Nn.Functional;

public sealed class BCEWithLogitsLoss<T> where T : struct, INumber<T>
{
    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> logits, ReverseGradTensor<T> targets)
    {
        if (logits == null) throw new ArgumentNullException(nameof(logits));
        if (targets == null) throw new ArgumentNullException(nameof(targets));

        int n = logits.Length;
        var oneBuf = ArrayPool<T>.Shared.Rent(n);
        NivaraColumn<T> oneCol;
        try
        {
            oneBuf.AsSpan(0, n).Fill(T.One);
            oneCol = NivaraColumn<T>.Create(oneBuf.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(oneBuf, clearArray: true);
        }
        var one = new ReverseGradTensor<T>(oneCol,
            requiresGrad: false, logits.shape);

        var maxX = GradOperations.Relu(logits);
        var negAbs = GradOperations.Negate(GradOperations.Abs(logits));
        var negExpNegAbs = GradOperations.Exp(negAbs);
        var log1pExp = GradOperations.Log(GradOperations.Add(one, negExpNegAbs));

        // loss = max(0, x) - x*z + log(1 + exp(-|x|))
        var loss = GradOperations.Add(
            GradOperations.Subtract(maxX, GradOperations.Multiply(logits, targets)),
            log1pExp);
        return GradOperations.Sum(loss);
    }
}
