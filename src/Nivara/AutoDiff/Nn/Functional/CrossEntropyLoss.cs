using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;

namespace Nivara.AutoDiff.Nn.Functional;

public sealed class CrossEntropyLoss<T> : Loss<T> where T : struct, IFloatingPointIeee754<T>
{
    public CrossEntropyLoss(Reduction reduction = Reduction.Mean) : base(reduction)
    {
    }

    public override ReverseGradTensor<T> Forward(
        ReverseGradTensor<T> logits,
        ReverseGradTensor<T> targets,
        Reduction reduction)
    {
        if (logits == null) throw new ArgumentNullException(nameof(logits));
        if (targets == null) throw new ArgumentNullException(nameof(targets));

        var logSoftmax = ReverseGradOperations.LogSoftmax(logits);
        var nll = ReverseGradOperations.Negate(ReverseGradOperations.Multiply(logSoftmax, targets));

        if (reduction == Reduction.None)
            return PerRowNll(nll, logits);

        int batchSize = logits.shape[0];
        return Reduce(nll, reduction, batchSize);
    }

    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> logits, int[] targets)
    {
        if (logits == null) throw new ArgumentNullException(nameof(logits));
        if (targets == null) throw new ArgumentNullException(nameof(targets));

        int batchSize = logits.shape[0];
        int numClasses = logits.Rank >= 2 ? logits.shape[1] : logits.Length;

        if (targets.Length != batchSize)
            throw new ArgumentException(
                $"Targets length ({targets.Length}) must match batch size ({batchSize}).",
                nameof(targets));

        for (int i = 0; i < targets.Length; i++)
        {
            if (targets[i] < 0 || targets[i] >= numClasses)
                throw new ArgumentOutOfRangeException(
                    nameof(targets),
                    $"Target at position {i} is {targets[i]}, must be in range [0, {numClasses}).");
        }

        // Build one-hot targets
        var oneHotData = new T[batchSize * numClasses];
        for (int i = 0; i < batchSize; i++)
            oneHotData[i * numClasses + targets[i]] = T.One;

        var oneHotCol = NivaraColumn<T>.Create(oneHotData);
        var oneHotTargets = new ReverseGradTensor<T>(oneHotCol, requiresGrad: false);
        oneHotTargets.Reshape(batchSize, numClasses);

        return Forward(logits, oneHotTargets);
    }

    // Per-sample NLL (shape [N]) matching PyTorch's reduction='none': sums the
    // class-weighted NLL within each row so the returned tensor has batch length.
    static ReverseGradTensor<T> PerRowNll(ReverseGradTensor<T> nll, ReverseGradTensor<T> logits)
    {
        int numClasses = logits.Rank >= 2 ? logits.shape[1] : logits.Length;
        var onesCol = GradientUtils.Full(numClasses, T.One);
        onesCol.Reshape(numClasses, 1);
        var perRow = ReverseGradOperations.MatMul(nll, onesCol);
        perRow.Reshape(logits.shape[0]);
        return perRow;
    }
}
