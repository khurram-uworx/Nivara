using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;

namespace Nivara.AutoDiff.Nn.Functional;

public sealed class BCEWithLogitsLoss<T> where T : struct, IFloatingPointIeee754<T>
{
    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> logits, ReverseGradTensor<T> targets)
        => Forward(logits, targets, reduceToMean: false);

    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> logits, ReverseGradTensor<T> targets, bool reduceToMean)
    {
        if (logits == null) throw new ArgumentNullException(nameof(logits));
        if (targets == null) throw new ArgumentNullException(nameof(targets));

        int n = logits.Length;
        var logitsData = new T[n];
        logits.Data.CopyTo(logitsData, default(T)!);
        var targetsData = new T[n];
        targets.Data.CopyTo(targetsData, default(T)!);

        var lossData = new T[n];
        for (int i = 0; i < n; i++)
        {
            T x = logitsData[i];
            T z = targetsData[i];
            T maxVal = T.Max(T.Zero, x);
            T negAbsX = -T.Abs(x);
            lossData[i] = maxVal - x * z + SoftPlus(negAbsX);
        }

        var lossCol = NivaraColumn<T>.Create(lossData);
        bool shouldTrack = GradientUtils.ShouldTrackGrad(logits, targets);
        var lossTensor = new ReverseGradTensor<T>(lossCol, shouldTrack, logits.shape);

        if (shouldTrack)
        {
            bool trackTargets = targets.RequiresGrad;
            var gradFn = new OpNode<T>("BCEWithLogits", new object[] { logits, targets }, (gradOutput) =>
            {
                var grad = new T[n];
                gradOutput.CopyTo(grad.AsSpan(), default(T)!);

                var sigmoidX = new T[n];
                GradKernels.Sigmoid(logitsData.AsSpan(), sigmoidX.AsSpan());

                var logitsGrad = new T[n];
                for (int i = 0; i < n; i++)
                    logitsGrad[i] = (sigmoidX[i] - targetsData[i]) * grad[i];
                ReverseGradOperations.AccumulateGradient(logits, NivaraColumn<T>.Create(logitsGrad));

                if (trackTargets)
                {
                    var targetsGrad = new T[n];
                    for (int i = 0; i < n; i++)
                        targetsGrad[i] = -logitsData[i] * grad[i];
                    ReverseGradOperations.AccumulateGradient(targets, NivaraColumn<T>.Create(targetsGrad));
                }
            });
            ComputationGraph.AddNode(lossTensor, gradFn);
        }

        var sumLoss = ReverseGradOperations.Sum(lossTensor);

        if (!reduceToMean)
            return sumLoss;

        var lengthTensor = GradientUtils.Full(1, T.CreateChecked(n));
        return ReverseGradOperations.Divide(sumLoss, lengthTensor);
    }

    static T SoftPlus(T x)
    {
        double dx = double.CreateChecked(x);
        if (dx > 30.0)
            return x;
        if (dx < -30.0)
            return T.CreateChecked(Math.Exp(dx));
        return T.CreateChecked(Math.Log(1.0 + Math.Exp(dx)));
    }
}
