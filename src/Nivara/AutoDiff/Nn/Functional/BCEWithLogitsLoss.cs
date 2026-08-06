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
        var logitsSpan = ModuleHelpers<T>.GetSpan(logits);
        var targetsSpan = ModuleHelpers<T>.GetSpan(targets);

        var lossData = new T[n];
        for (int i = 0; i < n; i++)
        {
            T x = logitsSpan[i];
            T z = targetsSpan[i];
            T maxVal = T.Max(T.Zero, x);
            T negAbsX = -T.Abs(x);
            lossData[i] = maxVal - x * z + SoftPlus(negAbsX);
        }

        var lossCol = NivaraColumn<T>.CreateFromOwnedArray(lossData);
        bool shouldTrack = GradientUtils.ShouldTrackGrad(logits, targets);
        var lossTensor = new ReverseGradTensor<T>(lossCol, shouldTrack, logits.shape);

        if (shouldTrack)
        {
            bool trackTargets = targets.RequiresGrad;
            var gradFn = new OpNode<T>("BCEWithLogits", new object[] { logits, targets }, (gradOutput) =>
            {
                var grad = new T[n];
                gradOutput.CopyTo(grad.AsSpan(), default(T)!);

                var logitsSpanBwd = ModuleHelpers<T>.GetSpan(logits);
                var targetsSpanBwd = ModuleHelpers<T>.GetSpan(targets);

                var sigmoidX = new T[n];
                GradKernels.Sigmoid(logitsSpanBwd, sigmoidX.AsSpan());

                var logitsGrad = new T[n];
                for (int i = 0; i < n; i++)
                    logitsGrad[i] = (sigmoidX[i] - targetsSpanBwd[i]) * grad[i];
                ReverseGradOperations.AccumulateGradient(logits, NivaraColumn<T>.CreateFromOwnedArray(logitsGrad));

                if (trackTargets)
                {
                    var targetsGrad = new T[n];
                    for (int i = 0; i < n; i++)
                        targetsGrad[i] = -logitsSpanBwd[i] * grad[i];
                    ReverseGradOperations.AccumulateGradient(targets, NivaraColumn<T>.CreateFromOwnedArray(targetsGrad));
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
