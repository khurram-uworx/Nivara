using System.Buffers;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class Dropout<T> : Module<T> where T : struct, INumber<T>
{
    readonly double probability;

    public double Probability => probability;

    public Dropout(double probability = 0.5)
    {
        if (probability < 0.0 || probability >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(probability), "Dropout probability must be in [0, 1).");
        this.probability = probability;
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        if (!IsTraining || probability <= 0.0)
            return input;

        var n = input.Length;
        var mask = new bool[n];
        var scale = T.CreateChecked(1.0 / (1.0 - probability));
        var random = Random.Shared;

        for (int i = 0; i < n; i++)
            mask[i] = random.NextDouble() >= probability;

        var data = input.Data;
        var resultBuf = ArrayPool<T>.Shared.Rent(n);

        try
        {
            if (!data.HasNulls)
            {
                data.TryGetSpan(out var span);
                for (int i = 0; i < n; i++)
                    resultBuf[i] = mask[i] ? span[i] * scale : T.Zero;
                var resultColumn = NivaraColumn<T>.Create(resultBuf.AsSpan(0, n));
                return new ReverseGradTensor<T>(resultColumn, input.RequiresGrad, input.shape);
            }

            data.CopyTo(resultBuf.AsSpan(0, n), T.Zero);
            data.TryGetNullMask(out var nullMask);

            var finalMask = ArrayPool<bool>.Shared.Rent(n);
            try
            {
                nullMask.CopyTo(finalMask.AsSpan(0, n));
                for (int i = 0; i < n; i++)
                {
                    if (!nullMask[i])
                        resultBuf[i] = mask[i] ? resultBuf[i] * scale : T.Zero;
                }
                var resultColumn = NivaraColumn<T>.CreateFromSpans(resultBuf.AsSpan(0, n), finalMask.AsSpan(0, n));
                return new ReverseGradTensor<T>(resultColumn, input.RequiresGrad, input.shape);
            }
            finally
            {
                ArrayPool<bool>.Shared.Return(finalMask, clearArray: true);
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(resultBuf, clearArray: true);
        }
    }
}
