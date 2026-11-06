namespace Nivara.Silk;

/// <summary>
/// Shared host-side pieces of the Silk training loops: the softmax cross-entropy loss and its
/// input gradient, the Adam step the host-resident embedding takes, and the grow-on-demand scratch
/// helper. The classes that keep their parameters on the device take their Adam steps there
/// (<c>adam_update</c>); the classifiers' word embedding stays host-side, because a vocab-sized
/// gradient upload per batch is not worth the round trip (see the classifiers' remarks).
///
/// <para>
/// The two-class overloads exist because the two text classifiers are binary and their numerics are
/// pinned by host-formula tests; they are kept verbatim rather than re-expressed through the
/// general form so that adding a class-count-agnostic overload cannot move them.
/// </para>
/// </summary>
internal static class SilkLoss
{
    /// <summary>Mean cross-entropy of two-class logits against the labels.</summary>
    public static float CrossEntropy(ReadOnlySpan<float> logits, ReadOnlySpan<int> labels, int n)
    {
        float loss = 0f;
        for (int i = 0; i < n; i++)
        {
            float e0 = MathF.Exp(logits[i * 2]);
            float e1 = MathF.Exp(logits[i * 2 + 1]);
            float p0 = e0 / (e0 + e1);
            loss += labels[i] == 0 ? -MathF.Log(p0) : -MathF.Log(1f - p0);
        }
        return loss / n;
    }

    /// <summary>Mean cross-entropy of <paramref name="numClasses"/>-way logits against the labels,
    /// with the row maximum subtracted before exponentiation for overflow safety.</summary>
    public static float CrossEntropy(ReadOnlySpan<float> logits, ReadOnlySpan<int> labels, int n, int numClasses)
    {
        float loss = 0f;
        for (int i = 0; i < n; i++)
        {
            int rowBase = i * numClasses;
            float max = float.NegativeInfinity;
            for (int c = 0; c < numClasses; c++)
                max = MathF.Max(max, logits[rowBase + c]);
            float sum = 0f;
            for (int c = 0; c < numClasses; c++)
                sum += MathF.Exp(logits[rowBase + c] - max);
            loss -= MathF.Log(MathF.Exp(logits[rowBase + labels[i]] - max) / sum);
        }
        return loss / n;
    }

    /// <summary>
    /// Softmax cross-entropy input gradient for two classes: <c>dZ[i,c] = (p[i,c] − onehot[i,c]) · scale</c>.
    /// The batch average (<c>scale = 1/n</c>) is folded in here so no downstream pass needs a
    /// scaling sweep.
    /// </summary>
    public static void InputGradient(ReadOnlySpan<float> logits, ReadOnlySpan<int> labels, int n, float scale, Span<float> dZ)
    {
        for (int i = 0; i < n; i++)
        {
            float e0 = MathF.Exp(logits[i * 2]);
            float e1 = MathF.Exp(logits[i * 2 + 1]);
            float z = e0 + e1;
            float p0 = e0 / z;
            int label = labels[i];
            dZ[i * 2 + 0] = (p0 - (label == 0 ? 1f : 0f)) * scale;
            dZ[i * 2 + 1] = (1f - p0 - (label == 1 ? 1f : 0f)) * scale;
        }
    }

    /// <summary>
    /// General-class form of <see cref="InputGradient(ReadOnlySpan{float}, ReadOnlySpan{int}, int, float, Span{float})"/>:
    /// <c>dZ[i,c] = (p[i,c] − onehot[i,c]) · scale</c> with the row maximum subtracted before
    /// exponentiating, matching <see cref="CrossEntropy(ReadOnlySpan{float}, ReadOnlySpan{int}, int, int)"/>.
    /// </summary>
    public static void InputGradient(
        ReadOnlySpan<float> logits, ReadOnlySpan<int> labels, int n, int numClasses, float scale, Span<float> dZ)
    {
        for (int i = 0; i < n; i++)
        {
            int rowBase = i * numClasses;
            float max = float.NegativeInfinity;
            for (int c = 0; c < numClasses; c++)
                max = MathF.Max(max, logits[rowBase + c]);
            float sum = 0f;
            for (int c = 0; c < numClasses; c++)
                sum += MathF.Exp(logits[rowBase + c] - max);

            int label = labels[i];
            for (int c = 0; c < numClasses; c++)
            {
                float p = MathF.Exp(logits[rowBase + c] - max) / sum;
                dZ[rowBase + c] = (p - (c == label ? 1f : 0f)) * scale;
            }
        }
    }

    /// <summary>
    /// One Adam step over a host-resident parameter tensor (β1 0.9, β2 0.999, ε 1e-8 — the same
    /// constants the device <c>adam_update</c> kernel carries). <paramref name="first"/> and
    /// <paramref name="second"/> are the persistent moment arrays, both the length of
    /// <paramref name="values"/>.
    /// </summary>
    public static void AdamStep(float[] values, float[] first, float[] second, ReadOnlySpan<float> gradient,
        float learningRate, float biasCorrection1, float biasCorrection2)
    {
        for (int i = 0; i < values.Length; i++)
        {
            first[i] = 0.9f * first[i] + 0.1f * gradient[i];
            second[i] = 0.999f * second[i] + 0.001f * gradient[i] * gradient[i];
            values[i] -= learningRate * (first[i] / biasCorrection1) / (MathF.Sqrt(second[i] / biasCorrection2) + 1e-8f);
        }
    }

    /// <summary>Returns <paramref name="buffer"/> when it already fits, otherwise a fresh array
    /// of exactly <paramref name="length"/>. Scratch arrays are reused across batches and never
    /// shrink.</summary>
    public static T[] Obtain<T>(ref T[] buffer, int length)
    {
        if (buffer.Length >= length) return buffer;
        buffer = new T[length];
        return buffer;
    }
}
