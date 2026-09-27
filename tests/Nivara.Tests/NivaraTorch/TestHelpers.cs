using Nivara.AutoDiff;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

public static class TestHelpers
{
    internal static string TestDataDir => Path.Combine(
        TestContext.CurrentContext.TestDirectory,
        "..", "..", "..", "..", "..", "samples", "data", "torch-comparison");

    internal static float[] LoadBin(string name)
    {
        var path = Path.Combine(TestDataDir, name);
        Assert.That(File.Exists(path), Is.True, $"Missing reference file: {path}");
        var bytes = File.ReadAllBytes(path);
        var floats = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }

    internal static int[] LoadInt64Bin(string name)
    {
        var path = Path.Combine(TestDataDir, name);
        Assert.That(File.Exists(path), Is.True, $"Missing reference file: {path}");
        var bytes = File.ReadAllBytes(path);
        var longs = new long[bytes.Length / 8];
        Buffer.BlockCopy(bytes, 0, longs, 0, bytes.Length);
        return longs.Select(l => (int)l).ToArray();
    }

    internal static void AssertTensorEqual(float[] expected, float[] actual, float absTol = 1e-5f, float relTol = 1e-4f, string? label = null)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length),
            $"{label}: length mismatch {actual.Length} vs {expected.Length}");

        int failCount = 0;
        float maxDiff = 0f;
        int maxDiffIdx = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float threshold = absTol + relTol * MathF.Abs(expected[i]);
            if (diff > threshold)
            {
                failCount++;
                if (diff > maxDiff)
                {
                    maxDiff = diff;
                    maxDiffIdx = i;
                }
            }
        }

        if (failCount > 0)
        {
            int showCount = Math.Min(5, failCount);
            var diffs = new List<string>();
            int shown = 0;
            for (int i = 0; i < expected.Length && shown < showCount; i++)
            {
                float diff = MathF.Abs(expected[i] - actual[i]);
                float threshold = absTol + relTol * MathF.Abs(expected[i]);
                if (diff > threshold)
                {
                    diffs.Add($"  [{i}] expected={expected[i]:G7} actual={actual[i]:G7} diff={diff:G7}");
                    shown++;
                }
            }
            Assert.Fail($"{label}: {failCount} elements differ (max diff={maxDiff:G7} at [{maxDiffIdx}]).\n" +
                        string.Join("\n", diffs));
        }
    }

    internal static float[] ExtractOutput(ReverseGradTensor<float> tensor)
    {
        var arr = new float[tensor.Length];
        for (int i = 0; i < tensor.Length; i++)
            arr[i] = tensor[i];
        return arr;
    }

    internal static float[] ExtractGrad(ReverseGradTensor<float> tensor)
    {
        var grad = tensor.Grad ?? throw new InvalidOperationException("Tensor has no gradient");
        var arr = new float[grad.Length];
        for (int i = 0; i < grad.Length; i++)
            arr[i] = grad[i];
        return arr;
    }

    internal static float ScalarOutput(ReverseGradTensor<float> tensor)
    {
        Assert.That(tensor.Length, Is.EqualTo(1), "Expected scalar tensor (length 1)");
        return tensor[0];
    }

    /// <summary>
    /// Compares two tensors while treating non-finite values as a first-class, assertable
    /// outcome. <see cref="AssertTensorEqual"/> cannot do this: it computes
    /// <c>MathF.Abs(expected - actual)</c> and skips anything failing <c>diff &gt; threshold</c>,
    /// but <c>NaN &gt; x</c> is false, so a reference NaN matched against a finite actual value
    /// passes silently - the exact discrimination a safe-softmax test needs (clamp to zeros vs
    /// propagate NaN) would be unverifiable.
    /// </summary>
    internal static void AssertTensorClose(float[] expected, float[] actual, float absTol = 1e-5f, float relTol = 1e-4f, string? label = null)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length),
            $"{label}: length mismatch {actual.Length} vs {expected.Length}");

        var mismatches = new List<string>();
        int nanPositions = 0;
        float maxDiff = 0f;

        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            bool eNan = float.IsNaN(e);
            bool aNan = float.IsNaN(a);

            if (eNan || aNan)
            {
                if (eNan && aNan) nanPositions++;
                else mismatches.Add($"  [{i}] NaN mismatch: expected={Format(e)} actual={Format(a)}");
                continue;
            }

            if (float.IsInfinity(e) || float.IsInfinity(a))
            {
                if (e != a) mismatches.Add($"  [{i}] infinity mismatch: expected={Format(e)} actual={Format(a)}");
                continue;
            }

            float diff = MathF.Abs(e - a);
            if (diff > maxDiff) maxDiff = diff;
            if (diff > absTol + relTol * MathF.Abs(e))
                mismatches.Add($"  [{i}] expected={Format(e)} actual={Format(a)} diff={diff:G7}");
        }

        if (mismatches.Count > 0)
        {
            int show = Math.Min(5, mismatches.Count);
            string extra = mismatches.Count > show ? $"\n  ... and {mismatches.Count - show} more" : string.Empty;
            Assert.Fail($"{label}: {mismatches.Count} element(s) differ " +
                        $"(max finite diff={maxDiff:G7}, matching NaN positions={nanPositions}).\n" +
                        string.Join("\n", mismatches.Take(show)) + extra);
        }
    }

    static string Format(float value) => float.IsNaN(value) ? "NaN" : value.ToString("G9");

    internal static void AssertScalarEqual(float expected, float actual, float absTol = 1e-4f, float relTol = 1e-3f, string? label = null)
    {
        float diff = MathF.Abs(expected - actual);
        float threshold = absTol + relTol * MathF.Abs(expected);
        if (diff > threshold)
            Assert.Fail($"{label}: expected={expected:G7} actual={actual:G7} diff={diff:G7}");
    }
}
