using Nivara.Samples;
using Nivara.Silk;

namespace Nivara.Tests.Qwen;

/// <summary>
/// Compatibility wrapper for the previous Silk-specific name. Prefer
/// <see cref="QwenTestWeights"/> for new tests.
/// </summary>
internal static class SilkQwenTestWeights
{
    public static LlamaConfig SmokeConfig => QwenTestWeights.SmokeConfig;

    public static (Dictionary<string, (float[] Data, int[] Shape)> tensors, LlamaConfig config)
        Synthesize(uint seed = 1234) => QwenTestWeights.Synthesize(seed);

    public static QwenSilkConfig ToSilkConfig(LlamaConfig config, int maxPos) =>
        QwenTestWeights.ToSilkConfig(config, maxPos);
}
