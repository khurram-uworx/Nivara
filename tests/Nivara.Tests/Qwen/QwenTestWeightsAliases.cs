using Nivara.Samples;
using Nivara.Silk;
using System.Numerics;

namespace Nivara.Tests.Qwen;

internal static class QwenSyntheticWeights
{
    public static LlamaConfig SmokeConfig => QwenTestWeights.SmokeConfig;
    public static (Dictionary<string, (float[] Data, int[] Shape)> tensors, LlamaConfig config) Synthesize(uint seed = 1234) => QwenTestWeights.Synthesize(seed);
    public static (Dictionary<string, (TWeight[] Data, int[] Shape)> tensors, LlamaConfig config) Synthesize<TWeight>(uint seed = 1234) where TWeight : struct, IFloatingPointIeee754<TWeight> => QwenTestWeights.Synthesize<TWeight>(seed);
    public static (Dictionary<string, (float[] Data, int[] Shape)> tensors, LlamaConfig config) SynthesizeFloat(uint seed = 1234) => QwenTestWeights.SynthesizeFloat(seed);
    public static (Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors, LlamaConfig config) SynthesizeBFloat16(uint seed = 1234) => QwenTestWeights.SynthesizeBFloat16(seed);
    public static Dictionary<string, (BFloat16[] Data, int[] Shape)> ToBFloat16(Dictionary<string, (float[] Data, int[] Shape)> tensors) => QwenTestWeights.ToBFloat16(tensors);
    public static Dictionary<string, (float[] Data, int[] Shape)> ToF32(Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors) => QwenTestWeights.ToF32(tensors);
    public static QwenSilkConfig ToSilkConfig(LlamaConfig config, int maxPos) => QwenTestWeights.ToSilkConfig(config, maxPos);
}

internal static class QwenSyntheticTestWeights
{
    public static LlamaConfig SmokeConfig => QwenTestWeights.SmokeConfig;
    public static (Dictionary<string, (float[] Data, int[] Shape)> tensors, LlamaConfig config) Synthesize(uint seed = 1234) => QwenTestWeights.Synthesize(seed);
    public static (Dictionary<string, (TWeight[] Data, int[] Shape)> tensors, LlamaConfig config) Synthesize<TWeight>(uint seed = 1234) where TWeight : struct, IFloatingPointIeee754<TWeight> => QwenTestWeights.Synthesize<TWeight>(seed);
    public static (Dictionary<string, (float[] Data, int[] Shape)> tensors, LlamaConfig config) SynthesizeFloat(uint seed = 1234) => QwenTestWeights.SynthesizeFloat(seed);
    public static (Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors, LlamaConfig config) SynthesizeBFloat16(uint seed = 1234) => QwenTestWeights.SynthesizeBFloat16(seed);
    public static Dictionary<string, (BFloat16[] Data, int[] Shape)> ToBFloat16(Dictionary<string, (float[] Data, int[] Shape)> tensors) => QwenTestWeights.ToBFloat16(tensors);
    public static Dictionary<string, (float[] Data, int[] Shape)> ToF32(Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors) => QwenTestWeights.ToF32(tensors);
    public static QwenSilkConfig ToSilkConfig(LlamaConfig config, int maxPos) => QwenTestWeights.ToSilkConfig(config, maxPos);
}

internal static class QwenSmokeWeights
{
    public static LlamaConfig SmokeConfig => QwenTestWeights.SmokeConfig;
    public static (Dictionary<string, (float[] Data, int[] Shape)> tensors, LlamaConfig config) Synthesize(uint seed = 1234) => QwenTestWeights.Synthesize(seed);
    public static (Dictionary<string, (TWeight[] Data, int[] Shape)> tensors, LlamaConfig config) Synthesize<TWeight>(uint seed = 1234) where TWeight : struct, IFloatingPointIeee754<TWeight> => QwenTestWeights.Synthesize<TWeight>(seed);
    public static (Dictionary<string, (float[] Data, int[] Shape)> tensors, LlamaConfig config) SynthesizeFloat(uint seed = 1234) => QwenTestWeights.SynthesizeFloat(seed);
    public static (Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors, LlamaConfig config) SynthesizeBFloat16(uint seed = 1234) => QwenTestWeights.SynthesizeBFloat16(seed);
    public static Dictionary<string, (BFloat16[] Data, int[] Shape)> ToBFloat16(Dictionary<string, (float[] Data, int[] Shape)> tensors) => QwenTestWeights.ToBFloat16(tensors);
    public static Dictionary<string, (float[] Data, int[] Shape)> ToF32(Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors) => QwenTestWeights.ToF32(tensors);
    public static QwenSilkConfig ToSilkConfig(LlamaConfig config, int maxPos) => QwenTestWeights.ToSilkConfig(config, maxPos);
}

internal static class QwenSmokeTestWeights
{
    public static LlamaConfig SmokeConfig => QwenTestWeights.SmokeConfig;
    public static (Dictionary<string, (float[] Data, int[] Shape)> tensors, LlamaConfig config) Synthesize(uint seed = 1234) => QwenTestWeights.Synthesize(seed);
    public static (Dictionary<string, (TWeight[] Data, int[] Shape)> tensors, LlamaConfig config) Synthesize<TWeight>(uint seed = 1234) where TWeight : struct, IFloatingPointIeee754<TWeight> => QwenTestWeights.Synthesize<TWeight>(seed);
    public static (Dictionary<string, (float[] Data, int[] Shape)> tensors, LlamaConfig config) SynthesizeFloat(uint seed = 1234) => QwenTestWeights.SynthesizeFloat(seed);
    public static (Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors, LlamaConfig config) SynthesizeBFloat16(uint seed = 1234) => QwenTestWeights.SynthesizeBFloat16(seed);
    public static Dictionary<string, (BFloat16[] Data, int[] Shape)> ToBFloat16(Dictionary<string, (float[] Data, int[] Shape)> tensors) => QwenTestWeights.ToBFloat16(tensors);
    public static Dictionary<string, (float[] Data, int[] Shape)> ToF32(Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors) => QwenTestWeights.ToF32(tensors);
    public static QwenSilkConfig ToSilkConfig(LlamaConfig config, int maxPos) => QwenTestWeights.ToSilkConfig(config, maxPos);
}

internal static class QwenWeights
{
    public static LlamaConfig SmokeConfig => QwenTestWeights.SmokeConfig;
    public static (Dictionary<string, (float[] Data, int[] Shape)> tensors, LlamaConfig config) Synthesize(uint seed = 1234) => QwenTestWeights.Synthesize(seed);
    public static (Dictionary<string, (TWeight[] Data, int[] Shape)> tensors, LlamaConfig config) Synthesize<TWeight>(uint seed = 1234) where TWeight : struct, IFloatingPointIeee754<TWeight> => QwenTestWeights.Synthesize<TWeight>(seed);
    public static (Dictionary<string, (float[] Data, int[] Shape)> tensors, LlamaConfig config) SynthesizeFloat(uint seed = 1234) => QwenTestWeights.SynthesizeFloat(seed);
    public static (Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors, LlamaConfig config) SynthesizeBFloat16(uint seed = 1234) => QwenTestWeights.SynthesizeBFloat16(seed);
    public static Dictionary<string, (BFloat16[] Data, int[] Shape)> ToBFloat16(Dictionary<string, (float[] Data, int[] Shape)> tensors) => QwenTestWeights.ToBFloat16(tensors);
    public static Dictionary<string, (float[] Data, int[] Shape)> ToF32(Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors) => QwenTestWeights.ToF32(tensors);
    public static QwenSilkConfig ToSilkConfig(LlamaConfig config, int maxPos) => QwenTestWeights.ToSilkConfig(config, maxPos);
}
