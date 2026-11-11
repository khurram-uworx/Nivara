using System.Numerics;

namespace Nivara.Silk;

public sealed partial class SilkMobileNetV2GpuRunner
{
    public SilkMobileNetV2GpuRunner(
        SilkRuntime runtime,
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors,
        int numClasses = 1001)
        : this(runtime, ToFloat(tensors), numClasses)
    {
    }

    static Dictionary<string, (float[] Data, int[] Shape)> ToFloat(
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors)
    {
        var result = new Dictionary<string, (float[] Data, int[] Shape)>(tensors.Count);
        foreach (var kv in tensors)
        {
            var src = kv.Value.Data;
            var dst = new float[src.Length];
            for (int i = 0; i < src.Length; i++)
                dst[i] = (float)src[i];
            result[kv.Key] = (dst, kv.Value.Shape);
        }
        return result;
    }
}
