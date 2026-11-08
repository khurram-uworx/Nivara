using System.Numerics;
using System.Runtime.InteropServices;

namespace Nivara.Silk;

public sealed partial class SilkLayaHeadGpuRunner
{
    public SilkLayaHeadGpuRunner(
        SilkRuntime runtime,
        Dictionary<string, (Half[] Data, int[] Shape)> tensors,
        int headLayerCount = 2,
        int actClassCount = 2)
        : this(runtime, ToFloat(tensors), headLayerCount, actClassCount)
    {
    }

    static Dictionary<string, (float[] Data, int[] Shape)> ToFloat(
        Dictionary<string, (Half[] Data, int[] Shape)> tensors)
    {
        var result = new Dictionary<string, (float[] Data, int[] Shape)>(tensors.Count);
        foreach (var kv in tensors)
        {
            var src = kv.Value.Data;
            var arr = MemoryMarshal.Cast<Half, ushort>(src.AsSpan());
            var dst = new float[src.Length];
            for (int i = 0; i < src.Length; i++)
                dst[i] = (float)BitConverter.Int16BitsToHalf((short)arr[i]);
            result[kv.Key] = (dst, kv.Value.Shape);
        }
        return result;
    }
}
