using System.Numerics;

namespace Nivara.Silk;

public sealed partial class SilkLayaHeadGpuRunner
{
    public SilkLayaHeadGpuRunner(
        SilkRuntime runtime,
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors,
        int headLayerCount = 2,
        int actClassCount = 2)
        : this(runtime, ToUshort(tensors), headLayerCount, actClassCount, isBf16: true, isFp16: false)
    {
    }

    internal static Dictionary<string, (ushort[] Data, int[] Shape)> ToUshort(
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors)
    {
        var result = new Dictionary<string, (ushort[] Data, int[] Shape)>(tensors.Count);
        foreach (var kv in tensors)
        {
            var src = kv.Value.Data;
            var arr = new ushort[src.Length];
            System.Runtime.InteropServices.MemoryMarshal.Cast<BFloat16, ushort>(src.AsSpan()).CopyTo(arr.AsSpan());
            result[kv.Key] = (arr, kv.Value.Shape);
        }
        return result;
    }
}
