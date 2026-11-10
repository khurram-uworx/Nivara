namespace Nivara.Silk;

public sealed partial class SilkLayaHeadGpuRunner
{
    public SilkLayaHeadGpuRunner(
        SilkRuntime runtime,
        Dictionary<string, (Half[] Data, int[] Shape)> tensors,
        int headLayerCount = 2,
        int actClassCount = 2)
        : this(runtime, ToUshort(tensors), headLayerCount, actClassCount, isBf16: false, isFp16: true)
    {
    }

    internal static Dictionary<string, (ushort[] Data, int[] Shape)> ToUshort(
        Dictionary<string, (Half[] Data, int[] Shape)> tensors)
    {
        var result = new Dictionary<string, (ushort[] Data, int[] Shape)>(tensors.Count);
        foreach (var kv in tensors)
        {
            var src = kv.Value.Data;
            var arr = new ushort[src.Length];
            System.Runtime.InteropServices.MemoryMarshal.Cast<Half, ushort>(src.AsSpan()).CopyTo(arr.AsSpan());
            result[kv.Key] = (arr, kv.Value.Shape);
        }
        return result;
    }
}
