using Nivara.AI;
using System.Numerics;

namespace Nivara.Silk;

public sealed partial class SilkModernBertGpuRunner
{
    public SilkModernBertGpuRunner(
        SilkRuntime runtime,
        ModernBertConfig config,
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors,
        string prefix = "model")
        : this(runtime, config, ToUshort(tensors), prefix, isBf16: true, isFp16: false)
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
            // BFloat16 and ushort share an identical 2-byte layout; bit-preserving reinterpretation for device storage.
            System.Runtime.InteropServices.MemoryMarshal.Cast<BFloat16, ushort>(src.AsSpan()).CopyTo(arr.AsSpan());
            result[kv.Key] = (arr, kv.Value.Shape);
        }
        return result;
    }
}
