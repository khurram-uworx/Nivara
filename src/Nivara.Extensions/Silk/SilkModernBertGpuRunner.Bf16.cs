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
        : this(runtime, config, ToFloat(tensors), prefix)
    {
    }

    static Dictionary<string, (float[] Data, int[] Shape)> ToFloat(
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors)
    {
        var result = new Dictionary<string, (float[] Data, int[] Shape)>(tensors.Count);
        foreach (var kv in tensors)
        {
            var src = kv.Value.Data;
            var u = System.Runtime.InteropServices.MemoryMarshal.Cast<BFloat16, ushort>(src.AsSpan());
            var dst = new float[src.Length];
            for (int i = 0; i < src.Length; i++)
                dst[i] = BitConverter.UInt32BitsToSingle((uint)u[i] << 16);
            result[kv.Key] = (dst, kv.Value.Shape);
        }
        return result;
    }
}
