using Nivara.AI;
using System.Runtime.InteropServices;

namespace Nivara.Silk;

public sealed partial class SilkBertEncoderRunner
{
    public SilkBertEncoderRunner(
        SilkRuntime runtime,
        BertConfig config,
        Dictionary<string, (Half[] Data, int[] Shape)> tensors,
        BertGpuNaming naming)
        : this(runtime, config, ToUshort(tensors), naming, isBf16: false, isFp16: true)
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
            MemoryMarshal.Cast<Half, ushort>(src.AsSpan()).CopyTo(arr.AsSpan());
            result[kv.Key] = (arr, kv.Value.Shape);
        }
        return result;
    }
}
