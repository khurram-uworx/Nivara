using Nivara.AI;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Nivara.Silk;

public sealed partial class SilkBertEncoderRunner
{
    public SilkBertEncoderRunner(
        SilkRuntime runtime,
        BertConfig config,
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors,
        BertGpuNaming naming)
        : this(runtime, config, ToUshort(tensors), naming, isBf16: true, isFp16: false)
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
            MemoryMarshal.Cast<BFloat16, ushort>(src.AsSpan()).CopyTo(arr.AsSpan());
            result[kv.Key] = (arr, kv.Value.Shape);
        }
        return result;
    }
}