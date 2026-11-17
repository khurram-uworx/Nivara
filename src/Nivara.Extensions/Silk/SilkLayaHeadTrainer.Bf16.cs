using Nivara.AI;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Nivara.Silk;

public sealed partial class SilkLayaHeadTrainer
{
    internal static Dictionary<string, (ushort[] Data, int[] Shape)> ToUshort(
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var result = new Dictionary<string, (ushort[] Data, int[] Shape)>(tensors.Count, StringComparer.Ordinal);
        foreach (var kv in tensors)
        {
            var src = kv.Value.Data;
            var arr = new ushort[src.Length];
            MemoryMarshal.Cast<BFloat16, ushort>(src.AsSpan()).CopyTo(arr.AsSpan());
            result[kv.Key] = (arr, kv.Value.Shape);
        }
        return result;
    }

    public SilkLayaHeadTrainer(
        SilkRuntime runtime,
        Dictionary<string, (ushort[] Data, int[] Shape)> tensors,
        int headLayerCount = 2,
        int actClassCount = 2,
        int maxSeqLen = 64,
        bool isBf16 = true,
        bool isFp16 = false)
        : this(runtime, new SilkLayaHeadWeights(runtime, tensors, headLayerCount, actClassCount, trainable: true, isBf16, isFp16), isBf16, isFp16, maxSeqLen)
    {
    }

    public SilkLayaHeadTrainer(
        SilkRuntime runtime,
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors,
        int headLayerCount = 2,
        int actClassCount = 2,
        int maxSeqLen = 64,
        bool trueBf16 = false)
        : this(runtime,
              trueBf16
                  ? new SilkLayaHeadWeights(runtime, ToUshort(tensors), headLayerCount, actClassCount, trainable: true, true, false)
                  : new SilkLayaHeadWeights(runtime, ToFloat(tensors), headLayerCount, actClassCount, trainable: true),
              trueBf16, false, maxSeqLen)
    {
    }

    internal static Dictionary<string, (float[] Data, int[] Shape)> ToFloat(
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var result = new Dictionary<string, (float[] Data, int[] Shape)>(tensors.Count, StringComparer.Ordinal);
        foreach (var kv in tensors)
        {
            var src = kv.Value.Data;
            var arr = new float[src.Length];
            for (int i = 0; i < src.Length; i++) arr[i] = (float)src[i];
            result[kv.Key] = (arr, kv.Value.Shape);
        }
        return result;
    }

    public static SilkLayaHeadTrainer CreateTrueBf16(
        SilkRuntime runtime,
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors,
        int headLayerCount = 2,
        int actClassCount = 2,
        int maxSeqLen = 64)
        => new SilkLayaHeadTrainer(runtime, tensors, headLayerCount, actClassCount, maxSeqLen, trueBf16: true);

    public static SilkLayaHeadTrainer CreateTrueBf16(
        SilkRuntime runtime,
        Dictionary<string, (ushort[] Data, int[] Shape)> tensors,
        int headLayerCount = 2,
        int actClassCount = 2,
        int maxSeqLen = 64)
        => new SilkLayaHeadTrainer(runtime, tensors, headLayerCount, actClassCount, maxSeqLen, true, false);
}
