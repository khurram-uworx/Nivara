using Nivara.AI;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Nivara.Silk;

public sealed partial class SilkDistilBertTrainer
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
            // BFloat16 and ushort share an identical 2-byte layout, so this is a
            // bit-preserving reinterpretation (no conversion) — safe to copy via Cast.
            MemoryMarshal.Cast<BFloat16, ushort>(src.AsSpan()).CopyTo(arr.AsSpan());
            result[kv.Key] = (arr, kv.Value.Shape);
        }
        return result;
    }

    /// <summary>True BF16 path: keeps matrix weights as BF16 (<c>ushort</c>) on device.</summary>
    public SilkDistilBertTrainer(
        SilkRuntime runtime,
        BertConfig config,
        Dictionary<string, (ushort[] Data, int[] Shape)> tensors,
        BertGpuNaming naming,
        int maxBatch,
        int maxSeqLen,
        bool isBf16 = true,
        bool isFp16 = false,
        int trainableLayerCount = -1)
        : this(runtime, config,
              new SilkBertWeights(runtime, config, tensors, naming, trainable: true, isBf16, isFp16, trainableLayerCount),
              isBf16, isFp16, maxBatch, maxSeqLen, trainableLayerCount)
    {
    }

    public SilkDistilBertTrainer(
        SilkRuntime runtime,
        BertConfig config,
        Dictionary<string, (ushort[] Data, int[] Shape)> tensors,
        BertGpuNaming naming,
        int maxBatch,
        int maxSeqLen,
        bool freezeEncoder,
        bool isBf16 = true,
        bool isFp16 = false)
        : this(runtime, config, tensors, naming, maxBatch, maxSeqLen, isBf16, isFp16, freezeEncoder ? 0 : -1)
    {
    }

    public SilkDistilBertTrainer(
        SilkRuntime runtime,
        BertConfig config,
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors,
        BertGpuNaming naming,
        int maxBatch,
        int maxSeqLen,
        int trainableLayerCount = -1,
        bool trueBf16 = false)
        : this(runtime, config,
              trueBf16
                  ? new SilkBertWeights(runtime, config, ToUshort(tensors), naming, trainable: true, true, false, trainableLayerCount)
                  : new SilkBertWeights(runtime, config, ToFloat(tensors), naming, trainable: true, trainableLayerCount: trainableLayerCount),
              trueBf16, false, maxBatch, maxSeqLen, trainableLayerCount)
    {
    }

    // freezeEncoder convenience for true BF16 is provided by CreateTrueBf16(..., freezeEncoder).

    public static SilkDistilBertTrainer CreateTrueBf16(
        SilkRuntime runtime, BertConfig config,
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors,
        BertGpuNaming naming, int maxBatch, int maxSeqLen,
        int trainableLayerCount = -1)
        => new SilkDistilBertTrainer(runtime, config, tensors, naming, maxBatch, maxSeqLen, trainableLayerCount, true);

    public static SilkDistilBertTrainer CreateTrueBf16(
        SilkRuntime runtime, BertConfig config,
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors,
        BertGpuNaming naming, int maxBatch, int maxSeqLen,
        bool freezeEncoder)
        => new SilkDistilBertTrainer(runtime, config, tensors, naming, maxBatch, maxSeqLen, freezeEncoder ? 0 : -1, true);

    public static SilkDistilBertTrainer CreateBf16(
        SilkRuntime runtime, BertConfig config,
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors,
        BertGpuNaming naming, int maxBatch, int maxSeqLen,
        int trainableLayerCount = -1)
        => CreateTrueBf16(runtime, config, tensors, naming, maxBatch, maxSeqLen, trainableLayerCount);

    public static SilkDistilBertTrainer CreateNarrowBf16(
        SilkRuntime runtime, BertConfig config,
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors,
        BertGpuNaming naming, int maxBatch, int maxSeqLen,
        int trainableLayerCount = -1)
        => CreateTrueBf16(runtime, config, tensors, naming, maxBatch, maxSeqLen, trainableLayerCount);

    public static SilkDistilBertTrainer CreateTrueBf16(
        SilkRuntime runtime, BertConfig config,
        Dictionary<string, (ushort[] Data, int[] Shape)> tensors,
        BertGpuNaming naming, int maxBatch, int maxSeqLen,
        int trainableLayerCount = -1)
        => new SilkDistilBertTrainer(runtime, config, tensors, naming, maxBatch, maxSeqLen, true, false, trainableLayerCount);

    public static SilkDistilBertTrainer CreateTrueBf16(
        SilkRuntime runtime, BertConfig config,
        Dictionary<string, (ushort[] Data, int[] Shape)> tensors,
        BertGpuNaming naming, int maxBatch, int maxSeqLen,
        bool freezeEncoder)
        => new SilkDistilBertTrainer(runtime, config, tensors, naming, maxBatch, maxSeqLen, true, false, freezeEncoder ? 0 : -1);
}
