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

    /// <summary>
    /// True BF16/FP16 path: keeps matrix weights narrow (<c>ushort</c>) on device.
    /// </summary>
    /// <param name="isBf16">
    /// How to interpret the <c>ushort</c> payload — BF16 bits when <see langword="true"/> (the
    /// default), FP16 bits when <see langword="false"/>. This is a bit-interpretation flag, not a
    /// precision preference; the <c>BFloat16</c> overload carries the widen-or-keep choice instead.
    /// </param>
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

    /// <summary>
    /// Ingests a <see cref="BFloat16"/> checkpoint. By default the weights are widened to F32 on
    /// load; pass <paramref name="trueBf16"/> to keep them BF16 on device.
    /// </summary>
    /// <param name="trueBf16">
    /// <see langword="false"/> (the default) widens the checkpoint to F32 and runs the F32 lane;
    /// <see langword="true"/> keeps the weights BF16 and runs the narrow lane. Widening is the
    /// default so that narrow precision is always an explicit choice — <see cref="CreateTrueBf16"/>
    /// names the same decision at the call site.
    /// </param>
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

    /// <summary>
    /// Keeps the checkpoint's matrix weights BF16 on device while activations, gradients and AdamW
    /// moments stay F32 — the narrow lane the parity gates measure. Equivalent to the
    /// <c>BFloat16</c> constructor with <c>trueBf16: true</c>, but named so the call site states the
    /// precision decision.
    /// </summary>
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
