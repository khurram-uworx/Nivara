using Nivara.AI;

namespace Nivara.Silk;

public sealed partial class SilkDistilBertTrainer
{
    /// <summary>
    /// Widens an FP16 (Half) checkpoint into the F32 tensor dictionary the trainer's device path
    /// consumes, so an FP16 checkpoint can be ingested (converted on load) without an external
    /// conversion step.
    /// </summary>
    internal static Dictionary<string, (float[] Data, int[] Shape)> ToFloat(
        Dictionary<string, (Half[] Data, int[] Shape)> tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var result = new Dictionary<string, (float[] Data, int[] Shape)>(tensors.Count, StringComparer.Ordinal);
        foreach (var kv in tensors)
        {
            var src = kv.Value.Data;
            var dst = new float[src.Length];
            for (int i = 0; i < src.Length; i++)
                dst[i] = (float)src[i];
            result[kv.Key] = (dst, kv.Value.Shape);
        }
        return result;
    }

    /// <summary>
    /// Ingests an FP16 (Half) checkpoint by converting its weights to F32 on load; training then
    /// runs on the device's F32 path. See the F32 constructor for parameter semantics.
    /// </summary>
    public SilkDistilBertTrainer(
        SilkRuntime runtime,
        BertConfig config,
        Dictionary<string, (Half[] Data, int[] Shape)> tensors,
        BertGpuNaming naming,
        int maxBatch,
        int maxSeqLen,
        int trainableLayerCount = -1)
        : this(runtime, config, ToFloat(tensors), naming, maxBatch, maxSeqLen, trainableLayerCount)
    {
    }

    /// <summary>
    /// Convenience constructor (FP16 checkpoint) that freezes the whole encoder (trains only the
    /// classifier head). See the F32 constructor for parameter semantics.
    /// </summary>
    public SilkDistilBertTrainer(
        SilkRuntime runtime,
        BertConfig config,
        Dictionary<string, (Half[] Data, int[] Shape)> tensors,
        BertGpuNaming naming,
        int maxBatch,
        int maxSeqLen,
        bool freezeEncoder)
        : this(runtime, config, ToFloat(tensors), naming, maxBatch, maxSeqLen, freezeEncoder ? 0 : -1)
    {
    }
}
