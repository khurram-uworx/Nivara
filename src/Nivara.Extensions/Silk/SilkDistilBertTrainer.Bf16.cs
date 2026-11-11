using System.Numerics;

namespace Nivara.Silk;

public sealed partial class SilkDistilBertTrainer
{
    /// <summary>
    /// Widens a BF16 checkpoint into the F32 tensor dictionary the trainer's device path consumes,
    /// so a BF16 checkpoint can be ingested (converted on load) without an external conversion step.
    /// </summary>
    internal static Dictionary<string, (float[] Data, int[] Shape)> ToFloat(
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors)
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
}
