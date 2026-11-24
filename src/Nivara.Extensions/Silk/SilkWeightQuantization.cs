namespace Nivara.Silk;

/// <summary>Weight storage mode for Silk inference runners.</summary>
public enum SilkWeightQuantization
{
    /// <summary>Use the checkpoint's native storage (F32 or BF16) without quantization.</summary>
    None = 0,
    /// <summary>Quantize decode weights to INT8 block-32 with in-kernel dequant (issue #1469).</summary>
    Int8Block32 = 1,
}
