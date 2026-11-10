namespace Nivara.Silk;

/// <summary>
/// The single, model-agnostic definition of Silk's opt-in partial/frozen training policy.
/// A trainer's <c>TrainableLayerCount</c> (top-down: head + top N transformer layers, frozen
/// embeddings in any partial mode) is the canonical input; this helper turns it into the per-group
/// trainable flags so that every Silk trainer (DistilBERT today, Laya/ModernBERT later) applies
/// exactly the same rule instead of reinventing a second freeze policy.
/// </summary>
/// <remarks>
/// Rules (mirroring <see cref="SilkDistilBertTrainer"/>):
/// <list type="bullet">
/// <item><description>With <c>trainable == false</c> nothing is trainable.</description></item>
/// <item><description>Negative or oversized counts are clamped to <c>[0, numLayers]</c>.</description></item>
/// <item><description>The classifier/decision head is always trainable when <c>trainable == true</c>.</description></item>
/// <item><description>Embeddings (and the encoder's input LayerNorm) are trainable only when every layer is.</description></item>
/// <item><description>Only the top <c>N</c> transformer layers are trainable; lower layers are frozen.</description></item>
/// </list>
/// </remarks>
internal static class SilkPartialTrainingPolicy
{
    /// <summary>Normalizes a requested trainable-layer count into <c>[0, numLayers]</c>.</summary>
    public static int Normalize(int requested, int numLayers, bool trainable)
    {
        if (!trainable) return 0;
        if (numLayers < 0) throw new ArgumentOutOfRangeException(nameof(numLayers), numLayers, "numLayers cannot be negative.");
        // Negative inputs are clamped to 0 (property semantics). The -1 "all layers" sentinel is
        // handled by callers (constructors) before reaching this helper.
        return Math.Clamp(requested, 0, numLayers);
    }

    /// <summary>Whether the embedding/input-norm parameters are trainable.</summary>
    public static bool EmbeddingsTrainable(int normalized, int numLayers, bool trainable)
        => trainable && normalized == numLayers;

    /// <summary>Whether transformer layer <paramref name="layerIndex"/> (0-based, bottom to top) is trainable.</summary>
    public static bool LayerTrainable(int layerIndex, int numLayers, int normalized, bool trainable)
    {
        if (!trainable) return false;
        if (layerIndex < 0 || layerIndex >= numLayers) throw new ArgumentOutOfRangeException(nameof(layerIndex), layerIndex, "layerIndex out of range.");
        int firstTrainable = numLayers - normalized;
        return layerIndex >= firstTrainable;
    }

    /// <summary>Whether the classifier/decision head parameters are trainable.</summary>
    public static bool HeadTrainable(bool trainable) => trainable;
}
