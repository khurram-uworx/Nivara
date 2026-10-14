namespace Nivara.AI;

/// <summary>Result of a BERT-family encoder GPU forward: the encoder's final hidden state
/// ([batch*seqLen, hiddenDim]) and, when the classification head weights are present
/// (pre_classifier/classifier keys), the [batch, numClasses] logits.</summary>
public sealed record BertEncoderGpuResult(float[] Hidden, float[]? Logits);

/// <summary>Weight-key naming style for a BERT-family encoder: HuggingFace 'distilbert.*' keys
/// (DistilBERT), or BERT-style keys already stripped of the 'bert.' module prefix during
/// provisioning (MiniLM — the sample weights hold 'embeddings.*' / 'encoder.layer.N.*' keys).</summary>
public enum BertGpuNaming
{
    DistilBert,
    Bert
}

/// <summary>
/// Resolves the model-specific weight key names to the roles the runner consumes (word/position
/// embeddings + embed LayerNorm, per-layer q/k/v/o projections, ffn1/ffn2, ln1/ln2). Both naming
/// styles map to the same layer roles, so a runner is config-driven across the BERT family.
/// Shared so every GPU backend derives the same keys from the same place.
/// </summary>
public static class BertGpuKeys
{
    public static string Embeddings(BertGpuNaming naming)
        => naming == BertGpuNaming.Bert ? "embeddings" : "distilbert.embeddings";

    public static string Layer(BertGpuNaming naming, int index)
        => naming == BertGpuNaming.Bert
            ? $"encoder.layer.{index}"
            : $"distilbert.transformer.layer.{index}";

    public static string Attention(BertGpuNaming naming, int index, char role) => naming switch
    {
        BertGpuNaming.Bert => role switch
        {
            'q' => $"{Layer(naming, index)}.attention.self.query",
            'k' => $"{Layer(naming, index)}.attention.self.key",
            'v' => $"{Layer(naming, index)}.attention.self.value",
            'o' => $"{Layer(naming, index)}.attention.output.dense",
            _ => throw new ArgumentOutOfRangeException(nameof(role), role, "Expected q/k/v/o.")
        },
        _ => role switch
        {
            'q' => $"{Layer(naming, index)}.attention.q_lin",
            'k' => $"{Layer(naming, index)}.attention.k_lin",
            'v' => $"{Layer(naming, index)}.attention.v_lin",
            'o' => $"{Layer(naming, index)}.attention.out_lin",
            _ => throw new ArgumentOutOfRangeException(nameof(role), role, "Expected q/k/v/o.")
        }
    };

    public static string Ffn(BertGpuNaming naming, int index, int which) => naming switch
    {
        BertGpuNaming.Bert => which == 1
            ? $"{Layer(naming, index)}.intermediate.dense"
            : $"{Layer(naming, index)}.output.dense",
        _ => which == 1
            ? $"{Layer(naming, index)}.ffn.lin1"
            : $"{Layer(naming, index)}.ffn.lin2"
    };

    public static string LayerNorm(BertGpuNaming naming, int index, int which) => naming switch
    {
        BertGpuNaming.Bert => which == 1
            ? $"{Layer(naming, index)}.attention.output.LayerNorm"
            : $"{Layer(naming, index)}.output.LayerNorm",
        _ => which == 1
            ? $"{Layer(naming, index)}.sa_layer_norm"
            : $"{Layer(naming, index)}.output_layer_norm"
    };
}
