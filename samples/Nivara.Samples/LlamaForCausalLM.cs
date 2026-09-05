using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.Samples;

/// <summary>
/// Llama-family causal language model: token embedding, <c>numHiddenLayers</c> decoder
/// blocks, a final RMS norm, and a tied-embedding LM head producing logits over the
/// vocabulary. Inference-by-default: running outside a <c>GradientUtils.Grad()</c> scope
/// builds no graph nodes. Matches the <c>LlamaForCausalLM</c> structure used by SmolLM.
/// </summary>
public sealed class LlamaForCausalLM<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>Gets the token embedding table (reused as the tied LM head).</summary>
    public Embedding<T> Embed { get; }
    internal readonly LlamaDecoderBlock<T>[] layers;
    internal readonly RMSNorm<T> finalNorm;
    internal readonly int vocabSize;
    internal readonly bool tieWordEmbeddings;

    /// <summary>Gets the model's vocabulary size.</summary>
    public int VocabSize => vocabSize;

    /// <summary>Gets whether the LM head reuses the input embedding weights.</summary>
    public bool TieWordEmbeddings => tieWordEmbeddings;

    public LlamaForCausalLM(
        int vocabSize,
        int hiddenSize,
        int numHiddenLayers,
        int numHeads,
        int numKeyValueHeads,
        int intermediateSize,
        float rmsNormEps = 1e-5f,
        int maxPositionEmbeddings = 2048,
        float ropeTheta = 10000f,
        bool tieWordEmbeddings = true,
        bool qkvBias = false)
    {
        this.vocabSize = vocabSize;
        this.tieWordEmbeddings = tieWordEmbeddings;

        Embed = new Embedding<T>(vocabSize, hiddenSize);
        layers = new LlamaDecoderBlock<T>[numHiddenLayers];
        for (int i = 0; i < numHiddenLayers; i++)
            layers[i] = new LlamaDecoderBlock<T>(
                hiddenSize, numHeads, numKeyValueHeads, intermediateSize,
                rmsNormEps, maxPositionEmbeddings, ropeTheta, qkvBias);
        finalNorm = new RMSNorm<T>(hiddenSize, rmsNormEps);

        var modules = new Module<T>[layers.Length + 2];
        modules[0] = Embed;
        modules[^1] = finalNorm;
        for (int i = 0; i < layers.Length; i++)
            modules[i + 1] = layers[i];
        RegisterModules(modules);
    }

    /// <summary>
    /// Runs the full causal-LM stack over token IDs, returning logits <c>[L, vocabSize]</c>.
    /// Token IDs are passed as exact integers so they survive narrow compute dtypes.
    /// </summary>
    /// <param name="inputIds">Token IDs (length L)</param>
    /// <returns>Logits with shape <c>[L, vocabSize]</c></returns>
    public ReverseGradTensor<T> Forward(int[] inputIds)
    {
        if (inputIds == null) throw new ArgumentNullException(nameof(inputIds));

        var h = Embed.Forward(inputIds); // [L, hidden]

        foreach (var layer in layers)
            h = layer.Forward(h);

        h = finalNorm.Forward(h); // [L, hidden]

        // Tied LM head: logits = h @ embedWeight^T, producing [L, vocab].
        var logits = ReverseGradOperations.MatMulTransposedB(h, Embed.Weight!.Tensor);
        logits.Reshape(h.Shape[0], vocabSize);
        return logits;
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
        => throw new NotImplementedException("Use Forward(int[] inputIds) for LlamaForCausalLM.");

    /// <summary>
    /// Runs the causal-LM stack for a <em>single</em> token during cached inference, returning
    /// logits <c>[1, vocabSize]</c>. The token's hidden state is embedded and passed through every
    /// layer's <see cref="LlamaDecoderBlock{T}.ForwardCached"/>, which appends the new key/value to
    /// the supplied <see cref="LlamaKVCache{T}"/> and attends against the cached prefix. Numerically
    /// identical to processing the token as the final position of a full
    /// <see cref="Forward(int[])"/> call.
    /// </summary>
    /// <param name="tokenId">The new token id</param>
    /// <param name="positionOffset">Absolute position of this token</param>
    /// <param name="cache">The shared KV cache (positions 0..<paramref name="positionOffset"/>-1)</param>
    /// <returns>Logits with shape <c>[1, vocabSize]</c></returns>
    public ReverseGradTensor<T> ForwardCached(int tokenId, int positionOffset, LlamaKVCache<T> cache)
    {
        if (tokenId < 0 || tokenId >= vocabSize) throw new ArgumentOutOfRangeException(nameof(tokenId));
        if (positionOffset < 0) throw new ArgumentOutOfRangeException(nameof(positionOffset));
        ArgumentNullException.ThrowIfNull(cache);

        cache.Ensure(positionOffset + 1);

        var h = Embed.Forward(tokenId); // [1, hidden]

        for (int i = 0; i < layers.Length; i++)
        {
            var layer = layers[i];
            h = layer.ForwardCached(h, positionOffset, cache.keys[i], cache.values[i], positionOffset);
        }

        h = finalNorm.Forward(h); // [1, hidden]

        var logits = ReverseGradOperations.MatMulTransposedB(h, Embed.Weight!.Tensor);
        logits.Reshape(1, vocabSize);
        return logits;
    }
}

/// <summary>
/// Holds the per-layer key/value caches for cached inference on
/// <see cref="LlamaForCausalLM{T}"/>. Each layer stores the RoPE'd keys and values (per-KV-head,
/// row-major) of every generated position. Buffers grow on demand and are reused across steps.
/// Call <see cref="Reset"/> before starting a new generation sequence.
/// </summary>
public sealed class LlamaKVCache<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
{
    internal readonly T[][] keys;
    internal readonly T[][] values;
    int capacity;
    bool disposed;

    /// <summary>Creates a fresh cache sized for <paramref name="numLayers"/> layers.</summary>
    public LlamaKVCache(int numLayers, int kvWidth, int initialCapacity = 16)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (kvWidth <= 0) throw new ArgumentOutOfRangeException(nameof(kvWidth));
        if (initialCapacity < 1) throw new ArgumentOutOfRangeException(nameof(initialCapacity));

        keys = new T[numLayers][];
        values = new T[numLayers][];
        capacity = initialCapacity;
        for (int i = 0; i < numLayers; i++)
        {
            keys[i] = new T[initialCapacity * kvWidth];
            values[i] = new T[initialCapacity * kvWidth];
        }
        KvWidth = kvWidth;
    }

    /// <summary>Gets the per-KV-head width of each cached row.</summary>
    public int KvWidth { get; }

    /// <summary>Returns the buffers to their initial size and clears the position counter.</summary>
    public void Reset()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        capacity = keys[0].Length / KvWidth;
        for (int i = 0; i < keys.Length; i++)
        {
            Array.Clear(keys[i]);
            Array.Clear(values[i]);
        }
    }

    internal void Ensure(int neededLength)
    {
        if (neededLength <= capacity) return;
        int newCap = Math.Max(capacity * 2, neededLength);
        for (int i = 0; i < keys.Length; i++)
        {
            var newK = new T[newCap * KvWidth];
            var newV = new T[newCap * KvWidth];
            Array.Copy(keys[i], newK, capacity * KvWidth);
            Array.Copy(values[i], newV, capacity * KvWidth);
            keys[i] = newK;
            values[i] = newV;
        }
        capacity = newCap;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (disposed) return;
        Array.Clear(keys);   // drop references so the pooled buffers can be collected
        Array.Clear(values);
        disposed = true;
    }
}

/// <summary>Configuration for a SmolLM / Llama-family causal LM.</summary>
public sealed record LlamaConfig : LLamaConfigLike
{
    public int HiddenSize { get; init; } = 576;
    public int NumHiddenLayers { get; init; } = 30;
    public int NumAttentionHeads { get; init; } = 9;
    public int NumKeyValueHeads { get; init; } = 3;
    public int IntermediateSize { get; init; } = 1536;
    public int VocabSize { get; init; } = 49152;
    public int MaxPositionEmbeddings { get; init; } = 2048;
    public float RmsNormEps { get; init; } = 1e-5f;
    public float RopeTheta { get; init; } = 10000f;
    public bool TieWordEmbeddings { get; init; } = true;
    public int BosTokenId { get; init; } = 1;
    public int EosTokenId { get; init; } = 2;
    public int PadTokenId { get; init; } = 2;
    public string HiddenAct { get; init; } = "silu";

    public static LlamaConfig FromJson(string json)
    {
        var config = new LlamaConfig();
        var props = typeof(LlamaConfig).GetProperties();
        foreach (var prop in props)
        {
            var key = SnakeKey(prop.Name);
            var idx = json.IndexOf($"\"{key}\"", StringComparison.OrdinalIgnoreCase);
            if (idx < 0) continue;
            var colonIdx = json.IndexOf(':', idx);
            var endIdx = json.IndexOfAny([',', '}', ']'], colonIdx);
            var valStr = json.Substring(colonIdx + 1, endIdx - colonIdx - 1).Trim();
            if (prop.PropertyType == typeof(int) && int.TryParse(valStr, out int intVal))
                prop.SetValue(config, intVal);
            else if (prop.PropertyType == typeof(float) && float.TryParse(valStr, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float floatVal))
                prop.SetValue(config, floatVal);
            else if (prop.PropertyType == typeof(bool) && bool.TryParse(valStr, out bool boolVal))
                prop.SetValue(config, boolVal);
            else if (prop.PropertyType == typeof(string))
                prop.SetValue(config, valStr.Trim('"'));
        }
        return config;
    }

    static string SnakeKey(string pascal)
    {
        if (pascal == "NumHiddenLayers") return "num_hidden_layers";
        if (pascal == "NumAttentionHeads") return "num_attention_heads";
        if (pascal == "NumKeyValueHeads") return "num_key_value_heads";
        if (pascal == "HiddenSize") return "hidden_size";
        if (pascal == "IntermediateSize") return "intermediate_size";
        if (pascal == "VocabSize") return "vocab_size";
        if (pascal == "MaxPositionEmbeddings") return "max_position_embeddings";
        if (pascal == "RmsNormEps") return "rms_norm_eps";
        if (pascal == "RopeTheta") return "rope_theta";
        if (pascal == "TieWordEmbeddings") return "tie_word_embeddings";
        if (pascal == "BosTokenId") return "bos_token_id";
        if (pascal == "EosTokenId") return "eos_token_id";
        if (pascal == "PadTokenId") return "pad_token_id";
        if (pascal == "HiddenAct") return "hidden_act";
        return pascal.ToLowerInvariant();
    }
}

/// <summary>
/// Loads Llama/SmolLM safetensors (or a pre-read tensor dictionary) into a
/// <see cref="LlamaForCausalLM{TModel}"/>. Key layout follows the HF Llama convention
/// (<c>model.embed_tokens.weight</c>, <c>model.layers.{i}.self_attn.{q,k,v,o}_proj.weight</c>,
/// <c>model.layers.{i}.mlp.{gate,up,down}_proj.weight</c>, <c>model.norm.weight</c>), and the
/// embedding weight is reused for the tied LM head.
/// </summary>
public static class LlamaLoader
{
    public static LlamaForCausalLM<TModel> Load<TModel, TWeight>(
        LLamaConfigLike config,
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        // Qwen2-style checkpoints carry bias on the query/key/value projections; canonical
        // Llama/SmolLM do not. The checkpoint is authoritative — mirror transformers' inference
        // of use_qkv_bias from the loaded weights rather than requiring a config flag.
        bool qkvBias = tensors.ContainsKey("model.layers.0.self_attn.q_proj.bias");

        var model = new LlamaForCausalLM<TModel>(
            config.VocabSize,
            config.HiddenSize,
            config.NumHiddenLayers,
            config.NumAttentionHeads,
            config.NumKeyValueHeads,
            config.IntermediateSize,
            config.RmsNormEps,
            config.MaxPositionEmbeddings,
            config.RopeTheta,
            config.TieWordEmbeddings,
            qkvBias);

        StateDictLoader.LoadEmbed(model.Embed, tensors, "model.embed_tokens.weight");
        StateDictLoader.LoadRMSNorm(model.finalNorm, tensors, "model.norm");

        for (int i = 0; i < model.layers.Length; i++)
        {
            var layer = model.layers[i];
            var p = $"model.layers.{i}";
            StateDictLoader.LoadRMSNorm(layer.InputNorm, tensors, $"{p}.input_layernorm");
            StateDictLoader.LoadRMSNorm(layer.PostNorm, tensors, $"{p}.post_attention_layernorm");
            StateDictLoader.LoadLinear(layer.Attention.QProj, tensors, $"{p}.self_attn.q_proj");
            StateDictLoader.LoadLinear(layer.Attention.KProj, tensors, $"{p}.self_attn.k_proj");
            StateDictLoader.LoadLinear(layer.Attention.VProj, tensors, $"{p}.self_attn.v_proj");
            StateDictLoader.LoadLinear(layer.Attention.OProj, tensors, $"{p}.self_attn.o_proj");
            StateDictLoader.LoadLinear(layer.GateProj, tensors, $"{p}.mlp.gate_proj");
            StateDictLoader.LoadLinear(layer.UpProj, tensors, $"{p}.mlp.up_proj");
            StateDictLoader.LoadLinear(layer.DownProj, tensors, $"{p}.mlp.down_proj");
        }

        model.Eval();
        return model;
    }
}

/// <summary>
/// Minimal structural view of <see cref="LlamaConfig"/> consumed by
/// <see cref="LlamaLoader.Load{TModel, TWeight}"/> to avoid a hard dependency between the
/// loader and the record.
/// </summary>
public interface LLamaConfigLike
{
    int VocabSize { get; }
    int HiddenSize { get; }
    int NumHiddenLayers { get; }
    int NumAttentionHeads { get; }
    int NumKeyValueHeads { get; }
    int IntermediateSize { get; }
    float RmsNormEps { get; }
    int MaxPositionEmbeddings { get; }
    float RopeTheta { get; }
    bool TieWordEmbeddings { get; }
}

