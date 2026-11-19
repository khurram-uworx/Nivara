using Microsoft.Extensions.AI;
using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using NivaraChat.Shared;

namespace NivaraChat.Transformer;

/// <summary>
/// An <see cref="IChatClient"/> backed by a trained <see cref="BatchedTransformer{T}"/>.
/// The model runs in eval mode; generation is autoregressive and re-entrant (each call
/// builds its own tensors), so concurrent requests are safe.
/// </summary>
internal sealed class BatchedChatClient : CausalLmChatClientBase
{
    readonly BatchedTransformer<float> model;
    readonly TextTokenizer tokenizer;

    public BatchedChatClient(
        BatchedTransformer<float> model,
        TextTokenizer tokenizer,
        float temperature = 0.8f,
        int maxNewTokens = 256,
        float topP = 1.0f,
        bool useKvCache = true)
        : base(temperature, topP, maxNewTokens, useKvCache, new Random(1234))
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);

        model.Eval();
        this.model = model;
        this.tokenizer = tokenizer;
    }

    public override ChatClientMetadata? Metadata { get; } =
        new ChatClientMetadata("Nivara", null, "BatchedTransformer-word-tokenizer");

    protected override string RenderPrompt(IEnumerable<ChatMessage> messages)
        => FormatConversation(messages);

    protected override IReadOnlyList<int> Encode(string prompt)
        => tokenizer.Encode(prompt, addBosEos: true);

    protected override string DecodeToken(int tokenId)
        => tokenizer.Decode([tokenId]);

    protected override string DecodeAll(IReadOnlyList<int> ids)
        => tokenizer.Decode(ids.ToArray());

    protected override CausalLmOps BuildOps()
    {
        TransformerKVCache<float>? cache = null;
        return new CausalLmOps(
            Prefill: w =>
            {
                cache ??= new TransformerKVCache<float>(model);
                return TensorToFloats(model.ForwardPrefill(w, cache));
            },
            Cached: (t, p) => TensorToFloats(model.ForwardCached(t, p, cache!)),
            LastLogits: ForwardLastRow,
            ResetCache: () => cache = new TransformerKVCache<float>(model),
            IsStopToken: id => id == tokenizer.EosToken,
            VocabSize: model.VocabSize,
            EosTokenId: tokenizer.EosToken,
            MaxSequenceLength: model.MaxSeqLen);
    }

    protected override bool WindowPrompt => true;
    protected override bool WindowNonCached => true;
    protected override int OverflowKeep => model.MaxSeqLen - 1;

    static string FormatConversation(IEnumerable<ChatMessage> messages)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var message in messages)
        {
            if (string.IsNullOrEmpty(message.Text))
                continue;

            string role = message.Role == ChatRole.User ? "user" : "assistant";
            sb.Append(role).Append(": ").Append(message.Text).Append('\n');
        }
        sb.Append("assistant:");
        return sb.ToString();
    }

    float[] ForwardLastRow(int[] window)
    {
        var data = new float[window.Length];
        for (int i = 0; i < window.Length; i++)
            data[i] = window[i];

        var column = NivaraColumn<float>.Create(data);
        var input = new ReverseGradTensor<float>(column, requiresGrad: false);
        input.Reshape(1, window.Length);

        var logits = model.Forward(input);
        int vocab = logits.Shape[^1];

        var last = new float[vocab];
        int rowStart = (window.Length - 1) * vocab;
        for (int v = 0; v < vocab; v++)
            last[v] = float.CreateChecked(logits[rowStart + v]);
        return last;
    }

    static int[] ToIntArray(List<int> window) => window.ToArray();

    static float[] TensorToFloats(ReverseGradTensor<float> logits)
    {
        logits.Data.TryGetSpan(out var span);
        int vocab = logits.Shape[^1];
        var last = new float[vocab];
        for (int v = 0; v < vocab; v++)
            last[v] = span.Length >= vocab ? span[v] : float.CreateChecked(logits[v]);
        return last;
    }
}
