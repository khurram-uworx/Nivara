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
internal sealed class BatchedChatClient : IChatClient
{
    readonly BatchedTransformer<float> model;
    readonly TextTokenizer tokenizer;
    readonly float temperature;
    readonly float topP;
    readonly int maxNewTokens;
    readonly bool useKvCache;
    readonly Random rng = new(1234);

    public BatchedChatClient(
        BatchedTransformer<float> model,
        TextTokenizer tokenizer,
        float temperature = 0.8f,
        int maxNewTokens = 256,
        float topP = 1.0f,
        bool useKvCache = false)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);

        model.Eval();
        this.model = model;
        this.tokenizer = tokenizer;
        this.temperature = temperature;
        this.topP = Math.Clamp(topP, 0f, 1f);
        this.maxNewTokens = maxNewTokens > 0 ? maxNewTokens : 1;
        this.useKvCache = useKvCache;
    }

    public ChatClientMetadata? Metadata { get; } =
        new ChatClientMetadata("Nivara", null, "BatchedTransformer-word-tokenizer");

    public object? GetService(Type serviceType, object? key = null) => null;

    public Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var prompt = FormatConversation(messages);
        var text = Generate(prompt, cancellationToken);
        return Task.FromResult(new ChatResponse([new ChatMessage(ChatRole.Assistant, text)]));
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var prompt = FormatConversation(messages);
        foreach (var chunk in GenerateStreaming(prompt, cancellationToken))
        {
            yield return new ChatResponseUpdate(ChatRole.Assistant, chunk);
        }
    }

    public void Dispose() { }

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

    string Generate(string prompt, CancellationToken cancellationToken)
    {
        var promptTokens = tokenizer.Encode(prompt, addBosEos: true);
        var ids = new List<int>(promptTokens);

        lock (rng)
        {
            if (useKvCache)
            {
                int seqLen = Math.Min(ids.Count, model.MaxSeqLen);
                var window = ids.GetRange(ids.Count - seqLen, seqLen);
                var cache = new TransformerKVCache<float>(model);
                var logitsTensor = model.ForwardPrefill(ToIntArray(window), cache);
                var logits = TensorToFloats(logitsTensor);
                int position = window.Count;

                for (int t = 0; t < maxNewTokens; t++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    int next = Sample(logits);
                    if (next == tokenizer.EosToken)
                        break;
                    ids.Add(next);

                    logits = TensorToFloats(model.ForwardCached(next, position++, cache));
                }
            }
            else
            {
                int seqLen = Math.Min(ids.Count, model.MaxSeqLen);
                for (int t = 0; t < maxNewTokens; t++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    var logits = ForwardLastRow(ids.GetRange(ids.Count - seqLen, seqLen));
                    int next = Sample(logits);
                    if (next == tokenizer.EosToken)
                        break;
                    ids.Add(next);
                }
            }
        }

        var generated = ids.Skip(promptTokens.Length).ToArray();
        return tokenizer.Decode(generated);
    }

    IEnumerable<string> GenerateStreaming(string prompt, CancellationToken cancellationToken)
    {
        var promptTokens = tokenizer.Encode(prompt, addBosEos: true);
        var ids = new List<int>(promptTokens);

        lock (rng)
        {
            if (useKvCache)
            {
                int seqLen = Math.Min(ids.Count, model.MaxSeqLen);
                var window = ids.GetRange(ids.Count - seqLen, seqLen);
                var cache = new TransformerKVCache<float>(model);
                var logits = TensorToFloats(model.ForwardPrefill(ToIntArray(window), cache));
                int position = window.Count;

                for (int t = 0; t < maxNewTokens; t++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    int next = Sample(logits);
                    if (next == tokenizer.EosToken)
                        yield break;

                    ids.Add(next);
                    yield return tokenizer.Decode([next]);

                    logits = TensorToFloats(model.ForwardCached(next, position++, cache));
                }
            }
            else
            {
                int seqLen = Math.Min(ids.Count, model.MaxSeqLen);
                for (int t = 0; t < maxNewTokens; t++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    var logits = ForwardLastRow(ids.GetRange(ids.Count - seqLen, seqLen));
                    int next = Sample(logits);
                    if (next == tokenizer.EosToken)
                        yield break;

                    ids.Add(next);
                    yield return tokenizer.Decode([next]);
                }
            }
        }
    }

    float[] ForwardLastRow(List<int> window)
    {
        var data = new float[window.Count];
        for (int i = 0; i < window.Count; i++)
            data[i] = window[i];

        var column = NivaraColumn<float>.Create(data);
        var input = new ReverseGradTensor<float>(column, requiresGrad: false);
        input.Reshape(1, window.Count);

        var logits = model.Forward(input);
        int vocab = logits.Shape[^1];

        var last = new float[vocab];
        int rowStart = (window.Count - 1) * vocab;
        for (int v = 0; v < vocab; v++)
            last[v] = float.CreateChecked(logits[rowStart + v]);
        return last;
    }

    static int[] ToIntArray(List<int> window)
    {
        var arr = new int[window.Count];
        for (int i = 0; i < window.Count; i++) arr[i] = window[i];
        return arr;
    }

    static float[] TensorToFloats(ReverseGradTensor<float> logits)
    {
        logits.Data.TryGetSpan(out var span);
        int vocab = logits.Shape[^1];
        var last = new float[vocab];
        for (int v = 0; v < vocab; v++)
            last[v] = span.Length >= vocab ? span[v] : float.CreateChecked(logits[v]);
        return last;
    }

    int Sample(float[] logits)
        => TokenSampling.Sample(logits, logits.Length, temperature, topP, rng);
}
