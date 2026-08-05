using Microsoft.Extensions.AI;
using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;

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
    readonly int maxNewTokens;
    readonly Random rng = new(1234);

    public BatchedChatClient(
        BatchedTransformer<float> model,
        TextTokenizer tokenizer,
        float temperature = 0.8f,
        int maxNewTokens = 256)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);

        model.Eval();
        this.model = model;
        this.tokenizer = tokenizer;
        this.temperature = temperature;
        this.maxNewTokens = maxNewTokens > 0 ? maxNewTokens : 1;
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
        int seqLen = Math.Min(ids.Count, model.MaxSeqLen);

        lock (rng)
        {
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

        var generated = ids.Skip(promptTokens.Length).ToArray();
        return tokenizer.Decode(generated);
    }

    IEnumerable<string> GenerateStreaming(string prompt, CancellationToken cancellationToken)
    {
        var promptTokens = tokenizer.Encode(prompt, addBosEos: true);
        var ids = new List<int>(promptTokens);
        int seqLen = Math.Min(ids.Count, model.MaxSeqLen);

        lock (rng)
        {
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

    int Sample(float[] logits)
    {
        float temperature = this.temperature;
        float max = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            max = logits[i] > max ? logits[i] : max;

        var exp = new float[logits.Length];
        float sum = 0f;
        for (int i = 0; i < logits.Length; i++)
        {
            exp[i] = MathF.Exp((logits[i] - max) / temperature);
            sum += exp[i];
        }

        if (sum <= 0f || !float.IsFinite(sum))
            return ArgMax(logits);

        double r = rng.NextDouble() * sum;
        double acc = 0;
        for (int i = 0; i < exp.Length; i++)
        {
            acc += exp[i];
            if (r <= acc)
                return i;
        }
        return ArgMax(logits);
    }

    static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > logits[best])
                best = i;
        }
        return best;
    }
}
