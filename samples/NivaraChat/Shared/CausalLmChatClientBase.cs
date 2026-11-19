using Microsoft.Extensions.AI;

namespace NivaraChat.Shared;

/// <summary>
/// Common plumbing for the sample chat clients: state, IChatClient implementation,
/// locking, and delegation to <see cref="CausalLmGenerator"/>.
/// </summary>
internal abstract class CausalLmChatClientBase : IChatClient
{
    protected readonly float temperature;
    protected readonly float topP;
    protected readonly int maxNewTokens;
    protected readonly bool useKvCache;
    protected readonly Random rng;

    protected CausalLmChatClientBase(
        float temperature,
        float topP,
        int maxNewTokens,
        bool useKvCache,
        Random rng)
    {
        this.temperature = temperature;
        this.topP = topP;
        this.maxNewTokens = maxNewTokens > 0 ? maxNewTokens : 1;
        this.useKvCache = useKvCache;
        this.rng = rng;
    }

    public abstract ChatClientMetadata? Metadata { get; }

    public object? GetService(Type serviceType, object? key = null) => null;

    public virtual void Dispose() { }

    protected abstract string RenderPrompt(IEnumerable<ChatMessage> messages);
    protected abstract IReadOnlyList<int> Encode(string prompt);
    protected abstract string DecodeToken(int tokenId);
    protected abstract string DecodeAll(IReadOnlyList<int> ids);
    protected abstract CausalLmOps BuildOps();

    protected virtual bool WindowPrompt => false;
    protected virtual bool WindowNonCached => false;
    protected virtual int OverflowKeep => 0;
    protected virtual bool BreakOnContextLimit => false;

    protected virtual IReadOnlyList<int> PostProcess(IReadOnlyList<int> generated) => generated;

    protected virtual IEnumerable<int> GenerateIds(IEnumerable<ChatMessage> messages, CancellationToken cancellationToken)
    {
        var prompt = RenderPrompt(messages);
        var promptIds = Encode(prompt);
        var ops = BuildOps();
        return CausalLmGenerator.Generate(
            ops,
            promptIds,
            maxNewTokens,
            temperature,
            topP,
            rng,
            useKvCache,
            cancellationToken,
            WindowPrompt,
            WindowNonCached,
            OverflowKeep,
            BreakOnContextLimit);
    }

    public virtual Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var generated = new List<int>();
        lock (rng)
        {
            foreach (var id in GenerateIds(messages, cancellationToken))
                generated.Add(id);
        }
        var final = PostProcess(generated);
        var text = DecodeAll(final);
        return Task.FromResult(new ChatResponse([new ChatMessage(ChatRole.Assistant, text)]));
    }

    public virtual async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        lock (rng)
        {
            foreach (var id in GenerateIds(messages, cancellationToken))
            {
                yield return new ChatResponseUpdate(ChatRole.Assistant, DecodeToken(id));
            }
        }
    }
}
