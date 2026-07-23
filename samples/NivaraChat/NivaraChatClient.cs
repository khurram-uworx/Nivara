using Microsoft.Extensions.AI;

namespace NivaraChat;

internal sealed class NivaraChatClient : IChatClient
{
    private readonly ITextModel _model;

    public NivaraChatClient(ITextModel model)
    {
        _model = model;
    }

    public object? GetService(Type serviceType, object? key = null) => null;

    public Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var originalInput = messages
            .Where(m => m.Role == ChatRole.User)
            .FirstOrDefault();

        var text = originalInput?.Text ?? string.Empty;
        var result = _model.Process(text);

        var responseMessage = new ChatMessage(ChatRole.Assistant, result);
        return Task.FromResult(new ChatResponse([responseMessage]));
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var originalInput = messages
            .Where(m => m.Role == ChatRole.User)
            .FirstOrDefault();

        var text = originalInput?.Text ?? string.Empty;
        var result = _model.Process(text);

        yield return new ChatResponseUpdate(ChatRole.Assistant, result);
    }

    public void Dispose() { }
}
