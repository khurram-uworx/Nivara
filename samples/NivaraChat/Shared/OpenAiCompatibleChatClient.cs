using Microsoft.Extensions.AI;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace NivaraChat.Shared;

/// <summary>
/// A minimal <see cref="IChatClient"/> over any OpenAI-compatible chat-completions endpoint
/// (OpenAI, Azure OpenAI v1, Ollama's <c>/v1</c> shim, vLLM, LM Studio, ...). Non-streaming
/// only; streaming is implemented as SSE parsing for text deltas.
/// </summary>
internal sealed class OpenAiCompatibleChatClient : IChatClient
{
    readonly Uri endpoint;
    readonly string model;
    readonly string apiKey;
    readonly string authMode;
    readonly HttpClient http;

    public OpenAiCompatibleChatClient(
        string baseUrl,
        string model,
        string? apiKey = null,
        string authMode = "bearer",
        HttpMessageHandler? handler = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(baseUrl);
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        if (authMode is not ("bearer" or "api-key"))
            throw new ArgumentException("authMode must be 'bearer' or 'api-key'.", nameof(authMode));

        this.endpoint = BuildEndpoint(baseUrl);
        this.model = model;
        this.apiKey = apiKey ?? string.Empty;
        this.authMode = authMode;
        this.http = handler is null ? new HttpClient() : new HttpClient(handler, disposeHandler: true);
    }

    public ChatClientMetadata Metadata { get; } = new("Nivara", null, "OpenAI-compatible");

    public async Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> chatMessages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(chatMessages);
        using var request = BuildRequest(chatMessages, options, stream: false);
        using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        var body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
            throw new HttpRequestException(
                $"OpenAI-compatible chat request failed: {(int)response.StatusCode} {response.ReasonPhrase}. Body: {Truncate(body)}",
                null,
                response.StatusCode);

        using var doc = JsonDocument.Parse(body);
        var root = doc.RootElement;
        var content = root.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString() ?? string.Empty;
        return new ChatResponse(new ChatMessage(ChatRole.Assistant, content));
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> chatMessages,
        ChatOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(chatMessages);
        using var request = BuildRequest(chatMessages, options, stream: true);
        using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            var body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            throw new HttpRequestException(
                $"OpenAI-compatible chat request failed: {(int)response.StatusCode} {response.ReasonPhrase}. Body: {Truncate(body)}",
                null,
                response.StatusCode);
        }

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        using var reader = new StreamReader(stream, Encoding.UTF8);
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var line = await reader.ReadLineAsync(cancellationToken).ConfigureAwait(false);
            if (line is null) break;
            if (string.IsNullOrWhiteSpace(line)) continue;
            if (!line.StartsWith("data:", StringComparison.Ordinal)) continue;
            var payload = line["data:".Length..].Trim();
            if (payload == "[DONE]") yield break;
            if (string.IsNullOrEmpty(payload)) continue;
            string? delta = null;
            try
            {
                using var doc = JsonDocument.Parse(payload);
                var choices = doc.RootElement.GetProperty("choices");
                if (choices.GetArrayLength() == 0) continue;
                var d = choices[0].GetProperty("delta");
                if (d.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String)
                    delta = c.GetString();
            }
            catch (JsonException)
            {
                continue;
            }
            if (!string.IsNullOrEmpty(delta))
                yield return new ChatResponseUpdate(ChatRole.Assistant, delta);
        }
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose() => http.Dispose();

    HttpRequestMessage BuildRequest(IEnumerable<ChatMessage> chatMessages, ChatOptions? options, bool stream)
    {
        var messages = new List<object>();
        foreach (var m in chatMessages)
            messages.Add(new { role = RoleName(m.Role), content = m.Text ?? string.Empty });

        var effectiveModel = options?.ModelId ?? model;
        var payload = new Dictionary<string, object>
        {
            ["model"] = effectiveModel,
            ["messages"] = messages,
        };
        if (stream) payload["stream"] = true;
        if (options?.Temperature is float t) payload["temperature"] = t;

        var request = new HttpRequestMessage(HttpMethod.Post, endpoint)
        {
            Content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json"),
        };
        if (!string.IsNullOrEmpty(apiKey))
        {
            if (authMode == "api-key")
                request.Headers.Add("api-key", apiKey);
            else
                request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
        }
        return request;
    }

    static string RoleName(ChatRole role)
    {
        if (role == ChatRole.System) return "system";
        if (role == ChatRole.User) return "user";
        if (role == ChatRole.Assistant) return "assistant";
        if (role == ChatRole.Tool) return "tool";
        return role.Value;
    }

    static Uri BuildEndpoint(string baseUrl)
    {
        var trimmed = baseUrl.TrimEnd('/');
        if (trimmed.EndsWith("/chat/completions", StringComparison.OrdinalIgnoreCase))
            return new Uri(trimmed);
        return new Uri(trimmed + "/chat/completions");
    }

    static string Truncate(string s) => s.Length <= 500 ? s : s[..500] + "...";
}
