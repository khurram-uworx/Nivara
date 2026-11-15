using Microsoft.Extensions.AI;
using NivaraChat.Modes;
using OpenAI;
using System.ClientModel;

namespace NivaraChat.Shared;

/// <summary>
/// Builds the <see cref="IChatClient"/> used by the plain-chat modes from a <see cref="ModeContext"/>,
/// using the OpenAI chat-completions protocol for both Ollama and OpenAI-compatible endpoints.
/// </summary>
static class LlmClientFactory
{
    public static IChatClient? Create(ModeContext ctx)
    {
        ArgumentNullException.ThrowIfNull(ctx);
        if (!ctx.UseOllama) return null;

        var baseUrl = string.IsNullOrWhiteSpace(ctx.OllamaUrl) ? "http://localhost:11434" : ctx.OllamaUrl!.TrimEnd('/');
        if (!baseUrl.EndsWith("/v1", StringComparison.OrdinalIgnoreCase))
            baseUrl += "/v1";

        string apiKey = "ollama";
        if (ctx.ApiKeyEnvVar is { Length: > 0 })
        {
            var envKey = Environment.GetEnvironmentVariable(ctx.ApiKeyEnvVar);
            if (string.IsNullOrWhiteSpace(envKey))
                throw new InvalidOperationException($"API key environment variable '{ctx.ApiKeyEnvVar}' is not set.");
            apiKey = envKey;
        }

        var options = new OpenAIClientOptions { Endpoint = new Uri(baseUrl) };
        var openAIClient = new OpenAIClient(new ApiKeyCredential(apiKey), options);

        if (openAIClient is not null && ctx.ModelName is { Length: > 0 })
            return openAIClient.GetChatClient(ctx.ModelName).AsIChatClient();

        return null;
    }
}
