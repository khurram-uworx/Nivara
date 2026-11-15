using Microsoft.Extensions.AI;
using NivaraChat.Modes;
using OpenAI;
using System.ClientModel;

namespace NivaraChat.Shared;

/// <summary>
/// Builds the <see cref="IChatClient"/> used by the plain-chat modes from a <see cref="ModeContext"/>,
/// selecting Ollama or an OpenAI-compatible endpoint based on <see cref="ModeContext.LlmProvider"/>.
/// </summary>
static class LlmClientFactory
{
    public static IChatClient? Create(ModeContext ctx)
    {
        ArgumentNullException.ThrowIfNull(ctx);

        OpenAIClient? openAIClient = null;

        if (ctx.OllamaUrl is { Length: > 0 })
        {
            if (ctx.ApiKeyEnvVar is null)
                throw new InvalidOperationException("API key environment variable name must be provided for OpenAI-compatible endpoints.");

            var key = Environment.GetEnvironmentVariable(ctx.ApiKeyEnvVar);
            if (key is null)
                throw new InvalidOperationException($"API key environment variable '{ctx.ApiKeyEnvVar}' is not set.");

            var options = new OpenAIClientOptions { Endpoint = new Uri(ctx.OllamaUrl) };
            openAIClient = new OpenAIClient(new ApiKeyCredential(key), options);
        }
        else
        {
            var options = new OpenAIClientOptions { Endpoint = new Uri("http://localhost:11434/v1") };
            openAIClient = new OpenAIClient(new ApiKeyCredential("x"), options);
        }

        if (openAIClient != null && ctx.ModelName is { Length: > 0 })
            return openAIClient.GetChatClient(ctx.ModelName).AsIChatClient();

        return null;
    }
}
