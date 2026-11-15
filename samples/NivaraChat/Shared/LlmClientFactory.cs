using System;
using Microsoft.Extensions.AI;
using NivaraChat.Modes;
using OllamaSharp;

namespace NivaraChat.Shared;

/// <summary>
/// Builds the <see cref="IChatClient"/> used by the plain-chat modes from a <see cref="ModeContext"/>,
/// selecting Ollama or an OpenAI-compatible endpoint based on <see cref="ModeContext.LlmProvider"/>.
/// </summary>
internal static class LlmClientFactory
{
    public const string ApiKeyEnvVar = "NIVARA_CHAT_API_KEY";

    public static IChatClient? Create(ModeContext ctx)
    {
        ArgumentNullException.ThrowIfNull(ctx);
        if (string.Equals(ctx.LlmProvider, "openai", StringComparison.OrdinalIgnoreCase))
        {
            if (string.IsNullOrWhiteSpace(ctx.OpenAiUrl))
                throw new InvalidOperationException("--llm openai requires --openai-url <base>.");
            var model = string.IsNullOrWhiteSpace(ctx.OpenAiModel) ? ctx.ModelName : ctx.OpenAiModel!;
            var key = Environment.GetEnvironmentVariable(ApiKeyEnvVar);
            return new OpenAiCompatibleChatClient(ctx.OpenAiUrl!, model, key, ctx.OpenAiAuth);
        }

        if (!ctx.UseOllama) return null;
        return new OllamaApiClient(new Uri(ctx.OllamaUrl), ctx.ModelName);
    }

    public static void RejectForUnsupportedMode(ModeContext ctx, string modeName)
    {
        if (string.Equals(ctx.LlmProvider, "openai", StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException(
                $"--llm openai is not supported for {modeName} yet (function calling / streaming on the OpenAI transport is a follow-up). Use --llm ollama.");
    }
}