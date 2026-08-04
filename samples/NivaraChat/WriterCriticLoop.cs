using Microsoft.Extensions.AI;
using OllamaSharp;
using System.Text.Json;

namespace NivaraChat;

internal sealed class WriterCriticLoop
{
    private readonly OllamaApiClient _chatClient;
    private readonly CriticExecutor _critic;
    private const int MaxIterations = 3;

    public WriterCriticLoop(OllamaApiClient chatClient, CriticExecutor critic)
    {
        _chatClient = chatClient;
        _critic = critic;
    }

    public async ValueTask<string> RunAsync(string query, CancellationToken cancellationToken = default)
    {
        string feedback = "";
        string lastResponseText = "";

        for (int i = 0; i < MaxIterations; i++)
        {
            var prompt = string.IsNullOrEmpty(feedback)
                ? $"Answer this question clearly and concisely: {query}"
                : $"Answer this question. Previous attempt scored poorly. Feedback: {feedback}\n\nQuestion: {query}";

            Console.WriteLine($"  [Writer] Attempt {i + 1}/{MaxIterations}...");
            var chatResponse = await _chatClient.GetResponseAsync(prompt);
            var responseText = chatResponse.ToString();
            lastResponseText = responseText;

            var critiqueInput = $"{query} || {responseText}";
            var critiqueJson = await _critic.ScoreAsync(critiqueInput, cancellationToken);
            var critique = JsonSerializer.Deserialize<CritiqueResult>(critiqueJson);

            Console.WriteLine($"  [Critic] Score: {critique!.Score:F2} ({critique.Verdict})");

            if (critique.Acceptable)
                return $"Attempt {i + 1} — Score: {critique.Score:F2} (PASS)\n\n{responseText}";

            feedback = $"Previous attempt scored {critique.Score:F2} ({critique.Verdict}). "
                     + "Improve on: clarity, accuracy, relevance to the query.";
        }

        return $"Attempt {MaxIterations} — Score: below threshold (max iterations reached)\n\n{lastResponseText}";
    }

    private sealed class CritiqueResult
    {
        public float Score { get; set; }
        public string Verdict { get; set; } = "";
        public bool Acceptable { get; set; }
    }
}
