using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using OllamaSharp;
using System.Linq;

namespace NivaraChat;

internal sealed class FactualExecutor : Executor<string, string>
{
    private readonly CommunityToolkit.VectorData.InMemory.InMemoryVectorStore _vectorStore;
    private readonly OllamaApiClient _chatClient;
    private readonly int _topK;

    public FactualExecutor(CommunityToolkit.VectorData.InMemory.InMemoryVectorStore vectorStore, OllamaApiClient chatClient, int topK = 3)
        : base("Factual RAG")
    {
        _vectorStore = vectorStore;
        _chatClient = chatClient;
        _topK = topK;
    }

    public override async ValueTask<string> HandleAsync(string input, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        try
        {
            var collection = _vectorStore.GetCollection<string, DocumentChunk>("nivaradocs");
            var searchResults = new List<(DocumentChunk Record, double? Score)>();
            await foreach (var result in collection.SearchAsync(input, _topK))
            {
                searchResults.Add((result.Record, result.Score));
            }

            string prompt;
            if (searchResults.Count > 0)
            {
                var contextText = string.Join("\n\n", searchResults.Select(r => r.Record.Text));
                prompt = $"Answer the following question based on the provided context.\n\nContext:\n{contextText}\n\nQuestion: {input}\n\nAnswer:";
            }
            else
            {
                prompt = $"Answer the following question.\n\nQuestion: {input}\n\nAnswer:";
            }

            var response = await _chatClient.GetResponseAsync(prompt);
            return response.ToString();
        }
        catch (Exception ex)
        {
            return $"Error in factual retrieval: {ex.Message}";
        }
    }
}