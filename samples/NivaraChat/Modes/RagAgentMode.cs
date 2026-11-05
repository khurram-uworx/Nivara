using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using NivaraChat.Helpers;
using OllamaSharp;

namespace NivaraChat.Modes;

public static class RagAgentMode
{
    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat — RAG Agent (TextSearchProvider) ===\n");

        using var index = await RagIndex.BuildAsync(ctx);
        if (index is null) return;

        Console.WriteLine($"Top-K: {ctx.TopK} chunks per query. TextSearchProvider auto-injects context before each LLM call.");
        Console.WriteLine("Type a question (or 'quit' to exit):\n");

        if (ctx.ShowRoundTrip && index.UseFrame)
            index.PrintRoundTrip();

        if (ctx.UseOllama)
        {
            await RunOllamaPath(ctx, index);
            return;
        }

        // No Ollama: list retrieved chunks only.
        Console.WriteLine("No --ollama: will list retrieved chunks only.\n");

        async Task RunQuery(string query)
        {
            if (!index.HasVisibleChunks)
            {
                Console.WriteLine($"No chunks visible under ACL group '{ctx.AclGroup}'.");
                return;
            }
            var (results, retrievalMs) = await index.RetrieveAsync(query);
            RagIndex.PrintRetrieved(results, retrievalMs);
            Console.WriteLine();
        }

        if (ctx.SingleShotText != null)
        {
            await RunQuery(ctx.SingleShotText);
        }
        else
        {
            while (true)
            {
                Console.Write("> ");
                var query = Console.ReadLine()?.Trim();
                if (string.IsNullOrEmpty(query) || query == "quit") break;
                await RunQuery(query);
            }
        }
    }

    static async Task RunOllamaPath(ModeContext ctx, RagIndex index)
    {
        Console.WriteLine($"Connecting to Ollama at {ctx.OllamaUrl} (model: {ctx.ModelName})...");
        var chatClient = new OllamaApiClient(new Uri(ctx.OllamaUrl), ctx.ModelName);
        Console.WriteLine("Ollama connected.\n");

        async Task<IEnumerable<TextSearchProvider.TextSearchResult>> SearchAsync(string query, CancellationToken ct)
        {
            var (results, retrievalMs) = await index.RetrieveAsync(query, ct);
            RagIndex.PrintRetrieved(results, retrievalMs);
            return results.Select(r => new TextSearchProvider.TextSearchResult { Text = r.Content, SourceName = r.Source });
        }

        var textSearch = new TextSearchProvider(SearchAsync, new TextSearchProviderOptions
        {
            SearchTime = TextSearchProviderOptions.TextSearchBehavior.BeforeAIInvoke,
            RecentMessageMemoryLimit = 2
        });

        var baseAgent = new ChatClientAgent(chatClient, name: "NivaraRagAgent",
            instructions: "You are a helpful assistant that answers questions about the Nivara project using the provided context. Always base your answers on the retrieved context.");

        var agent = new AIAgentBuilder(baseAgent).UseAIContextProviders(textSearch).Build();

        async Task RunQuery(string query)
        {
            var run = await InProcessExecution.RunAsync(new WorkflowBuilder(agent).WithOutputFrom(agent).Build(), query);
            Console.WriteLine("\n--- RAG Agent Response ---");
            foreach (var evt in run.NewEvents)
            {
                switch (evt)
                {
                    case AgentResponseUpdateEvent updateEvt:
                        if (updateEvt.Update?.Text is string text && !string.IsNullOrEmpty(text)) Console.Write(text);
                        break;
                    case AgentResponseEvent agentEvt:
                        if (agentEvt.Data is string data && !string.IsNullOrEmpty(data)) Console.WriteLine(data);
                        break;
                }
            }
            Console.WriteLine("\n");
        }

        if (ctx.SingleShotText != null)
            await RunQuery(ctx.SingleShotText);
        else
        {
            while (true)
            {
                Console.Write("> ");
                var query = Console.ReadLine()?.Trim();
                if (string.IsNullOrEmpty(query) || query == "quit") break;
                await RunQuery(query);
            }
        }
    }
}
