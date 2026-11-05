using Microsoft.Extensions.AI;
using NivaraChat.Helpers;
using OllamaSharp;

namespace NivaraChat.Modes;

/// <summary>
/// --rag: RAG pipeline. Chunks docs, embeds via MiniLM (CPU or Silk), stores in a
/// NivaraFrame (convention from #867) or legacy InMemoryVectorStore, retrieves Top-K
/// via RowCosineSimilarity, and optionally grounds an Ollama answer.
/// </summary>
public static class RagMode
{
    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat — RAG Pipeline ===\n");

        using var index = await RagIndex.BuildAsync(ctx);
        if (index is null) return;

        OllamaApiClient? chatClient = null;
        if (ctx.UseOllama)
        {
            Console.WriteLine($"Connecting to Ollama at {ctx.OllamaUrl} (model: {ctx.ModelName})...");
            chatClient = new OllamaApiClient(new Uri(ctx.OllamaUrl), ctx.ModelName);
            Console.WriteLine("Ollama connected.\n");
        }
        else
        {
            Console.WriteLine("No --ollama: will list retrieved chunks only.\n");
        }

        Console.WriteLine($"Top-K: {ctx.TopK} chunks per query. Type a question (or 'quit' to exit):\n");

        async Task RunQuery(string query)
        {
            if (!index.HasVisibleChunks)
            {
                Console.WriteLine($"No chunks visible under ACL group '{ctx.AclGroup}'.");
                return;
            }

            var (results, retrievalMs) = await index.RetrieveAsync(query);
            RagIndex.PrintRetrieved(results, retrievalMs);

            if (ctx.ShowRoundTrip && index.UseFrame)
                index.PrintRoundTrip();

            if (chatClient != null && results.Count > 0)
            {
                var context = string.Join("\n\n", results.Select(r => r.Content));
                var prompt = $"Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:";

                var sw = System.Diagnostics.Stopwatch.StartNew();
                var response = await chatClient.GetResponseAsync(prompt, new ChatOptions { ModelId = ctx.ModelName });
                var llmMs = sw.ElapsedMilliseconds;

                Console.WriteLine($"\n  LLM response ({llmMs} ms):");
                Console.WriteLine($"  {response.Text}");
                Console.WriteLine();
            }
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
}
