using Microsoft.Extensions.AI;
using NivaraChat.Helpers;
using OllamaSharp;

namespace NivaraChat.Modes;

/// <summary>
/// --rag-adaptive: retrieval-first RAG with an additional Laya decision layer that filters the top
/// result(s) by chunk relevance before answering.
/// </summary>
public static class RagAdaptiveMode
{
    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat — Adaptive RAG (#981) ===\n");

        using var index = await RagIndex.BuildAsync(ctx);
        if (index is null)
            return;

        if (!index.HasVisibleChunks)
        {
            Console.WriteLine($"No chunks visible under ACL group '{ctx.AclGroup}'.");
            return;
        }

        OllamaApiClient? chatClient = null;
        if (ctx.UseOllama)
        {
            Console.WriteLine($"Connecting to Ollama at {ctx.OllamaUrl} (model: {ctx.ModelName})...");
            chatClient = new OllamaApiClient(new Uri(ctx.OllamaUrl), ctx.ModelName);
            Console.WriteLine("Ollama connected.\n");
        }
        else
        {
            Console.WriteLine("No --ollama: the mode will report the selected chunk and the decision verdict only.\n");
        }

        Console.WriteLine($"Threshold: {ctx.ConfidenceThreshold:F2} | Top-K: {ctx.TopK} chunks per query.\n");

        async Task RunQuery(string query)
        {
            var (results, retrievalMs) = await index.RetrieveAsync(query);
            RagIndex.PrintRetrieved(results, retrievalMs);

            var chosen = RagDecisionRouting.SelectBestDecision(query, results, ctx.ConfidenceThreshold);
            Console.WriteLine($"\n  Adaptive decision: {chosen.Decision} (confidence {chosen.AnswerConfidence:F4}, retrieval {chosen.RetrievalScore:F4})");
            Console.WriteLine($"    {chosen.Source} :: {chosen.Title}");
            Console.WriteLine($"    {chosen.Reason}");
            Console.WriteLine($"    {chosen.ChunkId}");

            var context = chosen.Accepted ? chosen.Content : string.Join("\n\n", results.Select(r => r.Content));
            if (chatClient is not null && !string.IsNullOrWhiteSpace(context))
            {
                var prompt = $"Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:";
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var response = await chatClient.GetResponseAsync(prompt, new ChatOptions { ModelId = ctx.ModelName });
                var llmMs = sw.ElapsedMilliseconds;
                Console.WriteLine($"\n  LLM response ({llmMs} ms):");
                Console.WriteLine($"  {response.Text}\n");
            }
        }

        if (ctx.SingleShotText is not null)
        {
            await RunQuery(ctx.SingleShotText);
            return;
        }

        while (true)
        {
            Console.Write("> ");
            var query = Console.ReadLine()?.Trim();
            if (string.IsNullOrEmpty(query) || query == "quit")
                break;
            await RunQuery(query);
        }
    }
}
