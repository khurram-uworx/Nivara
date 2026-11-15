using Microsoft.Extensions.AI;
using NivaraChat.Helpers;
using NivaraChat.Shared;
using System.Diagnostics;

namespace NivaraChat.Modes;

/// <summary>
/// --rag-adaptive: retrieval-first RAG with a local Laya decision layer that filters the top
/// result(s) by chunk relevance before answering. Uses the multi-view corpus identity so chunk
/// ids line up with --rag-learn and --rag-eval.
/// </summary>
public static class RagAdaptiveMode
{
    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat — Adaptive RAG (#981) ===\n");

        using var index = await MultiViewIndex.BuildAsync(ctx);
        if (index is null)
            return;

        if (!index.HasVisibleChunks)
        {
            Console.WriteLine($"No chunks visible under ACL group '{ctx.AclGroup}'.");
            return;
        }

        var modelDir = Path.Combine(ModeHelpers.GetRepoRoot(), "samples", "data", "laya");
        LayaRetrievalDecisionHead.LayaDecisionEngine? engine = null;
        if (Directory.Exists(modelDir) && File.Exists(Path.Combine(modelDir, "model.safetensors")))
            engine = new LayaRetrievalDecisionHead.LayaDecisionEngine(modelDir);

        IChatClient? chatClient = LlmClientFactory.Create(ctx);
        if (chatClient is not null)
            Console.WriteLine($"LLM connected.\n");

        Console.WriteLine($"Threshold: {ctx.ConfidenceThreshold:F2} | Top-K: {ctx.TopK} chunks per query.\n");

        async Task RunQuery(string query)
        {
            var (hits, retrievalMs) = index.Retrieve(query, FusionWeights.Default, ctx.TopK);
            Console.WriteLine($"  Retrieved {hits.Count} chunks in {retrievalMs} ms:");
            foreach (var hit in hits)
                Console.WriteLine($"    [{hit.Fused?.ToString("0.000") ?? "n/a"}] {hit.Chunk.ChunkId} :: {hit.Chunk.Title}");

            var chunks = hits.Select(hit => new RetrievedChunk(
                hit.Chunk.ChunkId, hit.Chunk.Source, hit.Chunk.Title, hit.Chunk.Content,
                hit.Fused ?? 0d)).ToList();

            IReadOnlyList<LayaChunkDecision> decisions;
            if (engine is null)
                decisions = LayaRetrievalDecisionHead.Evaluate(query, chunks, ctx.ConfidenceThreshold);
            else
                decisions = LayaRetrievalDecisionHead.Evaluate(engine, query, chunks, ctx.ConfidenceThreshold);

            var chosen = RagDecisionRouting.SelectBestDecisionFrom(decisions);
            Console.WriteLine($"\n  Adaptive decision: {chosen.Decision} (confidence {chosen.AnswerConfidence:F4}, retrieval {chosen.RetrievalScore:F4})");
            Console.WriteLine($"    {chosen.Source} :: {chosen.Title}");
            Console.WriteLine($"    {chosen.Reason}");
            Console.WriteLine($"    {chosen.ChunkId}");

            var context = chosen.Accepted ? chosen.Content : string.Join("\n\n", chunks.Select(c => c.Content));
            if (chatClient is not null && !string.IsNullOrWhiteSpace(context))
            {
                var prompt = $"Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:";
                var sw = Stopwatch.StartNew();
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

        engine?.Dispose();
    }
}
