using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;
using OllamaSharp;

namespace NivaraChat.Helpers;

/// <summary>
/// LLM-as-judge grading using an Ollama model.
/// </summary>
public static class RagEvalJudge
{
    const int BatchSize = 8;

    public static async Task<IReadOnlyDictionary<string, LabelResult>> GradeChunksAsync(
        OllamaApiClient client,
        string query,
        IReadOnlyList<(string ChunkId, string Text)> passages,
        int runs,
        double temperature,
        CancellationToken ct = default)
    {
        runs = Math.Max(1, runs);
        var results = passages.ToDictionary(p => p.ChunkId, _ => new List<int>(), StringComparer.Ordinal);

        for (var run = 0; run < runs; run++)
        {
            for (var start = 0; start < passages.Count; start += BatchSize)
            {
                var batch = passages.Skip(start).Take(BatchSize).ToList();
                var grades = await GradeBatchAsync(client, query, batch, temperature, ct);

                foreach (var (chunkId, grade) in grades)
                {
                    if (results.TryGetValue(chunkId, out var list))
                        list.Add(Math.Clamp(grade, 0, 2));
                }
            }
        }

        var output = new Dictionary<string, LabelResult>(StringComparer.Ordinal);
        foreach (var (chunkId, grades) in results)
        {
            if (grades.Count == 0)
            {
                output[chunkId] = new LabelResult(0, 0, 0d);
                continue;
            }

            var majority = grades.GroupBy(g => g).OrderByDescending(g => g.Count()).ThenBy(g => g.Key).First().Key;
            var mean = grades.Average();
            var variance = grades.Sum(g => (g - mean) * (g - mean)) / grades.Count;
            var stddev = Math.Sqrt(variance);

            output[chunkId] = new LabelResult(majority, grades.Count, stddev);
        }

        return output;
    }

    static async Task<IReadOnlyDictionary<string, int>> GradeBatchAsync(
        OllamaApiClient client,
        string query,
        IReadOnlyList<(string ChunkId, string Text)> batch,
        double temperature,
        CancellationToken ct)
    {
        var sb = new StringBuilder();
        sb.AppendLine("You are a relevance judge for retrieval.");
        sb.AppendLine("Query:");
        sb.AppendLine(query);
        sb.AppendLine();
        sb.AppendLine("Grade each passage for relevance to the query: 0 = irrelevant, 1 = partially relevant, 2 = highly relevant.");
        sb.AppendLine("Return ONLY a JSON array of integers, one per passage, in the same order.");
        sb.AppendLine();
        for (var i = 0; i < batch.Count; i++)
        {
            sb.AppendLine($"Passage {i + 1}:");
            sb.AppendLine(batch[i].Text);
            sb.AppendLine();
        }

        var options = new ChatOptions
        {
            Temperature = (float)temperature
        };

        ChatResponse response;
        try
        {
            response = await client.GetResponseAsync(sb.ToString(), options, ct);
        }
        catch
        {
            return batch.ToDictionary(b => b.ChunkId, _ => 0, StringComparer.Ordinal);
        }

        var text = response.Text ?? string.Empty;
        var grades = ParseGrades(text, batch.Count);

        var output = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < batch.Count; i++)
            output[batch[i].ChunkId] = i < grades.Count ? grades[i] : 0;

        return output;
    }

    static List<int> ParseGrades(string text, int expected)
    {
        var result = new List<int>();
        try
        {
            var start = text.IndexOf('[');
            var end = text.LastIndexOf(']');
            if (start >= 0 && end > start)
                text = text[start..(end + 1)];

            using var doc = JsonDocument.Parse(text);
            foreach (var element in doc.RootElement.EnumerateArray())
            {
                if (element.ValueKind == JsonValueKind.Number && element.TryGetInt32(out var value))
                    result.Add(value);
                else if (int.TryParse(element.GetString(), out var parsed))
                    result.Add(parsed);
            }
        }
        catch
        {
            // Fall through; missing grades become 0.
        }

        while (result.Count < expected)
            result.Add(0);

        if (result.Count > expected)
            result = result.Take(expected).ToList();

        return result;
    }
}
