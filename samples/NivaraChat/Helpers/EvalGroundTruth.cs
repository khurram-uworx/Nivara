using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NivaraChat.Helpers;

/// <summary>
/// Loads the eval query TSV and can derive heading/title query pairs from a chunked corpus.
/// </summary>
public sealed record EvalQuery(string Query, IReadOnlyDictionary<string, int> Relevant, string Notes);

public static class EvalGroundTruth
{
    public static IReadOnlyList<EvalQuery> Load(string path)
    {
        var queries = new List<EvalQuery>();
        if (!File.Exists(path))
            return queries;

        foreach (var raw in File.ReadAllLines(path))
        {
            var line = raw.TrimEnd('\r');
            if (string.IsNullOrWhiteSpace(line) || line.TrimStart().StartsWith("#"))
                continue;

            var parts = line.Split('\t');
            if (parts.Length < 2)
                continue;

            var query = parts[0].Trim();
            if (string.IsNullOrWhiteSpace(query))
                continue;

            var relevant = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var token in parts[1].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                var grade = 1;
                var id = token;
                var colon = token.IndexOf(':');
                if (colon > 0 && colon < token.Length - 1 &&
                    int.TryParse(token[(colon + 1)..], out var parsed))
                {
                    id = token[..colon];
                    grade = parsed;
                }

                if (!string.IsNullOrWhiteSpace(id))
                    relevant[id] = grade;
            }

            var notes = parts.Length >= 3 ? parts[2].Trim() : string.Empty;
            queries.Add(new EvalQuery(query, relevant, notes));
        }

        return queries;
    }

    public static IReadOnlyList<EvalQuery> DeriveFromChunks(IReadOnlyList<MultiViewChunk> chunks)
    {
        var queries = new List<EvalQuery>();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        // Title-as-query: document title -> all chunks from that source.
        foreach (var group in chunks.GroupBy(c => c.SourcePath, StringComparer.Ordinal))
        {
            var title = group.FirstOrDefault()?.Title;
            if (string.IsNullOrWhiteSpace(title))
                continue;

            var relevant = group.ToDictionary(c => c.ChunkId, _ => 1, StringComparer.Ordinal);
            var key = title + "|" + string.Join(",", relevant.Keys.OrderBy(k => k, StringComparer.Ordinal));
            if (seen.Add(key))
                queries.Add(new EvalQuery(title!, relevant, "auto: title"));
        }

        // Heading-as-query: section tail -> all chunks sharing that section.
        foreach (var group in chunks.GroupBy(c => c.Section, StringComparer.Ordinal))
        {
            if (string.IsNullOrWhiteSpace(group.Key))
                continue;

            var section = group.Key;
            var heading = section.Contains(" > ", StringComparison.Ordinal)
                ? section[(section.LastIndexOf(" > ", StringComparison.Ordinal) + 3)..]
                : section;

            if (string.IsNullOrWhiteSpace(heading))
                continue;

            var relevant = group.ToDictionary(c => c.ChunkId, _ => 1, StringComparer.Ordinal);
            var key = heading + "|" + string.Join(",", relevant.Keys.OrderBy(k => k, StringComparer.Ordinal));
            if (seen.Add(key))
                queries.Add(new EvalQuery(heading, relevant, "auto: heading"));
        }

        return queries;
    }

    public static void Save(string path, IReadOnlyList<EvalQuery> queries)
    {
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        var lines = queries.Select(q =>
            q.Query + "\t" + string.Join(",", q.Relevant.Select(r => r.Value == 1 ? r.Key : $"{r.Key}:{r.Value}")) + "\t" + q.Notes);

        File.WriteAllLines(path, lines);
    }
}
