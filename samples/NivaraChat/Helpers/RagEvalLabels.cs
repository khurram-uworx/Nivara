namespace NivaraChat.Helpers;

/// <summary>
/// Persistent cache for LLM judge grades.
/// </summary>
public sealed record LabelResult(int Grade, int Runs, double StdDev);

public static class RagEvalLabels
{
    public static Dictionary<string, Dictionary<string, LabelResult>> Load(string path)
    {
        var result = new Dictionary<string, Dictionary<string, LabelResult>>(StringComparer.Ordinal);
        if (!File.Exists(path))
            return result;

        foreach (var raw in File.ReadAllLines(path))
        {
            var line = raw.TrimEnd('\r');
            if (string.IsNullOrWhiteSpace(line) || line.TrimStart().StartsWith("#"))
                continue;

            var parts = line.Split('\t');
            if (parts.Length < 4)
                continue;

            var query = parts[0];
            var chunkId = parts[1];
            if (!int.TryParse(parts[2], out var grade))
                continue;
            if (!int.TryParse(parts[3], out var runs))
                runs = 1;

            var stddev = 0d;
            if (parts.Length >= 5)
                double.TryParse(parts[4], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out stddev);

            if (!result.TryGetValue(query, out var map))
            {
                map = new Dictionary<string, LabelResult>(StringComparer.Ordinal);
                result[query] = map;
            }

            map[chunkId] = new LabelResult(grade, runs, stddev);
        }

        return result;
    }

    public static void Save(string path, IReadOnlyDictionary<string, Dictionary<string, LabelResult>> labels)
    {
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        var lines = new List<string>();
        foreach (var query in labels.Keys.OrderBy(q => q, StringComparer.Ordinal))
        {
            foreach (var chunk in labels[query].Keys.OrderBy(c => c, StringComparer.Ordinal))
            {
                var r = labels[query][chunk];
                lines.Add(string.Join("\t",
                    query,
                    chunk,
                    r.Grade,
                    r.Runs,
                    r.StdDev.ToString("0.0000", System.Globalization.CultureInfo.InvariantCulture)));
            }
        }

        File.WriteAllLines(path, lines);
    }
}
