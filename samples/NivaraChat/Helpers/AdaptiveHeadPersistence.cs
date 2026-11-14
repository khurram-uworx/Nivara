using Nivara.AI;
using Nivara.AutoDiff.Serialization;

namespace NivaraChat.Helpers;

public static class AdaptiveHeadPersistence
{
    public const string FileName = "laya_head.json";

    public static string DefaultPath(string repoRoot)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoRoot);
        return Path.Combine(repoRoot, "samples", "data", "nivarachat", "adaptive", FileName);
    }

    public static bool LoadIfPresent(LayaDecisionHead<float, float> head, string path)
    {
        ArgumentNullException.ThrowIfNull(head);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            return false;

        var state = ModelSerializer.JsonToStateDict<float>(File.ReadAllText(path));
        var expectedNames = head.Parameters().Keys.ToHashSet(StringComparer.Ordinal);
        if (state.Count != expectedNames.Count || expectedNames.Except(state.Keys).Any())
            throw new InvalidOperationException("Adaptive head file does not contain the complete Laya head state.");

        ModelSerializer.Load(head, path);
        return true;
    }

    public static void Save(LayaDecisionHead<float, float> head, string path)
    {
        ArgumentNullException.ThrowIfNull(head);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var fullPath = Path.GetFullPath(path);
        var directory = Path.GetDirectoryName(fullPath)
            ?? throw new ArgumentException("Adaptive head path must include a parent directory.", nameof(path));
        Directory.CreateDirectory(directory);

        var temporaryPath = Path.Combine(
            directory, $".{Path.GetFileName(fullPath)}.{Guid.NewGuid():N}.tmp");
        try
        {
            ModelSerializer.Save(head, temporaryPath);
            File.Move(temporaryPath, fullPath, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporaryPath))
                File.Delete(temporaryPath);
        }
    }
}
