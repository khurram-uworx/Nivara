using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NivaraFineTuning;

/// <summary>
/// Structured, machine-readable summary emitted by the ft-lora / ft-eval CLI paths.
/// Honest about D2a parity status: never claims release certification.
/// </summary>
public sealed class LoraSftRunReport
{
    [JsonPropertyName("mode")]
    public string Mode { get; set; } = "";

    [JsonPropertyName("backend")]
    public string Backend { get; set; } = ""; // "cpu" | "silk"

    [JsonPropertyName("device")]
    public string Device { get; set; } = "";

    [JsonPropertyName("model")]
    public string Model { get; set; } = "";

    [JsonPropertyName("template")]
    public string Template { get; set; } = "";

    [JsonPropertyName("rank")]
    public int Rank { get; set; }

    [JsonPropertyName("alpha")]
    public float Alpha { get; set; }

    [JsonPropertyName("targets")]
    public string Targets { get; set; } = "";

    [JsonPropertyName("maxLen")]
    public int MaxLen { get; set; }

    [JsonPropertyName("maxExamples")]
    public int MaxExamples { get; set; }

    [JsonPropertyName("seed")]
    public int Seed { get; set; }

    [JsonPropertyName("trainExamples")]
    public int TrainExamples { get; set; }

    [JsonPropertyName("valExamples")]
    public int ValExamples { get; set; }

    [JsonPropertyName("baseParams")]
    public int BaseParams { get; set; }

    [JsonPropertyName("trainableParams")]
    public int TrainableParams { get; set; }

    [JsonPropertyName("initialValLoss")]
    public float InitialValLoss { get; set; }

    [JsonPropertyName("bestValLoss")]
    public float BestValLoss { get; set; }

    [JsonPropertyName("bestEpoch")]
    public int BestEpoch { get; set; }

    [JsonPropertyName("elapsedMs")]
    public long ElapsedMs { get; set; }

    [JsonPropertyName("adapterPath")]
    public string? AdapterPath { get; set; }

    [JsonPropertyName("adapterFormat")]
    public string? AdapterFormat { get; set; }

    [JsonPropertyName("adapterSha256")]
    public string? AdapterSha256 { get; set; }

    [JsonPropertyName("parityStatus")]
    public string ParityStatus { get; set; } = "D2a in progress: synthetic + smoke parity evidence; not release-certified.";

    [JsonPropertyName("baseMetrics")]
    public string? BaseMetrics { get; set; }

    [JsonPropertyName("adaptedMetrics")]
    public string? AdaptedMetrics { get; set; }

    public static JsonSerializerOptions SerializerOptions { get; } = new() { WriteIndented = true };

    public void Save(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        File.WriteAllText(path, JsonSerializer.Serialize(this, SerializerOptions));
    }

    public static string? Sha256File(string? path)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path)) return null;
        using var sha = SHA256.Create();
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(sha.ComputeHash(stream)).ToLowerInvariant();
    }
}