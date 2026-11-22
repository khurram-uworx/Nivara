using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NivaraFineTuning;

/// <summary>
/// Structured, machine-readable summary emitted by the ft-lora / ft-eval CLI paths.
/// Includes D2a parity status.
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

    [JsonPropertyName("adapterParams")]
    public int AdapterParams { get; set; }

    [JsonPropertyName("trainableParams")]
    public int TrainableParams { get; set; }

    [JsonPropertyName("totalParams")]
    public int TotalParams { get; set; }

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
    public string ParityStatus { get; set; } = "CPU/Silk parity confirmed by controlled synthetic GPU-gated tests.";

    [JsonPropertyName("memoryBudgetBytes")]
    public long? MemoryBudgetBytes { get; set; }

    [JsonPropertyName("memoryBudgetMode")]
    public string MemoryBudgetMode { get; set; } = "unspecified"; // "advisory" | "unspecified"

    [JsonPropertyName("lossCurve")]
    public List<LoraSftEpochLoss> LossCurve { get; set; } = [];

    [JsonPropertyName("evalRan")]
    public bool EvalRan { get; set; }

    [JsonPropertyName("evalSplit")]
    public string? EvalSplit { get; set; }

    [JsonPropertyName("evalExamples")]
    public int EvalExamples { get; set; }

    [JsonPropertyName("baseMetrics")]
    public string? BaseMetrics { get; set; }

    [JsonPropertyName("adaptedMetrics")]
    public string? AdaptedMetrics { get; set; }

    [JsonPropertyName("evalDeltas")]
    public string? EvalDeltas { get; set; }

    [JsonPropertyName("baseWorklogMetrics")]
    public WorklogEvalMetrics? BaseWorklogMetrics { get; set; }

    [JsonPropertyName("adaptedWorklogMetrics")]
    public WorklogEvalMetrics? AdaptedWorklogMetrics { get; set; }

    [JsonPropertyName("evalElapsedMs")]
    public long EvalElapsedMs { get; set; }

    public static JsonSerializerOptions SerializerOptions { get; } = new() { WriteIndented = true };

    public void Save(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        File.WriteAllText(path, JsonSerializer.Serialize(this, SerializerOptions));
    }

    public void SaveMarkdown(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        File.WriteAllText(path, ToMarkdown());
    }

    public string ToMarkdown()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"# LoRA SFT Run Report");
        sb.AppendLine();
        sb.AppendLine($"- **Mode:** {Mode}");
        sb.AppendLine($"- **Backend:** {Backend}");
        sb.AppendLine($"- **Device:** {Device}");
        sb.AppendLine($"- **Model:** {Model}");
        sb.AppendLine($"- **Template:** {Template}");
        sb.AppendLine($"- **Rank:** {Rank}");
        sb.AppendLine($"- **Alpha:** {Alpha}");
        sb.AppendLine($"- **Targets:** {Targets}");
        sb.AppendLine($"- **MaxLen:** {MaxLen}");
        sb.AppendLine($"- **MaxExamples:** {MaxExamples}");
        sb.AppendLine($"- **Seed:** {Seed}");
        sb.AppendLine($"- **Train examples:** {TrainExamples}");
        sb.AppendLine($"- **Val examples:** {ValExamples}");
        sb.AppendLine();
        sb.AppendLine("## Parameters");
        sb.AppendLine();
        sb.AppendLine($"- **Base params:** {BaseParams:N0}");
        sb.AppendLine($"- **Adapter params:** {AdapterParams:N0}");
        sb.AppendLine($"- **Trainable params:** {TrainableParams:N0}");
        sb.AppendLine($"- **Total params:** {TotalParams:N0}");
        sb.AppendLine();
        sb.AppendLine("## Training");
        sb.AppendLine();
        sb.AppendLine($"- **Initial val loss:** {InitialValLoss:F4}");
        sb.AppendLine($"- **Best val loss:** {BestValLoss:F4}");
        sb.AppendLine($"- **Best epoch:** {BestEpoch}");
        sb.AppendLine($"- **Elapsed (ms):** {ElapsedMs}");
        if (LossCurve.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("### Loss curve");
            sb.AppendLine();
            sb.AppendLine("| Epoch | Train loss | Val loss |");
            sb.AppendLine("|---|---:|---:|");
            foreach (var e in LossCurve)
                sb.AppendLine($"| {e.Epoch} | {(e.TrainLoss.HasValue ? e.TrainLoss.Value.ToString("F4") : "n/a")} | {e.ValLoss:F4} |");
        }
        sb.AppendLine();
        sb.AppendLine("## Memory budget");
        sb.AppendLine();
        sb.AppendLine($"- **Requested (bytes):** {(MemoryBudgetBytes.HasValue ? MemoryBudgetBytes.Value.ToString("N0") : "unspecified")}");
        sb.AppendLine($"- **Mode:** {MemoryBudgetMode} (advisory only; not a hard runtime cap)");
        sb.AppendLine();
        sb.AppendLine("## Adapter");
        sb.AppendLine();
        sb.AppendLine($"- **Path:** {AdapterPath ?? "(none)"}");
        sb.AppendLine($"- **Format:** {AdapterFormat ?? "(none)"}");
        sb.AppendLine($"- **SHA256:** {AdapterSha256 ?? "(none)"}");
        sb.AppendLine();
        sb.AppendLine("## Eval");
        sb.AppendLine();
        sb.AppendLine($"- **Eval ran:** {EvalRan}");
        if (EvalRan)
        {
            sb.AppendLine($"- **Split:** {EvalSplit}");
            sb.AppendLine($"- **Examples:** {EvalExamples}");
            if (EvalElapsedMs > 0)
                sb.AppendLine($"- **Eval elapsed (ms):** {EvalElapsedMs}");
        }
        if (!string.IsNullOrWhiteSpace(BaseMetrics))
        {
            sb.AppendLine();
            sb.AppendLine("### Base metrics");
            sb.AppendLine();
            sb.AppendLine("```");
            sb.AppendLine(BaseMetrics);
            sb.AppendLine("```");
        }
        if (!string.IsNullOrWhiteSpace(AdaptedMetrics))
        {
            sb.AppendLine();
            sb.AppendLine("### Adapted metrics");
            sb.AppendLine();
            sb.AppendLine("```");
            sb.AppendLine(AdaptedMetrics);
            sb.AppendLine("```");
        }
        if (!string.IsNullOrWhiteSpace(EvalDeltas))
        {
            sb.AppendLine();
            sb.AppendLine("### Deltas");
            sb.AppendLine();
            sb.AppendLine("```");
            sb.AppendLine(EvalDeltas);
            sb.AppendLine("```");
        }
        sb.AppendLine();
        sb.AppendLine("## Parity status");
        sb.AppendLine();
        sb.AppendLine(ParityStatus);
        return sb.ToString();
    }

    public static string? Sha256File(string? path)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path)) return null;
        using var sha = SHA256.Create();
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(sha.ComputeHash(stream)).ToLowerInvariant();
    }
}

public sealed record LoraSftEpochLoss(int Epoch, float? TrainLoss, float ValLoss);
