using System.Text.Json;
using System.Text.Json.Serialization;

namespace Nivara.Silk;

/// <summary>
/// Canonical on-disk bundle for the multi-adapter LoRA SFT path shared by the CPU
/// reference trainer (<c>WorklogLoraSftTrainer</c>) and the Silk/iGPU trainer
/// (<see cref="SilkLlamaLoRATrainer"/>). Both producers serialize this exact shape so
/// adapters trained on either backend can be evaluated/reloaded by the other.
/// </summary>
/// <remarks>
/// Distinct from <c>Nivara.AutoDiff.Serialization.LoraAdapterSerializer</c>
/// (<c>nivara-lora-adapter-v1</c>), which serializes a single adapter against one
/// frozen base. This bundle carries one entry per (layer, target) projection.
/// </remarks>
public sealed class LoraSftAdapterBundle
{
    public const string FormatId = "nivara-lora-sft-adapters-v1";

    [JsonPropertyName("format")]
    public string Format { get; set; } = FormatId;

    [JsonPropertyName("baseModelId")]
    public string BaseModelId { get; set; } = "";

    [JsonPropertyName("template")]
    public string Template { get; set; } = "";

    [JsonPropertyName("rank")]
    public int Rank { get; set; }

    [JsonPropertyName("alpha")]
    public float Alpha { get; set; }

    [JsonPropertyName("targets")]
    public string Targets { get; set; } = "";

    [JsonPropertyName("trainableParams")]
    public int TrainableParams { get; set; }

    [JsonPropertyName("adapters")]
    public LoraSftAdapterEntry[] Adapters { get; set; } = [];

    public static JsonSerializerOptions SerializerOptions { get; } = new()
    {
        WriteIndented = true,
    };

    public static void Save(string path, LoraSftAdapterBundle bundle)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(bundle);
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        File.WriteAllText(path, JsonSerializer.Serialize(bundle, SerializerOptions));
    }

    public static LoraSftAdapterBundle Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"LoRA adapter bundle not found: {path}", path);

        var bundle = JsonSerializer.Deserialize<LoraSftAdapterBundle>(File.ReadAllText(path), SerializerOptions)
            ?? throw new InvalidOperationException("Failed to deserialize LoRA adapter bundle.");

        if (bundle.Format != FormatId)
            throw new InvalidOperationException(
                $"Unsupported LoRA adapter bundle format '{bundle.Format}'. Expected '{FormatId}'.");

        if (bundle.Rank <= 0)
            throw new InvalidOperationException($"Invalid adapter bundle rank {bundle.Rank}.");
        if (bundle.Adapters is null)
            throw new InvalidOperationException("Adapter bundle is missing the 'adapters' array.");

        return bundle;
    }

    public static byte[] Sha256(string path)
    {
        using var sha = System.Security.Cryptography.SHA256.Create();
        using var stream = File.OpenRead(path);
        return sha.ComputeHash(stream);
    }
}

public sealed class LoraSftAdapterEntry
{
    [JsonPropertyName("index")]
    public int Index { get; set; }

    [JsonPropertyName("layer")]
    public int Layer { get; set; } = -1;

    [JsonPropertyName("target")]
    public string Target { get; set; } = "";

    [JsonPropertyName("aShape")]
    public int[] AShape { get; set; } = [];

    [JsonPropertyName("bShape")]
    public int[] BShape { get; set; } = [];

    [JsonPropertyName("a")]
    public string A { get; set; } = "";

    [JsonPropertyName("b")]
    public string B { get; set; } = "";
}