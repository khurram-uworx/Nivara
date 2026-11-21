using Nivara.AutoDiff.Nn;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Numerics;

namespace Nivara.AutoDiff.Serialization;

/// <summary>
/// Serializes and deserializes LoRA adapter-only files containing the trainable
/// <c>A</c>/<c>B</c> tensors and adapter metadata, without base weights.
/// </summary>
public static class LoraAdapterSerializer
{
    public const string AdapterFormat = "nivara-lora-adapter-v1";

    static readonly JsonSerializerOptions s_options = new()
    {
        WriteIndented = true
    };

    /// <summary>
    /// Saves a LoRA adapter and its metadata to a versioned JSON file that contains
    /// only the adapter tensors and metadata, without base weights.
    /// </summary>
    /// <typeparam name="T">The adapter compute type.</typeparam>
    /// <param name="lora">The LoRA adapter to save.</param>
    /// <param name="path">The destination file path.</param>
    /// <param name="baseModelId">The base model identifier the adapter targets.</param>
    /// <param name="targets">The adapter target names.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="lora"/>, <paramref name="path"/>, or <paramref name="baseModelId"/> is null or blank.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the adapter has been merged.</exception>
    public static void Save<T>(LoRALinear<T> lora, string path, string baseModelId, IReadOnlyList<string>? targets = null)
        where T : struct, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(lora);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentException.ThrowIfNullOrWhiteSpace(baseModelId);

        if (lora.IsMerged)
            throw new InvalidOperationException("Cannot save a merged LoRA adapter as an adapter-only file.");

        var file = CreateAdapterFile(
            typeof(T).Name,
            lora.A.Tensor,
            lora.B.Tensor,
            lora.Rank,
            lora.Alpha,
            baseModelId,
            targets);

        File.WriteAllText(path, JsonSerializer.Serialize(file, s_options));
    }

    /// <summary>
    /// Saves a mixed-dtype LoRA adapter and its metadata to a versioned JSON file that
    /// contains only the adapter tensors and metadata, without base weights.
    /// </summary>
    /// <typeparam name="TModel">The adapter compute type (for example F32).</typeparam>
    /// <typeparam name="TWeight">The frozen base weight type (for example BF16).</typeparam>
    /// <param name="lora">The mixed-dtype LoRA adapter to save.</param>
    /// <param name="path">The destination file path.</param>
    /// <param name="baseModelId">The base model identifier the adapter targets.</param>
    /// <param name="targets">The adapter target names.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="lora"/>, <paramref name="path"/>, or <paramref name="baseModelId"/> is null or blank.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the adapter has been merged.</exception>
    public static void Save<TModel, TWeight>(
        LoRALinear<TModel, TWeight> lora,
        string path,
        string baseModelId,
        IReadOnlyList<string>? targets = null)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        ArgumentNullException.ThrowIfNull(lora);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentException.ThrowIfNullOrWhiteSpace(baseModelId);

        if (lora.IsMerged)
            throw new InvalidOperationException("Cannot save a merged LoRA adapter as an adapter-only file.");

        var file = CreateAdapterFile(
            typeof(TModel).Name,
            lora.A.Tensor,
            lora.B.Tensor,
            lora.Rank,
            lora.Alpha,
            baseModelId,
            targets);

        File.WriteAllText(path, JsonSerializer.Serialize(file, s_options));
    }

    /// <summary>
    /// Loads an adapter-only LoRA file into a LoRA adapter backed by a frozen base.
    /// </summary>
    /// <typeparam name="T">The adapter compute type.</typeparam>
    /// <param name="lora">The LoRA adapter to load into.</param>
    /// <param name="path">The adapter file path.</param>
    /// <param name="expectedBaseModelId">The base model identifier the adapter must target.</param>
    /// <param name="expectedTargets">The adapter target names the loaded adapter must serve.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="lora"/>, <paramref name="path"/>, or <paramref name="expectedBaseModelId"/> is null or blank.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the adapter file does not exist.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the format, rank, alpha, base model, or shapes are incompatible.</exception>
    public static void Load<T>(
        LoRALinear<T> lora,
        string path,
        string expectedBaseModelId,
        IReadOnlyList<string>? expectedTargets = null)
        where T : struct, IFloatingPointIeee754<T>
    {
        var file = ReadAdapterFile(path);
        ValidateAdapterFile(lora.Rank, file, GetDouble(lora.Alpha), expectedBaseModelId, expectedTargets);
        LoadAdapter(lora.A, lora.B, lora.Base.InFeatures, lora.Base.OutFeatures, lora.Rank, file);
    }

    /// <summary>
    /// Loads an adapter-only mixed-dtype LoRA file into a LoRA adapter backed by a frozen base.
    /// </summary>
    /// <typeparam name="TModel">The adapter compute type (for example F32).</typeparam>
    /// <typeparam name="TWeight">The frozen base weight type (for example BF16).</typeparam>
    /// <param name="lora">The mixed-dtype LoRA adapter to load into.</param>
    /// <param name="path">The adapter file path.</param>
    /// <param name="expectedBaseModelId">The base model identifier the adapter must target.</param>
    /// <param name="expectedTargets">The adapter target names the loaded adapter must serve.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="lora"/>, <paramref name="path"/>, or <paramref name="expectedBaseModelId"/> is null or blank.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the adapter file does not exist.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the format, rank, alpha, base model, or shapes are incompatible.</exception>
    public static void Load<TModel, TWeight>(
        LoRALinear<TModel, TWeight> lora,
        string path,
        string expectedBaseModelId,
        IReadOnlyList<string>? expectedTargets = null)
        where TModel : struct, IFloatingPointIeee754<TModel>
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        var file = ReadAdapterFile(path);
        ValidateAdapterFile(lora.Rank, file, GetDouble(lora.Alpha), expectedBaseModelId, expectedTargets);
        LoadAdapter(lora.A, lora.B, lora.Base.InFeatures, lora.Base.OutFeatures, lora.Rank, file);
    }

    static LoraAdapterFile CreateAdapterFile<T>(
        string type,
        GradTensor<T> a,
        GradTensor<T> b,
        int rank,
        T alpha,
        string baseModelId,
        IReadOnlyList<string>? targets)
        where T : struct, IFloatingPointIeee754<T>
    {
        return new LoraAdapterFile
        {
            Format = AdapterFormat,
            Type = type,
            Version = 1,
            BaseModelId = baseModelId,
            Rank = rank,
            Alpha = GetDouble(alpha),
            Targets = targets?.ToArray() ?? [],
            Parameters = new Dictionary<string, LoraParameterEntry>
            {
                ["A"] = new LoraParameterEntry
                {
                    Shape = a.Shape,
                    Values = SerializeBinary(GetValues<T>(a))
                },
                ["B"] = new LoraParameterEntry
                {
                    Shape = b.Shape,
                    Values = SerializeBinary(GetValues<T>(b))
                }
            }
        };
    }

    static void ValidateAdapterFile(
        int rank,
        LoraAdapterFile file,
        double expectedAlpha,
        string expectedBaseModelId,
        IReadOnlyList<string>? expectedTargets)
    {
        if (file.Format != AdapterFormat)
            throw new InvalidOperationException(
                $"Unsupported LoRA adapter format '{file.Format}'. Expected '{AdapterFormat}'.");

        if (file.BaseModelId != expectedBaseModelId)
            throw new InvalidOperationException(
                $"Cannot load LoRA adapter for base model '{expectedBaseModelId}': adapter was saved for base model '{file.BaseModelId}'.");

        if (file.Rank != rank)
            throw new InvalidOperationException(
                $"Cannot load LoRA adapter with rank {file.Rank}: adapter rank does not match the target rank {rank}.");

        if (file.Alpha != expectedAlpha)
            throw new InvalidOperationException(
                $"Cannot load LoRA adapter with alpha {file.Alpha}: adapter alpha does not match the target alpha {expectedAlpha}.");

        if (expectedTargets is not null && !expectedTargets.SequenceEqual(file.Targets ?? []))
            throw new InvalidOperationException(
                $"Cannot load LoRA adapter for targets [{string.Join(", ", expectedTargets)}]: adapter was saved for targets [{string.Join(", ", file.Targets ?? [])}].");
    }

    static void LoadAdapter<T>(
        Parameter<T> a,
        Parameter<T> b,
        int inFeatures,
        int outFeatures,
        int rank,
        LoraAdapterFile file)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (!file.Parameters.TryGetValue("A", out var aEntry) || !file.Parameters.TryGetValue("B", out var bEntry))
            throw new InvalidOperationException("LoRA adapter file must contain adapter tensors 'A' and 'B'.");

        ValidateAdapterTensor("A", aEntry, inFeatures, rank);
        ValidateAdapterTensor("B", bEntry, rank, outFeatures);

        var aData = DeserializeBinary<T>(aEntry.Values, GetElementCount(aEntry.Shape));
        var bData = DeserializeBinary<T>(bEntry.Values, GetElementCount(bEntry.Shape));

        a.Tensor = ReverseGradTensor<T>.FromMatrix(aData, inFeatures, rank, requiresGrad: true);
        b.Tensor = ReverseGradTensor<T>.FromMatrix(bData, rank, outFeatures, requiresGrad: true);
    }

    static void ValidateAdapterTensor(string name, LoraParameterEntry entry, int expectedRows, int expectedCols)
    {
        if (entry.Shape is null || entry.Shape.Length != 2 || entry.Shape[0] != expectedRows || entry.Shape[1] != expectedCols)
            throw new InvalidOperationException(
                $"LoRA adapter tensor '{name}' shape mismatch: expected [{expectedRows}, {expectedCols}], got [{string.Join(", ", entry.Shape ?? [])}].");
    }

    static LoraAdapterFile ReadAdapterFile(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        if (!File.Exists(path))
            throw new FileNotFoundException($"LoRA adapter file not found: {path}", path);

        var file = JsonSerializer.Deserialize<LoraAdapterFile>(File.ReadAllText(path), s_options)
            ?? throw new InvalidOperationException("Failed to deserialize LoRA adapter file.");

        if (file.Format != AdapterFormat)
            throw new InvalidOperationException(
                $"Unsupported LoRA adapter format '{file.Format}'. Expected '{AdapterFormat}'.");

        return file;
    }

    static int GetElementCount(int[]? shape)
    {
        if (shape is null || shape.Length == 0)
            throw new InvalidOperationException("LoRA adapter parameter shape must contain at least one dimension.");

        int length = 1;
        foreach (var dimension in shape)
        {
            if (dimension <= 0)
                throw new InvalidOperationException($"LoRA adapter parameter shape dimensions must be positive, got {dimension}.");

            length *= dimension;
        }

        return length;
    }

    static T[] GetValues<T>(GradTensor<T> tensor) where T : struct, IFloatingPointIeee754<T>
    {
        var values = new T[tensor.Length];
        tensor.Data.CopyTo(values, default);
        return values;
    }

    static string SerializeBinary<T>(T[] data) where T : struct
    {
        var bytes = MemoryMarshal.AsBytes(data.AsSpan());
        return Convert.ToBase64String(bytes);
    }

    static T[] DeserializeBinary<T>(string base64, int expectedLength) where T : struct
    {
        var bytes = Convert.FromBase64String(base64);
        var expectedBytes = expectedLength * Unsafe.SizeOf<T>();

        if (bytes.Length != expectedBytes)
            throw new InvalidOperationException(
                $"Binary data size mismatch for LoRA adapter parameter: expected {expectedBytes} bytes, got {bytes.Length} bytes.");

        return MemoryMarshal.Cast<byte, T>(bytes).ToArray();
    }

    static double GetDouble<T>(T value) where T : struct, IFloatingPointIeee754<T>
        => double.CreateChecked(value);

    sealed class LoraAdapterFile
    {
        public string Format { get; set; } = AdapterFormat;
        public string Type { get; set; } = "";
        public int Version { get; set; } = 1;
        public string BaseModelId { get; set; } = "";
        public int Rank { get; set; }
        public double Alpha { get; set; }
        public string[] Targets { get; set; } = [];
        public Dictionary<string, LoraParameterEntry> Parameters { get; set; } = new();
    }

    sealed class LoraParameterEntry
    {
        public int[] Shape { get; set; } = [];
        public string Values { get; set; } = "";
    }
}
