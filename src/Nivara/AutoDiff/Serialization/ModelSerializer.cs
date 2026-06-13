using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Training;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Nivara.AutoDiff.Serialization;

public static class ModelSerializer
{
    static readonly JsonSerializerOptions s_options = new()
    {
        WriteIndented = true
    };

    public static void Save<T>(Module<T> model, string path) where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var file = BuildModelFile(model);
        var json = JsonSerializer.Serialize(file, s_options);
        File.WriteAllText(path, json);
    }

    const string ExpectedModelFormat = "nivara-ss-v1";
    const string ExpectedCheckpointFormat = "nivara-ckpt-v1";

    public static void Load<T>(Module<T> model, string path) where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        if (!File.Exists(path))
            throw new FileNotFoundException($"Model file not found: {path}", path);

        var json = File.ReadAllText(path);
        var file = JsonSerializer.Deserialize<ModelFile>(json, s_options)
            ?? throw new InvalidOperationException("Failed to deserialize model file.");

        if (file.Format != ExpectedModelFormat)
            throw new InvalidOperationException(
                $"Unsupported model format '{file.Format}'. Expected '{ExpectedModelFormat}'.");

        var parameters = model.GetParameters();

        foreach (var (name, entry) in file.Parameters)
        {
            if (!parameters.TryGetValue(name, out var param))
                throw new InvalidOperationException(
                    $"Parameter '{name}' not found in model. " +
                    $"Available parameters: [{string.Join(", ", parameters.Keys)}]");

            if (entry.Shape.Length != param.Shape.Length)
                throw new InvalidOperationException(
                    $"Parameter '{name}' shape rank mismatch: " +
                    $"file has {entry.Shape.Length}D, model has {param.Shape.Length}D.");

            int expectedLength = 1;
            foreach (var d in entry.Shape)
                expectedLength *= d;

            if (expectedLength != param.Length)
                throw new InvalidOperationException(
                    $"Parameter '{name}' length mismatch: " +
                    $"file has {expectedLength}, model has {param.Length}.");

            var values = DeserializeBinary<T>(entry.Values, expectedLength);
            var column = entry.HasNulls && entry.NullMask != null
                ? NivaraColumn<T>.CreateFromSpans(
                    values.AsSpan(0, expectedLength),
                    DeserializeBinary<bool>(entry.NullMask, expectedLength).AsSpan(0, expectedLength))
                : NivaraColumn<T>.Create(values.AsSpan(0, expectedLength));

            param.Tensor = new ReverseGradTensor<T>(column, param.Tensor.RequiresGrad, param.Tensor.Shape);
        }
    }

    public static void SaveCheckpoint<T>(
        Module<T> model,
        EpochResult<T> epoch,
        string path) where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(epoch);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var file = BuildCheckpointFile(model, epoch);
        var json = JsonSerializer.Serialize(file, s_options);
        File.WriteAllText(path, json);
    }

    public static Checkpoint<T> LoadCheckpoint<T>(string path) where T : struct, INumber<T>
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        if (!File.Exists(path))
            throw new FileNotFoundException($"Checkpoint file not found: {path}", path);

        var json = File.ReadAllText(path);
        var file = JsonSerializer.Deserialize<CheckpointFile>(json, s_options)
            ?? throw new InvalidOperationException("Failed to deserialize checkpoint file.");

        if (file.Format != ExpectedCheckpointFormat)
            throw new InvalidOperationException(
                $"Unsupported checkpoint format '{file.Format}'. Expected '{ExpectedCheckpointFormat}'.");

        var parameters = new Dictionary<string, ParameterData<T>>();
        foreach (var (name, entry) in file.Parameters)
        {
            int length = 1;
            foreach (var d in entry.Shape)
                length *= d;

            var values = DeserializeBinary<T>(entry.Values, length);

            var nullMask = entry.HasNulls && entry.NullMask != null
                ? DeserializeBinary<bool>(entry.NullMask, length)
                : null;

            parameters[name] = new ParameterData<T>
            {
                Shape = entry.Shape,
                Values = values,
                NullMask = nullMask
            };
        }

        return new Checkpoint<T>
        {
            Epoch = file.Epoch,
            Loss = file.Loss,
            Parameters = parameters
        };
    }

    static ModelFile BuildModelFile<T>(Module<T> model) where T : struct, INumber<T>
    {
        var parameters = model.Parameters();
        var entries = new Dictionary<string, ParameterEntry>();

        foreach (var (name, tensor) in parameters)
        {
            var data = tensor.Data;
            int length = data.Length;
            var values = new T[length];
            data.CopyTo(values, T.Zero);

            bool hasNulls = data.HasNulls;
            bool[]? nullMask = null;

            if (hasNulls && data.TryGetNullMask(out var mask))
            {
                nullMask = new bool[length];
                mask.CopyTo(nullMask);
            }

            entries[name] = new ParameterEntry
            {
                Shape = tensor.Shape,
                Values = SerializeBinary(values),
                HasNulls = hasNulls,
                NullMask = nullMask != null ? SerializeBinary(nullMask) : null
            };
        }

        return new ModelFile
        {
            Type = typeof(T).Name,
            Parameters = entries
        };
    }

    static CheckpointFile BuildCheckpointFile<T>(
        Module<T> model,
        EpochResult<T> epoch) where T : struct, INumber<T>
    {
        var parameters = model.Parameters();
        var entries = new Dictionary<string, ParameterEntry>();

        foreach (var (name, tensor) in parameters)
        {
            var data = tensor.Data;
            int length = data.Length;
            var values = new T[length];
            data.CopyTo(values, T.Zero);

            bool hasNulls = data.HasNulls;
            bool[]? nullMask = null;

            if (hasNulls && data.TryGetNullMask(out var mask))
            {
                nullMask = new bool[length];
                mask.CopyTo(nullMask);
            }

            entries[name] = new ParameterEntry
            {
                Shape = tensor.Shape,
                Values = SerializeBinary(values),
                HasNulls = hasNulls,
                NullMask = nullMask != null ? SerializeBinary(nullMask) : null
            };
        }

        return new CheckpointFile
        {
            Type = typeof(T).Name,
            Epoch = epoch.Epoch,
            Loss = double.CreateChecked(epoch.Loss),
            Parameters = entries
        };
    }

    static string SerializeBinary<T>(T[] data) where T : struct
    {
        var bytes = MemoryMarshal.AsBytes(data.AsSpan());
        return Convert.ToBase64String(bytes);
    }

    static string SerializeBinary(bool[] data)
    {
        var bytes = MemoryMarshal.AsBytes(data.AsSpan());
        return Convert.ToBase64String(bytes);
    }

    static T[] DeserializeBinary<T>(string base64, int expectedLength) where T : struct
    {
        var bytes = Convert.FromBase64String(base64);
        int expectedBytes = expectedLength * Unsafe.SizeOf<T>();

        if (bytes.Length != expectedBytes)
            throw new InvalidOperationException(
                $"Binary data size mismatch for parameter: " +
                $"expected {expectedBytes} bytes ({expectedLength} × {Unsafe.SizeOf<T>()}), " +
                $"got {bytes.Length} bytes.");

        return MemoryMarshal.Cast<byte, T>(bytes).ToArray();
    }

    sealed class ModelFile
    {
        public string Format { get; set; } = "nivara-ss-v1";
        public string Type { get; set; } = "";
        public int Version { get; set; } = 1;
        public Dictionary<string, ParameterEntry> Parameters { get; set; } = new();
    }

    sealed class CheckpointFile
    {
        public string Format { get; set; } = "nivara-ckpt-v1";
        public string Type { get; set; } = "";
        public int Version { get; set; } = 1;
        public int Epoch { get; set; }
        public double Loss { get; set; }
        public Dictionary<string, ParameterEntry> Parameters { get; set; } = new();
    }

    sealed class ParameterEntry
    {
        public int[] Shape { get; set; } = [];
        public string Values { get; set; } = "";
        public bool HasNulls { get; set; }
        public string? NullMask { get; set; }
    }
}
