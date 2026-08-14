using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Training;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Nivara.AutoDiff.Serialization;

/// <summary>
/// Serializes and deserializes models and checkpoints as versioned JSON files, with parameter
/// values encoded as base64 in binary form. Formats: <c>nivara-ss-v2</c> (state dict) and
/// <c>nivara-ckpt-v2</c> (checkpoint).
/// </summary>
public static class ModelSerializer
{
    static readonly JsonSerializerOptions s_options = new()
    {
        WriteIndented = true
    };

    /// <summary>
    /// Saves a model's state dict to a JSON file.
    /// </summary>
    /// <param name="model">The model to save</param>
    /// <param name="path">The destination file path</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="model"/> is null</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="path"/> is null or blank</exception>
    public static void Save<T>(Module<T> model, string path) where T : struct, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        File.WriteAllText(path, StateDictToJson(model.StateDict()));
    }

    const string ExpectedModelFormat = "nivara-ss-v2";
    const string ExpectedCheckpointFormat = "nivara-ckpt-v2";

    /// <summary>
    /// Loads a state dict from a JSON file into an existing model.
    /// </summary>
    /// <param name="model">The model to populate</param>
    /// <param name="path">The model file path</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="model"/> is null</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="path"/> is null or blank</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist</exception>
    public static void Load<T>(Module<T> model, string path) where T : struct, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        if (!File.Exists(path))
            throw new FileNotFoundException($"Model file not found: {path}", path);

        model.LoadStateDict(JsonToStateDict<T>(File.ReadAllText(path)));
    }

    /// <summary>
    /// Serializes a state dict to versioned JSON text.
    /// </summary>
    /// <param name="stateDict">The state dict to serialize</param>
    /// <returns>The JSON representation</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="stateDict"/> is null</exception>
    public static string StateDictToJson<T>(
        IReadOnlyDictionary<string, ReverseGradTensor<T>> stateDict) where T : struct, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(stateDict);

        var file = BuildModelFile(stateDict);
        return JsonSerializer.Serialize(file, s_options);
    }

    /// <summary>
    /// Deserializes versioned JSON text into a state dict.
    /// </summary>
    /// <param name="json">The JSON text</param>
    /// <param name="requiresGrad">Whether the deserialized tensors should require gradients</param>
    /// <returns>The deserialized state dict</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="json"/> is null or blank</exception>
    /// <exception cref="InvalidOperationException">Thrown when the JSON is malformed or uses an unsupported format</exception>
    public static Dictionary<string, ReverseGradTensor<T>> JsonToStateDict<T>(
        string json,
        bool requiresGrad = false) where T : struct, IFloatingPointIeee754<T>
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);

        var file = JsonSerializer.Deserialize<ModelFile>(json, s_options)
            ?? throw new InvalidOperationException("Failed to deserialize model file.");

        if (file.Format != ExpectedModelFormat)
            throw new InvalidOperationException(
                $"Unsupported model format '{file.Format}'. Expected '{ExpectedModelFormat}'.");

        var state = new Dictionary<string, ReverseGradTensor<T>>();
        foreach (var (name, entry) in file.Parameters)
            state[name] = DeserializeTensor<T>(entry, requiresGrad);

        return state;
    }

    /// <summary>
    /// Saves a model, epoch metadata, and optimizer state as a JSON checkpoint file.
    /// </summary>
    /// <param name="model">The model to save</param>
    /// <param name="epoch">The epoch result to record</param>
    /// <param name="path">The destination file path</param>
    /// <param name="optimizerState">Optional optimizer state to include</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="model"/> or <paramref name="epoch"/> is null</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="path"/> is null or blank</exception>
    public static void SaveCheckpoint<T>(
        Module<T> model,
        EpochResult<T> epoch,
        string path,
        Dictionary<string, T[]>? optimizerState = null) where T : struct, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(epoch);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var file = BuildCheckpointFile(model, epoch, optimizerState);
        var json = JsonSerializer.Serialize(file, s_options);
        File.WriteAllText(path, json);
    }

    /// <summary>
    /// Loads a checkpoint from a JSON checkpoint file.
    /// </summary>
    /// <param name="path">The checkpoint file path</param>
    /// <returns>The deserialized checkpoint</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="path"/> is null or blank</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist</exception>
    /// <exception cref="InvalidOperationException">Thrown when the JSON is malformed or uses an unsupported format</exception>
    public static Checkpoint<T> LoadCheckpoint<T>(string path) where T : struct, IFloatingPointIeee754<T>
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

            parameters[name] = new ParameterData<T>
            {
                Shape = entry.Shape,
                Values = values
            };
        }

        var optimizerState = new Dictionary<string, T[]>();
        foreach (var (name, entry) in file.OptimizerState)
        {
            var values = DeserializeBinary<T>(entry.Values, entry.Length);
            optimizerState[name] = values;
        }

        return new Checkpoint<T>
        {
            Epoch = file.Epoch,
            Loss = file.Loss,
            Parameters = parameters,
            OptimizerState = optimizerState
        };
    }

    static ModelFile BuildModelFile<T>(Module<T> model) where T : struct, IFloatingPointIeee754<T>
    {
        return BuildModelFile(model.StateDict());
    }

    static ModelFile BuildModelFile<T>(
        IReadOnlyDictionary<string, ReverseGradTensor<T>> stateDict) where T : struct, IFloatingPointIeee754<T>
    {
        return new ModelFile
        {
            Type = typeof(T).Name,
            Parameters = BuildParameterEntries(stateDict)
        };
    }

    static CheckpointFile BuildCheckpointFile<T>(
        Module<T> model,
        EpochResult<T> epoch,
        Dictionary<string, T[]>? optimizerState) where T : struct, IFloatingPointIeee754<T>
    {
        var entries = BuildParameterEntries(model.StateDict());

        var optEntries = new Dictionary<string, OptimizerStateEntry>();
        if (optimizerState != null)
        {
            foreach (var (name, values) in optimizerState)
                optEntries[name] = new OptimizerStateEntry { Length = values.Length, Values = SerializeBinary(values) };
        }

        return new CheckpointFile
        {
            Type = typeof(T).Name,
            Epoch = epoch.Epoch,
            Loss = double.CreateChecked(epoch.Loss),
            Parameters = entries,
            OptimizerState = optEntries
        };
    }

    static Dictionary<string, ParameterEntry> BuildParameterEntries<T>(
        IReadOnlyDictionary<string, ReverseGradTensor<T>> stateDict) where T : struct, IFloatingPointIeee754<T>
    {
        var entries = new Dictionary<string, ParameterEntry>();

        foreach (var (name, tensor) in stateDict)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(name);
            ArgumentNullException.ThrowIfNull(tensor);

            var data = tensor.Data;
            int length = data.Length;
            var values = new T[length];
            data.CopyTo(values, T.Zero);

            entries[name] = new ParameterEntry
            {
                Shape = tensor.Shape,
                Values = SerializeBinary(values)
            };
        }

        return entries;
    }

    static ReverseGradTensor<T> DeserializeTensor<T>(
        ParameterEntry entry,
        bool requiresGrad) where T : struct, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(entry);

        int length = GetElementCount(entry.Shape);
        var values = DeserializeBinary<T>(entry.Values, length);
        var column = NivaraColumn<T>.CreateFromOwnedArray(values);

        return new ReverseGradTensor<T>(column, requiresGrad, entry.Shape);
    }

    static int GetElementCount(int[] shape)
    {
        if (shape.Length == 0)
            throw new InvalidOperationException("Parameter shape must contain at least one dimension.");

        int length = 1;
        foreach (var d in shape)
        {
            if (d <= 0)
                throw new InvalidOperationException($"Parameter shape dimensions must be positive, got {d}.");

            length *= d;
        }

        return length;
    }

    static string SerializeBinary<T>(T[] data) where T : struct
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
        public string Format { get; set; } = "nivara-ss-v2";
        public string Type { get; set; } = "";
        public int Version { get; set; } = 1;
        public Dictionary<string, ParameterEntry> Parameters { get; set; } = new();
    }

    sealed class CheckpointFile
    {
        public string Format { get; set; } = "nivara-ckpt-v2";
        public string Type { get; set; } = "";
        public int Version { get; set; } = 1;
        public int Epoch { get; set; }
        public double Loss { get; set; }
        public Dictionary<string, ParameterEntry> Parameters { get; set; } = new();
        public Dictionary<string, OptimizerStateEntry> OptimizerState { get; set; } = new();
    }

    sealed class ParameterEntry
    {
        public int[] Shape { get; set; } = [];
        public string Values { get; set; } = "";
    }

    sealed class OptimizerStateEntry
    {
        public int Length { get; set; }
        public string Values { get; set; } = "";
    }
}
