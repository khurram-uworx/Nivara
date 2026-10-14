namespace Nivara.AI;

/// <summary>
/// A BERT-family encoder forward the sample runner drives without naming a backend. ILGPU
/// (<c>BertEncoderGpuRunner</c>) and Silk.NET OpenCL (<c>SilkBertEncoderRunner</c>) both
/// implement it, so the MiniLM entry points in <c>NivaraInference</c> are written once and
/// pick a device at construction — the two paths stay comparable because they are the same
/// code with a different <see cref="Forward"/> behind it.
/// </summary>
public interface IBertEncoderRunner : IDisposable
{
    /// <summary>The device the backend selected, for the run's device line.</summary>
    string DeviceName { get; }

    /// <summary>
    /// Runs the encoder for a padded batch of token IDs. The mask is <c>0</c> at padding
    /// positions and positional ids are <c>0..seqLen-1</c> repeated per batch row.
    /// </summary>
    BertEncoderGpuResult Forward(int[] tokenIds, float[] attentionMask, int batch, int seqLen);
}

/// <summary>
/// A ModernBERT encoder forward the sample drives without naming a backend. The device buffer
/// stays behind the session: ILGPU's is a <c>MemoryBuffer1D</c>, Silk's is an <c>nint</c>.
/// </summary>
public interface IModernBertGpuSession : IDisposable
{
    string DeviceName { get; }

    float[] Forward(int[] tokenIds, int validLength);
}

/// <summary>
/// Encoder plus decision head, driven without naming a backend. The on-device hidden state
/// stays inside the session.
/// </summary>
public interface ILayaGpuSession : IDisposable
{
    string DeviceName { get; }

    LayaHeadOutput Forward(int[] ids, int validLength, LayaQuestionType type, IReadOnlyList<int> markers);
}

public delegate IModernBertGpuSession OpenModernBertSession(
    ModernBertConfig config,
    Dictionary<string, (float[] Data, int[] Shape)> tensors);

public delegate ILayaGpuSession OpenLayaSession(
    ModernBertConfig config,
    Dictionary<string, (float[] Data, int[] Shape)> tensors,
    int headLayerCount);
