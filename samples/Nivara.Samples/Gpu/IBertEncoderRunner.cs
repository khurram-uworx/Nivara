namespace Nivara.Samples.Gpu;

/// <summary>
/// A BERT-family encoder forward the sample runner drives without naming a backend. ILGPU
/// (<see cref="BertEncoderGpuRunner"/>) and Silk.NET OpenCL (<see cref="SilkBertEncoderRunner"/>)
/// both implement it, so the MiniLM entry points in <c>NivaraInference</c> are written once and
/// pick a device at construction — the two paths stay comparable because they are the same code
/// with a different <see cref="Forward"/> behind it.
/// </summary>
public interface IBertEncoderRunner : IDisposable
{
    /// <summary>The device the backend selected, for the run's device line.</summary>
    string DeviceName { get; }

    /// <summary>
    /// Runs the encoder for a padded batch of token IDs. The padding contract is documented on
    /// <see cref="BertEncoderGpuRunner.Forward"/> and is identical here: the mask is <c>0</c> at
    /// padding positions and positional ids are <c>0..seqLen-1</c> repeated per batch row.
    /// </summary>
    BertEncoderGpuResult Forward(int[] tokenIds, float[] attentionMask, int batch, int seqLen);
}
