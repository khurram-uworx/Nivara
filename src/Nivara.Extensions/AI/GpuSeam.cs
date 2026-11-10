using Nivara.Silk;
using System.Numerics;

namespace Nivara.AI;

/// <summary>
/// A BERT-family encoder forward the sample runner drives without naming a backend. The Silk.NET
/// OpenCL backend (<c>SilkBertEncoderRunner</c>) implements it, so the MiniLM entry points in
/// <c>NivaraInference</c> are written once and
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
/// stays behind the session (Silk uses an <c>nint</c> device buffer handle).
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

    /// <summary>Encode (ModernBERT) time of the most recent <see cref="Forward"/> call, in milliseconds.</summary>
    long LastEncodeMilliseconds { get; }

    /// <summary>Decision-head time of the most recent <see cref="Forward"/> call, in milliseconds.</summary>
    long LastHeadMilliseconds { get; }

    LayaHeadOutput Forward(int[] ids, int validLength, LayaQuestionType type, IReadOnlyList<int> markers);
}

public delegate IModernBertGpuSession OpenModernBertSession(
    ModernBertConfig config,
    Dictionary<string, (float[] Data, int[] Shape)> tensors);

public delegate IModernBertGpuSession OpenModernBertSessionBf16(
    ModernBertConfig config,
    Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors);

public delegate IModernBertGpuSession OpenModernBertSessionFp16(
    ModernBertConfig config,
    Dictionary<string, (Half[] Data, int[] Shape)> tensors);

public delegate ILayaGpuSession OpenLayaSession(
    ModernBertConfig config,
    Dictionary<string, (float[] Data, int[] Shape)> tensors,
    int headLayerCount);

public delegate ILayaGpuSession OpenLayaSessionBf16(
    ModernBertConfig config,
    Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors,
    int headLayerCount);

public delegate ILayaGpuSession OpenLayaSessionFp16(
    ModernBertConfig config,
    Dictionary<string, (Half[] Data, int[] Shape)> tensors,
    int headLayerCount);

/// <summary>
/// Seam-level static factory instances for the Silk BERT-family GPU sessions. Samples and other
/// callers select narrow precision through these delegates instead of naming
/// <c>SilkModernBertSession</c>/<c>SilkLayaSession</c> directly, so the backend stays behind the
/// <see cref="IModernBertGpuSession"/>/<see cref="ILayaGpuSession"/> seam.
/// </summary>
public static class GpuSeamFactories
{
    public static readonly OpenModernBertSession ModernBert = SilkModernBertSession.Open;
    public static readonly OpenModernBertSession ModernBertF32 = SilkModernBertSession.Open;
    public static readonly OpenModernBertSession OpenModernBert = SilkModernBertSession.Open;

    public static readonly OpenModernBertSessionBf16 ModernBertBf16 = SilkModernBertSession.Open;
    public static readonly OpenModernBertSessionBf16 OpenModernBertBf16 = SilkModernBertSession.Open;

    public static readonly OpenModernBertSessionFp16 ModernBertFp16 = SilkModernBertSession.Open;
    public static readonly OpenModernBertSessionFp16 OpenModernBertFp16 = SilkModernBertSession.Open;

    public static readonly OpenLayaSession Laya = SilkLayaSession.Open;
    public static readonly OpenLayaSession LayaF32 = SilkLayaSession.Open;
    public static readonly OpenLayaSession OpenLaya = SilkLayaSession.Open;

    public static readonly OpenLayaSessionBf16 LayaBf16 = SilkLayaSession.Open;
    public static readonly OpenLayaSessionBf16 OpenLayaBf16 = SilkLayaSession.Open;

    public static readonly OpenLayaSessionFp16 LayaFp16 = SilkLayaSession.Open;
    public static readonly OpenLayaSessionFp16 OpenLayaFp16 = SilkLayaSession.Open;
}
