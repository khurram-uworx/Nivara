namespace Nivara.Samples.Gpu;

/// <summary>
/// Encoder plus decision head, driven without naming a backend. The on-device hidden state
/// stays inside the session.
/// </summary>
public interface ILayaGpuSession : IDisposable
{
    string DeviceName { get; }

    LayaHeadOutput Forward(int[] ids, int validLength, LayaQuestionType type, IReadOnlyList<int> markers);
}

public delegate ILayaGpuSession OpenLayaSession(
    ModernBertConfig config,
    Dictionary<string, (float[] Data, int[] Shape)> tensors,
    int headLayerCount);

public sealed class IlgpuLayaSession : ILayaGpuSession
{
    readonly IlgpuRuntime runtime;
    readonly ModernBertGpuRunner encoder;
    readonly LayaHeadGpuRunner head;

    public IlgpuLayaSession(
        ModernBertConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        int headLayerCount)
    {
        runtime = new IlgpuRuntime();
        try
        {
            encoder = new ModernBertGpuRunner(runtime, config, tensors, prefix: "encoder");
            try
            {
                head = new LayaHeadGpuRunner(runtime, tensors, headLayerCount);
            }
            catch
            {
                encoder.Dispose();
                throw;
            }
        }
        catch
        {
            runtime.Dispose();
            throw;
        }
    }

    public static ILayaGpuSession Open(
        ModernBertConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        int headLayerCount)
        => new IlgpuLayaSession(config, tensors, headLayerCount);

    public string DeviceName => runtime.DeviceName;

    public LayaHeadOutput Forward(int[] ids, int validLength, LayaQuestionType type, IReadOnlyList<int> markers)
    {
        encoder.ForwardOnDevice(ids, validLength);
        return head.Forward(encoder.HiddenOnDevice, encoder.HiddenRows, type, markers, validLength);
    }

    public void Dispose()
    {
        head.Dispose();
        encoder.Dispose();
        runtime.Dispose();
    }
}

public sealed class SilkLayaSession : ILayaGpuSession
{
    readonly SilkRuntime runtime;
    readonly SilkModernBertGpuRunner encoder;
    readonly SilkLayaHeadGpuRunner head;

    public SilkLayaSession(
        ModernBertConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        int headLayerCount)
    {
        runtime = new SilkRuntime();
        try
        {
            encoder = new SilkModernBertGpuRunner(runtime, config, tensors, prefix: "encoder");
            try
            {
                head = new SilkLayaHeadGpuRunner(runtime, tensors, headLayerCount);
            }
            catch
            {
                encoder.Dispose();
                throw;
            }
        }
        catch
        {
            runtime.Dispose();
            throw;
        }
    }

    public static ILayaGpuSession Open(
        ModernBertConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        int headLayerCount)
        => new SilkLayaSession(config, tensors, headLayerCount);

    public string DeviceName => runtime.DeviceName;

    public LayaHeadOutput Forward(int[] ids, int validLength, LayaQuestionType type, IReadOnlyList<int> markers)
    {
        encoder.ForwardOnDevice(ids, validLength);
        return head.Forward(encoder.HiddenOnDevice, encoder.HiddenRows, type, markers, validLength);
    }

    public void Dispose()
    {
        head.Dispose();
        encoder.Dispose();
        runtime.Dispose();
    }
}
