namespace Nivara.Samples.Gpu;

/// <summary>
/// A ModernBERT encoder forward the sample drives without naming a backend. The device buffer
/// stays behind the session: ILGPU's is a <c>MemoryBuffer1D</c>, Silk's is an <c>nint</c>.
/// </summary>
public interface IModernBertGpuSession : IDisposable
{
    string DeviceName { get; }

    float[] Forward(int[] tokenIds, int validLength);
}

public delegate IModernBertGpuSession OpenModernBertSession(
    ModernBertConfig config,
    Dictionary<string, (float[] Data, int[] Shape)> tensors);

public sealed class IlgpuModernBertSession : IModernBertGpuSession
{
    readonly IlgpuRuntime runtime;
    readonly ModernBertGpuRunner runner;

    public IlgpuModernBertSession(
        ModernBertConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        string prefix = "model")
    {
        runtime = new IlgpuRuntime();
        try
        {
            runner = new ModernBertGpuRunner(runtime, config, tensors, prefix);
        }
        catch
        {
            runtime.Dispose();
            throw;
        }
    }

    public static IModernBertGpuSession Open(
        ModernBertConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors)
        => new IlgpuModernBertSession(config, tensors);

    public string DeviceName => runtime.DeviceName;

    public float[] Forward(int[] tokenIds, int validLength) => runner.Forward(tokenIds, validLength);

    public void Dispose()
    {
        runner.Dispose();
        runtime.Dispose();
    }
}

public sealed class SilkModernBertSession : IModernBertGpuSession
{
    readonly SilkRuntime runtime;
    readonly SilkModernBertGpuRunner runner;

    public SilkModernBertSession(
        ModernBertConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        string prefix = "model")
    {
        runtime = new SilkRuntime();
        try
        {
            runner = new SilkModernBertGpuRunner(runtime, config, tensors, prefix);
        }
        catch
        {
            runtime.Dispose();
            throw;
        }
    }

    public static IModernBertGpuSession Open(
        ModernBertConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors)
        => new SilkModernBertSession(config, tensors);

    public string DeviceName => runtime.DeviceName;

    public float[] Forward(int[] tokenIds, int validLength) => runner.Forward(tokenIds, validLength);

    public void Dispose()
    {
        runner.Dispose();
        runtime.Dispose();
    }
}
