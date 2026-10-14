using Nivara.AI;
using Nivara.Silk;

namespace Nivara.Samples.Gpu;

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
