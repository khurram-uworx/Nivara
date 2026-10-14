using Nivara.AI;

namespace Nivara.Silk;

public sealed class SilkModernBertSession : IModernBertGpuSession
{
    readonly SilkRuntime runtime;
    readonly SilkModernBertGpuRunner runner;

    public SilkModernBertSession(ModernBertConfig config, Dictionary<string,(float[] Data,int[] Shape)> tensors, string prefix = "model")
    {
        runtime = new SilkRuntime();
        try { runner = new SilkModernBertGpuRunner(runtime, config, tensors, prefix); }
        catch { runtime.Dispose(); throw; }
    }

    public static SilkModernBertSession Open(ModernBertConfig config, Dictionary<string,(float[] Data,int[] Shape)> tensors)
        => new SilkModernBertSession(config, tensors);

    public string DeviceName => runtime.DeviceName;
    public float[] Forward(int[] tokenIds, int validLength) => runner.Forward(tokenIds, validLength);
    public void Dispose() { runner.Dispose(); runtime.Dispose(); }
}
