using Nivara.AI;

namespace Nivara.Silk;

public sealed class SilkSmolLMSession
{
    readonly SilkRuntime runtime;
    readonly SilkSmolLMGpuRunner runner;

    public SilkSmolLMSession(SmolLMConfig config, Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        runtime = new SilkRuntime();
        try { runner = new SilkSmolLMGpuRunner(runtime, config, tensors); }
        catch { runtime.Dispose(); throw; }
    }

    public static SilkSmolLMSession Open(SmolLMConfig config, Dictionary<string, (float[] Data, int[] Shape)> tensors)
        => new SilkSmolLMSession(config, tensors);

    public string DeviceName => runtime.DeviceName;
    public float[] Forward(int[] tokenIds) => runner.Forward(tokenIds);
    public float[] ForwardLastLogits(int[] tokenIds) => runner.ForwardLastLogits(tokenIds);
    public void Dispose() { runner.Dispose(); runtime.Dispose(); }
}
