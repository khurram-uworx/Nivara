using Nivara.AI;

namespace Nivara.Silk;

public sealed class SilkQwenSession : IDisposable
{
    readonly SilkRuntime runtime;
    readonly SilkQwenGpuRunner runner;

    public SilkQwenSession(QwenSilkConfig config, Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        runtime = new SilkRuntime();
        try { runner = new SilkQwenGpuRunner(runtime, config, tensors); }
        catch { runtime.Dispose(); throw; }
    }

    public static SilkQwenSession Open(QwenSilkConfig config, Dictionary<string, (float[] Data, int[] Shape)> tensors)
        => new SilkQwenSession(config, tensors);

    public string DeviceName => runtime.DeviceName;
    public float[] Forward(int[] tokenIds) => runner.Forward(tokenIds);
    public float[] ForwardLastLogits(int[] tokenIds) => runner.ForwardLastLogits(tokenIds);
    public float[] ForwardPrefill(int[] tokenIds) => runner.ForwardPrefill(tokenIds);
    public float[] ForwardCached(int tokenId, int position) => runner.ForwardCached(tokenId, position);
    public void ResetCache() => runner.ResetCache();
    public void Dispose() { runner.Dispose(); runtime.Dispose(); }
}
