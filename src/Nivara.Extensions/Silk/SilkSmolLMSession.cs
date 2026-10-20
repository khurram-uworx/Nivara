using System.Numerics;

namespace Nivara.Silk;

public sealed class SilkSmolLMSession : IDisposable
{
    readonly SilkRuntime runtime;
    readonly SilkSmolLMGpuRunner runner;

    public SilkSmolLMSession(SmolLMConfig config, Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        runtime = new SilkRuntime();
        try { runner = new SilkSmolLMGpuRunner(runtime, config, tensors); }
        catch { runtime.Dispose(); throw; }
    }

    public SilkSmolLMSession(SmolLMConfig config, Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors)
    {
        runtime = new SilkRuntime();
        try { runner = new SilkSmolLMGpuRunner(runtime, config, tensors); }
        catch { runtime.Dispose(); throw; }
    }

    public static SilkSmolLMSession Open(SmolLMConfig config, Dictionary<string, (float[] Data, int[] Shape)> tensors)
        => new SilkSmolLMSession(config, tensors);

    public static SilkSmolLMSession Open(SmolLMConfig config, Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors)
        => new SilkSmolLMSession(config, tensors);

    public string DeviceName => runtime.DeviceName;
    public float[] Forward(int[] tokenIds) => runner.Forward(tokenIds);
    public float[] ForwardLastLogits(int[] tokenIds) => runner.ForwardLastLogits(tokenIds);
    public int ForwardLastArgmax(int[] tokenIds) => runner.ForwardLastArgmax(tokenIds);
    public int ForwardArgmax(int[] tokenIds) => runner.ForwardArgmax(tokenIds);
    public float[] ForwardPrefill(int[] tokenIds) => runner.ForwardPrefill(tokenIds);
    public int ForwardPrefillArgmax(int[] tokenIds) => runner.ForwardPrefillArgmax(tokenIds);
    public float[] ForwardCached(int tokenId, int position) => runner.ForwardCached(tokenId, position);
    public int ForwardCachedArgmax(int tokenId, int position) => runner.ForwardCachedArgmax(tokenId, position);
    public void ResetCache() => runner.ResetCache();
    public void Dispose() { runner.Dispose(); runtime.Dispose(); }
}
