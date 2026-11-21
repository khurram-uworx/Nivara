namespace Nivara.Silk;

public sealed class SilkTinyShakespeareSession : IDisposable
{
    readonly SilkRuntime runtime;
    readonly SilkTinyShakespeareGpuRunner runner;

    public SilkTinyShakespeareSession(TinyShakespeareConfig config, Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        runtime = new SilkRuntime();
        try { runner = new SilkTinyShakespeareGpuRunner(runtime, config, tensors); }
        catch { runtime.Dispose(); throw; }
    }

    public static SilkTinyShakespeareSession Open(TinyShakespeareConfig config, Dictionary<string, (float[] Data, int[] Shape)> tensors)
        => new SilkTinyShakespeareSession(config, tensors);

    public string DeviceName => runtime.DeviceName;
    public float[] ForwardPrefill(int[] tokenIds) => runner.ForwardPrefill(tokenIds);
    public float[] ForwardCached(int tokenId, int position) => runner.ForwardCached(tokenId, position);
    public void ResetCache() => runner.ResetCache();
    public int CachedLength => runner.CachedLength;
    public void Dispose() { runner.Dispose(); runtime.Dispose(); }
}
