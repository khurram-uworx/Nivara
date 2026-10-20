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


    public SilkQwenSession(QwenSilkConfig config, Dictionary<string, (System.Numerics.BFloat16[] Data, int[] Shape)> tensors)
    {
        runtime = new SilkRuntime();
        try { runner = new SilkQwenGpuRunner(runtime, config, tensors); }
        catch { runtime.Dispose(); throw; }
    }


    public static SilkQwenSession Open(QwenSilkConfig config, Dictionary<string, (float[] Data, int[] Shape)> tensors)
        => new SilkQwenSession(config, tensors);

    public static SilkQwenSession Open(QwenSilkConfig config, Dictionary<string, (System.Numerics.BFloat16[] Data, int[] Shape)> tensors)
        => new SilkQwenSession(config, tensors);

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
