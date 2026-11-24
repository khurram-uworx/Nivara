using System.Numerics;

namespace Nivara.Silk;

public sealed class SilkSmolLMSession : IDisposable
{
    readonly SilkRuntime runtime;
    readonly SilkSmolLMGpuRunner runner;

    public SilkSmolLMSession(SmolLMConfig config, Dictionary<string, (float[] Data, int[] Shape)> tensors, bool enableQueueProfiling = false, SilkWeightQuantization quantization = SilkWeightQuantization.None)
    {
        if (quantization == SilkWeightQuantization.None)
        {
            string? env = Environment.GetEnvironmentVariable("NIVARA_SILK_INT8");
            if (env is not null && (env == "1" || env.Equals("true", StringComparison.OrdinalIgnoreCase)))
                quantization = SilkWeightQuantization.Int8Block32;
        }
        runtime = new SilkRuntime(enableQueueProfiling: enableQueueProfiling);
        try { runner = new SilkSmolLMGpuRunner(runtime, config, tensors, quantization); }
        catch { runtime.Dispose(); throw; }
    }

    public SilkSmolLMSession(SmolLMConfig config, Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors, bool enableQueueProfiling = false, SilkWeightQuantization quantization = SilkWeightQuantization.None)
    {
        if (quantization == SilkWeightQuantization.None)
        {
            string? env = Environment.GetEnvironmentVariable("NIVARA_SILK_INT8");
            if (env is not null && (env == "1" || env.Equals("true", StringComparison.OrdinalIgnoreCase)))
                quantization = SilkWeightQuantization.Int8Block32;
        }
        runtime = new SilkRuntime(enableQueueProfiling: enableQueueProfiling);
        try { runner = new SilkSmolLMGpuRunner(runtime, config, tensors, quantization); }
        catch { runtime.Dispose(); throw; }
    }

    public static SilkSmolLMSession Open(SmolLMConfig config, Dictionary<string, (float[] Data, int[] Shape)> tensors, bool enableQueueProfiling = false)
        => new SilkSmolLMSession(config, tensors, enableQueueProfiling);

    public static SilkSmolLMSession Open(SmolLMConfig config, Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors, bool enableQueueProfiling = false)
        => new SilkSmolLMSession(config, tensors, enableQueueProfiling);

    public string DeviceName => runtime.DeviceName;

    /// <summary>Enables per-phase cached-decode profiling on the underlying runner.</summary>
    public bool ProfilingEnabled
    {
        get => runner.ProfilingEnabled;
        set => runner.ProfilingEnabled = value;
    }

    /// <summary>Whether the underlying <see cref="SilkRuntime"/> command queue was created with
    /// <c>CL_QUEUE_PROFILING_ENABLE</c> (diagnostic mode, opt-in at construction).</summary>
    public bool QueueProfilingEnabled => runtime.QueueProfilingEnabled;

    /// <summary>Clears accumulated cached-decode profiling counters.</summary>
    public void ResetDecodeProfile() => runner.ResetDecodeProfile();

    /// <summary>Returns the accumulated cached-decode profiling snapshot.</summary>
    public SilkDecodeProfile GetDecodeProfile() => runner.GetDecodeProfile();

    public float[] Forward(int[] tokenIds) => runner.Forward(tokenIds);
    public float[] ForwardLastLogits(int[] tokenIds) => runner.ForwardLastLogits(tokenIds);
    public int ForwardLastArgmax(int[] tokenIds) => runner.ForwardLastArgmax(tokenIds);
    public int ForwardArgmax(int[] tokenIds) => runner.ForwardArgmax(tokenIds);
    public float[] ForwardPrefill(int[] tokenIds) => runner.ForwardPrefill(tokenIds);
    public int ForwardPrefillArgmax(int[] tokenIds) => runner.ForwardPrefillArgmax(tokenIds);
    public float[] ForwardCached(int tokenId, int position) => runner.ForwardCached(tokenId, position);
    public int ForwardCachedArgmax(int tokenId, int position) => runner.ForwardCachedArgmax(tokenId, position);
    public void ResetCache() => runner.ResetCache();
    public void LoadAdapterBundle(string adapterPath) => runner.LoadAdapterBundle(adapterPath);
    public void Dispose() { runner.Dispose(); runtime.Dispose(); }
}
