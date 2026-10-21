using Nivara.AI;
using System.Diagnostics;
using System.Numerics;

namespace Nivara.Silk;

public sealed class SilkLayaSession : ILayaGpuSession
{
    readonly SilkRuntime runtime;
    readonly SilkModernBertGpuRunner encoder;
    readonly SilkLayaHeadGpuRunner head;

    public SilkLayaSession(ModernBertConfig config, Dictionary<string, (float[] Data, int[] Shape)> tensors, int headLayerCount)
    {
        runtime = new SilkRuntime();
        try
        {
            encoder = new SilkModernBertGpuRunner(runtime, config, tensors, prefix: "encoder");
            try { head = new SilkLayaHeadGpuRunner(runtime, tensors, headLayerCount); }
            catch { encoder.Dispose(); throw; }
        }
        catch { runtime.Dispose(); throw; }
    }

    public SilkLayaSession(ModernBertConfig config, Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors, int headLayerCount)
    {
        runtime = new SilkRuntime();
        try
        {
            encoder = new SilkModernBertGpuRunner(runtime, config, tensors, prefix: "encoder");
            try { head = new SilkLayaHeadGpuRunner(runtime, tensors, headLayerCount); }
            catch { encoder.Dispose(); throw; }
        }
        catch { runtime.Dispose(); throw; }
    }

    public static SilkLayaSession Open(ModernBertConfig config, Dictionary<string, (float[] Data, int[] Shape)> tensors, int headLayerCount)
        => new SilkLayaSession(config, tensors, headLayerCount);

    public static SilkLayaSession Open(ModernBertConfig config, Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors, int headLayerCount)
        => new SilkLayaSession(config, tensors, headLayerCount);

    public long LastEncodeMilliseconds { get; private set; }
    public long LastHeadMilliseconds { get; private set; }

    public string DeviceName => runtime.DeviceName;
    public LayaHeadOutput Forward(int[] ids, int validLength, LayaQuestionType type, IReadOnlyList<int> markers)
    {
        var sw = Stopwatch.StartNew();
        encoder.ForwardOnDevice(ids, validLength);
        LastEncodeMilliseconds = sw.ElapsedMilliseconds;
        sw.Restart();
        var result = head.Forward(encoder.HiddenOnDevice, encoder.HiddenRows, type, markers, validLength);
        LastHeadMilliseconds = sw.ElapsedMilliseconds;
        return result;
    }
    public void Dispose() { head.Dispose(); encoder.Dispose(); runtime.Dispose(); }
}
