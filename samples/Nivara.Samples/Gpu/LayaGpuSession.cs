using Nivara.AI;
using System.Diagnostics;

namespace Nivara.Samples.Gpu;

public sealed class IlgpuLayaSession : ILayaGpuSession
{
    readonly IlgpuRuntime runtime;
    readonly ModernBertGpuRunner encoder;
    readonly LayaHeadGpuRunner head;

    public IlgpuLayaSession(
        ModernBertConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        int headLayerCount)
    {
        runtime = new IlgpuRuntime();
        try
        {
            encoder = new ModernBertGpuRunner(runtime, config, tensors, prefix: "encoder");
            try
            {
                head = new LayaHeadGpuRunner(runtime, tensors, headLayerCount);
            }
            catch
            {
                encoder.Dispose();
                throw;
            }
        }
        catch
        {
            runtime.Dispose();
            throw;
        }
    }

    public static ILayaGpuSession Open(
        ModernBertConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        int headLayerCount)
        => new IlgpuLayaSession(config, tensors, headLayerCount);

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

    public void Dispose()
    {
        head.Dispose();
        encoder.Dispose();
        runtime.Dispose();
    }
}
