using Nivara.AI;

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

    public string DeviceName => runtime.DeviceName;

    public LayaHeadOutput Forward(int[] ids, int validLength, LayaQuestionType type, IReadOnlyList<int> markers)
    {
        encoder.ForwardOnDevice(ids, validLength);
        return head.Forward(encoder.HiddenOnDevice, encoder.HiddenRows, type, markers, validLength);
    }

    public void Dispose()
    {
        head.Dispose();
        encoder.Dispose();
        runtime.Dispose();
    }
}
