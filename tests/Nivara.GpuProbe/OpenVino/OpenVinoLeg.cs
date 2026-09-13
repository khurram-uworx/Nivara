using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using Nivara.GpuProbe.Kernels;

namespace Nivara.GpuProbe.OpenVino;

/// <summary>
/// The OpenVINO leg: loads the runtime, emits the three SmolLM-shaped IR models
/// (dot16, silu, gemv) for the chosen precision, compiles them on <c>GPU</c> and
/// gates the outputs against the production Nivara CPU kernels. Two configurations
/// are exposed, mirroring the validated Python flow:
/// <list type="bullet">
/// <item><b>RunBf16</b> — BF16-declared models, no INFERENCE_PRECISION_HINT
/// (the GPU plugin resolves the hint itself; gates pass bit-exact for reductions).</item>
/// <item><b>RunF32</b> — F32-declared models with INFERENCE_PRECISION_HINT=f32
/// (without the hint the plugin silently lowers to f16 and the gate fails).</item>
/// </list>
/// A fresh CACHE_DIR temp folder per (config, run) isolates cache state: earlier
/// prototyping crashed (0xC0000005) when compiled caches were reused across configs.
/// </summary>
internal static class OpenVinoLeg
{
    private const int TimingPasses = 25;

    public static LegResults? RunBf16(KernelFixtures fixtures) => Run(fixtures, bf16: true, hint: null);

    public static LegResults? RunF32(KernelFixtures fixtures) => Run(fixtures, bf16: false, hint: "f32");

    private static LegResults? Run(KernelFixtures fixtures, bool bf16, string? hint)
    {
        using OpenVinoNative? ov = OpenVinoRunner.TryLoad();
        if (ov is null)
            return null;

        string config = bf16 ? "bf16" : "f32";
        Console.WriteLine($"  OpenVINO config: {config} ({ov.OpenVinoVersion()})");
        Console.WriteLine($"  device: {ov.GetCoreProperty("GPU", "FULL_DEVICE_NAME")}");

        string work = Path.Combine(Path.GetTempPath(), "opencode", "ov", config, Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(work);
        ov.SetProperty("GPU", "CACHE_DIR", work);
        if (hint is not null)
            ov.SetProperty("GPU", "INFERENCE_PRECISION_HINT", hint);

        var dot = RunModel(ov, work, OvIrModels.Dot16(fixtures, bf16), Feed(fixtures.Dot16A, bf16),
            new long[] { 1, KernelFixtures.Dot16Length }, bf16);
        var silu = RunModel(ov, work, OvIrModels.Silu(fixtures, bf16), Feed(fixtures.SiluX, bf16),
            new long[] { KernelFixtures.HiddenSize }, bf16);
        var gemv = RunModel(ov, work, OvIrModels.Gemv(fixtures, bf16), Feed(fixtures.GemvX, bf16),
            new long[] { 1, KernelFixtures.HiddenSize }, bf16);

        return new LegResults(dot.Values[0], silu.Values, gemv.Values, dot.BestUs, silu.BestUs, gemv.BestUs);
    }

    private static byte[] Feed(ReadOnlySpan<BFloat16> values, bool bf16)
    {
        if (bf16)
            return MemoryMarshal.AsBytes(values).ToArray();
        return MemoryMarshal.AsBytes(CpuLeg.Widen(values).AsSpan()).ToArray();
    }

    private static (float[] Values, double BestUs) RunModel(
        OpenVinoNative ov, string work, OvIrModel model, byte[] input, long[] inDims, bool bf16)
    {
        string xmlPath = Path.Combine(work, model.Name + ".xml");
        string binPath = Path.Combine(work, model.Name + ".bin");
        File.WriteAllText(xmlPath, model.Xml);
        if (model.Weights is null)
            File.WriteAllBytes(binPath, Array.Empty<byte>());
        else
            File.WriteAllBytes(binPath, model.Weights);

        var sw = new Stopwatch();

        sw.Restart();
        IntPtr modelPtr = ov.ReadModel(xmlPath, binPath);
        (int modelInputs, int modelOutputs) = ov.ModelIoCounts(modelPtr);
        IntPtr compiled = ov.CompileModel("GPU", modelPtr);
        double compileUs = sw.Elapsed.TotalMicroseconds;

        Console.WriteLine($"  [{model.Name}] model inputs {modelInputs}, outputs {modelOutputs}");

        // Assert real GPU execution — no silent CPU fallback.
        string execDevices = ov.GetCompiledModelProperty(compiled, "EXECUTION_DEVICES");
        if (!execDevices.Contains("GPU", StringComparison.Ordinal))
            throw new InvalidOperationException($"model compiled but EXECUTION_DEVICES = \"{execDevices}\" (not GPU)");
        string hint = ov.GetCompiledModelProperty(compiled, "INFERENCE_PRECISION_HINT");
        Console.WriteLine($"  [{model.Name}] compile {compileUs,8:F0} µs | hint {hint,-6} | exec {execDevices}");

        IntPtr request = ov.CreateInferRequest(compiled);
        IntPtr inputTensor = ov.CreateTensor(bf16 ? OvElementType.Bf16 : OvElementType.F32, inDims);
        ov.FillTensor(inputTensor, input);
        ov.SetInputTensor(request, inputTensor);

        ov.Infer(request);

        double bestUs = double.PositiveInfinity;
        for (int i = 0; i < TimingPasses; i++)
        {
            sw.Restart();
            ov.Infer(request);
            sw.Stop();
            bestUs = Math.Min(bestUs, sw.Elapsed.TotalMicroseconds);
        }

        IntPtr outputTensor = ov.GetOutputTensor(request);
        try
        {
            int count = ov.ElementCount(outputTensor);
            IntPtr data = ov.TensorDataPtr(outputTensor);
            float[] values = KernelFixtures.ReadF32(data, count);
            Console.WriteLine($"  [{model.Name}] infer best {bestUs,8:F1} µs | out {count} elems");
            return (values, bestUs);
        }
        finally
        {
            ov.TensorFree(outputTensor);
            ov.TensorFree(inputTensor);
            ov.InferRequestFree(request);
            ov.CompiledModelFree(compiled);
            ov.ModelFree(modelPtr);
        }
    }
}