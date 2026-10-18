using Nivara.AI;
using Nivara.Samples;
using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Qwen;

[TestFixture]
public class SilkQwenToolsWeatherLoopTests
{
    static string ModelDir
        => Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..",
            "samples", "data", "qwen2.5-0.5b-instruct");

    static readonly Dictionary<string, (float[] Data, int[] Shape)>? CachedTensors;

    static SilkQwenToolsWeatherLoopTests()
    {
        var safetensors = Path.Combine(ModelDir, "model.safetensors");
        if (!File.Exists(safetensors))
            return;
        CachedTensors = SafeTensorsLoader.Read<float>(safetensors);
    }

    [Test]
    public void ToolsWeather_Silk_Loop_ClosesWithCleanFinalAnswer_UseKvCacheOn()
    {
        if (CachedTensors is null)
            Assert.Ignore("Qwen model files absent; skipping Silk tool-loop acceptance verification.");

        try
        {
            using var probe = new SilkRuntime();
        }
        catch
        {
            Assert.Ignore("No OpenCL GPU available; skipping Silk tool-loop acceptance verification.");
        }

        var (model, config) = LoadConfig();
        _ = model;

        var oldOut = Console.Out;
        var writer = new StringWriter();
        Console.SetOut(writer);
        int code = 2;
        try
        {
            code = NivaraInference.Qwen.RunToolsSilk(CachedTensors!, ModelDir, true, string.Empty);
        }
        finally
        {
            Console.SetOut(oldOut);
        }

        var output = writer.ToString();
        Assert.That(code, Is.EqualTo(0));
        Assert.That(output, Does.Contain("partly cloudy"));
    }

    static (LlamaForCausalLM<float> Model, LlamaConfig Config) LoadConfig()
    {
        var tensors = CachedTensors!;
        var config = LlamaConfig.FromJson(File.ReadAllText(Path.Combine(ModelDir, "config.json")));
        var model = LlamaLoader.Load<float, float>(config, tensors);
        return (model, config);
    }
}
