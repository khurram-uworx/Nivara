using Microsoft.Extensions.AI;
using Microsoft.ML.Tokenizers;
using Nivara.AI;
using Nivara.Silk;

namespace Nivara.Samples;

/// <summary>
/// Shared MiniLM (all-MiniLM-L6-v2) embedding factory used by NivaraChat (and others).
/// Supports CPU (via <see cref="MiniLMEmbeddingGenerator"/>) and Silk OpenCL backends.
/// </summary>
public static class MiniLMEmbedder
{
    public sealed class Handle : IDisposable
    {
        public Func<string, float[]> Embed { get; }
        public int Dimension { get; }
        private readonly IDisposable? _lifetime;
        public Handle(Func<string, float[]> embed, int dimension, IDisposable? lifetime)
        {
            Embed = embed;
            Dimension = dimension;
            _lifetime = lifetime;
        }
        public void Dispose() => _lifetime?.Dispose();
    }

    public static Handle Create(string modelDir, bool useSilk)
    {
        if (useSilk)
            return CreateSilk(modelDir);
        return CreateCpu(modelDir);
    }

    static Handle CreateCpu(string modelDir)
    {
        var gen = MiniLMEmbeddingGenerator.Create(modelDir);
        Func<string, float[]> embed = text =>
        {
            var v = gen.GenerateVectorAsync(text).GetAwaiter().GetResult();
            return v.ToArray();
        };
        return new Handle(embed, gen.EmbeddingDimension, null);
    }

    static Handle CreateSilk(string modelDir)
    {
        var config = BertConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json")));
        var tensors = SafeTensorsLoader.Read(Path.Combine(modelDir, "model.safetensors"));
        var tokenizer = MiniLMTokenizer.Load(Path.Combine(modelDir, "vocab.txt"));

        var runtime = new SilkRuntime();
        SilkBertEncoderRunner runner;
        try
        {
            runner = new SilkBertEncoderRunner(runtime, config, tensors, BertGpuNaming.Bert);
        }
        catch
        {
            runtime.Dispose();
            throw;
        }

        var lifetime = new CompositeDisposable(runtime, runner);

        Func<string, float[]> embed = text =>
        {
            var (tokenIds, attnMask, _) = MiniLMTokenizer.Encode(tokenizer, text, maxLen: 128);
            var intIds = Array.ConvertAll(tokenIds, x => (int)x);
            var result = runner.Forward(intIds, attnMask, 1, intIds.Length);
            var cls = ExtractCls(result.Hidden, config.HiddenSize);
            L2Normalize(cls);
            return cls;
        };

        return new Handle(embed, config.HiddenSize, lifetime);
    }

    static float[] ExtractCls(float[] hidden, int hiddenSize)
    {
        var result = new float[hiddenSize];
        Array.Copy(hidden, 0, result, 0, hiddenSize);
        return result;
    }

    static void L2Normalize(float[] v)
    {
        float norm = 0f;
        foreach (var x in v) norm += x * x;
        norm = MathF.Sqrt(norm);
        if (norm > 1e-12f)
        {
            float inv = 1f / norm;
            for (int i = 0; i < v.Length; i++) v[i] *= inv;
        }
    }

    sealed class CompositeDisposable : IDisposable
    {
        private readonly IDisposable _a;
        private readonly IDisposable _b;
        public CompositeDisposable(IDisposable a, IDisposable b) { _a = a; _b = b; }
        public void Dispose() { _b.Dispose(); _a.Dispose(); }
    }
}
