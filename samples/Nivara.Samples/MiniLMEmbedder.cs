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
        public Func<IEnumerable<string>, float[][]> EmbedBatch { get; }
        public int Dimension { get; }
        private readonly IDisposable? _lifetime;

        public Handle(Func<string, float[]> embed, int dimension, IDisposable? lifetime)
            : this(embed, texts => EmbedBatchFallback(embed, texts), dimension, lifetime)
        {
        }

        public Handle(
            Func<string, float[]> embed,
            Func<IEnumerable<string>, float[][]> embedBatch,
            int dimension,
            IDisposable? lifetime)
        {
            Embed = embed;
            EmbedBatch = embedBatch;
            Dimension = dimension;
            _lifetime = lifetime;
        }

        public void Dispose() => _lifetime?.Dispose();
    }

    static float[][] EmbedBatchFallback(Func<string, float[]> embed, IEnumerable<string> texts)
    {
        var input = texts?.ToList() ?? [];
        var result = new float[input.Count][];
        for (int i = 0; i < input.Count; i++)
        {
            var text = input[i];
            if (string.IsNullOrWhiteSpace(text))
            {
                result[i] = null!;
                continue;
            }
            try { result[i] = embed(text); }
            catch { result[i] = null!; }
        }
        return result;
    }

    public static Handle Create(string modelDir, bool useSilk)
    {
        if (useSilk)
            return CreateSilk(modelDir);
        return CreateCpu(modelDir);
    }

    public static float[][] EmbedBatch(Handle embedder, IEnumerable<string> texts)
    {
        ArgumentNullException.ThrowIfNull(embedder);
        return embedder.EmbedBatch(texts);
    }

    static Handle CreateCpu(string modelDir)
    {
        var gen = MiniLMEmbeddingGenerator.Create(modelDir);
        Func<string, float[]> embed = text =>
        {
            var v = gen.GenerateVectorAsync(text).GetAwaiter().GetResult();
            return v.ToArray();
        };

        Func<IEnumerable<string>, float[][]> embedBatch = texts =>
        {
            var input = texts?.ToList() ?? [];
            var result = new float[input.Count][];
            var toEmbed = new List<(int Index, string Text)>();
            for (int i = 0; i < input.Count; i++)
            {
                var text = input[i];
                if (string.IsNullOrWhiteSpace(text))
                {
                    result[i] = null!;
                    continue;
                }
                toEmbed.Add((i, text));
            }

            if (toEmbed.Count > 0)
            {
                var generated = gen.GenerateAsync(toEmbed.Select(t => t.Text)).GetAwaiter().GetResult();
                for (int k = 0; k < toEmbed.Count; k++)
                    result[toEmbed[k].Index] = generated[k].Vector.ToArray();
            }
            return result;
        };

        return new Handle(embed, embedBatch, gen.EmbeddingDimension, null);
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

        const int maxLen = 128;
        const int maxBatch = 8;

        Func<string, float[]> embed = text =>
        {
            var (tokenIds, attnMask, _) = MiniLMTokenizer.Encode(tokenizer, text, maxLen);
            var intIds = Array.ConvertAll(tokenIds, x => (int)x);
            var result = runner.Forward(intIds, attnMask, 1, maxLen);
            var cls = ExtractCls(result.Hidden, config.HiddenSize);
            L2Normalize(cls);
            return cls;
        };

        Func<IEnumerable<string>, float[][]> embedBatch = texts =>
        {
            var input = texts?.ToList() ?? [];
            var result = new float[input.Count][];
            var work = new List<(int Index, string Text)>();
            for (int i = 0; i < input.Count; i++)
            {
                if (string.IsNullOrWhiteSpace(input[i]))
                {
                    result[i] = null!;
                    continue;
                }
                work.Add((i, input[i]));
            }

            int cursor = 0;
            while (cursor < work.Count)
            {
                int chunkSize = Math.Min(maxBatch, work.Count - cursor);
                var flatIds = new int[chunkSize * maxLen];
                var flatMask = new float[chunkSize * maxLen];

                for (int b = 0; b < chunkSize; b++)
                {
                    var (tokenIds, attnMask, _) = MiniLMTokenizer.Encode(tokenizer, work[cursor + b].Text, maxLen);
                    for (int s = 0; s < maxLen; s++)
                    {
                        flatIds[b * maxLen + s] = (int)tokenIds[s];
                        flatMask[b * maxLen + s] = attnMask[s];
                    }
                }

                var forward = runner.Forward(flatIds, flatMask, chunkSize, maxLen);
                for (int b = 0; b < chunkSize; b++)
                {
                    var cls = new float[config.HiddenSize];
                    Array.Copy(forward.Hidden, b * maxLen * config.HiddenSize, cls, 0, config.HiddenSize);
                    L2Normalize(cls);
                    result[work[cursor + b].Index] = cls;
                }

                cursor += chunkSize;
            }

            return result;
        };

        return new Handle(embed, embedBatch, config.HiddenSize, lifetime);
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
