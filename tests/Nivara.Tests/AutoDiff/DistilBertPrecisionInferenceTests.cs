using Nivara.AutoDiff;
using Nivara.AutoDiff.Utilities;
using Nivara.Samples;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// End-to-end regression coverage for narrow-precision (BFloat16 / Half) DistilBERT
/// SST-2 inference: it must preserve every prediction of the F32 model. The exact-int
/// token-ID path (<see cref="DistilBertForSequenceClassification{T}.Forward(int[], ReverseGradTensor{T}, int, int)"/>)
/// is required because narrow dtypes cannot represent a ~30k vocabulary.
/// These tests are weight-heavy (~268 MB checkpoint) and run only when the model files
/// exist locally; otherwise they Assert.Ignore so default/CI runs stay fast.
/// </summary>
[TestFixture]
public class DistilBertPrecisionInferenceTests
{
    static readonly string[] CompareSentences =
    [
        "This movie was an absolute joy from start to finish.",
        "A complete waste of time, boring and predictable.",
        "The acting was brilliant and the plot kept me on the edge of my seat.",
        "Terrible script, awful performances, I want my money back.",
        "An emotional masterpiece that will stay with you long after the credits.",
        "Not funny at all, the jokes fall completely flat.",
        "Visually stunning with a captivating story to match.",
        "Poorly paced and overlong, nothing happens for the first hour.",
    ];

    static string SstModelDir
        => Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..",
            "samples", "data", "distilbert_sst");

    static void ResolveSstFiles(out string safetensorsPath, out string modelDir)
    {
        modelDir = SstModelDir;
        safetensorsPath = Path.Combine(modelDir, "model.safetensors");
        string configPath = Path.Combine(modelDir, "config.json");
        string vocabPath = Path.Combine(modelDir, "vocab.txt");
        if (!File.Exists(safetensorsPath) || !File.Exists(configPath) || !File.Exists(vocabPath))
            Assert.Ignore("DistilBERT SST-2 weight files not found; skipping precision inference test.");
    }

    [Test]
    public void DistilBertSst_HalfInference_PreservesFloatArgmax()
    {
        ResolveSstFiles(out string safetensorsPath, out string modelDir);

        Dictionary<string, (float[] Data, int[] Shape)> f32;
        Dictionary<string, (Half[] Data, int[] Shape)> half;
        try
        {
            f32 = SafeTensorsLoader.Read(safetensorsPath);
            half = SafeTensorsLoader.Read<Half>(safetensorsPath);
        }
        catch (NotSupportedException ex)
        {
            Assert.Ignore($"Cannot load DistilBERT SST-2 weights: {ex.Message}");
            return;
        }

        var config = DistilBertConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json"))).ToBertConfig();
        var f32Model = new DistilBertForSequenceClassification<float>(config, numClasses: 2);
        f32Model.LoadWeights(f32);
        var halfModel = new DistilBertForSequenceClassification<Half>(config, numClasses: 2);
        halfModel.LoadWeights(half);

        var tokenizer = MiniLMTokenizer.Load(Path.Combine(modelDir, "vocab.txt"));
        int agree = 0;
        for (int s = 0; s < CompareSentences.Length; s++)
        {
            var (f32Ids, f32Mask, _) = MiniLMTokenizer.Encode(tokenizer, CompareSentences[s], maxLen: 128);
            var f32Logits = f32Model.Forward(Array.ConvertAll(f32Ids, x => (int)x),
                GradientUtils.Constant(f32Mask), 1, f32Ids.Length);

            var (ids, mask, _) = MiniLMTokenizer.Encode(tokenizer, CompareSentences[s], maxLen: 128);
            var halfLogits = halfModel.Forward(Array.ConvertAll(ids, x => (int)x),
                GradientUtils.Constant(Array.ConvertAll(mask, x => (Half)x)), 1, ids.Length);

            if (ArgMax(f32Logits.Data[0], f32Logits.Data[1]) == ArgMax((float)halfLogits.Data[0], (float)halfLogits.Data[1]))
                agree++;
        }

        Assert.That(agree, Is.EqualTo(CompareSentences.Length),
            "Half (fp16) inference must preserve the F32 argmax for every sentence.");
    }

    [Test]
    public void DistilBertSst_BFloat16Inference_PreservesFloatArgmax()
    {
        ResolveSstFiles(out string safetensorsPath, out string modelDir);

        Dictionary<string, (float[] Data, int[] Shape)> f32;
        Dictionary<string, (BFloat16[] Data, int[] Shape)> bf16;
        try
        {
            f32 = SafeTensorsLoader.Read(safetensorsPath);
            bf16 = SafeTensorsLoader.Read<BFloat16>(safetensorsPath);
        }
        catch (NotSupportedException ex)
        {
            Assert.Ignore($"Cannot load DistilBERT SST-2 weights: {ex.Message}");
            return;
        }

        var config = DistilBertConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json"))).ToBertConfig();
        var f32Model = new DistilBertForSequenceClassification<float>(config, numClasses: 2);
        f32Model.LoadWeights(f32);
        var bf16Model = new DistilBertForSequenceClassification<BFloat16>(config, numClasses: 2);
        bf16Model.LoadWeights(bf16);

        var tokenizer = MiniLMTokenizer.Load(Path.Combine(modelDir, "vocab.txt"));
        int agree = 0;
        for (int s = 0; s < CompareSentences.Length; s++)
        {
            var (f32Ids, f32Mask, _) = MiniLMTokenizer.Encode(tokenizer, CompareSentences[s], maxLen: 128);
            var f32Logits = f32Model.Forward(Array.ConvertAll(f32Ids, x => (int)x),
                GradientUtils.Constant(f32Mask), 1, f32Ids.Length);

            var (ids, mask, _) = MiniLMTokenizer.Encode(tokenizer, CompareSentences[s], maxLen: 128);
            var bf16Logits = bf16Model.Forward(Array.ConvertAll(ids, x => (int)x),
                GradientUtils.Constant(Array.ConvertAll(mask, x => (BFloat16)x)), 1, ids.Length);

            if (ArgMax(f32Logits.Data[0], f32Logits.Data[1]) == ArgMax((float)bf16Logits.Data[0], (float)bf16Logits.Data[1]))
                agree++;
        }

        Assert.That(agree, Is.EqualTo(CompareSentences.Length),
            "BFloat16 inference must preserve the F32 argmax for every sentence.");
    }

    static int ArgMax(float a, float b) => b > a ? 1 : 0;
}
