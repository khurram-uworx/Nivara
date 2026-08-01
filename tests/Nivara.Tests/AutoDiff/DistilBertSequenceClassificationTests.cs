using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using Nivara.Samples;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Regression coverage for the DistilBERT sequence-classification head.
/// DistilBertForSequenceClassification applies pre_classifier -> ReLU -> classifier
/// (HF architecture); a previous implementation used GeluExact on the head, which
/// produced logits off by ~0.05 from the reference. Also pins the ForwardBatched
/// input contract: input/attention-mask tensors must be length batchSize*seqLen.
/// </summary>
[TestFixture]
public class DistilBertSequenceClassificationTests
{
    const int Hidden = 8;
    const int Vocab = 16;
    const int MaxPos = 16;
    const int NumClasses = 2;

    static float[] Values(ReverseGradTensor<float> tensor)
    {
        var result = new float[tensor.Length];
        for (int i = 0; i < result.Length; i++)
            result[i] = tensor[i];
        return result;
    }

    static BertConfig TinyConfig() => new()
    {
        HiddenSize = Hidden,
        NumAttentionHeads = 2,
        NumHiddenLayers = 0,
        IntermediateSize = 16,
        MaxPositionEmbeddings = MaxPos,
        VocabSize = Vocab,
        LayerNormEps = 1e-12f,
    };

    static Dictionary<string, (float[] Data, int[] Shape)> FabricateWeights()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>();
        void Add(string key, int[] shape, Func<int, float> value)
        {
            int n = 1;
            foreach (var d in shape) n *= d;
            var data = new float[n];
            for (int i = 0; i < n; i++)
                data[i] = value(i);
            tensors[key] = (data, shape);
        }

        Add("distilbert.embeddings.word_embeddings.weight", [Vocab, Hidden], i => ((i * 7) % 13 - 6) * 0.25f);
        Add("distilbert.embeddings.position_embeddings.weight", [MaxPos, Hidden], i => ((i * 3) % 11 - 5) * 0.2f);
        Add("distilbert.embeddings.LayerNorm.weight", [Hidden], _ => 1f);
        Add("distilbert.embeddings.LayerNorm.bias", [Hidden], _ => 0f);
        Add("pre_classifier.weight", [Hidden, Hidden], i => (i % Hidden) == (i / Hidden) ? 1f : 0f);
        Add("pre_classifier.bias", [Hidden], i => (i % 2) * 0.5f - 0.25f);
        Add("classifier.weight", [NumClasses, Hidden], i => (i / Hidden) == 0 ? 1f : (i % 2 == 0 ? 1f : -1f));
        Add("classifier.bias", [NumClasses], i => i * 0.1f);
        return tensors;
    }

    static (ReverseGradTensor<float> InputIds, ReverseGradTensor<float> Mask) Inputs(int seqLen)
    {
        var tokenIds = new float[seqLen];
        for (int i = 0; i < seqLen; i++)
            tokenIds[i] = (i * 2 + 1) % Vocab;
        var attnMask = new float[seqLen];
        Array.Fill(attnMask, 1f);
        return (GradientUtils.Constant(tokenIds), GradientUtils.Constant(attnMask));
    }

    [Test]
    public void Forward_HeadActivation_MatchesEncoderPlusReluReference()
    {
        var tensors = FabricateWeights();
        var config = TinyConfig();
        using var model = new DistilBertForSequenceClassification<float>(config, NumClasses);
        model.LoadWeights(tensors);

        var (inputIds, mask) = Inputs(seqLen: 8);
        var logits = model.Forward(inputIds, mask, 1, 8);

        // Independent reference: encoder -> cls token -> pre_classifier -> ReLU -> classifier.
        using var encoder = new BertEncoder<float>(config, includeTokenTypeEmbedding: false);
        DistilBertLoader.LoadEncoderWeights<float, float>(encoder, tensors, "distilbert");
        var encoded = encoder.ForwardBatched(inputIds, mask, 1, 8);
        var cls = ReverseGradOperations.Gather(encoded, [0], axis: 0);

        using var preClassifier = new Linear<float>(Hidden, Hidden, bias: true);
        using var classifier = new Linear<float>(Hidden, NumClasses, bias: true);
        StateDictLoader.LoadLinear(preClassifier, tensors, "pre_classifier");
        StateDictLoader.LoadLinear(classifier, tensors, "classifier");

        var h = preClassifier.Forward(cls);
        h = ReverseGradOperations.Relu(h);
        var expected = classifier.Forward(h);

        Assert.That(logits.Shape, Is.EqualTo(new[] { 1, NumClasses }));
        Assert.That(Values(logits), Is.EqualTo(Values(expected)).Within(1e-4f));
    }

    [Test]
    public void Forward_HeadActivation_GeluHead_ProducesDifferentLogits()
    {
        // Sanity check that this fixture actually discriminates ReLU from GELU:
        // if the head were GELU, logits must diverge (negative pre-classifier activations).
        var tensors = FabricateWeights();
        var config = TinyConfig();
        using var model = new DistilBertForSequenceClassification<float>(config, NumClasses);
        model.LoadWeights(tensors);

        var (inputIds, mask) = Inputs(seqLen: 8);
        var logits = model.Forward(inputIds, mask, 1, 8);

        using var encoder = new BertEncoder<float>(config, includeTokenTypeEmbedding: false);
        DistilBertLoader.LoadEncoderWeights<float, float>(encoder, tensors, "distilbert");
        var cls = ReverseGradOperations.Gather(encoder.ForwardBatched(inputIds, mask, 1, 8), [0], axis: 0);

        using var preClassifier = new Linear<float>(Hidden, Hidden, bias: true);
        using var classifier = new Linear<float>(Hidden, NumClasses, bias: true);
        StateDictLoader.LoadLinear(preClassifier, tensors, "pre_classifier");
        StateDictLoader.LoadLinear(classifier, tensors, "classifier");

        var geluHead = classifier.Forward(ReverseGradOperations.GeluExact(preClassifier.Forward(cls)));

        Assert.That(Values(logits), Is.Not.EqualTo(Values(geluHead)).Within(1e-3f));
    }

    [Test]
    public void Forward_FullPaddedInput_ProducesLeafLogits()
    {
        var tensors = FabricateWeights();
        var config = TinyConfig();
        using var model = new DistilBertForSequenceClassification<float>(config, NumClasses);
        model.LoadWeights(tensors);

        // Padded input: seqLen passed equals the tensor length; mask zeroes the padding.
        int maxLen = 16;
        var tokenIds = new float[maxLen];
        for (int i = 0; i < 8; i++)
            tokenIds[i] = (i * 2 + 1) % Vocab;
        var attnMask = new float[maxLen];
        Array.Fill(attnMask, 1f, 0, 8);
        var inputIds = GradientUtils.Constant(tokenIds);
        var mask = GradientUtils.Constant(attnMask);

        var logits = model.Forward(inputIds, mask, 1, maxLen);

        Assert.That(logits.Shape, Is.EqualTo(new[] { 1, NumClasses }));
        Assert.That(logits.IsLeaf, Is.True);
    }

    [Test]
    public void Forward_SeqLenShorterThanInputTensor_Throws()
    {
        var tensors = FabricateWeights();
        var config = TinyConfig();
        using var model = new DistilBertForSequenceClassification<float>(config, NumClasses);
        model.LoadWeights(tensors);

        // The ForwardBatched contract requires input/attention-mask tensors of length
        // batchSize*seqLen. Passing the actual token count with a padded tensor throws.
        int maxLen = 16;
        var tokenIds = new float[maxLen];
        var attnMask = new float[maxLen];
        Array.Fill(attnMask, 1f, 0, 8);
        var inputIds = GradientUtils.Constant(tokenIds);
        var mask = GradientUtils.Constant(attnMask);

        var ex = Assert.Throws<ArgumentException>(() => model.Forward(inputIds, mask, 1, 8));
        Assert.That(ex!.Message, Does.Contain("different lengths"));
    }
}
