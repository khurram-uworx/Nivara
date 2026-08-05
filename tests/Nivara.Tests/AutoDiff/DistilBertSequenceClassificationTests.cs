using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
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

    [Test]
    public void ForwardBatched_BatchedEqualsIndependentSingleSequenceRuns()
    {
        // The batched path isolates sequences per batch element (previously via a
        // block-diagonal mask on the flattened input). Batched B=2 must produce the
        // same logits as running each sequence independently at B=1.
        var tensors = FabricateWeights();
        var config = TinyConfig() with { NumHiddenLayers = 1 };
        using var model = new DistilBertForSequenceClassification<float>(config, NumClasses);
        model.LoadWeights(tensors);

        int L = 8;
        var ids1 = new float[L];
        var ids2 = new float[L];
        for (int i = 0; i < L; i++)
        {
            ids1[i] = (i * 2 + 1) % Vocab;
            ids2[i] = (i * 3 + 2) % Vocab;
        }
        var ones = new float[L];
        Array.Fill(ones, 1f);

        // Sequence 2 has its trailing half padded to exercise key-position masking.
        var mask2 = new float[L];
        Array.Fill(mask2, 1f, 0, L / 2);

        var single1 = model.Forward(GradientUtils.Constant(ids1), GradientUtils.Constant(ones), 1, L);
        var single2 = model.Forward(GradientUtils.Constant(ids2), GradientUtils.Constant(mask2), 1, L);

        var concatIds = new float[2 * L];
        var concatMask = new float[2 * L];
        Array.Copy(ids1, 0, concatIds, 0, L);
        Array.Copy(ids2, 0, concatIds, L, L);
        Array.Copy(ones, 0, concatMask, 0, L);
        Array.Copy(mask2, 0, concatMask, L, L);
        var batched = model.Forward(GradientUtils.Constant(concatIds), GradientUtils.Constant(concatMask), 2, L);

        Assert.That(batched.Shape, Is.EqualTo(new[] { 2, NumClasses }));
        for (int c = 0; c < NumClasses; c++)
        {
            Assert.That(batched[c], Is.EqualTo(single1[c]).Within(1e-4f));
            Assert.That(batched[NumClasses + c], Is.EqualTo(single2[c]).Within(1e-4f));
        }
    }

    [Test]
    public void ForwardBatched_Backward_ProducesParameterGradients()
    {
        // Regression for the training path: ForwardBatched reshapes Q/K/V from
        // [B*L, D] to [B, L, D] in place and feeds BatchedMultiHeadAttention, whose
        // backward accumulates into the reshaped tensors. Gradient must flow all the
        // way back through the MatMul projections to every parameter.
        var tensors = FabricateWeights();
        var config = TinyConfig() with { NumHiddenLayers = 1 };
        using var model = new DistilBertForSequenceClassification<float>(config, NumClasses);
        model.LoadWeights(tensors);

        int L = 8;
        var ids = new float[2 * L];
        var mask = new float[2 * L];
        var labels = new int[] { 0, 1 };
        for (int i = 0; i < 2 * L; i++)
        {
            ids[i] = (i * 2 + 1) % Vocab;
            mask[i] = 1f;
        }

        using (GradientUtils.Grad())
        {
            var logits = model.Forward(GradientUtils.Constant(ids), GradientUtils.Constant(mask), 2, L);
            var loss = new CrossEntropyLoss<float>().Forward(logits, labels);
            loss.Backward();
        }

        var parameters = model.GetParameters().Values;
        Assert.That(parameters, Is.Not.Empty);
        foreach (var param in parameters)
        {
            Assert.That(param.Tensor.Grad, Is.Not.Null,
                $"Parameter '{param.Name}' received no gradient through the batched attention path.");
            var grad = param.Tensor.Grad!;
            for (int i = 0; i < grad.Length; i++)
                Assert.That(float.IsFinite(grad[i]), Is.True,
                    $"Parameter '{param.Name}' has a non-finite gradient at index {i}.");
        }
    }
}
