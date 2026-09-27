using Nivara.Samples;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Composition tests for the Laya decision head, covering the parts whose shape is fixed by the
/// reference rather than by the checkpoint: the head count, the marker gather, the four act-head
/// features, and the ReLU-not-GELU activation.
/// </summary>
/// <remarks>
/// These run on a fabricated checkpoint of the right shapes, so they are fast and always run. The
/// numeric parity of the whole head against the wheel's <c>DecisionModel</c> is a separate gate
/// (<c>Python/laya_compare.py</c> plus <c>laya compare</c>), which needs the 822 MB checkpoint.
/// </remarks>
[TestFixture]
public class LayaDecisionHeadTests
{
    const int Hidden = 64;
    const int Intermediate = 4 * Hidden;
    const int ActHidden = LayaDecisionHead<float>.ActHeadHidden;
    const int FeatCount = LayaHeadOutput.FeatureCount;

    /// <summary>
    /// Deterministic pseudo-random weights. Seeded rather than random so a failure is reproducible;
    /// the values only need to be non-degenerate, since nothing here asserts magnitudes.
    /// </summary>
    static float[] Weights(int count, ref int seed)
    {
        var state = (uint)(0x9E3779B9u * (uint)++seed);
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            state = state * 1664525u + 1013904223u;
            values[i] = ((state >> 8) / (float)(1 << 24)) * 2.0f - 1.0f;
        }
        return values;
    }

    /// <summary>
    /// A synthetic checkpoint laid out exactly as the real one, at a width small enough for a unit
    /// test. Going through <see cref="LayaDecisionHead{T}.LoadWeights{TModel, TWeight}"/> rather than
    /// poking fields means these tests also pin the checkpoint key names and the fused QKV slicing.
    /// </summary>
    static Dictionary<string, (float[], int[])> SyntheticTensors(
        int headLayers = 2, int actClasses = 2, int? includeTemperature = null)
    {
        int seed = 0;
        var tensors = new Dictionary<string, (float[], int[])>();

        void Add(string key, int rows, int cols)
            => tensors[key] = (Weights(rows * cols, ref seed), [rows, cols]);

        void AddVec(string key, int length)
            => tensors[key] = (Weights(length, ref seed), [length]);

        tensors["type_emb.weight"] = (Weights(3 * Hidden, ref seed), [3, Hidden]);
        for (int i = 0; i < headLayers; i++)
        {
            string p = $"head.layers.{i}";
            AddVec($"{p}.norm1.weight", Hidden);
            AddVec($"{p}.norm1.bias", Hidden);
            AddVec($"{p}.norm2.weight", Hidden);
            AddVec($"{p}.norm2.bias", Hidden);
            Add($"{p}.self_attn.in_proj_weight", 3 * Hidden, Hidden);
            AddVec($"{p}.self_attn.in_proj_bias", 3 * Hidden);
            Add($"{p}.self_attn.out_proj.weight", Hidden, Hidden);
            AddVec($"{p}.self_attn.out_proj.bias", Hidden);
            Add($"{p}.linear1.weight", Intermediate, Hidden);
            AddVec($"{p}.linear1.bias", Intermediate);
            Add($"{p}.linear2.weight", Hidden, Intermediate);
            AddVec($"{p}.linear2.bias", Hidden);
        }

        AddVec("scorer.0.weight", Hidden);
        AddVec("scorer.0.bias", Hidden);
        Add("scorer.1.weight", Hidden, Hidden);
        AddVec("scorer.1.bias", Hidden);
        Add("scorer.3.weight", 1, Hidden);
        AddVec("scorer.3.bias", 1);

        Add("act_head.0.weight", ActHidden, Hidden + FeatCount);
        AddVec("act_head.0.bias", ActHidden);
        Add("act_head.2.weight", actClasses, ActHidden);
        AddVec("act_head.2.bias", actClasses);

        if (includeTemperature is int t)
        {
            var values = Weights(3, ref seed);
            for (int i = 0; i < 3; i++) values[i] = t + i;
            tensors["temperature"] = (values, [3]);
        }

        return tensors;
    }

    static LayaDecisionHead<float> NewHead(
        int headLayers = 2, int actClasses = 2, int? temperature = null)
        => LayaDecisionHead<float>.LoadWeights<float, float>(
            SyntheticTensors(headLayers, actClasses, temperature), headLayers, actClasses);

    /// <summary>A plausible encoder output: distinct rows, nothing degenerate.</summary>
    static float[,] HiddenState(int seqLen)
    {
        int seed = 777;
        var row = Weights(Hidden, ref seed);
        var values = new float[seqLen, Hidden];
        for (int r = 0; r < seqLen; r++)
            for (int c = 0; c < Hidden; c++)
                values[r, c] = row[c] + r;
        return values;
    }

    static Nivara.AutoDiff.ReverseGradTensor<float> AsTensor(float[,] values, int rows, int cols)
    {
        var flat = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                flat[r * cols + c] = values[r, c];
        return Nivara.AutoDiff.ReverseGradTensor<float>.FromMatrix(flat, rows, cols);
    }

    [Test]
    public void NumHeads_FollowsTheReferenceFloorOfSixtyFourPerHead()
    {
        // max(1, d // 64): the shipped 1024 gives 16 heads of 64, which is also the head width the GPU
        // buffers cap at. A narrow test width must not be read as "1 head" or, worse, 0.
        Assert.That(new LayaDecisionHead<float>(1024).NumHeads, Is.EqualTo(16));
        Assert.That(new LayaDecisionHead<float>(768).NumHeads, Is.EqualTo(12));
        Assert.That(new LayaDecisionHead<float>(64).NumHeads, Is.EqualTo(1));
        Assert.That(new LayaDecisionHead<float>(32).NumHeads, Is.EqualTo(1));
    }

    [Test]
    public void Constructor_RejectsAWidthThatDoesNotDivideIntoWholeHeads()
    {
        // Any width below 128 gives 1 head by the max(1, ...), and 1 divides everything, so the
        // rejection cannot fire there. It fires when d // 64 is 2 or more and does not divide d:
        // 200 // 64 == 3 and 200 % 3 == 2, so 3 heads of 66.67 is not a thing. The shipped 1024 gives
        // 16 heads of 64 and is fine.
        Assert.That(new LayaDecisionHead<float>(96).NumHeads, Is.EqualTo(1));
        Assert.That(new LayaDecisionHead<float>(40).NumHeads, Is.EqualTo(1));
        Assert.That(new LayaDecisionHead<float>(192).NumHeads, Is.EqualTo(3));
        Assert.Throws<ArgumentException>(() => new LayaDecisionHead<float>(200, headLayerCount: 1));
    }

    [Test]
    public void HeadLayerCount_ZeroIsAccepted()
    {
        // head_layers is a config key and `if head_layers > 0 else None` is the reference's own
        // branch, so zero layers is a real configuration rather than a degenerate one.
        var head = new LayaDecisionHead<float>(Hidden, headLayerCount: 0);
        Assert.That(head.HeadLayerCount, Is.EqualTo(0));
    }

    [Test]
    public void CheckpointTemperature_DefaultsToOneAndIsReadWhenPresent()
    {
        var bare = new LayaDecisionHead<float>(Hidden, headLayerCount: 0);
        Assert.That(bare.CheckpointTemperature, Has.Count.EqualTo(3));
        Assert.That(bare.CheckpointTemperature, Is.All.EqualTo(1.0f),
            "the reference registers torch.ones(3); a checkpoint with no buffer must stay at that");

        var loaded = NewHead(headLayers: 1, temperature: 100);
        Assert.That(loaded.CheckpointTemperature[0], Is.EqualTo(100.0f));
        Assert.That(loaded.CheckpointTemperature[2], Is.EqualTo(102.0f));
    }

    [Test]
    public void LoadWeights_ThrowsWithoutATypeEmbedding()
    {
        // A stock ModernBERT checkpoint is the likely mistake, and it would otherwise load as a head
        // with all-identity scorer weights and produce confident nonsense.
        var ex = Assert.Throws<KeyNotFoundException>(() =>
            LayaDecisionHead<float>.LoadWeights<float, float>(
                new Dictionary<string, (float[], int[])>(), headLayerCount: 0));
        Assert.That(ex!.Message, Does.Contain("type_emb"));
    }

    [Test]
    public void Forward_ProducesOneLogitPerMarker()
    {
        var head = NewHead();
        var hidden = HiddenState(12);

        var output = head.Forward(AsTensor(hidden, 12, Hidden), LayaQuestionType.Choice, [2, 5, 9], 12);

        Assert.That(output.Logits, Has.Length.EqualTo(3));
        Assert.That(output.MarkerProbabilities, Has.Length.EqualTo(3));
        Assert.That(output.Features, Has.Length.EqualTo(FeatCount));
        Assert.That(output.ActionClassCount, Is.EqualTo(2));
    }

    [Test]
    public void Forward_MarkerProbabilitiesFormADistribution()
    {
        var head = NewHead();
        var hidden = HiddenState(12);

        var output = head.Forward(AsTensor(hidden, 12, Hidden), LayaQuestionType.Choice, [2, 5, 9], 12);

        Assert.That(output.MarkerProbabilities, Is.All.InRange(0.0, 1.0));
        Assert.That(output.MarkerProbabilities.Sum(), Is.EqualTo(1.0).Within(1e-5));
    }

    [Test]
    public void Forward_FeaturesUseTheReferenceLayoutAndRanges()
    {
        var head = NewHead();
        var hidden = HiddenState(12);

        var output = head.Forward(AsTensor(hidden, 12, Hidden), LayaQuestionType.Choice, [2, 5, 9], 12);

        // [top1, top1 - top2, ent, max(2, k) / 255] with k = 3 here.
        double top1 = output.MarkerProbabilities.Max();
        double top2 = output.MarkerProbabilities.OrderByDescending(v => v).Skip(1).First();

        Assert.That(output.Features[LayaHeadOutput.FeatureTop1], Is.EqualTo(top1).Within(1e-6));
        Assert.That(output.Features[LayaHeadOutput.FeatureGap], Is.EqualTo(top1 - top2).Within(1e-6));
        Assert.That(output.Features[LayaHeadOutput.FeatureOptionCount], Is.EqualTo(3.0 / 255.0).Within(1e-12));
        Assert.That(output.Features[LayaHeadOutput.FeatureEntropy], Is.InRange(0.0, 1.0),
            "normalized entropy is 0 for a point mass and 1 for a uniform distribution");
    }

    [Test]
    public void Forward_ClampsTheOptionCountFeatureAtTwoForASingleMarker()
    {
        var head = NewHead();
        var hidden = HiddenState(8);

        var output = head.Forward(AsTensor(hidden, 8, Hidden), LayaQuestionType.Choice, [3], 8);

        // clamp(min=2) is load-bearing here: log(1) is -inf, so the entropy divisor must not be 1.
        Assert.That(output.Features[LayaHeadOutput.FeatureOptionCount], Is.EqualTo(2.0 / 255.0).Within(1e-12));
        Assert.That(double.IsFinite(output.Features[LayaHeadOutput.FeatureEntropy]), Is.True);
    }

    [Test]
    public void Forward_SingleMarkerPadsTopTwoWithZero()
    {
        var head = NewHead();
        var hidden = HiddenState(8);

        var output = head.Forward(AsTensor(hidden, 8, Hidden), LayaQuestionType.Choice, [3], 8);

        // Softmax over one logit is 1.0 whatever the logit is, so the reference's zero pad gives
        // top1 - top2 == 1.0, the same "fully decided" signal an unambiguous question produces.
        Assert.That(output.MarkerProbabilities[0], Is.EqualTo(1.0).Within(1e-9));
        Assert.That(output.Features[LayaHeadOutput.FeatureTop1], Is.EqualTo(1.0).Within(1e-9));
        Assert.That(output.Features[LayaHeadOutput.FeatureGap], Is.EqualTo(1.0).Within(1e-9));
        Assert.That(output.Features[LayaHeadOutput.FeatureEntropy], Is.EqualTo(0.0).Within(1e-9));
    }

    [Test]
    public void Forward_PaddingDoesNotChangeAValidMarkersLogit()
    {
        var head = NewHead();
        int valid = 6;
        var hidden = HiddenState(6);

        var shortRun = head.Forward(AsTensor(hidden, valid, Hidden), LayaQuestionType.Choice, [1, 4], valid);

        // Same real tokens, more padding after them. A valid query never reads a padded key, so the
        // marker logits must be bit-identical; if they are not, the key-padding mask is inverted or
        // absent. This is the same check the ModernBERT GPU gate makes about its -inf clamp.
        var padded = new float[16, Hidden];
        for (int r = 0; r < valid; r++)
            for (int c = 0; c < Hidden; c++)
                padded[r, c] = hidden[r, c];
        int seed = 4242;
        var noise = Weights((16 - valid) * Hidden, ref seed);
        for (int i = 0; i < noise.Length; i++)
            padded[valid + i / Hidden, i % Hidden] = noise[i] * 10.0f;

        var paddedRun = head.Forward(AsTensor(padded, 16, Hidden), LayaQuestionType.Choice, [1, 4], valid);

        Assert.That(paddedRun.Logits, Is.EqualTo(shortRun.Logits).Within(1e-5));
        Assert.That(paddedRun.ActionProbability, Is.EqualTo(shortRun.ActionProbability).Within(1e-5));
    }

    [Test]
    public void Forward_UsesTheQuestionTypeEmbeddingRow()
    {
        var head = NewHead();
        var hidden = HiddenState(10);

        var choice = head.Forward(AsTensor(hidden, 10, Hidden), LayaQuestionType.Choice, [2, 5], 10);
        var score = head.Forward(AsTensor(hidden, 10, Hidden), LayaQuestionType.Score, [2, 5], 10);
        var noul = head.Forward(AsTensor(hidden, 10, Hidden), LayaQuestionType.Noul, [2, 5], 10);

        Assert.That(choice.Logits, Is.Not.EqualTo(score.Logits));
        Assert.That(choice.Logits, Is.Not.EqualTo(noul.Logits));
    }

    [Test]
    public void Forward_ZeroHeadLayersStillScoresTheTypeEmbedding()
    {
        // The `head_layers: 0` configuration runs encoder -> +type_emb -> scorer with no stack, so
        // the two head layers must actually contribute when they are configured.
        var none = NewHead(headLayers: 0);
        var stacked = NewHead(headLayers: 2);
        var hidden = HiddenState(10);

        var withoutStack = none.Forward(AsTensor(hidden, 10, Hidden), LayaQuestionType.Choice, [2, 5], 10);
        var withStack = stacked.Forward(AsTensor(hidden, 10, Hidden), LayaQuestionType.Choice, [2, 5], 10);

        Assert.That(withoutStack.Logits, Is.Not.EqualTo(withStack.Logits));
    }

    [Test]
    public void Forward_RejectsAHeadWidthThatDoesNotMatchTheEncoderOutput()
    {
        var head = NewHead(headLayers: 1);
        int tooWide = Hidden + 4;
        int seed = 31337;
        var wide = Weights(8 * tooWide, ref seed);

        Assert.Throws<ArgumentException>(() =>
            head.Forward(Nivara.AutoDiff.ReverseGradTensor<float>.FromMatrix(wide, 8, tooWide),
                LayaQuestionType.Choice, [2], 8));
    }

    [Test]
    public void Forward_RejectsAnEmptyMarkerList()
    {
        var head = NewHead(headLayers: 1);
        var hidden = HiddenState(8);

        // Every option's [MASK] pushed out by the budget: the reference's agent rejects the question
        // rather than scoring zero options.
        Assert.Throws<ArgumentException>(() =>
            head.Forward(AsTensor(hidden, 8, Hidden), LayaQuestionType.Choice, [], 8));
    }

    [Test]
    public void Forward_RejectsAValidLengthOutsideTheSequence()
    {
        var head = NewHead(headLayers: 1);
        var hidden = HiddenState(8);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            head.Forward(AsTensor(hidden, 8, Hidden), LayaQuestionType.Choice, [2], 9));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            head.Forward(AsTensor(hidden, 8, Hidden), LayaQuestionType.Choice, [2], 0));
    }

    [Test]
    public void Forward_RejectsAMarkerBeyondTheEndOfTheSequence()
    {
        var head = NewHead(headLayers: 1);
        var hidden = HiddenState(8);

        // Gather validates the index; a marker at or past seqLen means the prompt and the head
        // disagree about the sequence length.
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            head.Forward(AsTensor(hidden, 8, Hidden), LayaQuestionType.Choice, [2, 8], 8));
    }

    [Test]
    public void HeadLayer_AppliesReluNotGeluToItsFeedForward()
    {
        // nn.TransformerEncoderLayer defaults to activation=ReLU while every other block in the
        // stack uses GELU, so the shape of linear1/linear2 does not disambiguate and the default is
        // easy to carry over from a neighbouring model. The two differ by ~0.17 per negative unit,
        // which is far above any tolerance here, so pinning the exact output pins the activation.
        var layer = new LayaHeadLayer<float>(Hidden, Intermediate, 4, 1e-5f);

        // linear1 alternates sign down its output rows, and within a row flips sign halfway across,
        // so its dot with a normalized row is a predictable sum of the first half of the row minus
        // the second half. A uniform row would give exactly zero, because a normalized row has zero
        // mean — which is why the obvious probe of a constant input cannot detect anything at all.
        var w1 = new float[Intermediate * Hidden];
        for (int i = 0; i < Intermediate; i++)
            for (int c = 0; c < Hidden; c++)
                w1[i * Hidden + c] = (i % 2 == 0 ? 1.0f : -1.0f) * (c < Hidden / 2 ? 1.0f : -1.0f);
        var w2 = new float[Hidden * Intermediate];
        Array.Fill(w2, 1.0f);

        StateDictLoader.LoadLinear(layer.linear1, new Dictionary<string, (float[], int[])>
        {
            ["linear1.weight"] = (w1, [Intermediate, Hidden]),
            ["linear1.bias"] = (new float[Intermediate], [Intermediate])
        }, "linear1");
        StateDictLoader.LoadLinear(layer.linear2, new Dictionary<string, (float[], int[])>
        {
            ["linear2.weight"] = (w2, [Hidden, Intermediate]),
            ["linear2.bias"] = (new float[Hidden], [Hidden])
        }, "linear2");

        // A single row: attention over one key softmaxes to 1, so the attention output is exactly
        // oProj(v). Zeroing oProj's weight and bias removes it, which leaves the residual as the
        // input and makes the whole layer a plain norm -> linear -> ReLU -> linear -> residual.
        StateDictLoader.LoadLinear(layer.attn.oProj, new Dictionary<string, (float[], int[])>
        {
            ["out_proj.weight"] = (new float[Hidden * Hidden], [Hidden, Hidden]),
            ["out_proj.bias"] = (new float[Hidden], [Hidden])
        }, "out_proj");

        // A non-constant input, so the first LayerNorm does not flatten it to zero.
        var input = new float[Hidden];
        for (int c = 0; c < Hidden; c++)
            input[c] = (float)Math.Cos(c * 0.37);

        var output = layer.Forward(Nivara.AutoDiff.ReverseGradTensor<float>.FromMatrix(input, 1, Hidden));

        // The reference calculation, done independently in scalar form.
        double mean = 0.0;
        for (int c = 0; c < Hidden; c++) mean += input[c];
        mean /= Hidden;
        double variance = 0.0;
        for (int c = 0; c < Hidden; c++)
        {
            double d = input[c] - mean;
            variance += d * d;
        }
        variance /= Hidden;
        double std = Math.Sqrt(variance + 1e-5);

        var normalized = new double[Hidden];
        for (int c = 0; c < Hidden; c++)
            normalized[c] = (input[c] - mean) / std;

        int survivors = 0;
        double survivingSum = 0.0;
        for (int i = 0; i < Intermediate; i++)
        {
            double dot = 0.0;
            for (int c = 0; c < Hidden; c++)
                dot += w1[i * Hidden + c] * normalized[c];
            if (dot <= 0) continue;
            survivors++;
            survivingSum += dot;
        }

        // linear2 has weight 1 everywhere, so every output element gets the same sum of surviving
        // activations, and the residual adds input[c] back at position c.
        Assert.That(survivors, Is.GreaterThan(0).And.LessThan(Intermediate),
            "the probe must activate some rows and suppress others, or it cannot tell ReLU from GELU");

        output.Data.TryGetSpan(out var span);
        for (int c = 0; c < Hidden; c++)
            Assert.That(span[c], Is.EqualTo(input[c] + survivingSum).Within(1e-3),
                "a negative linear1 output must clamp to zero rather than pass through a GELU tail");

        // Cross-checked against torch.nn.LayerNorm(64, eps=1e-5) and relu on the same weights, which
        // gives 196.504028 at position 0. GELU would give 171.740021, so the two are ~25 apart here
        // and the choice cannot be ambiguous.
        Assert.That(span[0], Is.EqualTo(196.50403f).Within(1e-3));
    }
}
