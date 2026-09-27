using Nivara.AutoDiff;
using Nivara.Samples;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Covers the additive mask that ModernBERT needs and a plain causal mask does not: a
/// bidirectional <em>band</em> fused together with right padding, into one dense <c>[L, L]</c>
/// tensor of 0 / -inf.
/// </summary>
/// <remarks>
/// The reference semantics are HuggingFace's <c>create_bidirectional_sliding_window_mask</c>
/// (<c>masking_utils.py:141</c>), which builds <c>abs(q_idx - kv_idx) &lt;= sliding_window</c> where
/// <c>config.sliding_window = local_attention // 2</c>. The <c>+1</c> in
/// <c>ModernBertAttention.__init__</c> is flash-attention window-size semantics and must not be
/// applied here; the arithmetic in <see cref="MaskCount_MatchesHuggingFaceForTheFixtureShape"/>
/// pins the exact entry counts for the parity fixture's shape.
/// </remarks>
[TestFixture]
public class ModernBertMaskTests
{
    static float[] MaskData(ReverseGradTensor<float> mask, int seqLen)
    {
        Assert.That(mask.Data.TryGetSpan(out var span), Is.True, "mask storage should expose a span");
        var data = span.ToArray();
        Assert.That(data.Length, Is.EqualTo(seqLen * seqLen));
        return data;
    }

    static bool Visible(float[] data, int seqLen, int query, int key)
        => data[query * seqLen + key] == 0.0f;

    [Test]
    public void Build_NegativeBand_AppliesNoDistanceLimit()
    {
        const int seqLen = 8;
        var data = MaskData(ModernBertMasks.Build<float>(seqLen, -1, seqLen), seqLen);

        for (int i = 0; i < seqLen; i++)
            for (int j = 0; j < seqLen; j++)
                Assert.That(Visible(data, seqLen, i, j), Is.True, $"query {i}, key {j}");
    }

    [Test]
    public void Build_BandBoundary_IsInclusive()
    {
        const int seqLen = 16;
        const int band = 3;
        var data = MaskData(ModernBertMasks.Build<float>(seqLen, band, seqLen), seqLen);

        for (int i = 0; i < seqLen; i++)
        {
            Assert.That(Visible(data, seqLen, i, i), Is.True, "a query always sees itself");
            if (i + band < seqLen)
                Assert.That(Visible(data, seqLen, i, i + band), Is.True, $"query {i} to key {i + band} (|d| == band)");
            if (i - band >= 0)
                Assert.That(Visible(data, seqLen, i, i - band), Is.True, $"query {i} to key {i - band} (|d| == band)");
            if (i + band + 1 < seqLen)
                Assert.That(Visible(data, seqLen, i, i + band + 1), Is.False, $"query {i} to key {i + band + 1}");
            if (i - band - 1 >= 0)
                Assert.That(Visible(data, seqLen, i, i - band - 1), Is.False, $"query {i} to key {i - band - 1}");
        }
    }

    [Test]
    public void Build_PaddedKeys_AreSuppressedForEveryQuery()
    {
        const int seqLen = 10;
        const int validLength = 4;
        var data = MaskData(ModernBertMasks.Build<float>(seqLen, -1, validLength), seqLen);

        for (int i = 0; i < seqLen; i++)
        {
            for (int j = 0; j < validLength; j++)
                Assert.That(Visible(data, seqLen, i, j), Is.True, $"query {i}, valid key {j}");
            for (int j = validLength; j < seqLen; j++)
                Assert.That(Visible(data, seqLen, i, j), Is.False, $"query {i}, padded key {j}");
        }
    }

    [Test]
    public void Build_PaddedQueries_StillAttendToValidKeysInAFullAttentionLayer()
    {
        // The padding mask is a *key* mask, not a query mask: a padded query row still sees every
        // valid key, which is what HuggingFace's bidirectional overlay does. Rows are only fully
        // masked when the band also puts every valid key out of reach.
        const int seqLen = 10;
        const int validLength = 4;
        var data = MaskData(ModernBertMasks.Build<float>(seqLen, -1, validLength), seqLen);

        for (int i = validLength; i < seqLen; i++)
        {
            for (int j = 0; j < validLength; j++)
                Assert.That(Visible(data, seqLen, i, j), Is.True, $"padded query {i}, valid key {j}");
            for (int j = validLength; j < seqLen; j++)
                Assert.That(Visible(data, seqLen, i, j), Is.False, $"padded query {i}, padded key {j}");
        }
    }

    [Test]
    public void Build_ClipsBandToTheValidRange()
    {
        const int seqLen = 6;
        const int validLength = 3;
        const int band = 10;
        var data = MaskData(ModernBertMasks.Build<float>(seqLen, band, validLength), seqLen);

        for (int i = 0; i < seqLen; i++)
            for (int j = 0; j < validLength; j++)
                Assert.That(Visible(data, seqLen, i, j), Is.True, $"query {i}, valid key {j}");
    }

    [Test]
    public void Build_SlidingLayer_LeavesDistantRowsFullyMasked()
    {
        // With the parity fixture's 26 valid tokens and a band of 64, every query at or past
        // validLength + band has its whole valid range outside the band.
        const int seqLen = 128;
        const int validLength = 26;
        const int band = 64;
        var data = MaskData(ModernBertMasks.Build<float>(seqLen, band, validLength), seqLen);

        for (int i = 0; i < validLength; i++)
            Assert.That(Visible(data, seqLen, i, 0), Is.True, $"query {i} still sees the first valid key");
        for (int i = validLength + band; i < seqLen; i++)
            for (int j = 0; j < validLength; j++)
                Assert.That(Visible(data, seqLen, i, j), Is.False, $"query {i}, valid key {j}");
    }

    [Test]
    public void Build_ShapeIsSquare()
    {
        var mask = ModernBertMasks.Build<float>(12, 4, 8);
        Assert.That(mask.Shape, Is.EqualTo(new[] { 12, 12 }));
    }

    [Test]
    public void Build_SuppressedEntriesAreNegativeInfinity()
    {
        const int seqLen = 6;
        var data = MaskData(ModernBertMasks.Build<float>(seqLen, 2, seqLen), seqLen);

        foreach (float value in data)
        {
            if (value != 0.0f)
                Assert.That(float.IsNegativeInfinity(value), Is.True, $"expected 0 or -inf, got {value}");
        }
    }

    [Test]
    public void MaskCount_MatchesHuggingFaceForTheFixtureShape()
    {
        const int seqLen = 128;
        const int validLength = 26;
        const int band = 64;

        // Full: 128 queries x 102 padded keys = 13056.
        Assert.That(CountSuppressed(ModernBertMasks.Build<float>(seqLen, -1, validLength), seqLen), Is.EqualTo(13056));

        // Sliding: the same 13056, plus 1313 valid keys pushed out of band
        // (sum 1..25 over queries 65..89, then 26 for each of queries 90..127 = 325 + 988).
        Assert.That(CountSuppressed(ModernBertMasks.Build<float>(seqLen, band, validLength), seqLen), Is.EqualTo(14369));
    }

    [Test]
    public void Build_SequenceLengthBeyondTheDenseLimit_Throws()
    {
        int tooLong = ModernBertMasks.MaxDenseLength + 1;
        var ex = Assert.Throws<ArgumentOutOfRangeException>(
            () => ModernBertMasks.Build<float>(tooLong, -1, tooLong));
        Assert.That(ex!.Message, Does.Contain("banded attention kernel"));
    }

    [Test]
    public void Build_InvalidArguments_Throw()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ModernBertMasks.Build<float>(0, -1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => ModernBertMasks.Build<float>(8, -1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => ModernBertMasks.Build<float>(8, -1, 9));
    }

    static int CountSuppressed(ReverseGradTensor<float> mask, int seqLen)
    {
        var data = MaskData(mask, seqLen);
        int count = 0;
        foreach (float value in data)
            if (value != 0.0f)
                count++;
        return count;
    }

    //  ── config derivation ──────────────────────────────────────────────────────

    [Test]
    public void Config_DerivesLayerTypesAndTheSlidingHalfWindow()
    {
        const string json = """
            {
              "hidden_size": 1024, "num_attention_heads": 16, "num_hidden_layers": 28,
              "intermediate_size": 2624, "local_attention": 128,
              "global_attn_every_n_layers": 3,
              "global_rope_theta": 160000, "local_rope_theta": 10000
            }
            """;

        var config = ModernBertConfig.FromJson(json);

        Assert.That(config.HeadDim, Is.EqualTo(64));
        Assert.That(config.SlidingWindow, Is.EqualTo(64));
        Assert.That(config.NumHiddenLayers, Is.EqualTo(28));
        Assert.That(config.LayerTypes.Count(n => n == "full_attention"), Is.EqualTo(10));
        Assert.That(config.LayerTypes.Count(n => n == "sliding_attention"), Is.EqualTo(18));
        Assert.That(config.IsFullAttention(0), Is.True);
        Assert.That(config.IsFullAttention(27), Is.True);
        Assert.That(config.IsFullAttention(1), Is.False);
        Assert.That(config.IsFullAttention(26), Is.False);
    }

    [Test]
    public void Config_RopeThetaAndBand_ArePerLayerType()
    {
        const string json = """
            {
              "num_hidden_layers": 6, "local_attention": 128, "global_attn_every_n_layers": 3,
              "global_rope_theta": 160000, "local_rope_theta": 10000
            }
            """;

        var config = ModernBertConfig.FromJson(json);

        Assert.That(config.RopeTheta(0), Is.EqualTo(160000f));
        Assert.That(config.RopeTheta(1), Is.EqualTo(10000f));
        Assert.That(config.Band(0), Is.EqualTo(-1));
        Assert.That(config.Band(1), Is.EqualTo(64));
    }

    [Test]
    public void Config_PrefersExplicitLayerTypesAndRopeParameters()
    {
        const string json = """
            {
              "num_hidden_layers": 2, "local_attention": 128,
              "layer_types": ["sliding_attention", "full_attention"],
              "rope_parameters": {
                "full_attention": { "rope_theta": 50000.0 },
                "sliding_attention": { "rope_theta": 500.0 }
              }
            }
            """;

        var config = ModernBertConfig.FromJson(json);

        Assert.That(config.IsFullAttention(0), Is.False);
        Assert.That(config.IsFullAttention(1), Is.True);
        Assert.That(config.RopeTheta(0), Is.EqualTo(500f));
        Assert.That(config.RopeTheta(1), Is.EqualTo(50000f));
    }

    [Test]
    public void Config_UnsupportedActivation_Throws()
    {
        const string json = """{ "hidden_activation": "silu" }""";

        var ex = Assert.Throws<NotSupportedException>(() => ModernBertConfig.FromJson(json));
        Assert.That(ex!.Message, Does.Contain("gelu"));
    }
}
