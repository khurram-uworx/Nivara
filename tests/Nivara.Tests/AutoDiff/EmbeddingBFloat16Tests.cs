using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Regression coverage for exact integer token-ID handling in <see cref="Embedding{T}"/>.
/// Narrow-precision dtypes (BFloat16 / Half) cannot represent typical vocabularies
/// (30k+), so token IDs must be passed as <see cref="int"/> and never round-tripped
/// through the compute dtype. See the BFloat16 transformer token-index fix.
/// </summary>
[TestFixture]
public class EmbeddingBFloat16Tests
{
    [Test]
    public void Forward_IntTokenIds_PreservedForNonRepresentableIds()
    {
        var emb = new Embedding<BFloat16>(2000, 4);

        // 501 is odd and > 256, so it is NOT exactly representable in BFloat16
        // (integers are exact only up to 256; 257..512 round to even values).
        const int id = 501;

        var exact = emb.Forward(new[] { id });

        var corruptInput = ReverseGradTensor<BFloat16>.FromArray(new BFloat16[] { (BFloat16)id }, requiresGrad: false);
        var corrupt = emb.Forward(corruptInput);

        exact.Data.TryGetSpan(out var exactSpan);
        corrupt.Data.TryGetSpan(out var corruptSpan);

        // The int[] path looks up the true row (501); the BF16-tensor path looks up the
        // rounded row (500 or 502). They must differ, proving integer indices must be
        // passed exactly for correct BF16/Half transformer inference.
        Assert.That(exactSpan[0], Is.Not.EqualTo(corruptSpan[0]));
    }

    [Test]
    public void Forward_IntTokenIds_MatchTensorPathForRepresentableIds()
    {
        var emb = new Embedding<BFloat16>(2000, 4);

        // 500 is even and within [256, 512], so it IS exactly representable in BFloat16.
        const int repId = 500;

        var repExact = emb.Forward(new[] { repId });
        var repCorruptInput = ReverseGradTensor<BFloat16>.FromArray(new BFloat16[] { (BFloat16)repId }, requiresGrad: false);
        var repCorrupt = emb.Forward(repCorruptInput);

        repExact.Data.TryGetSpan(out var repExactSpan);
        repCorrupt.Data.TryGetSpan(out var repCorruptSpan);

        // A representable id round-trips through BFloat16, so both paths agree.
        Assert.That(repExactSpan[0], Is.EqualTo(repCorruptSpan[0]));
    }
}
