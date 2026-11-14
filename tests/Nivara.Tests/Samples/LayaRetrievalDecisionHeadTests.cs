using Nivara.Tests.AutoDiff;
using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

[TestFixture]
public sealed class LayaRetrievalDecisionHeadTests
{
    [Test]
    public void BuildPrompt_UsesRetrievedChunkAsState_ForSameQuery()
    {
        var first = LayaRetrievalDecisionHead.BuildPrompt(
            ByteTokenizerFixture.Tokenizer,
            "Where is the library?",
            "chunk-1",
            "The library is on Main Street.",
            maxLen: 512,
            headMaxLen: 192);
        var second = LayaRetrievalDecisionHead.BuildPrompt(
            ByteTokenizerFixture.Tokenizer,
            "Where is the library?",
            "chunk-2",
            "The library closes at six.",
            maxLen: 512,
            headMaxLen: 192);

        int stateStart = Array.IndexOf(
            first.Sequence.Ids, ByteTokenizerFixture.SepId, first.Sequence.Markers[1]) + 1;

        Assert.Multiple(() =>
        {
            Assert.That(first.Question.Instructions, Is.EqualTo(second.Question.Instructions));
            Assert.That(first.Sequence.Ids[stateStart..^1],
                Is.EqualTo(ByteTokenizerFixture.Bytes("The library is on Main Street.")));
            Assert.That(second.Sequence.Ids[stateStart..^1],
                Is.EqualTo(ByteTokenizerFixture.Bytes("The library closes at six.")));
            Assert.That(first.Sequence.Ids, Is.Not.EqualTo(second.Sequence.Ids));
        });
    }
}
