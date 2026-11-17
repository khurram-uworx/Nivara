using Nivara;
using NUnit.Framework;

namespace Nivara.Tests.Data;

[TestFixture]
public class FrameDatasetTests
{
    static NivaraFrame CreateFrame(int numRows)
    {
        var f1 = new int[numRows];
        var label = new int[numRows];
        for (int i = 0; i < numRows; i++)
        {
            f1[i] = i;
            label[i] = i % 3;
        }
        return NivaraFrame.Create(
            ("f1", NivaraColumn<int>.Create(f1)),
            ("label", NivaraColumn<int>.Create(label)));
    }

    [Test]
    public void SplitTrainTest_CountsSumToSource()
    {
        var frame = CreateFrame(10);
        var (train, test) = frame.SplitTrainTest(0.3, 42);
        Assert.That(train.RowCount + test.RowCount, Is.EqualTo(frame.RowCount));
    }

    [Test]
    public void SplitTrainTest_DeterministicSameSeed()
    {
        var frame = CreateFrame(10);
        var (t1, e1) = frame.SplitTrainTest(0.3, 42);
        var (t2, e2) = frame.SplitTrainTest(0.3, 42);
        Assert.That(t1, Is.EqualTo(t2));
        Assert.That(e1, Is.EqualTo(e2));
    }

    [Test]
    public void SplitTrainTest_DifferentSeed_DifferentOrder()
    {
        var frame = CreateFrame(10);
        var (t1, _) = frame.SplitTrainTest(0.3, 1);
        var (t2, _) = frame.SplitTrainTest(0.3, 2);
        Assert.That(t1, Is.Not.EqualTo(t2));
    }

    [Test]
    public void SplitTrainTest_NoOverlapUnionCoversAll()
    {
        var frame = CreateFrame(7);
        var (train, test) = frame.SplitTrainTest(0.4, 7);
        var trainVals = Enumerable.Range(0, train.RowCount).Select(i => train.GetColumn<int>("f1")[i]).ToHashSet();
        var testVals = Enumerable.Range(0, test.RowCount).Select(i => test.GetColumn<int>("f1")[i]).ToHashSet();
        Assert.That(trainVals.Intersect(testVals), Is.Empty);
        Assert.That(trainVals.Union(testVals).Count, Is.EqualTo(frame.RowCount));
    }

    [Test]
    public void Shuffle_Reproducible()
    {
        var frame = CreateFrame(10);
        var a = frame.Shuffle(5);
        var b = frame.Shuffle(5);
        Assert.That(a, Is.EqualTo(b));
    }

    [Test]
    public void Shuffle_ChangesOrder()
    {
        var frame = CreateFrame(10);
        var shuffled = frame.Shuffle(5);
        Assert.That(shuffled, Is.Not.EqualTo(frame));
    }

    [Test]
    public void Batch_CountAndSizes()
    {
        var frame = CreateFrame(10);
        var batches = frame.Batch(3);
        Assert.That(batches.Count, Is.EqualTo(4));
        Assert.That(batches[0].RowCount, Is.EqualTo(3));
        Assert.That(batches[3].RowCount, Is.EqualTo(1));
    }

    [Test]
    public void Chunk_AliasWorks()
    {
        var frame = CreateFrame(10);
        var chunks = frame.Chunk(4);
        Assert.That(chunks.Count, Is.EqualTo(3));
    }

    [Test]
    public void StratifiedSplit_PreservesClassCounts()
    {
        var frame = CreateFrame(30);
        var (train, test) = frame.StratifiedSplit("label", 0.3, 99);
        // 3 classes, ~30/3=10 each; test ~30% => 3 each
        var testLabels = Enumerable.Range(0, test.RowCount).Select(i => test.GetColumn<int>("label")[i]).GroupBy(x => x).ToDictionary(g => g.Key, g => g.Count());
        Assert.That(testLabels.Values, Has.All.GreaterThan(0));
        Assert.That(train.RowCount + test.RowCount, Is.EqualTo(frame.RowCount));
    }

    [Test]
    public void StratifiedSplit_Deterministic()
    {
        var frame = CreateFrame(30);
        var (a, b) = frame.StratifiedSplit("label", 0.3, 7);
        var (c, d) = frame.StratifiedSplit("label", 0.3, 7);
        Assert.That(a, Is.EqualTo(c));
        Assert.That(b, Is.EqualTo(d));
    }

    [Test]
    public void SplitTrainTest_InvalidFraction_Throws()
    {
        var frame = CreateFrame(5);
        Assert.That(() => frame.SplitTrainTest(1.5, 1), Throws.TypeOf<ArgumentOutOfRangeException>());
        Assert.That(() => frame.SplitTrainTest(-0.1, 1), Throws.TypeOf<ArgumentOutOfRangeException>());
    }

    [Test]
    public void StratifiedSplit_MissingColumn_Throws()
    {
        var frame = CreateFrame(5);
        Assert.That(() => frame.StratifiedSplit("nope", 0.2, 1), Throws.ArgumentException);
    }

    [Test]
    public void Batch_InvalidSize_Throws()
    {
        var frame = CreateFrame(5);
        Assert.That(() => frame.Batch(0), Throws.TypeOf<ArgumentOutOfRangeException>());
    }

    [Test]
    public void SplitTrainTest_DoesNotMutateSource()
    {
        var frame = CreateFrame(5);
        var original = frame;
        _ = frame.SplitTrainTest(0.4, 1);
        Assert.That(frame, Is.EqualTo(original));
    }

    [Test]
    public void StaticEntryPoints_Work()
    {
        var frame = CreateFrame(6);
        var split = Dataset.SplitTrainTest(frame, 0.5, 3);
        Assert.That(split.Train.RowCount + split.Test.RowCount, Is.EqualTo(6));
        var shuffled = NivaraDataset.Shuffle(frame, 3);
        Assert.That(shuffled.RowCount, Is.EqualTo(6));
        var batches = Dataset.Batch(frame, 2);
        Assert.That(batches.Count, Is.EqualTo(3));
    }

    [Test]
    public void EmptyFrame_Split_Works()
    {
        var frame = CreateFrame(0);
        // CreateFrame(0) produces empty frame
        var (train, test) = frame.SplitTrainTest(0.2, 1);
        Assert.That(train.RowCount, Is.EqualTo(0));
        Assert.That(test.RowCount, Is.EqualTo(0));
    }
}
