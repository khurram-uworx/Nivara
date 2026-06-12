using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Training;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class TrainingTests
{
    static DataLoader<float> CreateLoader(int numRows, int batchSize, bool shuffle, int? seed = null)
    {
        var frame = CreateTestFrame(numRows);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        return new DataLoader<float>(dataset, batchSize, shuffle, seed);
    }

    static NivaraFrame CreateTestFrame(int numRows)
    {
        var f1 = new float[numRows];
        var label = new float[numRows];

        for (int i = 0; i < numRows; i++)
        {
            f1[i] = i;
            label[i] = 2 * i + 1;
        }

        return NivaraFrame.Create(
            ("f1", (IColumn)NivaraColumn<float>.Create(f1)),
            ("label", (IColumn)NivaraColumn<float>.Create(label)));
    }

    static List<float> FlattenBatches(DataLoader<float> loader)
    {
        var result = new List<float>();
        foreach (var batch in loader)
        {
            for (int i = 0; i < batch.Size; i++)
                result.Add(batch.Features[i]);
        }
        return result;
    }

    // --- Batch tests ---

    [Test]
    public void Batch_Creation_StoresFeaturesAndLabels()
    {
        var features = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([1f, 2f, 3f]));
        var labels = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([4f, 5f, 6f]));

        var batch = new Batch<float>(features, labels);

        Assert.That(batch.Features, Is.SameAs(features));
        Assert.That(batch.Labels, Is.SameAs(labels));
        Assert.That(batch.Size, Is.EqualTo(3));
    }

    [Test]
    public void Batch_NullFeatures_Throws()
    {
        var labels = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([1f]));
        Assert.That(() => new Batch<float>(null!, labels), Throws.ArgumentNullException);
    }

    [Test]
    public void Batch_NullLabels_Throws()
    {
        var features = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create([1f]));
        Assert.That(() => new Batch<float>(features, null!), Throws.ArgumentNullException);
    }

    // --- TensorDataset tests ---

    [Test]
    public void TensorDataset_Count_ReturnsRowCount()
    {
        var frame = CreateTestFrame(10);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");

        Assert.That(dataset.Count, Is.EqualTo(10));
    }

    [Test]
    public void TensorDataset_SingleFeature_GetBatch_ReturnsCorrectShape()
    {
        var frame = CreateTestFrame(10);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");

        var batch = dataset.GetBatch([0, 1, 2]);

        Assert.That(batch.Size, Is.EqualTo(3));
        Assert.That(batch.Features.Shape, Is.EqualTo(new[] { 3, 1 }));
        Assert.That(batch.Labels.Shape, Is.EqualTo(new[] { 3, 1 }));
        Assert.That(batch.Features[0], Is.EqualTo(0f));
        Assert.That(batch.Features[1], Is.EqualTo(1f));
        Assert.That(batch.Features[2], Is.EqualTo(2f));
    }

    [Test]
    public void TensorDataset_MultipleFeatures_CombinesColumns()
    {
        var frame = NivaraFrame.Create(
            ("f1", (IColumn)NivaraColumn<float>.Create([1f, 2f, 3f])),
            ("f2", (IColumn)NivaraColumn<float>.Create([10f, 20f, 30f])),
            ("label", (IColumn)NivaraColumn<float>.Create([100f, 200f, 300f])));

        var dataset = new TensorDataset<float>(frame, ["f1", "f2"], "label");
        var batch = dataset.GetBatch([0, 1]);

        Assert.That(batch.Features.Shape, Is.EqualTo(new[] { 2, 2 }));
        Assert.That(batch.Features.Length, Is.EqualTo(4));
        Assert.That(batch.Features[0], Is.EqualTo(1f));
        Assert.That(batch.Features[1], Is.EqualTo(10f));
        Assert.That(batch.Features[2], Is.EqualTo(2f));
        Assert.That(batch.Features[3], Is.EqualTo(20f));
        Assert.That(batch.Labels.Shape, Is.EqualTo(new[] { 2, 1 }));
    }

    [Test]
    public void TensorDataset_GetBatch_NullPropagation()
    {
        var frame = NivaraFrame.Create(
            ("f1", (IColumn)NivaraColumn<float>.CreateFromSpans(
                new float[] { 1f, 0f, 3f },
                new bool[] { false, true, false })),
            ("label", (IColumn)NivaraColumn<float>.CreateFromSpans(
                new float[] { 10f, 20f, 30f },
                new bool[] { false, false, false })));

        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var batch = dataset.GetBatch([0, 1, 2]);

        Assert.That(batch.Features.HasNulls, Is.True);
        Assert.That(batch.Features.IsNull(0), Is.False);
        Assert.That(batch.Features.IsNull(1), Is.True);
        Assert.That(batch.Features.IsNull(2), Is.False);
        Assert.That(batch.Features[1], Is.EqualTo(0f));
        Assert.That(batch.Labels.HasNulls, Is.False);
    }

    [Test]
    public void TensorDataset_ColumnNameMismatch_Throws()
    {
        var frame = CreateTestFrame(10);

        Assert.That(() => new TensorDataset<float>(frame, ["nonexistent"], "label"),
            Throws.ArgumentException);
    }

    [Test]
    public void TensorDataset_EmptyFeatureColumns_Throws()
    {
        var frame = CreateTestFrame(10);

        Assert.That(() => new TensorDataset<float>(frame, [], "label"),
            Throws.ArgumentException);
    }

    [Test]
    public void TensorDataset_EmptyLabelColumns_Throws()
    {
        var frame = CreateTestFrame(10);

        Assert.That(() => new TensorDataset<float>(frame, ["f1"], []),
            Throws.ArgumentException);
    }

    [Test]
    public void TensorDataset_EmptyFrame_Throws()
    {
        var frame = NivaraFrame.Create(
            ("f1", (IColumn)NivaraColumn<float>.Create([])),
            ("label", (IColumn)NivaraColumn<float>.Create([])));

        Assert.That(() => new TensorDataset<float>(frame, ["f1"], "label"),
            Throws.ArgumentException);
    }

    [Test]
    public void TensorDataset_NullFrame_Throws()
    {
        Assert.That(() => new TensorDataset<float>(null!, ["f1"], "label"),
            Throws.ArgumentNullException);
    }

    // --- DataLoader tests ---

    [Test]
    public void DataLoader_YieldsCorrectNumberOfBatches()
    {
        var loader = CreateLoader(10, 3, shuffle: false);
        var batches = new List<Batch<float>>();

        foreach (var batch in loader)
            batches.Add(batch);

        Assert.That(batches.Count, Is.EqualTo(4));
        Assert.That(batches[0].Size, Is.EqualTo(3));
        Assert.That(batches[1].Size, Is.EqualTo(3));
        Assert.That(batches[2].Size, Is.EqualTo(3));
        Assert.That(batches[3].Size, Is.EqualTo(1));
    }

    [Test]
    public void DataLoader_NoShuffle_SequentialOrder()
    {
        var loader = CreateLoader(10, 3, shuffle: false);
        var data = FlattenBatches(loader);

        Assert.That(data, Is.EqualTo(new[] { 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f }));
    }

    [Test]
    public void DataLoader_DeterministicSeed_SameOrder()
    {
        var frame = CreateTestFrame(10);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var loader1 = new DataLoader<float>(dataset, 3, shuffle: true, seed: 42);
        var loader2 = new DataLoader<float>(dataset, 3, shuffle: true, seed: 42);

        Assert.That(FlattenBatches(loader1), Is.EqualTo(FlattenBatches(loader2)));
    }

    [Test]
    public void DataLoader_DifferentSeed_DifferentOrder()
    {
        var frame = CreateTestFrame(10);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var loader1 = new DataLoader<float>(dataset, 3, shuffle: true, seed: 42);
        var loader2 = new DataLoader<float>(dataset, 3, shuffle: true, seed: 123);

        Assert.That(FlattenBatches(loader1), Is.Not.EqualTo(FlattenBatches(loader2)));
    }

    [Test]
    public void DataLoader_BatchSizeLargerThanDataset_OneBatch()
    {
        var frame = CreateTestFrame(10);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var loader = new DataLoader<float>(dataset, 100, shuffle: false);

        var batches = loader.ToList();

        Assert.That(batches.Count, Is.EqualTo(1));
        Assert.That(batches[0].Size, Is.EqualTo(10));
    }

    [Test]
    public void DataLoader_ExactMultiple_SizeLastBatchFull()
    {
        var frame = CreateTestFrame(10);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var loader = new DataLoader<float>(dataset, 5, shuffle: false);

        var batches = loader.ToList();

        Assert.That(batches.Count, Is.EqualTo(2));
        Assert.That(batches[0].Size, Is.EqualTo(5));
        Assert.That(batches[1].Size, Is.EqualTo(5));
    }

    [Test]
    public void DataLoader_RepeatedEnumeration_ProducesNewShuffle()
    {
        var frame = CreateTestFrame(10);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var loader = new DataLoader<float>(dataset, 5, shuffle: true);

        var first = FlattenBatches(loader);
        var second = FlattenBatches(loader);

        Assert.That(first, Is.Not.EqualTo(second));
    }

    [Test]
    public void DataLoader_NullDataset_Throws()
    {
        Assert.That(() => new DataLoader<float>(null!, 5), Throws.ArgumentNullException);
    }

    [Test]
    public void DataLoader_BatchSizeZero_Throws()
    {
        var frame = CreateTestFrame(10);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");

        Assert.That(() => new DataLoader<float>(dataset, 0), Throws.ArgumentException);
    }

    [Test]
    public void DataLoader_BatchSizeNegative_Throws()
    {
        var frame = CreateTestFrame(10);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");

        Assert.That(() => new DataLoader<float>(dataset, -1), Throws.ArgumentException);
    }

    [Test]
    public void DataLoader_MultipleLabelColumns_CombinesCorrectly()
    {
        var frame = NivaraFrame.Create(
            ("f1", (IColumn)NivaraColumn<float>.Create([1f, 2f, 3f])),
            ("l1", (IColumn)NivaraColumn<float>.Create([10f, 20f, 30f])),
            ("l2", (IColumn)NivaraColumn<float>.Create([100f, 200f, 300f])));

        var dataset = new TensorDataset<float>(frame, ["f1"], ["l1", "l2"]);
        var batch = dataset.GetBatch([0, 1]);

        Assert.That(batch.Labels.Shape, Is.EqualTo(new[] { 2, 2 }));
        Assert.That(batch.Labels[0], Is.EqualTo(10f));
        Assert.That(batch.Labels[1], Is.EqualTo(100f));
        Assert.That(batch.Labels[2], Is.EqualTo(20f));
        Assert.That(batch.Labels[3], Is.EqualTo(200f));
    }

    // --- TrainingLoop tests ---

    [Test]
    public void TrainingLoop_Run_CompletesEpochs()
    {
        var frame = CreateTestFrame(20);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var loader = new DataLoader<float>(dataset, 5, shuffle: false);

        using var model = new Linear<float>(1, 1);
        var lossFn = new MSELoss<float>().Forward;
        var optimizer = new SGD<float>(0.001f);
        optimizer.AddParameterGroup(model.GetParameters().Values, 0.001f);

        using var loop = new TrainingLoop<float>(model, loader, lossFn, optimizer, epochs: 3);
        var result = loop.Run();

        Assert.That(result.Epochs.Count, Is.EqualTo(3));
        Assert.That(result.Epochs[0].Loss, Is.GreaterThan(0f));
        Assert.That(result.Epochs[^1].Loss, Is.LessThan(result.Epochs[0].Loss));
        Assert.That(result.TotalElapsed.TotalMilliseconds, Is.GreaterThan(0d));
    }

    [Test]
    public void TrainingLoop_EpochResults_HaveCorrectStructure()
    {
        var frame = CreateTestFrame(10);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var loader = new DataLoader<float>(dataset, 10, shuffle: false);

        using var model = new Linear<float>(1, 1);
        var lossFn = new MSELoss<float>().Forward;
        var optimizer = new SGD<float>(0.01f);
        optimizer.AddParameterGroup(model.GetParameters().Values, 0.01f);

        using var loop = new TrainingLoop<float>(model, loader, lossFn, optimizer, epochs: 2);
        var result = loop.Run();

        foreach (var epoch in result.Epochs)
        {
            Assert.That(epoch.Epoch, Is.GreaterThan(0));
            Assert.That(epoch.Batches, Is.GreaterThan(0));
            Assert.That(epoch.Loss, Is.GreaterThan(0f));
            Assert.That(epoch.Elapsed.TotalMilliseconds, Is.GreaterThan(0d));
        }
    }

    [Test]
    public void TrainingLoop_MultipleBatches_WeightsUpdateEachStep()
    {
        var frame = CreateTestFrame(20);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var loader = new DataLoader<float>(dataset, 5, shuffle: false);

        using var model = new Linear<float>(1, 1);
        var lossFn = new MSELoss<float>().Forward;
        var optimizer = new SGD<float>(0.01f);
        optimizer.AddParameterGroup(model.GetParameters().Values, 0.01f);

        using var en = loader.GetEnumerator();
        en.MoveNext();
        var batch1 = en.Current;

        var wBefore = model.Weight[0];
        var bBefore = model.Bias![0];

        var output1 = model.Forward(batch1.Features);
        var loss1 = lossFn(output1, batch1.Labels);
        loss1.Backward();

        optimizer.Step();
        optimizer.ZeroGrad();

        Assert.That(model.Weight[0], Is.Not.EqualTo(wBefore), "Weight should change after step");
        Assert.That(model.Bias![0], Is.Not.EqualTo(bBefore), "Bias should change after step");
    }

    [Test]
    public void TrainingLoop_WithAdam_Converges()
    {
        var frame = CreateTestFrame(100);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var loader = new DataLoader<float>(dataset, 50, shuffle: false);

        using var model = new Linear<float>(1, 1);
        var lossFn = new MSELoss<float>().Forward;
        var optimizer = new Adam<float>();
        optimizer.AddParameterGroup(model.GetParameters().Values, 0.01f);

        using var loop = new TrainingLoop<float>(model, loader, lossFn, optimizer, epochs: 20);
        var result = loop.Run();

        Assert.That(result.Epochs[^1].Loss, Is.LessThan(result.Epochs[0].Loss));
    }

    [Test]
    public void TrainingLoop_MultiLayer_Converges()
    {
        var frame = NivaraFrame.Create(
            ("f1", (IColumn)NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f })),
            ("f2", (IColumn)NivaraColumn<float>.Create(new float[] { 5f, 6f, 7f, 8f, 1f, 2f, 3f, 4f })),
            ("label", (IColumn)NivaraColumn<float>.Create(new float[] { 11f, 17f, 25f, 35f, 11f, 17f, 25f, 35f })));
        var dataset = new TensorDataset<float>(frame, ["f1", "f2"], "label");
        var loader = new DataLoader<float>(dataset, 8, shuffle: false);

        using var model = new Sequential<float>(
            new Linear<float>(2, 3),
            new Linear<float>(3, 1));
        var lossFn = new MSELoss<float>().Forward;
        var optimizer = new SGD<float>(0.01f);
        optimizer.AddParameterGroup(model.GetParameters().Values, 0.01f);

        using var loop = new TrainingLoop<float>(model, loader, lossFn, optimizer, epochs: 50);
        var result = loop.Run();

        Assert.That(result.Epochs[^1].Loss, Is.LessThan(result.Epochs[0].Loss),
            "Multi-layer model should converge (all layers training)");
    }

    private sealed class HookTrackingTrainingLoop : TrainingLoop<float>
    {
        public int EpochStarts { get; private set; }
        public int BatchEnds { get; private set; }
        public int EpochEnds { get; private set; }

        public HookTrackingTrainingLoop(
            Module<float> model, DataLoader<float> loader,
            Func<ReverseGradTensor<float>, ReverseGradTensor<float>, ReverseGradTensor<float>> lossFn,
            Optimizer<float> optimizer, int epochs)
            : base(model, loader, lossFn, optimizer, epochs)
        {
        }

        protected override void OnEpochStart(int epoch) => EpochStarts++;
        protected override void OnBatchEnd(int epoch, int batch, float lossValue) => BatchEnds++;
        protected override void OnEpochEnd(int epoch, EpochResult<float> result) => EpochEnds++;
    }

    [Test]
    public void TrainingLoop_VirtualMethods_AreCalled()
    {
        var frame = CreateTestFrame(10);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var loader = new DataLoader<float>(dataset, 5, shuffle: false);

        using var model = new Linear<float>(1, 1);
        var lossFn = new MSELoss<float>().Forward;
        var optimizer = new SGD<float>(0.01f);
        optimizer.AddParameterGroup(model.GetParameters().Values, 0.01f);

        using var loop = new HookTrackingTrainingLoop(
            model, loader, lossFn, optimizer, epochs: 3);

        loop.Run();

        Assert.That(loop.EpochStarts, Is.EqualTo(3));
        Assert.That(loop.EpochEnds, Is.EqualTo(3));
        Assert.That(loop.BatchEnds, Is.EqualTo(6));
    }

    [Test]
    public void TrainingLoop_NullModel_Throws()
    {
        var frame = CreateTestFrame(10);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var loader = new DataLoader<float>(dataset, 10, shuffle: false);

        var lossFn = new MSELoss<float>().Forward;
        var optimizer = new SGD<float>(0.01f);

        Assert.That(() => new TrainingLoop<float>(null!, loader, lossFn, optimizer, 1),
            Throws.ArgumentNullException);
    }

    [Test]
    public void TrainingLoop_NullLoader_Throws()
    {
        using var model = new Linear<float>(1, 1);
        var lossFn = new MSELoss<float>().Forward;
        var optimizer = new SGD<float>(0.01f);

        Assert.That(() => new TrainingLoop<float>(model, null!, lossFn, optimizer, 1),
            Throws.ArgumentNullException);
    }

    [Test]
    public void TrainingLoop_NullLossFn_Throws()
    {
        var frame = CreateTestFrame(10);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var loader = new DataLoader<float>(dataset, 10, shuffle: false);
        using var model = new Linear<float>(1, 1);
        var optimizer = new SGD<float>(0.01f);

        Assert.That(() => new TrainingLoop<float>(model, loader, null!, optimizer, 1),
            Throws.ArgumentNullException);
    }

    [Test]
    public void TrainingLoop_NullOptimizer_Throws()
    {
        var frame = CreateTestFrame(10);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var loader = new DataLoader<float>(dataset, 10, shuffle: false);
        using var model = new Linear<float>(1, 1);
        var lossFn = new MSELoss<float>().Forward;

        Assert.That(() => new TrainingLoop<float>(model, loader, lossFn, null!, 1),
            Throws.ArgumentNullException);
    }

    [Test]
    public void TrainingLoop_ZeroEpochs_Throws()
    {
        var frame = CreateTestFrame(10);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var loader = new DataLoader<float>(dataset, 10, shuffle: false);
        using var model = new Linear<float>(1, 1);
        var lossFn = new MSELoss<float>().Forward;
        var optimizer = new SGD<float>(0.01f);

        Assert.That(() => new TrainingLoop<float>(model, loader, lossFn, optimizer, 0),
            Throws.ArgumentException);
    }

    [Test]
    public void PrintSummary_DoesNotThrow()
    {
        var frame = CreateTestFrame(10);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var loader = new DataLoader<float>(dataset, 10, shuffle: false);

        using var model = new Linear<float>(1, 1);
        var lossFn = new MSELoss<float>().Forward;
        var optimizer = new SGD<float>(0.01f);
        optimizer.AddParameterGroup(model.GetParameters().Values, 0.01f);

        using var loop = new TrainingLoop<float>(model, loader, lossFn, optimizer, epochs: 1);
        var result = loop.Run();

        Assert.DoesNotThrow(() => result.PrintSummary());
    }

    // --- Minibatch test gaps (I5) ---

    [Test]
    public void TrainingLoop_PartialBatch_TrainsToCompletion()
    {
        var frame = CreateTestFrame(10);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var loader = new DataLoader<float>(dataset, 3, shuffle: false);

        using var model = new Linear<float>(1, 1);
        var lossFn = new MSELoss<float>().Forward;
        var optimizer = new SGD<float>(0.01f);
        optimizer.AddParameterGroup(model.GetParameters().Values, 0.01f);

        using var loop = new TrainingLoop<float>(model, loader, lossFn, optimizer, epochs: 20);
        var result = loop.Run();

        Assert.That(result.Epochs[^1].Loss, Is.LessThan(result.Epochs[0].Loss));
    }

    [Test]
    public void DataLoader_PartialBatch_LastBatchHasCorrectSize()
    {
        var frame = CreateTestFrame(10);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var loader = new DataLoader<float>(dataset, 3, shuffle: false);

        var batches = loader.ToList();

        Assert.That(batches.Count, Is.EqualTo(4));
        Assert.That(batches[0].Size, Is.EqualTo(3));
        Assert.That(batches[1].Size, Is.EqualTo(3));
        Assert.That(batches[2].Size, Is.EqualTo(3));
        Assert.That(batches[3].Size, Is.EqualTo(1));
    }

    [Test]
    public void TrainingLoop_MultiEpochShuffle_Converges()
    {
        var frame = CreateTestFrame(20);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        var loader = new DataLoader<float>(dataset, 5, shuffle: true, seed: 42);

        using var model = new Linear<float>(1, 1);
        var lossFn = new MSELoss<float>().Forward;
        var optimizer = new SGD<float>(0.01f);
        optimizer.AddParameterGroup(model.GetParameters().Values, 0.01f);

        using var loop = new TrainingLoop<float>(model, loader, lossFn, optimizer, epochs: 30);
        var result = loop.Run();

        Assert.That(result.Epochs[^1].Loss, Is.LessThan(result.Epochs[0].Loss),
            "Shuffled minibatch training should converge");
    }
}
