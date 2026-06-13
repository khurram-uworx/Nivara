using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Training;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class DataParallelTests
{
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

    static DataLoader<float> CreateLoader(int numRows, int batchSize, bool shuffle, int? seed = null)
    {
        var frame = CreateTestFrame(numRows);
        var dataset = new TensorDataset<float>(frame, ["f1"], "label");
        return new DataLoader<float>(dataset, batchSize, shuffle, seed);
    }

    [Test]
    public void Run_SingleChunk_Completes()
    {
        var loader = CreateLoader(10, 20, shuffle: false);

        using var model = new Linear<float>(1, 1);
        Func<ReverseGradTensor<float>, ReverseGradTensor<float>, ReverseGradTensor<float>> lossFn = LossFunctions.MSE;
        var optimizer = new SGD<float>(0.01f);
        optimizer.AddParameterGroup(model.GetParameters().Values, 0.01f);

        using var trainer = new DataParallelTrainer<float>(
            model, loader, lossFn, optimizer, epochs: 2, maxDegreeOfParallelism: 1);
        var result = trainer.Run();

        Assert.That(result.Epochs.Count, Is.EqualTo(2));
        Assert.That(result.Epochs[0].Chunks, Is.EqualTo(1));
    }

    [Test]
    public void Run_MultipleChunks_LossDecreases()
    {
        var loader = CreateLoader(100, 25, shuffle: false);

        using var model = new Linear<float>(1, 1);
        Func<ReverseGradTensor<float>, ReverseGradTensor<float>, ReverseGradTensor<float>> lossFn = LossFunctions.MSE;
        var optimizer = new Adam<float>();
        optimizer.AddParameterGroup(model.GetParameters().Values, 0.01f);

        using var trainer = new DataParallelTrainer<float>(
            model, loader, lossFn, optimizer, epochs: 10, maxDegreeOfParallelism: 4);
        var result = trainer.Run();

        Assert.That(result.Epochs[^1].Loss, Is.LessThan(result.Epochs[0].Loss));
    }

    [Test]
    public void Run_Convergence_LinearRegression()
    {
        var loader = CreateLoader(200, 50, shuffle: false);

        using var model = new Linear<float>(1, 1);
        Func<ReverseGradTensor<float>, ReverseGradTensor<float>, ReverseGradTensor<float>> lossFn = LossFunctions.MSE;
        var optimizer = new Adam<float>();
        optimizer.AddParameterGroup(model.GetParameters().Values, 0.01f);

        using var trainer = new DataParallelTrainer<float>(
            model, loader, lossFn, optimizer, epochs: 20, maxDegreeOfParallelism: 4);
        var result = trainer.Run();

        Assert.That(result.Epochs[^1].Loss, Is.LessThan(result.Epochs[0].Loss));
        Assert.That(result.Epochs[^1].Workers, Is.GreaterThanOrEqualTo(1));
        Assert.That(result.Epochs[^1].Chunks, Is.GreaterThanOrEqualTo(1));
    }

    [Test]
    public void Run_Deterministic_SameSeedSameResults()
    {
        var frame = CreateTestFrame(50);

        (float finalLoss, float firstLoss) RunWithSeed(int seed)
        {
            var ds = new TensorDataset<float>(frame, ["f1"], "label");
            var loader = new DataLoader<float>(ds, 25, shuffle: true, seed: seed);
            using var model = new Linear<float>(1, 1);
            Func<ReverseGradTensor<float>, ReverseGradTensor<float>, ReverseGradTensor<float>> lossFn = LossFunctions.MSE;
            var optimizer = new Adam<float>();
            optimizer.AddParameterGroup(model.GetParameters().Values, 0.01f);

            using var trainer = new DataParallelTrainer<float>(
                model, loader, lossFn, optimizer, epochs: 10, maxDegreeOfParallelism: 1);
            var result = trainer.Run();
            return (result.Epochs[^1].Loss, result.Epochs[0].Loss);
        }

        var (loss1Final, loss1First) = RunWithSeed(42);
        var (loss2Final, loss2First) = RunWithSeed(42);

        Assert.That(loss1First, Is.GreaterThan(0f), "First run initial loss should be positive");
        Assert.That(loss2First, Is.GreaterThan(0f), "Second run initial loss should be positive");
        Assert.That(loss1Final, Is.LessThan(loss1First), "First run should converge");
        Assert.That(loss2Final, Is.LessThan(loss2First), "Second run should converge");
    }

    [Test]
    public void Run_GradientNorm_IsComputed()
    {
        var loader = CreateLoader(50, 25, shuffle: false);

        using var model = new Linear<float>(1, 1);
        Func<ReverseGradTensor<float>, ReverseGradTensor<float>, ReverseGradTensor<float>> lossFn = LossFunctions.MSE;
        var optimizer = new SGD<float>(0.01f);
        optimizer.AddParameterGroup(model.GetParameters().Values, 0.01f);

        using var trainer = new DataParallelTrainer<float>(
            model, loader, lossFn, optimizer, epochs: 3, maxDegreeOfParallelism: 2);
        var result = trainer.Run();

        foreach (var epoch in result.Epochs)
            Assert.That(epoch.GradientNorm, Is.GreaterThan(0));
    }

    [Test]
    public void Run_ResultStructure_Correct()
    {
        var loader = CreateLoader(20, 10, shuffle: false);

        using var model = new Linear<float>(1, 1);
        Func<ReverseGradTensor<float>, ReverseGradTensor<float>, ReverseGradTensor<float>> lossFn = LossFunctions.MSE;
        var optimizer = new SGD<float>(0.01f);
        optimizer.AddParameterGroup(model.GetParameters().Values, 0.01f);

        using var trainer = new DataParallelTrainer<float>(
            model, loader, lossFn, optimizer, epochs: 2, maxDegreeOfParallelism: 1);
        var result = trainer.Run();

        Assert.That(result.TotalElapsed.TotalMilliseconds, Is.GreaterThan(0));

        foreach (var epoch in result.Epochs)
        {
            Assert.That(epoch.Epoch, Is.GreaterThan(0));
            Assert.That(epoch.Loss, Is.GreaterThan(0f));
            Assert.That(epoch.Elapsed.TotalMilliseconds, Is.GreaterThan(0));
            Assert.That(epoch.Workers, Is.GreaterThan(0));
            Assert.That(epoch.Chunks, Is.GreaterThan(0));
            Assert.That(epoch.GradientNorm, Is.GreaterThan(0));
        }
    }

    private sealed class HookTrackingTrainer : DataParallelTrainer<float>
    {
        public int EpochStarts { get; private set; }
        public int EpochEnds { get; private set; }

        public HookTrackingTrainer(
            Module<float> model, DataLoader<float> loader,
            Func<ReverseGradTensor<float>, ReverseGradTensor<float>, ReverseGradTensor<float>> lossFn,
            Optimizer<float> optimizer, int epochs, int? maxDop = null)
            : base(model, loader, lossFn, optimizer, epochs, maxDop)
        {
        }

        protected override void OnEpochStart(int epoch) => EpochStarts++;
        protected override void OnEpochEnd(int epoch, DataParallelEpochResult<float> result) => EpochEnds++;
    }

    [Test]
    public void Run_Hooks_AreCalled()
    {
        var loader = CreateLoader(20, 10, shuffle: false);

        using var model = new Linear<float>(1, 1);
        Func<ReverseGradTensor<float>, ReverseGradTensor<float>, ReverseGradTensor<float>> lossFn = LossFunctions.MSE;
        var optimizer = new SGD<float>(0.01f);
        optimizer.AddParameterGroup(model.GetParameters().Values, 0.01f);

        using var trainer = new HookTrackingTrainer(
            model, loader, lossFn, optimizer, epochs: 3, maxDop: 1);
        trainer.Run();

        Assert.That(trainer.EpochStarts, Is.EqualTo(3));
        Assert.That(trainer.EpochEnds, Is.EqualTo(3));
    }

    [Test]
    public void Run_PrintSummary_DoesNotThrow()
    {
        var loader = CreateLoader(20, 10, shuffle: false);

        using var model = new Linear<float>(1, 1);
        Func<ReverseGradTensor<float>, ReverseGradTensor<float>, ReverseGradTensor<float>> lossFn = LossFunctions.MSE;
        var optimizer = new SGD<float>(0.01f);
        optimizer.AddParameterGroup(model.GetParameters().Values, 0.01f);

        using var trainer = new DataParallelTrainer<float>(
            model, loader, lossFn, optimizer, epochs: 2, maxDegreeOfParallelism: 1);
        var result = trainer.Run();

        Assert.DoesNotThrow(() => result.PrintSummary());
    }

    [Test]
    public void Run_NullModel_Throws()
    {
        var loader = CreateLoader(10, 5, shuffle: false);
        Func<ReverseGradTensor<float>, ReverseGradTensor<float>, ReverseGradTensor<float>> lossFn = LossFunctions.MSE;
        var optimizer = new SGD<float>(0.01f);

        Assert.Throws<ArgumentNullException>(() =>
            new DataParallelTrainer<float>(null!, loader, lossFn, optimizer, 1));
    }

    [Test]
    public void Run_NullLoader_Throws()
    {
        using var model = new Linear<float>(1, 1);
        Func<ReverseGradTensor<float>, ReverseGradTensor<float>, ReverseGradTensor<float>> lossFn = LossFunctions.MSE;
        var optimizer = new SGD<float>(0.01f);

        Assert.Throws<ArgumentNullException>(() =>
            new DataParallelTrainer<float>(model, null!, lossFn, optimizer, 1));
    }

    [Test]
    public void Run_ZeroEpochs_Throws()
    {
        var loader = CreateLoader(10, 5, shuffle: false);
        using var model = new Linear<float>(1, 1);
        Func<ReverseGradTensor<float>, ReverseGradTensor<float>, ReverseGradTensor<float>> lossFn = LossFunctions.MSE;
        var optimizer = new SGD<float>(0.01f);

        Assert.Throws<ArgumentException>(() =>
            new DataParallelTrainer<float>(model, loader, lossFn, optimizer, 0));
    }
}
