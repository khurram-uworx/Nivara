using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Training;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class SerializationTests
{
    static readonly string TestDir = Path.Combine(Path.GetTempPath(), "nivara_serialization_tests");

    [OneTimeSetUp]
    public void Setup() => Directory.CreateDirectory(TestDir);

    [OneTimeTearDown]
    public void Cleanup()
    {
        if (Directory.Exists(TestDir))
            Directory.Delete(TestDir, recursive: true);
    }

    [Test]
    public void SaveLoad_LinearFloat_RoundTrip()
    {
        var model = new Linear<float>(4, 3);
        var path = Path.Combine(TestDir, "linear_float.json");

        ModelSerializer.Save(model, path);

        var loaded = new Linear<float>(4, 3);
        ModelSerializer.Load(loaded, path);

        Assert.That(model.Weight.Length, Is.EqualTo(loaded.Weight.Length));
        for (int i = 0; i < model.Weight.Length; i++)
            Assert.That(loaded.Weight[i], Is.EqualTo(model.Weight[i]).Within(1e-6f));

        Assert.That(model.Bias, Is.Not.Null);
        Assert.That(loaded.Bias, Is.Not.Null);
        for (int i = 0; i < model.Bias!.Length; i++)
            Assert.That(loaded.Bias[i], Is.EqualTo(model.Bias[i]).Within(1e-6f));

        File.Delete(path);
    }

    [Test]
    public void SaveLoad_LinearDouble_RoundTrip()
    {
        var model = new Linear<double>(3, 2);
        var path = Path.Combine(TestDir, "linear_double.json");

        ModelSerializer.Save(model, path);

        var loaded = new Linear<double>(3, 2);
        ModelSerializer.Load(loaded, path);

        for (int i = 0; i < model.Weight.Length; i++)
            Assert.That(loaded.Weight[i], Is.EqualTo(model.Weight[i]).Within(1e-12));

        File.Delete(path);
    }

    [Test]
    public void SaveLoad_MultiLayerModel_RoundTrip()
    {
        var frame = NivaraFrame.Create(
            ("f1", (IColumn)NivaraColumn<float>.Create([1f, 2f, 3f, 4f])),
            ("f2", (IColumn)NivaraColumn<float>.Create([5f, 6f, 7f, 8f])),
            ("label", (IColumn)NivaraColumn<float>.Create([10f, 20f, 30f, 40f])));

        using var model = new Linear<float>(2, 1);
        Func<ReverseGradTensor<float>, ReverseGradTensor<float>, ReverseGradTensor<float>> lossFn = LossFunctions.MSE;
        var optimizer = new SGD<float>(0.01f);
        optimizer.AddParameterGroup(model.GetParameters().Values, 0.01f);

        var dataset = new TensorDataset<float>(frame, ["f1", "f2"], "label");
        var loader = new DataLoader<float>(dataset, 4, shuffle: false);
        using var loop = new TrainingLoop<float>(model, loader, lossFn, optimizer, epochs: 5);
        loop.Run();

        var path = Path.Combine(TestDir, "trained_model.json");
        ModelSerializer.Save(model, path);

        var loaded = new Linear<float>(2, 1);
        ModelSerializer.Load(loaded, path);

        for (int i = 0; i < model.Weight.Length; i++)
            Assert.That(loaded.Weight[i], Is.EqualTo(model.Weight[i]).Within(1e-6f));

        File.Delete(path);
    }

    [Test]
    public void SaveCheckpoint_RoundTrip()
    {
        using var model = new Linear<float>(2, 1);
        var path = Path.Combine(TestDir, "checkpoint.json");
        var epoch = new EpochResult<float>(10, 0.123f, TimeSpan.FromSeconds(1.5), 5);

        ModelSerializer.SaveCheckpoint(model, epoch, path);

        var checkpoint = ModelSerializer.LoadCheckpoint<float>(path);

        Assert.That(checkpoint.Epoch, Is.EqualTo(10));
        Assert.That(checkpoint.Loss, Is.EqualTo(0.123).Within(1e-6));
        Assert.That(checkpoint.Parameters.Count, Is.EqualTo(2));
        Assert.That(checkpoint.Parameters.ContainsKey("Weight"), Is.True);
        Assert.That(checkpoint.Parameters.ContainsKey("Bias"), Is.True);

        File.Delete(path);
    }

    [Test]
    public void LoadCheckpoint_CheckpointValues()
    {
        using var model = new Linear<float>(3, 2);
        var path = Path.Combine(TestDir, "checkpoint_values.json");
        var epoch = new EpochResult<float>(5, 0.5f, TimeSpan.FromSeconds(2), 3);

        ModelSerializer.SaveCheckpoint(model, epoch, path);

        var checkpoint = ModelSerializer.LoadCheckpoint<float>(path);

        var weightData = checkpoint.Parameters["Weight"];
        Assert.That(weightData.Shape, Is.EqualTo(model.Weight.Shape));
        Assert.That(weightData.Values.Length, Is.EqualTo(model.Weight.Length));

        File.Delete(path);
    }

    [Test]
    public void Load_DifferentParameterNames_Throws()
    {
        using var model = new Linear<float>(3, 2);
        var path = Path.Combine(TestDir, "mismatch.json");
        ModelSerializer.Save(model, path);

        // Linear<float>(2, 3) has Weight[3,2] and Bias[1,3] vs Weight[2,3] and Bias[1,2]
        var differentModel = new Linear<float>(4, 1);

        Assert.Throws<InvalidOperationException>(() =>
            ModelSerializer.Load(differentModel, path));

        File.Delete(path);
    }

    [Test]
    public void Load_NonexistentFile_Throws()
    {
        using var model = new Linear<float>(2, 1);

        Assert.Throws<FileNotFoundException>(() =>
            ModelSerializer.Load(model, "nonexistent.json"));
    }

    [Test]
    public void LoadCheckpoint_NonexistentFile_Throws()
    {
        Assert.Throws<FileNotFoundException>(() =>
            ModelSerializer.LoadCheckpoint<float>("nonexistent.json"));
    }

    [Test]
    public void Save_NullModel_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelSerializer.Save<float>(null!, "path.json"));
    }

    [Test]
    public void Save_NullPath_Throws()
    {
        using var model = new Linear<float>(2, 1);

        Assert.Throws<ArgumentNullException>(() =>
            ModelSerializer.Save(model, null!));
    }

    [Test]
    public void SaveCheckpoint_NullModel_Throws()
    {
        var epoch = new EpochResult<float>(1, 0f, TimeSpan.Zero, 0);
        Assert.Throws<ArgumentNullException>(() =>
            ModelSerializer.SaveCheckpoint<float>(null!, epoch, "path.json"));
    }

    [Test]
    public void SaveCheckpoint_NullEpoch_Throws()
    {
        using var model = new Linear<float>(2, 1);
        Assert.Throws<ArgumentNullException>(() =>
            ModelSerializer.SaveCheckpoint(model, null!, "path.json"));
    }

    [Test]
    public void Save_FileContainsCorrectFormat()
    {
        using var model = new Linear<float>(2, 1);
        var path = Path.Combine(TestDir, "format_check.json");

        ModelSerializer.Save(model, path);
        var json = File.ReadAllText(path);

        Assert.That(json, Does.Contain("nivara-ss-v1"));
        Assert.That(json, Does.Contain("Weight"));
        Assert.That(json, Does.Contain("Bias"));
        Assert.That(json, Does.Contain("Shape"));
        Assert.That(json, Does.Contain("Values"));
        Assert.That(json, Does.Contain("HasNulls"));

        File.Delete(path);
    }

    [Test]
    public void SaveCheckpoint_FileContainsCorrectFormat()
    {
        using var model = new Linear<float>(2, 1);
        var path = Path.Combine(TestDir, "ckpt_format.json");
        var epoch = new EpochResult<float>(3, 0.5f, TimeSpan.FromSeconds(1), 2);

        ModelSerializer.SaveCheckpoint(model, epoch, path);
        var json = File.ReadAllText(path);

        Assert.That(json, Does.Contain("nivara-ckpt-v1"));
        Assert.That(json, Does.Contain("Epoch"));

        File.Delete(path);
    }

    [Test]
    public void SaveLoad_Sequential_MultiLayer_RoundTrip()
    {
        using var model = new Sequential<float>(
            new Linear<float>(2, 3),
            new Linear<float>(3, 1));
        var path = Path.Combine(TestDir, "sequential_multilayer.json");

        ModelSerializer.Save(model, path);

        using var loaded = new Sequential<float>(
            new Linear<float>(2, 3),
            new Linear<float>(3, 1));
        ModelSerializer.Load(loaded, path);

        var modelParams = model.Parameters();
        var loadedParams = loaded.Parameters();
        Assert.That(loadedParams.Count, Is.EqualTo(modelParams.Count));

        foreach (var (name, tensor) in modelParams)
        {
            Assert.That(loadedParams.ContainsKey(name), Is.True,
                $"Parameter '{name}' should exist in loaded model");
            for (int i = 0; i < tensor.Length; i++)
                Assert.That(loadedParams[name][i], Is.EqualTo(tensor[i]).Within(1e-6f));
        }

        File.Delete(path);
    }

    [Test]
    public void SaveLoad_NoBiasLinear_RoundTrip()
    {
        var model = new Linear<float>(3, 2, bias: false);
        var path = Path.Combine(TestDir, "no_bias.json");

        ModelSerializer.Save(model, path);

        var loaded = new Linear<float>(3, 2, bias: false);
        ModelSerializer.Load(loaded, path);

        for (int i = 0; i < model.Weight.Length; i++)
            Assert.That(loaded.Weight[i], Is.EqualTo(model.Weight[i]).Within(1e-6f));

        Assert.That(model.Bias, Is.Null);
        Assert.That(loaded.Bias, Is.Null);

        File.Delete(path);
    }
}
