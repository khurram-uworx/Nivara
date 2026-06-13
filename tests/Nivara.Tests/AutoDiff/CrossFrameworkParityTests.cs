using NUnit.Framework;
using System.Text.Json;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class CrossFrameworkParityTests
{
    static string RepoRoot => Path.GetFullPath(
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));

    static string DataDir => Path.Combine(RepoRoot, "examples", "data");
    static string OutDir => Path.Combine(RepoRoot, "examples", "pytorch");

    [Test]
    public void LossCurves_AreWithinOnePercentRelativeDifference()
    {
        var pytPath = Path.Combine(OutDir, "epoch_losses_pytorch.json");
        var nivPath = Path.Combine(OutDir, "epoch_losses_nivara.json");

        if (!File.Exists(pytPath) || !File.Exists(nivPath))
            Assert.Ignore(
                "Run the Python and Nivara training scripts first. " +
                "See examples/README.md for setup instructions.");

        var pytLosses = JsonSerializer.Deserialize<double[]>(File.ReadAllBytes(pytPath))!;
        var nivLosses = JsonSerializer.Deserialize<double[]>(File.ReadAllBytes(nivPath))!;

        Assert.That(nivLosses.Length, Is.EqualTo(pytLosses.Length),
            "Loss arrays must have the same length (both should be 50 epochs)");

        double maxRelDiff = 0;
        for (int i = 0; i < pytLosses.Length; i++)
        {
            var denom = Math.Max(Math.Abs(pytLosses[i]), 1e-10);
            var rel = Math.Abs(pytLosses[i] - nivLosses[i]) / denom * 100;
            if (rel > maxRelDiff) maxRelDiff = rel;
        }

        Assert.That(maxRelDiff, Is.LessThan(1.0),
            $"Max relative loss difference ({maxRelDiff:F4}%) exceeds 1% threshold");
    }

    [Test]
    public void LossCurves_ShowConvergenceInBothFrameworks()
    {
        var pytPath = Path.Combine(OutDir, "epoch_losses_pytorch.json");
        var nivPath = Path.Combine(OutDir, "epoch_losses_nivara.json");

        if (!File.Exists(pytPath) || !File.Exists(nivPath))
            Assert.Ignore(
                "Run the Python and Nivara training scripts first. " +
                "See examples/README.md for setup instructions.");

        var pytLosses = JsonSerializer.Deserialize<double[]>(File.ReadAllBytes(pytPath))!;
        var nivLosses = JsonSerializer.Deserialize<double[]>(File.ReadAllBytes(nivPath))!;

        Assert.That(pytLosses[^1], Is.LessThan(pytLosses[0] * 0.5),
            "PyTorch loss should at least halve over 50 epochs");
        Assert.That(nivLosses[^1], Is.LessThan(nivLosses[0] * 0.5),
            "Nivara loss should at least halve over 50 epochs");
    }

    [Test]
    public void PredictionAgreement_IsAbove80Percent()
    {
        var pytPath = Path.Combine(OutDir, "test_preds_pytorch.csv");
        var nivPath = Path.Combine(OutDir, "test_preds_nivara.csv");

        if (!File.Exists(pytPath) || !File.Exists(nivPath))
            Assert.Ignore(
                "Run the Python and Nivara training scripts first. " +
                "See examples/README.md for setup instructions.");

        var pytLines = File.ReadAllLines(pytPath);
        var nivLines = File.ReadAllLines(nivPath);

        Assert.That(nivLines.Length, Is.EqualTo(pytLines.Length),
            "Prediction CSVs must have the same number of rows");

        int matches = 0;
        for (int i = 1; i < pytLines.Length; i++)
        {
            var pytParts = pytLines[i].Split(',');
            var nivParts = nivLines[i].Split(',');
            var pytProb = float.Parse(pytParts[^1]);
            var nivProb = float.Parse(nivParts[^1]);
            if (MathF.Round(pytProb, 3) == MathF.Round(nivProb, 3))
                matches++;
        }

        double agreement = (double)matches / (pytLines.Length - 1) * 100;
        Assert.That(agreement, Is.GreaterThan(80.0),
            $"Prediction agreement ({agreement:F1}%) is below 80% threshold");
    }

    [Test]
    public void InitialWeights_ExistForSeedBridge()
    {
        var initPath = Path.Combine(DataDir, "initial_weights.json");
        var normPath = Path.Combine(DataDir, "norm_params.json");

        if (!File.Exists(initPath) || !File.Exists(normPath))
            Assert.Ignore(
                "Run the PyTorch training script first (Step 3 in examples/README.md) " +
                "to generate initial_weights.json and norm_params.json.");

        var weights = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(
            File.ReadAllBytes(initPath))!;

        Assert.That(weights, Does.ContainKey("L1.Weight"));
        Assert.That(weights, Does.ContainKey("L1.Bias"));
        Assert.That(weights, Does.ContainKey("L2.Weight"));
        Assert.That(weights, Does.ContainKey("L2.Bias"));
        Assert.That(weights, Does.ContainKey("L3.Weight"));
        Assert.That(weights, Does.ContainKey("L3.Bias"));

        var norm = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(
            File.ReadAllBytes(normPath))!;
        Assert.That(norm, Does.ContainKey("means"));
        Assert.That(norm, Does.ContainKey("stds"));
    }
}
