using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Training;
using System.Text.Json;

namespace Nivara.SampleApp;

/// <summary>
/// Cross-framework FraudNet — corresponds to examples/pytorch/train_fraud_pytorch.py.
///
/// Trains a 3-layer MLP on synthetic fraud data using BCEWithLogitsLoss + Adam.
/// Run the PyTorch example first to generate input CSVs.
/// </summary>
class FraudNet : Module<float>
{
    public Linear<float> L1 { get; }
    public Linear<float> L2 { get; }
    public Linear<float> L3 { get; }

    public FraudNet()
    {
        L1 = new Linear<float>(8, 64);
        L2 = new Linear<float>(64, 32);
        L3 = new Linear<float>(32, 1);
        RegisterModules(L1, L2, L3);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> x)
    {
        var h = GradOperations.Relu(L1.Forward(x));
        h = GradOperations.Relu(L2.Forward(h));
        return L3.Forward(h);
    }

    /// <summary>
    /// Injects weights from a PyTorch interchange JSON into this model.
    /// </summary>
    public void LoadWeightsFromJson(string jsonPath)
    {
        var json = File.ReadAllText(jsonPath);
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        var parameters = GetParameters();

        var nameMap = new (string pyt, string niv, int rows, int cols)[]
        {
            ("L1.Weight", "Module_0.Weight", 64, 8),
            ("L1.Bias",   "Module_0.Bias",   1, 64),
            ("L2.Weight", "Module_1.Weight", 32, 64),
            ("L2.Bias",   "Module_1.Bias",   1, 32),
            ("L3.Weight", "Module_2.Weight", 1,  32),
            ("L3.Bias",   "Module_2.Bias",   1,  1),
        };

        foreach (var (pyt, niv, rows, cols) in nameMap)
        {
            if (!root.TryGetProperty(pyt, out var el)) continue;
            if (!parameters.TryGetValue(niv, out var param)) continue;

            var flat = FlattenJsonArray(el);
            param.Tensor = ReverseGradTensor<float>.FromMatrix(flat, rows, cols, requiresGrad: true);
        }
    }

    static float[] FlattenJsonArray(JsonElement el)
    {
        var result = new List<float>();
        FlattenInto(el, result);
        return result.ToArray();
    }

    static void FlattenInto(JsonElement el, List<float> result)
    {
        if (el.ValueKind == JsonValueKind.Number)
            result.Add(el.GetSingle());
        else if (el.ValueKind == JsonValueKind.Array)
            foreach (var child in el.EnumerateArray())
                FlattenInto(child, result);
    }
}

public static class CrossFrameworkFraudNet
{
    static string DataDir => Path.GetFullPath(
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "examples", "data"));

    static string OutDir => Path.GetFullPath(
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "examples", "pytorch"));

    static readonly string[] FeatureCols =
        ["amount", "hour", "distance", "prev_attempts",
         "country_change", "device_new", "amount_ratio", "velocity"];

    public static void Run()
    {
        Console.WriteLine("=== Cross-Framework FraudNet (Nivara) ===");
        Console.WriteLine();

        // ── Load & normalize data ──────────────────────────────
        var trainFrame = LoadCsv(Path.Combine(DataDir, "train_fraud.csv"));
        var testFrame = LoadCsv(Path.Combine(DataDir, "test_fraud.csv"));

        var norm = LoadNormParams(Path.Combine(DataDir, "norm_params.json"));
        trainFrame = Normalize(trainFrame, norm, FeatureCols);
        testFrame = Normalize(testFrame, norm, FeatureCols);

        Console.WriteLine($"Train: {trainFrame.RowCount} rows");
        Console.WriteLine($"Test:  {testFrame.RowCount} rows");
        Console.WriteLine();

        // ── Init model ─────────────────────────────────────────
        using var model = new FraudNet();

        // Optionally load initial weights from PyTorch:
        var initPath = Path.Combine(DataDir, "initial_weights.json");
        if (File.Exists(initPath))
        {
            model.LoadWeightsFromJson(initPath);
            Console.WriteLine("Loaded initial weights from PyTorch.");
        }

        // ── Training ───────────────────────────────────────────
        var loader = new DataLoader<float>(
            new TensorDataset<float>(trainFrame, FeatureCols, "is_fraud"),
            batchSize: 32, shuffle: false);

        var optimizer = new Adam<float>(beta1: 0.9, beta2: 0.999);
        optimizer.AddParameterGroup(model.GetParameters().Values, learningRate: 0.001f);

        var loop = new TrainingLoop<float>(
            model, loader,
            (pred, target) => new BCEWithLogitsLoss<float>().Forward(pred, target),
            optimizer,
            epochs: 50);

        var result = loop.Run();
        result.PrintSummary();
        Console.WriteLine();

        // ── Save trained model (Nivara native format) ──────────
        var nivaraPath = Path.Combine(OutDir, "nivara_trained_model.json");
        ModelSerializer.Save(model, nivaraPath);
        Console.WriteLine($"Saved Nivara model to {nivaraPath}");

        // ── Save trained weights (interchange JSON) ────────────
        SaveWeightsToJson(model, Path.Combine(OutDir, "trained_weights_nivara.json"));

        // ── Save epoch losses ──────────────────────────────────
        var losses = result.Epochs.Select(e => double.CreateChecked(e.Loss)).ToList();
        var lossesJson = JsonSerializer.Serialize(losses, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.Combine(OutDir, "epoch_losses_nivara.json"), lossesJson);

        // ── Inference on test set ──────────────────────────────
        model.Eval();
        var testLoader = new DataLoader<float>(
            new TensorDataset<float>(testFrame, FeatureCols, "is_fraud"),
            batchSize: testFrame.RowCount, shuffle: false);

        var batch = testLoader.First();
        var logits = model.Forward(batch.Features);

        var logitArr = new float[logits.Length];
        logits.Data.CopyTo(logitArr, 0.0f);

        var probArr = new float[logitArr.Length];
        for (int i = 0; i < probArr.Length; i++)
            probArr[i] = 1.0f / (1.0f + MathF.Exp(-logitArr[i]));

        var lines = new List<string> { "logit,prob" };
        for (int i = 0; i < logitArr.Length; i++)
            lines.Add($"{logitArr[i]:F6},{probArr[i]:F6}");
        File.WriteAllLines(Path.Combine(OutDir, "test_preds_nivara.csv"), lines);

        Console.WriteLine("Saved test predictions to test_preds_nivara.csv");
        Console.WriteLine($"First 5 logits: {string.Join(", ", logitArr.Take(5).Select(x => $"{x:F4}"))}");
        Console.WriteLine($"First 5 probs:  {string.Join(", ", probArr.Take(5).Select(x => $"{x:F4}"))}");
        Console.WriteLine();
        Console.WriteLine("=== Cross-Framework FraudNet Complete ===");
    }

    static NivaraFrame LoadCsv(string path)
    {
        var lines = File.ReadAllLines(path);
        var header = lines[0].Split(',');
        var rows = lines.Skip(1).Select(l => l.Split(',')).ToArray();

        var columns = new List<(string Name, Nivara.IColumn Column)>();
        for (int i = 0; i < header.Length; i++)
        {
            var data = rows.Select(r => float.Parse(r[i])).ToArray();
            columns.Add((header[i].Trim(), NivaraColumn<float>.Create(data)));
        }
        return NivaraFrame.Create(columns.ToArray());
    }

    static NormParams LoadNormParams(string path)
    {
        var json = File.ReadAllText(path);
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        return new NormParams(
            root.GetProperty("means").EnumerateArray().Select(x => x.GetSingle()).ToArray(),
            root.GetProperty("stds").EnumerateArray().Select(x => x.GetSingle()).ToArray());
    }

    static NivaraFrame Normalize(NivaraFrame frame, NormParams norm, string[] featureCols)
    {
        var columns = new List<(string Name, Nivara.IColumn Column)>(frame.ColumnCount);
        foreach (var name in frame.ColumnNames)
        {
            var col = frame.GetColumn<float>(name);
            int idx = Array.IndexOf(featureCols, name);
            if (idx >= 0)
            {
                var data = new float[col.Length];
                col.CopyTo(data, 0.0f);
                float mean = norm.Means[idx], std = norm.Stds[idx];
                for (int i = 0; i < data.Length; i++)
                    data[i] = (data[i] - mean) / std;
                columns.Add((name, NivaraColumn<float>.Create(data)));
            }
            else
            {
                columns.Add((name, col));
            }
        }
        return NivaraFrame.Create(columns.ToArray());
    }

    static void SaveWeightsToJson(Module<float> model, string path)
    {
        var parameters = model.Parameters();

        using var stream = new MemoryStream();
        using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });

        var layerMap = new (string nivaraPrefix, string pytorchPrefix)[]
        {
            ("Module_0.", "L1."),
            ("Module_1.", "L2."),
            ("Module_2.", "L3."),
        };

        writer.WriteStartObject();
        foreach (var (nivPrefix, pytPrefix) in layerMap)
        {
            foreach (var paramName in new[] { "Weight", "Bias" })
            {
                var key = nivPrefix + paramName;
                if (!parameters.TryGetValue(key, out var tensor))
                    continue;

                var data = new float[tensor.Length];
                tensor.Data.CopyTo(data, 0.0f);

                writer.WritePropertyName(pytPrefix + paramName);
                var shape = tensor.Shape;
                if (shape.Length == 2)
                {
                    writer.WriteStartArray();
                    for (int r = 0; r < shape[0]; r++)
                    {
                        writer.WriteStartArray();
                        for (int c = 0; c < shape[1]; c++)
                            writer.WriteNumberValue(data[r * shape[1] + c]);
                        writer.WriteEndArray();
                    }
                    writer.WriteEndArray();
                }
                else
                {
                    writer.WriteStartArray();
                    foreach (var v in data) writer.WriteNumberValue(v);
                    writer.WriteEndArray();
                }
            }
        }
        writer.WriteEndObject();
        writer.Flush();
        File.WriteAllBytes(path, stream.ToArray());
    }

    record NormParams(float[] Means, float[] Stds);
}
