using Nivara;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Training;
using Nivara.Metrics;
using Nivara.Samples;
using NivaraChat.Data;

namespace NivaraChat.Training;

public static class IntentTrainer
{
    public static (TextClassifierModel<float> model, TextTokenizer tokenizer) Train(
        int epochs = 20, int batchSize = 32, int numSamples = 1000, string saveDir = "models", int seed = 42)
    {
        var (texts, labels) = IntentDataGenerator.GenerateIntentData(numSamples, seed);
        var tokenizer = TextTokenizer.FromDocuments(texts, maxVocabSize: 5000);

        int maxSeqLen = 20;
        int trainCount = (int)(texts.Length * 0.8);
        var trainTexts = texts.AsSpan(0, trainCount).ToArray();
        var trainLabels = labels.AsSpan(0, trainCount).ToArray();
        var testTexts = texts.AsSpan(trainCount).ToArray();
        var testLabels = labels.AsSpan(trainCount).ToArray();

        var trainTokens = new int[trainCount * maxSeqLen];
        for (int i = 0; i < trainCount; i++)
        {
            var encoded = tokenizer.Encode(trainTexts[i], fixedLength: maxSeqLen);
            Array.Copy(encoded, 0, trainTokens, i * maxSeqLen, maxSeqLen);
        }

        using var model = new TextClassifierModel<float>(tokenizer.VocabSize, 32, 64, numClasses: 5, maxSeqLen);
        using var optimizer = new Adam<float>(learningRate: 0.001f);
        optimizer.AddParameterGroup(model.GetParameters().Values);

        var lossFn = new CrossEntropyLoss<float>();
        var trainFrame = FrameBuilder.BuildDocumentClassificationFrame(trainTokens, trainLabels, trainCount, maxSeqLen);
        var featureColumns = Enumerable.Range(0, maxSeqLen).Select(d => $"tok_{d}").ToArray();
        var trainDataset = new TensorDataset<float>(trainFrame, featureColumns, ["label"]);
        var trainLoader = new DataLoader<float>(trainDataset, batchSize, shuffle: true, seed: seed);

        Console.WriteLine($"  Training intent classifier: {trainCount} samples, {tokenizer.VocabSize} vocab, 5 classes");

        var trainLoop = new TrainingLoop<float>(
            model, trainLoader,
            (logits, lbls) =>
            {
                int bs = logits.Length / 5;
                var targets = new int[bs];
                for (int i = 0; i < bs; i++)
                    targets[i] = int.CreateChecked(lbls.Data[i]);
                return lossFn.Forward(logits, targets);
            },
            optimizer, epochs);

        var result = trainLoop.Run();

        var predictedLabels = new int[testLabels.Length];
        var actualLabels = (int[])testLabels.Clone();
        for (int i = 0; i < testLabels.Length; i++)
        {
            var encoded = tokenizer.Encode(testTexts[i], fixedLength: maxSeqLen);
            var preds = model.Predict(encoded);
            predictedLabels[i] = preds[0];
        }

        var perClassFrame = NivaraMetrics.PerClassMetricsFrame(
            NivaraColumn<int>.Create(predictedLabels),
            NivaraColumn<int>.Create(actualLabels));

        long correct = 0;
        for (int r = 0; r < perClassFrame.RowCount; r++)
            correct += perClassFrame.GetColumn<long>("Tp")[r];

        var classRowByLabel = new Dictionary<int, int>(perClassFrame.RowCount);
        var classLabels = perClassFrame.GetColumn<int>("Class");
        var tpColumn = perClassFrame.GetColumn<long>("Tp");
        var fnColumn = perClassFrame.GetColumn<long>("Fn");
        for (int r = 0; r < perClassFrame.RowCount; r++)
            classRowByLabel[classLabels[r]] = r;

        Console.WriteLine($"  Test accuracy: {(double)correct / testLabels.Length:P1}");
        for (int c = 0; c < 5; c++)
        {
            string intent = c switch { 0 => "factual", 1 => "question", 2 => "command", 3 => "complaint", _ => "chitchat" };
            if (classRowByLabel.TryGetValue(c, out int row))
            {
                long classCorrect = tpColumn[row];
                long classTotal = tpColumn[row] + fnColumn[row];
                double acc = classTotal > 0 ? (double)classCorrect / classTotal : 0;
                Console.WriteLine($"    {intent}: {acc:P1} ({classCorrect}/{classTotal})");
            }
            else
            {
                Console.WriteLine($"    {intent}: {0.0:P1} (0/0)");
            }
        }

        Directory.CreateDirectory(saveDir);
        ModelSerializer.Save(model, Path.Combine(saveDir, "intent_model.json"));
        tokenizer.Save(Path.Combine(saveDir, "intent_tokenizer.json"));
        trainLoop.SaveCheckpoint(
            Path.Combine(saveDir, "intent_checkpoint.json"),
            result.Epochs[^1].Epoch, result.Epochs[^1].Loss);
        Console.WriteLine($"  Saved to {saveDir}/");

        return (model, tokenizer);
    }

    public static (TextClassifierModel<float> model, TextTokenizer tokenizer) TrainIncremental(
        string[] feedbackTexts, int[] feedbackLabels,
        string modelsDir, int additionalEpochs = 5, int batchSize = 16)
    {
        var checkpointPath = Path.Combine(modelsDir, "intent_checkpoint.json");
        bool hasCheckpoint = File.Exists(checkpointPath);

        var tokenizer = TextTokenizer.Load(Path.Combine(modelsDir, "intent_tokenizer.json"));
        var model = new TextClassifierModel<float>(tokenizer.VocabSize, 32, 64, numClasses: 5, maxSeqLen: 20);
        if (hasCheckpoint)
        {
            Console.WriteLine("  Loading intent checkpoint (weights + optimizer state).");
        }
        else
        {
            ModelSerializer.Load(model, Path.Combine(modelsDir, "intent_model.json"));
            Console.WriteLine("  No checkpoint found; warm-starting weights with a fresh optimizer.");
        }

        int maxSeqLen = 20;
        var allTexts = new List<string>();
        var allLabels = new List<int>();

        var (origTexts, origLabels) = IntentDataGenerator.GenerateIntentData(500, seed: 42);
        allTexts.AddRange(origTexts);
        allLabels.AddRange(origLabels);

        allTexts.AddRange(feedbackTexts);
        allLabels.AddRange(feedbackLabels);

        var allTokens = new int[allTexts.Count * maxSeqLen];
        for (int i = 0; i < allTexts.Count; i++)
        {
            var encoded = tokenizer.Encode(allTexts[i], fixedLength: maxSeqLen);
            Array.Copy(encoded, 0, allTokens, i * maxSeqLen, maxSeqLen);
        }

        var frame = FrameBuilder.BuildDocumentClassificationFrame(allTokens, allLabels.ToArray(), allTexts.Count, maxSeqLen);
        var featureColumns = Enumerable.Range(0, maxSeqLen).Select(d => $"tok_{d}").ToArray();
        var dataset = new TensorDataset<float>(frame, featureColumns, ["label"]);
        var loader = new DataLoader<float>(dataset, batchSize, shuffle: true, seed: 42);

        using var optimizer = new Adam<float>(learningRate: 0.0005f);
        optimizer.AddParameterGroup(model.GetParameters().Values);

        var lossFn = new CrossEntropyLoss<float>();
        var loop = new TrainingLoop<float>(
            model, loader,
            (logits, lbls) =>
            {
                int bs = logits.Length / 5;
                var targets = new int[bs];
                for (int i = 0; i < bs; i++)
                    targets[i] = int.CreateChecked(lbls.Data[i]);
                return lossFn.Forward(logits, targets);
            },
            optimizer, additionalEpochs);

        var result = hasCheckpoint
            ? ResumeFromCheckpoint(loop, checkpointPath, additionalEpochs)
            : loop.Run();

        var lastLoss = result.Epochs[^1].Loss;
        Console.WriteLine($"  Incremental training complete: {result.Epochs.Count} epochs, final loss {lastLoss:F4}");

        ModelSerializer.Save(model, Path.Combine(modelsDir, "intent_model.json"));
        loop.SaveCheckpoint(checkpointPath, result.Epochs[^1].Epoch, result.Epochs[^1].Loss);
        Console.WriteLine($"  Saved updated model to {modelsDir}/");

        model.Eval();
        return (model, tokenizer);
    }

    static TrainingResult<float> ResumeFromCheckpoint(
        TrainingLoop<float> loop, string checkpointPath, int additionalEpochs)
    {
        loop.LoadCheckpoint(checkpointPath);
        return loop.Continue(additionalEpochs);
    }
}
