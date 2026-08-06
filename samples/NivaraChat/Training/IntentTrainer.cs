using Nivara;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Training;
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

        int correct = 0;
        var classCorrect = new int[5];
        var classTotal = new int[5];
        for (int i = 0; i < testLabels.Length; i++)
        {
            var encoded = tokenizer.Encode(testTexts[i], fixedLength: maxSeqLen);
            var preds = model.Predict(encoded);
            classTotal[testLabels[i]]++;
            if (preds[0] == testLabels[i])
            {
                correct++;
                classCorrect[testLabels[i]]++;
            }
        }
        Console.WriteLine($"  Test accuracy: {(double)correct / testLabels.Length:P1}");
        for (int c = 0; c < 5; c++)
        {
            string intent = c switch { 0 => "factual", 1 => "question", 2 => "command", 3 => "complaint", _ => "chitchat" };
            double acc = classTotal[c] > 0 ? (double)classCorrect[c] / classTotal[c] : 0;
            Console.WriteLine($"    {intent}: {acc:P1} ({classCorrect[c]}/{classTotal[c]})");
        }

        Directory.CreateDirectory(saveDir);
        ModelSerializer.Save(model, Path.Combine(saveDir, "intent_model.json"));
        trainLoop.SaveCheckpoint(
            Path.Combine(saveDir, "intent_checkpoint.json"),
            result.Epochs[^1].Epoch, result.Epochs[^1].Loss);
        tokenizer.Save(Path.Combine(saveDir, "intent_tokenizer.json"));
        Console.WriteLine($"  Saved to {saveDir}/ (model + checkpoint with optimizer state)");

        return (model, tokenizer);
    }

    public static (TextClassifierModel<float> model, TextTokenizer tokenizer) TrainIncremental(
        string[] feedbackTexts, int[] feedbackLabels,
        string modelsDir, int additionalEpochs = 5, int batchSize = 16)
    {
        var tokenizer = TextTokenizer.Load(Path.Combine(modelsDir, "intent_tokenizer.json"));
        var model = new TextClassifierModel<float>(tokenizer.VocabSize, 32, 64, numClasses: 5, maxSeqLen: 20);

        var checkpointPath = Path.Combine(modelsDir, "intent_checkpoint.json");
        bool hasCheckpoint = File.Exists(checkpointPath);
        var checkpoint = hasCheckpoint ? ModelSerializer.LoadCheckpoint<float>(checkpointPath) : null;

        if (hasCheckpoint)
            Console.WriteLine($"  Loaded checkpoint: epoch {checkpoint!.Epoch}, loss {checkpoint.Loss:F4}.");
        else
        {
            ModelSerializer.Load(model, Path.Combine(modelsDir, "intent_model.json"));
            Console.WriteLine("  Checkpoint not found — loaded model weights, starting fresh optimizer.");
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
            optimizer,
            epochs: hasCheckpoint ? checkpoint!.Epoch : additionalEpochs);

        TrainingResult<float> result;
        if (hasCheckpoint)
        {
            loop.LoadCheckpoint(checkpointPath);
            result = loop.Continue(additionalEpochs);
        }
        else
        {
            result = loop.Run();
        }

        var lastLoss = result.Epochs[^1].Loss;
        Console.WriteLine($"  Incremental training complete: {result.Epochs.Count} epochs, final loss {lastLoss:F4}");

        loop.SaveCheckpoint(checkpointPath, result.Epochs[^1].Epoch, lastLoss);
        ModelSerializer.Save(model, Path.Combine(modelsDir, "intent_model.json"));
        Console.WriteLine($"  Saved updated model + checkpoint to {modelsDir}/");

        model.Eval();
        return (model, tokenizer);
    }
}