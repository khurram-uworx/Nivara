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
        tokenizer.Save(Path.Combine(saveDir, "intent_tokenizer.json"));
        Console.WriteLine($"  Saved to {saveDir}/");

        return (model, tokenizer);
    }
}